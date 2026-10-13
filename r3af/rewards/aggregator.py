import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch

from .base_reward import RewardOutput
from .answer_reward import AnswerReward
from .prm_reward import PRMReward
from .anchor_reward import AnchorReward
from .corrigibility_reward import CorrigibilityReward
from .length_reward import LengthReward
from ..anchors.extractor import AnchorExtractor
from ..anchors.nli_scorer import NLIScorer

logger = logging.getLogger(__name__)


@dataclass
class R3AFAggregatedRewardOutput:
    """Aggregated reward output for GDPO/GRPO."""

    component_rewards: torch.Tensor  # [batch, 5]
    combined_rewards: torch.Tensor  # [batch]
    advantages: torch.Tensor  # [batch]
    component_names: List[str] = field(
        default_factory=lambda: [
            "R_prm", "R_anchor", "R_corrigible", "R_ans", "R_len"
        ]
    )
    metrics: Dict[str, float] = field(default_factory=dict)


class R3AFRewardAggregator:
    """Aggregator for all 5 R3AF reward components.

    Owns shared PRM + NLI models. Computes all 5 components and returns
    component_rewards [batch, 5] for GDPO and advantages [batch] for GRPO.
    """

    def __init__(self, config):
        self.config = config
        self.weights = config.get_reward_weights()

        # Shared models
        self.nli_scorer = NLIScorer(
            model_name=config.nli_model_name,
            entailment_threshold=config.nli_entailment_threshold,
            contradiction_threshold=config.nli_contradiction_threshold,
        )
        self.anchor_extractor = AnchorExtractor(
            threshold=config.prm_threshold
        )

        # Initialize reward components
        self.prm_reward = PRMReward(prm_model_name=config.prm_model_name)
        self.anchor_reward = AnchorReward(
            extractor=self.anchor_extractor,
            nli_scorer=self.nli_scorer,
        )
        self.corrigibility_reward = CorrigibilityReward()
        self.answer_reward = AnswerReward()
        self.length_reward = LengthReward()

        self.component_names = [
            "R_prm", "R_anchor", "R_corrigible", "R_ans", "R_len"
        ]

    def compute(
        self,
        transcripts: list,
        group_size: Optional[int] = None,
        loss_type: str = "gdpo",
    ) -> R3AFAggregatedRewardOutput:
        """Compute all 5 reward components for a batch of transcripts.

        Args:
            transcripts: List of Transcript objects.
            group_size: Group size for GRPO normalization.
            loss_type: "grpo" or "gdpo".

        Returns:
            R3AFAggregatedRewardOutput with component_rewards, combined, advantages.
        """
        batch_size = len(transcripts)

        # Compute PRM scores first (needed by anchor reward)
        prm_output = self.prm_reward.compute(transcripts)
        prm_scores_list = prm_output.metadata.get("step_scores", None)

        # Compute all components
        anchor_output = self.anchor_reward.compute(
            transcripts, prm_scores_list=prm_scores_list
        )
        corrigibility_output = self.corrigibility_reward.compute(transcripts)
        answer_output = self.answer_reward.compute(transcripts)
        length_output = self.length_reward.compute(transcripts)

        # Stack into [batch, 5]
        component_rewards = torch.tensor(
            [
                prm_output.values,
                anchor_output.values,
                corrigibility_output.values,
                answer_output.values,
                length_output.values,
            ],
            dtype=torch.float,
        ).T  # [batch, 5]

        # Weighted sum
        weight_tensor = torch.tensor(
            [
                self.weights["R_prm"],
                self.weights["R_anchor"],
                self.weights["R_corrigible"],
                self.weights["R_ans"],
                self.weights["R_len"],
            ],
            dtype=torch.float,
        )

        combined = (component_rewards * weight_tensor.unsqueeze(0)).sum(dim=-1)

        # Compute advantages
        if loss_type == "grpo":
            advantages = self._group_normalize(combined, group_size)
        else:
            advantages = combined  # GDPO handles normalization in loss

        # Collect metrics
        metrics = self._collect_metrics(
            component_rewards, combined, prm_output, anchor_output,
            corrigibility_output, answer_output, length_output,
        )

        return R3AFAggregatedRewardOutput(
            component_rewards=component_rewards,
            combined_rewards=combined,
            advantages=advantages,
            component_names=self.component_names,
            metrics=metrics,
        )

    def _group_normalize(
        self,
        rewards: torch.Tensor,
        group_size: Optional[int],
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Normalize rewards within groups for GRPO."""
        if group_size is None or group_size <= 1:
            mean = rewards.mean()
            std = rewards.std().clamp(min=eps)
            return (rewards - mean) / std

        batch_size = rewards.shape[0]
        normalized = torch.zeros_like(rewards)

        for g in range(0, batch_size, group_size):
            end = min(g + group_size, batch_size)
            group = rewards[g:end]
            if group.numel() > 1:
                mean = group.mean()
                std = group.std().clamp(min=eps)
                normalized[g:end] = (group - mean) / std
            else:
                normalized[g:end] = 0.0

        return normalized

    def _collect_metrics(
        self, component_rewards, combined, *outputs
    ) -> Dict[str, float]:
        """Collect logging metrics from all reward components."""
        metrics = {}

        with torch.no_grad():
            for i, name in enumerate(self.component_names):
                col = component_rewards[:, i]
                metrics[f"reward/{name}_mean"] = col.mean().item()
                metrics[f"reward/{name}_std"] = col.std().item() if col.numel() > 1 else 0.0

            metrics["reward/combined_mean"] = combined.mean().item()
            metrics["reward/combined_std"] = combined.std().item() if combined.numel() > 1 else 0.0

        # Add answer accuracy from answer_output
        for output in outputs:
            if hasattr(output, "metadata") and isinstance(output.metadata, dict):
                if "accuracy" in output.metadata:
                    metrics["reward/accuracy"] = output.metadata["accuracy"]

        return metrics
