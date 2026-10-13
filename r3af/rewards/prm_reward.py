import logging
from typing import Optional, List

import torch

from .base_reward import BaseRewardComponent, RewardOutput

logger = logging.getLogger(__name__)


class PRMReward(BaseRewardComponent):
    """R_PRM = (1/N) * sum_i V(s_i)

    Length-normalized average of step scores on the post-attack trace
    using a Process Reward Model (PRM).
    """

    def __init__(self, prm_model=None, prm_model_name: str = None):
        self._prm_model = prm_model
        self._prm_model_name = prm_model_name
        self._prm_tokenizer = None
        self._load_attempted = False

    def _load_prm(self):
        """Lazy load PRM model (called at most once)."""
        if self._load_attempted:
            return
        self._load_attempted = True

        if self._prm_model is not None:
            return

        if not self._prm_model_name:
            logger.info("No PRM model specified, using uniform scores")
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading PRM: {self._prm_model_name}")
            self._prm_tokenizer = AutoTokenizer.from_pretrained(self._prm_model_name)
            self._prm_model = AutoModelForSequenceClassification.from_pretrained(
                self._prm_model_name
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._prm_model = self._prm_model.to(device)
            self._prm_model.eval()
        except Exception as e:
            logger.warning(f"Could not load PRM model: {e}")

    def score_steps(self, problem: str, steps: list) -> List[float]:
        """Score individual reasoning steps using PRM.

        Args:
            problem: The problem text.
            steps: List of ReasoningStep objects.

        Returns:
            List of scores per step.
        """
        if not steps:
            return []

        if self._prm_model is None:
            self._load_prm()

        if self._prm_model is None:
            return [0.5] * len(steps)

        scores = []
        device = next(self._prm_model.parameters()).device

        for step in steps:
            text = f"{problem}\n{step.content}"
            inputs = self._prm_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = self._prm_model(**inputs)
                # Assume binary classification: [negative, positive]
                probs = torch.softmax(outputs.logits, dim=-1)
                score = probs[0, -1].item()  # positive class probability
            scores.append(score)

        return scores

    def compute(self, transcripts, **kwargs) -> RewardOutput:
        values = []
        all_step_scores = []

        for transcript in transcripts:
            post_trace = transcript.post_attack_trace
            if not post_trace:
                values.append(0.0)
                all_step_scores.append([])
                continue

            step_scores = self.score_steps(transcript.problem, post_trace.steps)
            all_step_scores.append(step_scores)

            # Length-normalized average
            avg_score = sum(step_scores) / max(len(step_scores), 1)
            values.append(avg_score)

        return RewardOutput(
            values=values,
            metadata={"step_scores": all_step_scores},
        )
