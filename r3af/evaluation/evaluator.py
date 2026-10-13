import logging
from typing import List, Optional
import numpy as np

from ..models.config import R3AFConfig
from ..models.base import R3AFModel
from ..environment.trg_game import TRGGame
from ..environment.budget import AttackBudget
from ..attacks.attack_mixer import AttackMixer
from ..parsing.trace_parser import TraceParser
from ..verifiers import get_verifier
from ..tasks.base import TaskRegistry
from .metrics import R3AFMetrics

logger = logging.getLogger(__name__)


class R3AFEvaluator:
    """R3AF evaluation pipeline.

    Plays TRG games on eval set, computes all metrics.
    """

    def __init__(self, config: R3AFConfig, model: R3AFModel):
        self.config = config
        self.model = model
        self.trace_parser = TraceParser()
        self.attack_mixer = AttackMixer(config)
        self.trg_game = TRGGame(
            retainer_model=model,
            attack_fn=self.attack_mixer,
            trace_parser=self.trace_parser,
            budget=AttackBudget(max_tokens=config.max_attack_tokens),
            config=config,
        )

    def evaluate(
        self,
        dataset_name: str = "gsm8k",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> R3AFMetrics:
        """Run evaluation on a dataset."""
        task = TaskRegistry.get(dataset_name)
        examples = task.load_data(split=split, max_samples=max_samples)
        verifier = get_verifier(dataset_name)

        logger.info(f"Evaluating on {len(examples)} {dataset_name} examples")

        self.model.model.eval()

        # Play TRG games
        transcripts = self.trg_game.play_batch(
            problems=[e.query for e in examples],
            ground_truths=[e.ground_truth for e in examples],
            datasets=[dataset_name] * len(examples),
            problem_ids=[e.id for e in examples],
        )

        # Compute metrics
        clean_correct = []
        post_correct = []
        attack_names = []
        length_inflations = []

        for transcript in transcripts:
            cc = verifier.verify(
                str(transcript.clean_answer) if transcript.clean_answer else "",
                transcript.ground_truth,
            )
            pc = verifier.verify(
                str(transcript.post_attack_answer)
                if transcript.post_attack_answer
                else "",
                transcript.ground_truth,
            )
            clean_correct.append(cc)
            post_correct.append(pc)

            attack_turn = transcript.attack_turn
            if attack_turn and attack_turn.metadata:
                attack_names.append(
                    attack_turn.metadata.get("attack_name", "unknown")
                )
            else:
                attack_names.append("unknown")

            clean_tokens = (
                transcript.clean_turn.token_count
                if transcript.clean_turn
                else 0
            )
            post_tokens = (
                transcript.post_attack_turn.token_count
                if transcript.post_attack_turn
                else 0
            )
            length_inflations.append(max(0, post_tokens - clean_tokens))

        n = len(transcripts)
        num_clean_correct = sum(clean_correct)
        num_post_correct = sum(post_correct)

        # ASR = fraction of correct answers flipped to wrong
        flips = sum(
            1 for cc, pc in zip(clean_correct, post_correct) if cc and not pc
        )
        asr = flips / max(num_clean_correct, 1)

        # Beneficial revisions
        beneficial = sum(
            1 for cc, pc in zip(clean_correct, post_correct) if not cc and pc
        )
        num_clean_wrong = n - num_clean_correct
        brr = beneficial / max(num_clean_wrong, 1)

        # Over-stability
        over_stable = sum(
            1 for cc, pc in zip(clean_correct, post_correct) if not cc and not pc
        )
        ose = over_stable / max(num_clean_wrong, 1)

        inflations = np.array(length_inflations) if length_inflations else np.array([0])

        metrics = R3AFMetrics(
            robust_accuracy=num_post_correct / max(n, 1),
            clean_accuracy=num_clean_correct / max(n, 1),
            attack_success_rate=asr,
            beneficial_revision_rate=brr,
            over_stability_error=ose,
            length_inflation_mean=float(inflations.mean()),
            length_inflation_p90=float(np.percentile(inflations, 90)),
            length_inflation_p95=float(np.percentile(inflations, 95)),
            num_samples=n,
            num_clean_correct=num_clean_correct,
            num_post_correct=num_post_correct,
        )

        # Per-attack metrics
        per_attack = {}
        for name in set(attack_names):
            indices = [i for i, a in enumerate(attack_names) if a == name]
            attack_flips = sum(
                1 for i in indices if clean_correct[i] and not post_correct[i]
            )
            attack_correct = sum(1 for i in indices if clean_correct[i])
            per_attack[name] = {
                "count": len(indices),
                "asr": attack_flips / max(attack_correct, 1),
            }
        metrics.per_attack_metrics = per_attack

        logger.info(f"Evaluation results: {metrics.summary()}")
        return metrics
