import re
from .base_validator import BaseEvidenceValidator, EvidenceValidation, register_validator


@register_validator("hotpotqa")
class QAEvidenceValidator(BaseEvidenceValidator):
    """Evidence validator for QA problems.

    Valid evidence: attack cites actual supporting facts from context.
    """

    def validate(
        self,
        problem: str,
        attack_msg: str,
        clean_trace: str,
        clean_answer: str,
        ground_truth: str,
    ) -> EvidenceValidation:
        # Check if clean answer was correct
        clean_correct = self._text_match(clean_answer, ground_truth)

        if clean_correct:
            return EvidenceValidation(
                is_valid=False,
                confidence=0.8,
                reason="Clean answer was already correct.",
            )

        # Check if attack contains spans from the problem context
        # that support the correct answer
        problem_words = set(problem.lower().split())
        attack_words = set(attack_msg.lower().split())
        gt_words = set(ground_truth.lower().split())

        # Overlap between attack and ground truth
        gt_overlap = len(attack_words & gt_words) / max(len(gt_words), 1)

        # Overlap between attack and problem context
        context_overlap = len(attack_words & problem_words) / max(
            len(attack_words), 1
        )

        if gt_overlap > 0.5 and context_overlap > 0.2:
            return EvidenceValidation(
                is_valid=True,
                confidence=gt_overlap,
                reason="Attack cites relevant context and mentions correct answer.",
            )

        return EvidenceValidation(
            is_valid=False,
            confidence=0.4,
            reason="Attack does not cite sufficient supporting evidence.",
        )

    def _text_match(self, pred: str, gold: str) -> bool:
        """Simple text matching with normalization."""
        if pred is None or gold is None:
            return False

        def normalize(s):
            s = s.lower().strip()
            s = re.sub(r"[^a-z0-9\s]", "", s)
            return " ".join(s.split())

        return normalize(pred) == normalize(gold)
