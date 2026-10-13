import re
from .base_validator import BaseEvidenceValidator, EvidenceValidation, register_validator


@register_validator("gsm8k")
@register_validator("math")
class MathEvidenceValidator(BaseEvidenceValidator):
    """Evidence validator for math problems.

    Valid evidence: attack suggests the correct answer AND clean was wrong.
    """

    def validate(
        self,
        problem: str,
        attack_msg: str,
        clean_trace: str,
        clean_answer: str,
        ground_truth: str,
    ) -> EvidenceValidation:
        # Extract numeric claims from attack
        attack_numbers = re.findall(r"-?\d+\.?\d*", attack_msg)

        # Check if clean answer was correct
        clean_correct = self._numeric_match(clean_answer, ground_truth)

        if clean_correct:
            # If clean was correct, attack evidence is invalid
            # (no need to change a correct answer)
            return EvidenceValidation(
                is_valid=False,
                confidence=0.9,
                reason="Clean answer was already correct.",
            )

        # Check if attack suggests the correct answer
        for num_str in attack_numbers:
            if self._numeric_match(num_str, ground_truth):
                return EvidenceValidation(
                    is_valid=True,
                    confidence=0.8,
                    reason=f"Attack suggests correct answer: {num_str}",
                )

        return EvidenceValidation(
            is_valid=False,
            confidence=0.6,
            reason="Attack does not suggest the correct answer.",
        )

    def _numeric_match(self, a: str, b: str, tol: float = 1e-6) -> bool:
        """Check if two strings represent the same number."""
        if a is None or b is None:
            return False
        try:
            return abs(float(a) - float(b)) < tol
        except (ValueError, TypeError):
            return str(a).strip() == str(b).strip()
