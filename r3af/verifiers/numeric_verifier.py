import re
from typing import Optional

from .base_verifier import BaseVerifier


class NumericVerifier(BaseVerifier):
    """Numeric comparison verifier for GSM8K and MATH."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def verify(self, pred: str, gold: str) -> bool:
        pred_num = self._parse_number(pred)
        gold_num = self._parse_number(gold)

        if pred_num is not None and gold_num is not None:
            return abs(pred_num - gold_num) < self.tolerance

        # Fallback to string comparison
        return self._normalize(pred) == self._normalize(gold)

    def extract_answer(self, text: str) -> Optional[str]:
        # Try #### pattern
        match = re.search(r"####\s*(.+?)(?:\n|$)", text)
        if match:
            return match.group(1).strip()

        # Try boxed
        match = re.search(r"\\boxed\{([^}]*)\}", text)
        if match:
            return match.group(1).strip()

        # Try "answer is" pattern
        match = re.search(
            r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        return None

    def _parse_number(self, s: str) -> Optional[float]:
        if s is None:
            return None
        s = s.strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            return float(s)
        except ValueError:
            # Try extracting just the number
            match = re.search(r"-?\d+\.?\d*", s)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
            return None

    def _normalize(self, s: str) -> str:
        if s is None:
            return ""
        return s.strip().lower().replace(",", "").replace(" ", "")
