from abc import ABC, abstractmethod
from typing import Optional


class BaseVerifier(ABC):
    """Abstract base class for answer verification."""

    @abstractmethod
    def verify(self, pred: str, gold: str) -> bool:
        """Verify if prediction matches ground truth.

        Args:
            pred: Predicted answer string.
            gold: Ground truth answer string.

        Returns:
            True if prediction is correct.
        """
        ...

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from model output text.

        Args:
            text: Full model output.

        Returns:
            Extracted answer string, or None if not found.
        """
        return text.strip() if text else None
