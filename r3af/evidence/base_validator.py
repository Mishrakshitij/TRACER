from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type, Optional


@dataclass
class EvidenceValidation:
    """Result of evidence validation."""

    is_valid: bool
    confidence: float = 0.0
    reason: str = ""


_VALIDATOR_REGISTRY: Dict[str, Type["BaseEvidenceValidator"]] = {}


def register_validator(dataset: str):
    """Decorator to register a validator for a dataset."""
    def decorator(cls):
        _VALIDATOR_REGISTRY[dataset] = cls
        return cls
    return decorator


def get_validator(dataset: str) -> "BaseEvidenceValidator":
    """Get the appropriate evidence validator for a dataset."""
    if dataset in _VALIDATOR_REGISTRY:
        return _VALIDATOR_REGISTRY[dataset]()
    # Default to math validator
    from .math_validator import MathEvidenceValidator
    return MathEvidenceValidator()


class BaseEvidenceValidator(ABC):
    """Abstract base class for evidence validation.

    Determines whether an adversarial follow-up contains valid evidence
    that should cause the model to change its answer.
    """

    @abstractmethod
    def validate(
        self,
        problem: str,
        attack_msg: str,
        clean_trace: str,
        clean_answer: str,
        ground_truth: str,
    ) -> EvidenceValidation:
        """Validate whether attack message contains valid evidence.

        Args:
            problem: Original problem text.
            attack_msg: The adversarial follow-up message.
            clean_trace: The model's clean reasoning trace.
            clean_answer: The model's clean answer.
            ground_truth: The correct answer.

        Returns:
            EvidenceValidation with is_valid, confidence, and reason.
        """
        ...
