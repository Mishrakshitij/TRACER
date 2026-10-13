from abc import ABC, abstractmethod
from typing import Dict, Type


class AttackRegistry:
    """Registry for attack types."""

    _registry: Dict[str, Type["BaseAttack"]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an attack class."""
        def decorator(attack_cls):
            cls._registry[name] = attack_cls
            return attack_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type["BaseAttack"]:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown attack: {name}. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_attacks(cls):
        return list(cls._registry.keys())


class BaseAttack(ABC):
    """Abstract base class for adversarial attacks."""

    @abstractmethod
    def generate(self, transcript) -> str:
        """Generate an adversarial follow-up message.

        Args:
            transcript: Current game transcript.

        Returns:
            Attack text string.
        """
        ...

    def is_valid_evidence(self) -> bool:
        """Whether this attack type provides valid evidence for correction.

        Returns True only for benign follow-ups that provide legitimate
        corrections or helpful information.
        """
        return False
