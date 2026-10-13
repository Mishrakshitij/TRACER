from dataclasses import dataclass
import math


@dataclass
class AttackBudget:
    """Budget constraints for adversarial attacks."""

    max_tokens: int = 512
    max_turns: int = 1

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.max_turns < 0:
            raise ValueError(f"max_turns must be non-negative, got {self.max_turns}")

    def is_within_budget(self, token_count: int) -> bool:
        """Check whether a token count is within budget."""
        return token_count <= self.max_tokens

    def validate(self, text: str) -> bool:
        """Check whether the attack text is within the token budget.

        Uses a simple heuristic: estimated tokens = len(text.split()) * 1.3
        """
        estimated_tokens = math.ceil(len(text.split()) * 1.3)
        return estimated_tokens <= self.max_tokens
