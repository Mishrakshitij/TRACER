from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RewardOutput:
    """Output from a reward component computation."""

    values: List[float]  # One scalar per sample
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRewardComponent(ABC):
    """Abstract base class for R3AF reward components."""

    @abstractmethod
    def compute(
        self,
        transcripts: list,
        **kwargs,
    ) -> RewardOutput:
        """Compute reward for a batch of transcripts.

        Args:
            transcripts: List of Transcript objects.
            **kwargs: Additional arguments.

        Returns:
            RewardOutput with values and metadata.
        """
        ...
