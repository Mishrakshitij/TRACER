from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Type


@dataclass
class TaskExample:
    """A single task example."""

    id: str
    query: str
    ground_truth: str
    dataset: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskRegistry:
    """Registry for task types."""

    _registry: Dict[str, Type["BaseTask"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(task_cls):
            cls._registry[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> "BaseTask":
        if name not in cls._registry:
            raise ValueError(
                f"Unknown task: {name}. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]()

    @classmethod
    def list_tasks(cls):
        return list(cls._registry.keys())


class BaseTask(ABC):
    """Abstract base class for task datasets."""

    @abstractmethod
    def load_data(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[TaskExample]:
        """Load task examples.

        Args:
            split: Dataset split ("train", "test", "validation").
            max_samples: Maximum number of samples to load.

        Returns:
            List of TaskExample objects.
        """
        ...
