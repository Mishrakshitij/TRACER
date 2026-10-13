from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from torch.utils.data import Dataset


@dataclass
class R3AFExample:
    """A single R3AF training example."""

    id: str
    query: str
    ground_truth: str
    dataset: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class R3AFDataset(Dataset):
    """Dataset for R3AF training."""

    def __init__(
        self,
        examples: List[R3AFExample],
        tokenizer=None,
        max_prompt_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "id": example.id,
            "query": example.query,
            "ground_truth": example.ground_truth,
            "dataset": example.dataset,
            "metadata": example.metadata,
        }
