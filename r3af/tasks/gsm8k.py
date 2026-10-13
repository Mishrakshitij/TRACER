import re
import logging
from typing import List, Optional

from .base import BaseTask, TaskExample, TaskRegistry

logger = logging.getLogger(__name__)


@TaskRegistry.register("gsm8k")
class GSM8KTask(BaseTask):
    """GSM8K grade school math task."""

    def load_data(
        self, split: str = "train", max_samples: Optional[int] = None
    ) -> List[TaskExample]:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split=split)

        examples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            answer = self._extract_answer(item["answer"])
            examples.append(
                TaskExample(
                    id=f"gsm8k_{split}_{i}",
                    query=item["question"],
                    ground_truth=answer,
                    dataset="gsm8k",
                    metadata={"full_solution": item["answer"]},
                )
            )

        logger.info(f"Loaded {len(examples)} GSM8K examples ({split})")
        return examples

    def _extract_answer(self, solution: str) -> str:
        """Extract numeric answer after ####."""
        match = re.search(r"####\s*(.+)", solution)
        if match:
            return match.group(1).strip().replace(",", "")
        return solution.strip().split("\n")[-1].strip()
