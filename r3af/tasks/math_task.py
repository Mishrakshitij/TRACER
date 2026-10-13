import re
import logging
from typing import List, Optional

from .base import BaseTask, TaskExample, TaskRegistry

logger = logging.getLogger(__name__)


@TaskRegistry.register("math")
class MATHTask(BaseTask):
    """MATH competition dataset with optional PRM800K process labels."""

    def load_data(
        self, split: str = "train", max_samples: Optional[int] = None
    ) -> List[TaskExample]:
        from datasets import load_dataset

        try:
            ds = load_dataset("hendrycks/competition_math", split=split)
        except Exception:
            ds = load_dataset("lighteval/MATH", split=split)

        examples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            answer = self._extract_answer(item.get("solution", item.get("answer", "")))
            examples.append(
                TaskExample(
                    id=f"math_{split}_{i}",
                    query=item["problem"],
                    ground_truth=answer,
                    dataset="math",
                    metadata={
                        "level": item.get("level", ""),
                        "type": item.get("type", ""),
                        "full_solution": item.get("solution", ""),
                    },
                )
            )

        logger.info(f"Loaded {len(examples)} MATH examples ({split})")
        return examples

    def _extract_answer(self, solution: str) -> str:
        """Extract answer from \\boxed{...}."""
        match = re.search(r"\\boxed\{([^}]*)\}", solution)
        if match:
            return match.group(1).strip()
        return solution.strip().split("\n")[-1].strip()
