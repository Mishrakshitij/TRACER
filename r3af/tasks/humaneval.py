import logging
from typing import List, Optional

from .base import BaseTask, TaskExample, TaskRegistry

logger = logging.getLogger(__name__)


@TaskRegistry.register("humaneval")
class HumanEvalTask(BaseTask):
    """HumanEval code generation task."""

    def load_data(
        self, split: str = "test", max_samples: Optional[int] = None
    ) -> List[TaskExample]:
        from datasets import load_dataset

        ds = load_dataset("openai_humaneval", split=split)

        examples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            examples.append(
                TaskExample(
                    id=f"humaneval_{item['task_id']}",
                    query=item["prompt"],
                    ground_truth=item["canonical_solution"],
                    dataset="humaneval",
                    metadata={
                        "test": item["test"],
                        "entry_point": item["entry_point"],
                        "task_id": item["task_id"],
                    },
                )
            )

        logger.info(f"Loaded {len(examples)} HumanEval examples ({split})")
        return examples
