import logging
from typing import List, Optional

from .base import BaseTask, TaskExample, TaskRegistry

logger = logging.getLogger(__name__)


@TaskRegistry.register("mbpp")
class MBPPTask(BaseTask):
    """MBPP (Mostly Basic Python Problems) code task."""

    def load_data(
        self, split: str = "train", max_samples: Optional[int] = None
    ) -> List[TaskExample]:
        from datasets import load_dataset

        ds = load_dataset("mbpp", split=split)

        examples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            examples.append(
                TaskExample(
                    id=f"mbpp_{item['task_id']}",
                    query=item["text"],
                    ground_truth=item["code"],
                    dataset="mbpp",
                    metadata={
                        "test_list": item.get("test_list", []),
                        "task_id": item["task_id"],
                    },
                )
            )

        logger.info(f"Loaded {len(examples)} MBPP examples ({split})")
        return examples
