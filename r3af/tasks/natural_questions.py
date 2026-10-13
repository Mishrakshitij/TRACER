import logging
from typing import List, Optional

from .base import BaseTask, TaskExample, TaskRegistry

logger = logging.getLogger(__name__)


@TaskRegistry.register("nq")
class NaturalQuestionsTask(BaseTask):
    """Natural Questions (open-domain) task.

    Uses the nq_open subset: (question, answer_list) pairs without
    long document context — suitable for RAG-framed evaluation where
    retrieved documents are injected as adversarial context.
    """

    def load_data(
        self, split: str = "train", max_samples: Optional[int] = None
    ) -> List[TaskExample]:
        from datasets import load_dataset

        # nq_open has train and validation splits
        ds_split = "validation" if split in ("test", "validation") else "train"
        ds = load_dataset("nq_open", split=ds_split)

        examples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            # nq_open answers are a list of valid answer strings
            answers = item["answer"]
            if not answers:
                continue
            # Encode all valid answers as pipe-separated for QAVerifier
            ground_truth = "|".join(answers)

            examples.append(
                TaskExample(
                    id=f"nq_{ds_split}_{i}",
                    query=item["question"],
                    ground_truth=ground_truth,
                    dataset="nq",
                    metadata={
                        "all_answers": answers,
                    },
                )
            )

        logger.info(f"Loaded {len(examples)} NQ examples ({ds_split})")
        return examples
