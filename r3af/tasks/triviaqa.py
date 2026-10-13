import logging
from typing import List, Optional

from .base import BaseTask, TaskExample, TaskRegistry

logger = logging.getLogger(__name__)


@TaskRegistry.register("triviaqa")
class TriviaQATask(BaseTask):
    """TriviaQA reading comprehension task.

    Uses the 'rc' config (reading comprehension with evidence documents).
    Answers include aliases for robust verification.
    """

    def load_data(
        self, split: str = "train", max_samples: Optional[int] = None
    ) -> List[TaskExample]:
        from datasets import load_dataset

        ds = load_dataset("trivia_qa", "rc", split=split)

        examples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            answer_obj = item.get("answer", {})
            # Primary answer
            primary_answer = answer_obj.get("value", "")
            # All valid aliases
            aliases = answer_obj.get("aliases", [])
            all_answers = [primary_answer] + aliases if primary_answer else aliases
            if not all_answers:
                continue
            # Encode all valid answers as pipe-separated for QAVerifier
            ground_truth = "|".join(all_answers)

            examples.append(
                TaskExample(
                    id=f"triviaqa_{split}_{i}",
                    query=item["question"],
                    ground_truth=ground_truth,
                    dataset="triviaqa",
                    metadata={
                        "all_answers": all_answers,
                        "normalized_aliases": answer_obj.get(
                            "normalized_aliases", []
                        ),
                    },
                )
            )

        logger.info(f"Loaded {len(examples)} TriviaQA examples ({split})")
        return examples
