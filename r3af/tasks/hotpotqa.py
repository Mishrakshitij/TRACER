import logging
from typing import List, Optional

from .base import BaseTask, TaskExample, TaskRegistry

logger = logging.getLogger(__name__)


@TaskRegistry.register("hotpotqa")
class HotpotQATask(BaseTask):
    """HotpotQA multi-hop QA with supporting facts."""

    def load_data(
        self, split: str = "train", max_samples: Optional[int] = None
    ) -> List[TaskExample]:
        from datasets import load_dataset

        ds = load_dataset("hotpot_qa", "fullwiki", split=split)

        examples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            # Build context from paragraphs
            context_parts = []
            for title, sentences in zip(
                item.get("context", {}).get("title", []),
                item.get("context", {}).get("sentences", []),
            ):
                context_parts.append(f"{title}: {''.join(sentences)}")
            context = "\n".join(context_parts)

            query = f"Context:\n{context}\n\nQuestion: {item['question']}"

            examples.append(
                TaskExample(
                    id=f"hotpotqa_{split}_{i}",
                    query=query,
                    ground_truth=item["answer"],
                    dataset="hotpotqa",
                    metadata={
                        "supporting_facts": item.get("supporting_facts", {}),
                        "type": item.get("type", ""),
                        "level": item.get("level", ""),
                    },
                )
            )

        logger.info(f"Loaded {len(examples)} HotpotQA examples ({split})")
        return examples
