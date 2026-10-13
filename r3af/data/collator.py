from typing import List, Dict, Any
import torch
from .prompt_templates import RETAINER_SYSTEM_PROMPT, get_system_prompt


class R3AFCollator:
    """Collator for R3AF multi-turn aware batching.

    Left-pads for generation, carries metadata through.
    """

    def __init__(self, tokenizer, max_prompt_length: int = 512, framing: str = "followup"):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.framing = framing

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = [item["query"] for item in batch]
        ground_truths = [item["ground_truth"] for item in batch]
        datasets = [item["dataset"] for item in batch]
        ids = [item["id"] for item in batch]

        # Format as chat messages
        framing = self.framing[0] if isinstance(self.framing, list) else self.framing
        system_prompt = get_system_prompt(framing)
        prompts = []
        for query in queries:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"System: {system_prompt}\nUser: {query}\nAssistant:"
            prompts.append(prompt)

        # Tokenize with left padding for generation
        self.tokenizer.padding_side = "left"
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "ground_truths": ground_truths,
            "datasets": datasets,
            "ids": ids,
            "queries": queries,
        }
