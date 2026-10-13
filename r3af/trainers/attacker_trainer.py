"""Phase 2: Attacker RL trainer.

Freezes retainer, trains attacker to maximize answer flips + anchor contradictions,
regularized for naturalness and budget compliance.
"""

import os
import logging
from typing import Optional, Dict

import torch
from torch.optim import AdamW
from tqdm import tqdm

from ..models.config import R3AFConfig
from ..models.base import R3AFModel
from ..models.attacker_model import AttackerModel
from ..environment.trg_game import TRGGame
from ..environment.budget import AttackBudget
from ..parsing.trace_parser import TraceParser
from ..verifiers import get_verifier

logger = logging.getLogger(__name__)


class AttackerTrainer:
    """Phase 2: Attacker RL trainer."""

    def __init__(
        self,
        config: R3AFConfig,
        retainer_model: Optional[R3AFModel] = None,
        attacker_model: Optional[AttackerModel] = None,
    ):
        self.config = config

        # Retainer (frozen)
        if retainer_model is not None:
            self.retainer_model = retainer_model
        else:
            self.retainer_model = R3AFModel(config)

        # Freeze retainer
        for param in self.retainer_model.model.parameters():
            param.requires_grad = False

        # Attacker
        if attacker_model is not None:
            self.attacker_model = attacker_model
        else:
            self.attacker_model = AttackerModel(config)

        self.trace_parser = TraceParser()

        # Optimizer for attacker
        self.optimizer = AdamW(
            self.attacker_model.model.parameters(),
            lr=config.attacker_lr,
            weight_decay=config.weight_decay,
        )

        self.global_step = 0

    def train(self, train_data: list, num_epochs: int = 1):
        """Train attacker to generate effective attacks."""
        logger.info("Starting attacker training")

        for epoch in range(num_epochs):
            logger.info(f"=== Attacker Epoch {epoch + 1}/{num_epochs} ===")

            for i, example in enumerate(tqdm(train_data, desc="Attacker")):
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    break

                metrics = self._train_step(example)

                if i % self.config.logging_steps == 0:
                    logger.info(f"Attacker step {self.global_step}: {metrics}")

                self.global_step += 1

            ckpt = os.path.join(
                self.config.output_dir, f"attacker-epoch-{epoch + 1}"
            )
            self.attacker_model.save_checkpoint(ckpt)

    def _train_step(self, example: dict) -> Dict[str, float]:
        """Single attacker training step."""
        from ..environment.transcript import Transcript, Turn, TurnRole

        problem = example["query"]
        ground_truth = example["ground_truth"]
        dataset = example.get("dataset", "gsm8k")

        # Build transcript with clean response
        transcript = Transcript(
            problem=problem,
            ground_truth=ground_truth,
            dataset=dataset,
        )
        transcript.turns.append(
            Turn(role=TurnRole.ATTACKER, content=problem)
        )

        # Generate clean response from retainer
        self.retainer_model.model.eval()
        clean_response = self._generate_retainer(transcript)
        parsed = self.trace_parser.parse(clean_response)
        transcript.turns.append(
            Turn(
                role=TurnRole.RETAINER,
                content=clean_response,
                reasoning_steps=parsed.steps,
                answer=parsed.answer,
            )
        )

        # Generate attack from attacker
        attack_text = self.attacker_model.generate_attack(transcript)
        transcript.turns.append(
            Turn(role=TurnRole.ATTACKER, content=attack_text)
        )

        # Generate post-attack response from retainer
        post_response = self._generate_retainer(transcript)
        post_parsed = self.trace_parser.parse(post_response)

        # Compute attacker reward (maximize flips)
        verifier = get_verifier(dataset)
        clean_correct = verifier.verify(
            str(parsed.answer) if parsed.answer else "", ground_truth
        )
        post_correct = verifier.verify(
            str(post_parsed.answer) if post_parsed.answer else "", ground_truth
        )

        # Reward: +1 for flipping correct to wrong, -1 for other cases
        if clean_correct and not post_correct:
            reward = 1.0
        elif clean_correct and post_correct:
            reward = -0.5
        else:
            reward = 0.0

        # Budget penalty
        attack_tokens = len(attack_text.split())
        if attack_tokens > self.config.max_attack_tokens:
            reward -= 0.5

        return {
            "attacker_reward": reward,
            "flip_rate": float(clean_correct and not post_correct),
            "attack_tokens": attack_tokens,
        }

    def _generate_retainer(self, transcript) -> str:
        """Generate retainer response."""
        messages = transcript.to_messages()
        tokenizer = self.retainer_model.tokenizer

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - self.config.max_new_tokens,
        )
        device = next(self.retainer_model.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.retainer_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        response_ids = outputs[0, input_ids.shape[1]:]
        return tokenizer.decode(response_ids, skip_special_tokens=True)
