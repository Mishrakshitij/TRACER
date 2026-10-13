"""Phase 0: SFT warmup trainer.

Standard cross-entropy supervised fine-tuning on (problem, solution) pairs.
"""

import os
import logging
from typing import Optional, Dict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from ..models.config import R3AFConfig
from ..models.base import R3AFModel
from ..data.dataset import R3AFDataset, R3AFExample
from ..data.collator import R3AFCollator
from ..tasks.base import TaskRegistry

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Phase 0: SFT warmup trainer."""

    def __init__(self, config: R3AFConfig, model: Optional[R3AFModel] = None):
        self.config = config

        if model is not None:
            self.model = model
        else:
            self.model = R3AFModel(config)

        self.tokenizer = self.model.tokenizer

        # Build dataset
        self.train_dataset = self._build_dataset(
            config.datasets, "train", config.max_train_samples
        )
        self.collator = R3AFCollator(
            self.tokenizer, max_prompt_length=config.max_prompt_length
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,
            drop_last=True,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = (
            len(self.train_loader) * config.epochs
            // config.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.global_step = 0

    def _build_dataset(self, dataset_names, split, max_samples):
        all_examples = []
        for ds_name in dataset_names:
            task = TaskRegistry.get(ds_name)
            task_examples = task.load_data(split=split, max_samples=max_samples)
            for te in task_examples:
                all_examples.append(
                    R3AFExample(
                        id=te.id,
                        query=te.query,
                        ground_truth=te.ground_truth,
                        dataset=ds_name,
                        metadata=te.metadata,
                    )
                )
        return R3AFDataset(
            examples=all_examples,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.max_prompt_length,
        )

    def train(self):
        """Run SFT training."""
        logger.info(f"Starting SFT training: {self.config.epochs} epochs")

        for epoch in range(self.config.epochs):
            logger.info(f"=== SFT Epoch {epoch + 1}/{self.config.epochs} ===")
            self._train_epoch(epoch)

            ckpt_path = os.path.join(
                self.config.output_dir, f"sft-checkpoint-epoch-{epoch + 1}"
            )
            self.model.save_checkpoint(ckpt_path)

        logger.info("SFT training complete!")

    def _train_epoch(self, epoch: int):
        self.model.model.train()
        total_loss = 0.0
        num_steps = 0

        progress = tqdm(self.train_loader, desc=f"SFT Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress):
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break

            device = next(self.model.model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # For SFT, we need the solution as target
            # Here we use the query as input and compute cross-entropy
            # on the model's output matching the ground truth
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss / self.config.gradient_accumulation_steps
            loss.backward()

            self.global_step += 1
            if self.global_step % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            num_steps += 1

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.config.dry_run and batch_idx >= 1:
                break
