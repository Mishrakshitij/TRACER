"""Phases 1/3: Retainer RL trainer.

Core training loop:
1. Load batch of B prompts
2. Expand B -> B*G for group rollouts
3. Play TRG games: Turn 1 (generate clean) -> Turn 2 (attack) -> Turn 3 (generate post-attack)
4. Parse all traces
5. Compute all 5 rewards via aggregator -> component_rewards [B*G, 5]
6. Build Turn 3 sequences, compute log_probs (current, old, ref)
7. For TCR-KL: two extra forward passes, build committed_mask + gate
8. Compute combined loss
9. Backward + optimizer step
10. Log metrics to file + wandb
"""

import csv
import json
import os
import logging
import time
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from ..models.config import R3AFConfig
from ..models.base import R3AFModel
from ..environment.trg_game import TRGGame, BatchedTRGGame
from ..environment.budget import AttackBudget
from ..attacks.attack_mixer import AttackMixer
from ..parsing.trace_parser import TraceParser
from ..rewards.aggregator import R3AFRewardAggregator
from ..losses import LOSS_REGISTRY
from ..losses.tcr_kl_loss import TCRKLLoss
from ..data.dataset import R3AFDataset, R3AFExample
from ..data.collator import R3AFCollator
from ..data.prompt_templates import RETAINER_SYSTEM_PROMPT, get_system_prompt
from ..tasks.base import TaskRegistry
from ..verifiers import get_verifier
from ..evidence import get_evidence_validator

logger = logging.getLogger(__name__)


class RetainerTrainer:
    """Core R3AF retainer trainer for Phases 1 and 3."""

    def __init__(self, config: R3AFConfig, model: Optional[R3AFModel] = None):
        self.config = config

        # Model
        if model is not None:
            self.model = model
        else:
            self.model = R3AFModel(config)
        self.tokenizer = self.model.tokenizer

        # Loss function
        if config.loss_type == "combined":
            from ..losses.combined_loss import CombinedR3AFLoss

            # Resolve effective lambda_reg (backward compat with lambda_tcr)
            effective_lambda_reg = config.lambda_reg
            if config.lambda_tcr > 0 and config.reg_loss_type == "none":
                effective_lambda_reg = config.lambda_tcr

            self.loss_fn = CombinedR3AFLoss(
                beta=config.beta,
                epsilon=config.epsilon,
                policy_loss_type=config.policy_loss_type,
                reg_loss_type=config.reg_loss_type,
                lambda_reg=effective_lambda_reg,
                lambda_tcr=config.lambda_tcr,
                lambda_sdpo=config.lambda_sdpo,
                tcr_temperature=config.tcr_temperature,
                tcr_gate_mode=config.gate_mode,
                atrc_alpha=config.atrc_alpha,
            )
        else:
            loss_cls = LOSS_REGISTRY.get(config.loss_type)
            if loss_cls is None:
                raise ValueError(f"Unknown loss: {config.loss_type}")
            self.loss_fn = loss_cls(beta=config.beta, epsilon=config.epsilon)

        # Reward aggregator
        self.reward_aggregator = R3AFRewardAggregator(config)

        # Attack mixer
        self.attack_mixer = AttackMixer(config)

        # Trace parser
        self.trace_parser = TraceParser()

        # TRG Game (vLLM-accelerated or HF sequential)
        self.vllm_worker = None
        budget = AttackBudget(max_tokens=config.max_attack_tokens)

        if config.use_vllm_generation:
            from ..models.vllm_model import VLLMGenerationWorker

            logger.info(
                f"Initializing vLLM generation worker on GPU {config.vllm_gpu_id}"
            )
            self.vllm_worker = VLLMGenerationWorker(
                config, gpu_id=config.vllm_gpu_id
            )
            self.trg_game = BatchedTRGGame(
                retainer_model=self.model,
                attack_fn=self.attack_mixer,
                trace_parser=self.trace_parser,
                budget=budget,
                config=config,
                vllm_worker=self.vllm_worker,
            )
            # Export initial LoRA adapter so vLLM uses the same weights
            initial_adapter = self._export_lora_adapter("init")
            self.vllm_worker.update_lora(initial_adapter)
            logger.info("vLLM worker initialized with initial LoRA adapter")
        else:
            self.trg_game = TRGGame(
                retainer_model=self.model,
                attack_fn=self.attack_mixer,
                trace_parser=self.trace_parser,
                budget=budget,
                config=config,
            )

        # Dataset and data loader
        self.train_dataset = self._build_dataset(
            config.datasets, "train", config.max_train_samples
        )
        self.collator = R3AFCollator(
            self.tokenizer,
            max_prompt_length=config.max_prompt_length,
            framing=getattr(config, "attack_framing", "followup"),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,
            drop_last=True,
        )

        # Optimizer and scheduler
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

        # Accelerate (optional)
        try:
            from accelerate import Accelerator

            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )
            (
                self.model.model,
                self.optimizer,
                self.train_loader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model.model,
                self.optimizer,
                self.train_loader,
                self.scheduler,
            )
            self._use_accelerator = True
        except Exception as e:
            logger.warning(f"Could not init Accelerator: {e}")
            self._use_accelerator = False

        self.global_step = 0
        self._wandb_initialized = False

        # Resume from checkpoint if specified
        if getattr(config, "resume_from_checkpoint", None):
            self.model.load_checkpoint(config.resume_from_checkpoint)
            logger.info(f"Resumed from checkpoint: {config.resume_from_checkpoint}")

        # File-based logging
        self._log_file = None
        self._csv_writer = None
        self._csv_header_written = False
        self._init_log_file()

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

    def _expand_for_group(self, tensor: torch.Tensor) -> torch.Tensor:
        G = self.config.group_size
        return (
            tensor.unsqueeze(1)
            .expand(-1, G, -1)
            .reshape(-1, tensor.shape[-1])
        )

    def _expand_list(self, lst: list) -> list:
        G = self.config.group_size
        return [item for item in lst for _ in range(G)]

    def _update_beta(self, new_beta: float):
        """Update beta across all loss sub-components."""
        old_beta = self.loss_fn.beta
        if abs(new_beta - old_beta) < 1e-6:
            return
        self.loss_fn.beta = new_beta
        # Update sub-losses if combined
        if hasattr(self.loss_fn, "gdpo_loss"):
            self.loss_fn.gdpo_loss.beta = new_beta
        logger.info(f"Adaptive KL: beta {old_beta:.4f} -> {new_beta:.4f}")

    def _export_lora_adapter(self, name: str) -> str:
        """Save current PEFT adapter for the vLLM worker to load."""
        adapter_dir = os.path.join(
            self.config.output_dir, ".vllm_adapters", f"adapter_{name}"
        )
        os.makedirs(adapter_dir, exist_ok=True)
        self.model.model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)
        logger.info(f"Exported LoRA adapter to {adapter_dir}")
        return adapter_dir

    def _init_log_file(self):
        """Initialize CSV log file for per-step metrics."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        log_path = os.path.join(self.config.output_dir, "training_log.csv")
        self._log_file = open(log_path, "a", newline="", buffering=1)
        self._csv_writer = csv.writer(self._log_file)
        # Header written on first log call (once we know all metric keys)
        self._csv_header_written = os.path.getsize(log_path) > 0
        logger.info(f"Training log file: {log_path}")

    def _log_step_metrics(self, step: int, epoch: int, metrics: Dict[str, float]):
        """Write a row to the CSV log and log to console."""
        if self._csv_writer is None:
            return

        # Add step/epoch metadata
        row_data = {"step": step, "epoch": epoch}
        row_data.update(metrics)

        if not self._csv_header_written:
            self._csv_writer.writerow(list(row_data.keys()))
            self._csv_header_written = True

        self._csv_writer.writerow(list(row_data.values()))
        self._log_file.flush()

    def _compute_attack_metrics(self, transcripts) -> Dict[str, float]:
        """Compute per-step attack success metrics from transcripts."""
        attack_metrics = {}
        if not transcripts:
            return attack_metrics

        clean_correct = 0
        post_correct = 0
        flips = 0  # correct -> wrong (attack succeeded)
        beneficial = 0  # wrong -> correct
        attack_counts = {}
        attack_flips = {}

        for transcript in transcripts:
            verifier = get_verifier(transcript.dataset)
            cc = verifier.verify(
                str(transcript.clean_answer) if transcript.clean_answer else "",
                transcript.ground_truth,
            )
            pc = verifier.verify(
                str(transcript.post_attack_answer)
                if transcript.post_attack_answer
                else "",
                transcript.ground_truth,
            )

            if cc:
                clean_correct += 1
            if pc:
                post_correct += 1
            if cc and not pc:
                flips += 1
            if not cc and pc:
                beneficial += 1

            # Per-attack tracking
            attack_turn = transcript.attack_turn
            attack_name = "unknown"
            if attack_turn and attack_turn.metadata:
                attack_name = attack_turn.metadata.get("attack_name", "unknown")
            attack_counts[attack_name] = attack_counts.get(attack_name, 0) + 1
            if cc and not pc:
                attack_flips[attack_name] = attack_flips.get(attack_name, 0) + 1

        n = len(transcripts)
        attack_metrics["attack/clean_accuracy"] = clean_correct / max(n, 1)
        attack_metrics["attack/post_attack_accuracy"] = post_correct / max(n, 1)
        attack_metrics["attack/ASR"] = flips / max(clean_correct, 1)
        attack_metrics["attack/beneficial_revision_rate"] = beneficial / max(n - clean_correct, 1)
        attack_metrics["attack/num_flips"] = flips
        attack_metrics["attack/num_samples"] = n

        # Per-attack ASR
        for name, count in attack_counts.items():
            af = attack_flips.get(name, 0)
            attack_metrics[f"attack/{name}_count"] = count
            attack_metrics[f"attack/{name}_flips"] = af

        return attack_metrics

    def train(self):
        """Run retainer training."""
        logger.info(
            f"Starting retainer training: {self.config.epochs} epochs, "
            f"loss={self.config.loss_type}"
        )

        start_epoch = getattr(self.config, "start_epoch", 0)
        for epoch in range(start_epoch, self.config.epochs):
            logger.info(f"=== Epoch {epoch + 1}/{self.config.epochs} ===")
            epoch_metrics = self._train_epoch(epoch)

            ckpt_path = os.path.join(
                self.config.output_dir, f"checkpoint-epoch-{epoch + 1}"
            )
            self.model.save_checkpoint(ckpt_path)

            logger.info(f"Epoch {epoch + 1} metrics: {epoch_metrics}")

        final_path = os.path.join(self.config.output_dir, "checkpoint-final")
        self.model.save_checkpoint(final_path)

        # Shut down vLLM worker
        if self.vllm_worker is not None:
            self.vllm_worker.shutdown()
            self.vllm_worker = None

        # Close log file
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

        logger.info("Training complete!")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.model.train()
        epoch_metrics_accum: Dict[str, float] = {}
        num_batches = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress):
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break

            step_start = time.time()
            step_metrics = self._train_step(batch)
            step_time = time.time() - step_start

            if step_metrics is not None:
                step_metrics["step_time_sec"] = round(step_time, 1)
                step_metrics["lr"] = self.scheduler.get_last_lr()[0]

                for k, v in step_metrics.items():
                    epoch_metrics_accum[k] = epoch_metrics_accum.get(k, 0.0) + v
                num_batches += 1

                progress.set_postfix({
                    "loss": f"{step_metrics.get('total_loss', 0):.4f}",
                    "acc": f"{step_metrics.get('reward/accuracy', 0):.3f}",
                    "ASR": f"{step_metrics.get('attack/ASR', 0):.3f}",
                    "t": f"{step_time:.0f}s",
                })

                # Per-step file logging
                logging_steps = getattr(self.config, "logging_steps", 10)
                if self.global_step % logging_steps == 0 or self.global_step == 1:
                    self._log_step_metrics(self.global_step, epoch + 1, step_metrics)
                    logger.info(
                        f"Step {self.global_step} | "
                        f"loss={step_metrics.get('total_loss', 0):.4f} | "
                        f"acc={step_metrics.get('reward/accuracy', 0):.3f} | "
                        f"ASR={step_metrics.get('attack/ASR', 0):.3f} | "
                        f"R_ans={step_metrics.get('reward/R_ans_mean', 0):.3f} | "
                        f"R_corr={step_metrics.get('reward/R_corrigible_mean', 0):.3f} | "
                        f"R_anchor={step_metrics.get('reward/R_anchor_mean', 0):.3f} | "
                        f"R_prm={step_metrics.get('reward/R_prm_mean', 0):.3f} | "
                        f"R_len={step_metrics.get('reward/R_len_mean', 0):.3f} | "
                        f"combined={step_metrics.get('reward/combined_mean', 0):.3f} | "
                        f"lr={step_metrics.get('lr', 0):.2e} | "
                        f"time={step_time:.0f}s"
                    )

            # Periodic checkpoint saving
            save_steps = getattr(self.config, "save_steps", 500)
            if self.global_step > 0 and self.global_step % save_steps == 0:
                ckpt_path = os.path.join(
                    self.config.output_dir, f"checkpoint-step-{self.global_step}"
                )
                self.model.save_checkpoint(ckpt_path)
                logger.info(f"Checkpoint saved at step {self.global_step}: {ckpt_path}")

            # Sync LoRA adapter to vLLM worker
            if (
                self.vllm_worker is not None
                and self.global_step > 0
                and self.global_step % self.config.vllm_lora_sync_steps == 0
            ):
                try:
                    adapter_path = self._export_lora_adapter(
                        f"step_{self.global_step}"
                    )
                    self.vllm_worker.update_lora(adapter_path)
                    logger.info(
                        f"vLLM LoRA synced at step {self.global_step}"
                    )
                except Exception as e:
                    logger.warning(f"vLLM LoRA sync failed: {e}")

            if self.config.dry_run and batch_idx >= 1:
                break

        if num_batches > 0:
            return {k: v / num_batches for k, v in epoch_metrics_accum.items()}
        return {}

    def _train_step(self, batch: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Single training step: TRG game -> rewards -> loss -> backward."""
        device = next(self.model.model.parameters()).device
        B = len(batch["ground_truths"])
        G = self.config.group_size

        # Expand for group
        queries = self._expand_list(
            [batch["ids"][i] for i in range(B)]
        )
        ground_truths = self._expand_list(batch["ground_truths"])
        datasets = self._expand_list(batch["datasets"])

        # Get the raw query texts from the collator
        raw_queries = self._expand_list(batch["queries"])

        # Play TRG games
        self.model.model.eval()
        attack_framing = getattr(self.config, "attack_framing", "followup")
        if isinstance(attack_framing, list):
            # Multi-framing: sample one framing per batch (keeps vLLM batching intact)
            import random as _rand
            batch_framing = _rand.choice(attack_framing)
            if hasattr(self.trg_game, 'attack_fn'):
                self.trg_game.attack_fn.framing = batch_framing
            transcripts = self.trg_game.play_batch(
                problems=raw_queries,
                ground_truths=ground_truths,
                datasets=datasets,
                system_prompt=get_system_prompt(batch_framing),
            )
        else:
            transcripts = self.trg_game.play_batch(
                problems=raw_queries,
                ground_truths=ground_truths,
                datasets=datasets,
                system_prompt=get_system_prompt(attack_framing),
            )
        self.model.model.train()

        if not transcripts:
            return None

        # Compute rewards
        reward_output = self.reward_aggregator.compute(
            transcripts=transcripts,
            group_size=G,
            loss_type=self.config.loss_type,
        )

        # Build sequences for Turn 3 (post-attack response)
        all_input_ids = []
        all_attention_masks = []
        all_response_masks = []

        for transcript in transcripts:
            messages = transcript.to_messages()
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                prompt = transcript.to_prompt_text()

            encodings = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
            all_input_ids.append(encodings["input_ids"])
            all_attention_masks.append(encodings["attention_mask"])

            # Create response mask (1 for Turn 3 tokens)
            # Approximate: last portion of the sequence
            resp_mask = torch.zeros_like(encodings["input_ids"])
            post_turn = transcript.post_attack_turn
            if post_turn:
                post_len = min(
                    post_turn.token_count,
                    self.config.max_seq_length,
                )
                resp_mask[0, -post_len:] = 1.0
            all_response_masks.append(resp_mask)

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)
        response_mask = torch.cat(all_response_masks, dim=0).to(device).float()

        # Free generation KV caches before heavy forward passes
        torch.cuda.empty_cache()

        # Compute log probabilities
        log_probs = self.model.compute_log_probs(
            input_ids, attention_mask, response_mask
        )
        old_log_probs = log_probs.detach()

        torch.cuda.empty_cache()
        ref_log_probs = self.model.compute_ref_log_probs(
            input_ids, attention_mask, response_mask
        )

        # Build loss kwargs
        component_weights_tensor = torch.tensor(
            [
                self.config.w_prm,
                self.config.w_anchor,
                self.config.w_corrigible,
                self.config.w_ans,
                self.config.w_len,
            ],
            device=device,
            dtype=torch.float,
        )

        loss_kwargs: Dict[str, Any] = {
            "component_rewards": reward_output.component_rewards.to(device),
            "component_weights": component_weights_tensor,
            "group_size": G,
        }

        # Regularization (TCR/ATR/ATRC): extra forward passes
        torch.cuda.empty_cache()
        effective_lambda_reg = self.config.lambda_reg
        if self.config.lambda_tcr > 0 and self.config.reg_loss_type == "none":
            effective_lambda_reg = self.config.lambda_tcr
        reg_enabled = (
            effective_lambda_reg > 0
            and self.config.loss_type == "combined"
            and self.config.reg_loss_type != "none"
        )
        # Backward compat: also enable if old-style lambda_tcr > 0
        if not reg_enabled and self.config.lambda_tcr > 0 and self.config.loss_type == "combined":
            reg_enabled = True
        if reg_enabled:
            reg_kwargs = self._compute_reg_inputs(
                transcripts, input_ids, attention_mask, device
            )
            loss_kwargs.update(reg_kwargs)

        # SDPO: feedback-conditioned forward passes (Phase 4)
        if self.config.lambda_sdpo > 0 and self.config.loss_type == "combined":
            sdpo_kwargs = self._compute_sdpo_inputs(
                transcripts, input_ids, attention_mask, response_mask, device
            )
            if sdpo_kwargs is not None:
                loss_kwargs.update(sdpo_kwargs)

        # Compute loss
        loss_output = self.loss_fn.compute_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=reward_output.advantages.to(device),
            mask=response_mask,
            **loss_kwargs,
        )

        # Backward
        loss = loss_output.total_loss / self.config.gradient_accumulation_steps
        if self._use_accelerator:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        # Optimizer step
        self.global_step += 1
        if self.global_step % self.config.gradient_accumulation_steps == 0:
            if self._use_accelerator:
                self.accelerator.clip_grad_norm_(
                    self.model.model.parameters(), self.config.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Collect metrics
        metrics: Dict[str, float] = {
            "total_loss": loss_output.total_loss.item(),
            "policy_loss": loss_output.policy_loss.item(),
            "kl_loss": loss_output.kl_loss.item(),
        }
        metrics.update(loss_output.metrics)
        metrics.update(reward_output.metrics)

        # Attack success metrics from transcripts
        attack_metrics = self._compute_attack_metrics(transcripts)
        metrics.update(attack_metrics)

        return metrics

    def _compute_reg_inputs(
        self, transcripts, input_ids, attention_mask, device
    ) -> Dict[str, torch.Tensor]:
        """Compute regularization inputs (TCR/ATR/ATRC) with aligned response spans.

        Builds two genuinely different sequences per transcript:
        - Clean seq:    [SYSTEM | PROBLEM | CLEAN_RESPONSE]
        - Attacked seq: [SYSTEM | PROBLEM | ATTACK | CLEAN_RESPONSE]

        **Critical fix**: Extracts response-span logits from both sequences and
        aligns them by response-relative position, rather than applying a single
        positional mask to full-length logits (which was misaligned because the
        attacked prefix is longer than the clean prefix).

        Returns:
            clean_logits:    [batch, max_resp_len, vocab] — detached
            attacked_logits: [batch, max_resp_len, vocab] — has gradients
            committed_mask:  [batch, max_resp_len] — relative to response start
            tcr_gate:        [batch] — per-sample gate
            clean_token_ids: [batch, max_resp_len] — target token IDs for ATR
        """
        batch_size = len(transcripts)

        clean_input_ids_list = []
        clean_attn_list = []
        attacked_input_ids_list = []
        attacked_attn_list = []
        # Track where clean response tokens start/end in each sequence
        clean_resp_offsets = []   # (start, end) in clean seq
        attacked_resp_offsets = []  # (start, end) in attacked seq
        resp_lengths = []  # actual response length per sample

        for transcript in transcripts:
            clean_turn = transcript.clean_turn
            attack_turn = transcript.attack_turn

            # Build clean messages: [system, problem, clean_response]
            clean_messages = []
            for turn in transcript.turns:
                if turn.role.value == "system":
                    clean_messages.append({"role": "system", "content": turn.content})
                elif turn.role.value == "problem":
                    clean_messages.append({"role": "user", "content": turn.content})
                    break

            clean_response_text = clean_turn.content if clean_turn else ""
            clean_messages.append({"role": "assistant", "content": clean_response_text})

            # Build attacked messages: [system, problem, attack, clean_response]
            attacked_messages = list(clean_messages[:-1])  # system + problem
            attack_text = attack_turn.content if attack_turn else ""
            attacked_messages.append({"role": "user", "content": attack_text})
            attacked_messages.append({"role": "assistant", "content": clean_response_text})

            # Tokenize to find prefix boundaries
            if hasattr(self.tokenizer, "apply_chat_template"):
                clean_prefix_msgs = clean_messages[:-1]
                clean_prefix = self.tokenizer.apply_chat_template(
                    clean_prefix_msgs, tokenize=False, add_generation_prompt=True
                )
                clean_full = self.tokenizer.apply_chat_template(
                    clean_messages, tokenize=False, add_generation_prompt=False
                )
                attacked_prefix_msgs = attacked_messages[:-1]
                attacked_prefix = self.tokenizer.apply_chat_template(
                    attacked_prefix_msgs, tokenize=False, add_generation_prompt=True
                )
                attacked_full = self.tokenizer.apply_chat_template(
                    attacked_messages, tokenize=False, add_generation_prompt=False
                )
            else:
                clean_prefix = "\n".join(
                    f"{m['role']}: {m['content']}" for m in clean_messages[:-1]
                ) + "\nassistant: "
                clean_full = clean_prefix + clean_response_text
                attacked_prefix = "\n".join(
                    f"{m['role']}: {m['content']}" for m in attacked_messages[:-1]
                ) + "\nassistant: "
                attacked_full = attacked_prefix + clean_response_text

            # Tokenize both full sequences (no padding — we'll extract spans first)
            clean_enc = self.tokenizer(
                clean_full, return_tensors="pt", truncation=True,
                max_length=self.config.max_seq_length, padding="max_length",
            )
            attacked_enc = self.tokenizer(
                attacked_full, return_tensors="pt", truncation=True,
                max_length=self.config.max_seq_length, padding="max_length",
            )

            # Find where the clean response starts in each sequence
            clean_prefix_len = len(self.tokenizer.encode(
                clean_prefix, add_special_tokens=False
            ))
            attacked_prefix_len = len(self.tokenizer.encode(
                attacked_prefix, add_special_tokens=False
            ))
            clean_resp_len = len(self.tokenizer.encode(
                clean_response_text, add_special_tokens=False
            ))

            # Clamp to actual sequence length
            seq_len_clean = clean_enc["input_ids"].shape[1]
            seq_len_attacked = attacked_enc["input_ids"].shape[1]
            clean_start = min(clean_prefix_len, seq_len_clean)
            clean_end = min(clean_prefix_len + clean_resp_len, seq_len_clean)
            attacked_start = min(attacked_prefix_len, seq_len_attacked)
            attacked_end = min(attacked_prefix_len + clean_resp_len, seq_len_attacked)

            # Use the shorter of the two actual spans to stay aligned
            actual_resp_len = min(clean_end - clean_start, attacked_end - attacked_start)
            actual_resp_len = max(actual_resp_len, 0)

            clean_resp_offsets.append((clean_start, clean_start + actual_resp_len))
            attacked_resp_offsets.append((attacked_start, attacked_start + actual_resp_len))
            resp_lengths.append(actual_resp_len)

            clean_input_ids_list.append(clean_enc["input_ids"])
            clean_attn_list.append(clean_enc["attention_mask"])
            attacked_input_ids_list.append(attacked_enc["input_ids"])
            attacked_attn_list.append(attacked_enc["attention_mask"])

        clean_ids = torch.cat(clean_input_ids_list, dim=0).to(device)
        clean_attn = torch.cat(clean_attn_list, dim=0).to(device)
        attacked_ids = torch.cat(attacked_input_ids_list, dim=0).to(device)
        attacked_attn = torch.cat(attacked_attn_list, dim=0).to(device)

        # Forward passes on full sequences
        with torch.no_grad():
            clean_full_logits = self.model.get_logits(clean_ids, clean_attn)

        attacked_full_logits = self.model.get_logits(attacked_ids, attacked_attn)

        # --- Extract and align response spans ---
        max_resp_len = max(resp_lengths) if resp_lengths else 1
        max_resp_len = max(max_resp_len, 1)  # ensure at least 1
        vocab_size = clean_full_logits.shape[-1]

        # Pre-allocate aligned tensors
        clean_resp_logits = torch.zeros(
            batch_size, max_resp_len, vocab_size,
            device=device, dtype=clean_full_logits.dtype,
        )
        attacked_resp_logits = torch.zeros(
            batch_size, max_resp_len, vocab_size,
            device=device, dtype=attacked_full_logits.dtype,
        )
        clean_token_ids = torch.zeros(
            batch_size, max_resp_len,
            device=device, dtype=torch.long,
        )
        resp_padding_mask = torch.zeros(
            batch_size, max_resp_len, device=device,
        )

        for i in range(batch_size):
            rlen = resp_lengths[i]
            if rlen == 0:
                continue
            c_start, c_end = clean_resp_offsets[i]
            a_start, a_end = attacked_resp_offsets[i]

            # Extract response spans (now aligned by response-relative position)
            clean_resp_logits[i, :rlen, :] = clean_full_logits[i, c_start:c_end, :].detach()
            attacked_resp_logits[i, :rlen, :] = attacked_full_logits[i, a_start:a_end, :]
            clean_token_ids[i, :rlen] = clean_ids[i, c_start:c_end]
            resp_padding_mask[i, :rlen] = 1.0

        # Build committed mask relative to response start (not absolute seq position)
        anchor_positions = []
        for idx, transcript in enumerate(transcripts):
            positions = []
            clean_trace = transcript.clean_trace
            if clean_trace is None:
                anchor_positions.append([])
                continue
            rlen = resp_lengths[idx]
            for step in clean_trace.steps:
                # Step positions are relative to the response text already
                s = step.start_token_pos
                e = step.end_token_pos
                if s < rlen and e <= rlen:
                    positions.append((s, e))
            anchor_positions.append(positions)

        committed_mask = TCRKLLoss.build_committed_mask(
            anchor_positions=anchor_positions,
            seq_len=max_resp_len,
            device=device,
            batch_size=batch_size,
        )
        # Intersect with response padding mask (don't include padded positions)
        committed_mask = committed_mask * resp_padding_mask

        # Build gate
        clean_correct = []
        evidence_invalid = []
        for transcript in transcripts:
            verifier = get_verifier(transcript.dataset)
            cc = verifier.verify(
                str(transcript.clean_answer) if transcript.clean_answer else "",
                transcript.ground_truth,
            )
            clean_correct.append(float(cc))

            attack_turn = transcript.attack_turn
            if attack_turn and attack_turn.metadata:
                ei = not attack_turn.metadata.get("is_valid_evidence", True)
            else:
                ei = True
            evidence_invalid.append(float(ei))

        gate = TCRKLLoss.build_gate(
            clean_correct=torch.tensor(clean_correct, device=device),
            evidence_invalid=torch.tensor(evidence_invalid, device=device),
            mode=self.config.gate_mode,
        )

        return {
            "clean_logits": clean_resp_logits,
            "attacked_logits": attacked_resp_logits,
            "committed_mask": committed_mask,
            "tcr_gate": gate,
            "clean_token_ids": clean_token_ids,
        }

    def _compute_sdpo_inputs(
        self, transcripts, input_ids, attention_mask, response_mask, device
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute SDPO inputs: base_logits, feedback_logits, response_mask.

        For each transcript, builds:
        - base_logits:     model(Turn 3 context) — standard generation context
        - feedback_logits: model(Turn 3 context + feedback) — feedback-conditioned

        Feedback comes from transcript metadata (e.g. compiler errors, test output,
        judge critique). Only applicable for code tasks or tasks with rich feedback.
        Returns None if no transcripts have feedback available.
        """
        # Check if any transcripts have feedback metadata
        has_feedback = False
        for transcript in transcripts:
            post_turn = transcript.post_attack_turn
            if post_turn and post_turn.metadata.get("feedback"):
                has_feedback = True
                break

        if not has_feedback:
            return None

        # base_logits: forward pass on the standard Turn 3 sequence (reuse input_ids)
        with torch.no_grad():
            base_logits = self.model.get_logits(input_ids, attention_mask)

        # feedback_logits: forward pass on Turn 3 context + feedback appended
        feedback_input_ids_list = []
        feedback_attn_list = []

        for transcript in transcripts:
            post_turn = transcript.post_attack_turn
            feedback_text = ""
            if post_turn and post_turn.metadata.get("feedback"):
                feedback_text = post_turn.metadata["feedback"]

            # Build messages with feedback appended as an extra user turn
            messages = transcript.to_messages()
            if feedback_text:
                messages.append({"role": "user", "content": feedback_text})
                messages.append({"role": "assistant", "content": post_turn.content})

            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                prompt = "\n".join(
                    f"{m['role']}: {m['content']}" for m in messages
                )

            enc = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.config.max_seq_length, padding="max_length",
            )
            feedback_input_ids_list.append(enc["input_ids"])
            feedback_attn_list.append(enc["attention_mask"])

        feedback_ids = torch.cat(feedback_input_ids_list, dim=0).to(device)
        feedback_attn = torch.cat(feedback_attn_list, dim=0).to(device)

        # Feedback logits: teacher (detached)
        with torch.no_grad():
            feedback_logits = self.model.get_logits(feedback_ids, feedback_attn)

        return {
            "base_logits": base_logits,
            "feedback_logits": feedback_logits,
            "response_mask": response_mask,
        }
