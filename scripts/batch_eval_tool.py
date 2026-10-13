#!/usr/bin/env python3
"""Batch evaluate tool-output trained checkpoints as they become available.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/batch_eval_tool.py

Scans the tool-output training output directory for new checkpoints,
evaluates each with tool-output framing, and saves results.
"""

import glob
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r3af.models.config import R3AFConfig
from r3af.models.vllm_model import VLLMModel
from r3af.evaluation.vllm_evaluator import VLLMEvaluator
from r3af.utils.logging import setup_logging
from r3af.utils.seed import set_seed

TRAIN_DIR = "outputs/r3af/phase1_tool_grpo_atrc"
EVAL_DIR = "outputs/r3af/eval_tool_checkpoints"
CONFIG_PATH = "configs/experiments/eval_gsm8k_tool.yaml"
MAX_SAMPLES = 200
POLL_INTERVAL = 300  # 5 minutes


def get_checkpoints(train_dir):
    """Find all checkpoint directories."""
    pattern = os.path.join(train_dir, "checkpoint-*")
    return sorted(glob.glob(pattern))


def get_evaluated(eval_dir):
    """Find already-evaluated checkpoints."""
    if not os.path.isdir(eval_dir):
        return set()
    return {
        f.replace(".json", "")
        for f in os.listdir(eval_dir)
        if f.endswith(".json")
    }


def main():
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    os.makedirs(EVAL_DIR, exist_ok=True)

    config = R3AFConfig.from_yaml(CONFIG_PATH)
    # Override framing to tool_output for eval
    config.attack_framing = "tool_output"
    set_seed(config.seed)

    logger.info("Starting batch evaluation of tool-output checkpoints")
    logger.info(f"Training dir: {TRAIN_DIR}")
    logger.info(f"Eval dir: {EVAL_DIR}")

    # Import merge function
    from scripts.evaluate_vllm import merge_lora_checkpoint

    while True:
        checkpoints = get_checkpoints(TRAIN_DIR)
        evaluated = get_evaluated(EVAL_DIR)

        pending = []
        for ckpt_path in checkpoints:
            ckpt_name = os.path.basename(ckpt_path)
            if ckpt_name not in evaluated:
                # Check if checkpoint is fully saved (has config.json)
                if os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
                    pending.append((ckpt_name, ckpt_path))

        if not pending:
            logger.info(f"No new checkpoints. {len(evaluated)} already evaluated. Waiting {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)
            continue

        for ckpt_name, ckpt_path in pending:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {ckpt_name}")
            logger.info(f"{'='*60}")

            try:
                # Merge LoRA
                base_model = "Qwen/Qwen2.5-3B-Instruct"
                merged_dir = merge_lora_checkpoint(base_model, ckpt_path)

                # Update config
                config.model_name = merged_dir

                # Load vLLM and evaluate
                vllm_model = VLLMModel(config)
                evaluator = VLLMEvaluator(config, vllm_model)
                metrics = evaluator.evaluate(
                    dataset_name="gsm8k",
                    split="test",
                    max_samples=MAX_SAMPLES,
                )

                results = metrics.summary()
                results["checkpoint"] = ckpt_name
                results["checkpoint_path"] = ckpt_path

                # Save results
                out_path = os.path.join(EVAL_DIR, f"{ckpt_name}.json")
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

                logger.info(f"Results for {ckpt_name}:")
                logger.info(f"  Clean: {results['clean_accuracy']*100:.1f}%")
                logger.info(f"  Robust: {results['robust_accuracy']*100:.1f}%")
                logger.info(f"  ASR: {results['ASR']*100:.1f}%")

                # Cleanup
                del vllm_model, evaluator
                import shutil
                shutil.rmtree(merged_dir, ignore_errors=True)
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error evaluating {ckpt_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Check if training is done (final checkpoint exists)
        if os.path.exists(os.path.join(TRAIN_DIR, "checkpoint-final")):
            remaining = get_checkpoints(TRAIN_DIR)
            re_evaluated = get_evaluated(EVAL_DIR)
            all_names = {os.path.basename(c) for c in remaining}
            if all_names.issubset(re_evaluated):
                logger.info("All checkpoints evaluated. Done!")
                break

        logger.info(f"Waiting {POLL_INTERVAL}s for more checkpoints...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
