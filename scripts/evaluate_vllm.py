#!/usr/bin/env python3
"""R3AF evaluation using vLLM for fast batched inference.

Usage (base model — no training):
    python scripts/evaluate_vllm.py \
        --config configs/experiments/eval_baseline.yaml \
        --dataset gsm8k --split test --max_samples 100

For a trained model (merge LoRA first, then evaluate):
    python scripts/evaluate_vllm.py \
        --config configs/experiments/eval_baseline.yaml \
        --dataset gsm8k --split test --max_samples 100 \
        --checkpoint outputs/r3af/phase1_full/checkpoint-final

The --checkpoint flag triggers automatic LoRA merge: the base model and
adapter are loaded via PEFT, merged with merge_and_unload(), saved to a
temporary directory, and then loaded into vLLM for evaluation.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r3af.models.config import R3AFConfig
from r3af.models.vllm_model import VLLMModel
from r3af.evaluation.vllm_evaluator import VLLMEvaluator
from r3af.utils.logging import setup_logging
from r3af.utils.seed import set_seed

logger = logging.getLogger(__name__)


def merge_lora_checkpoint(base_model_name: str, checkpoint_path: str) -> str:
    """Merge LoRA adapters into the base model and save to a temp directory.

    Args:
        base_model_name: HuggingFace model name or local path.
        checkpoint_path: Path to the PEFT/LoRA adapter checkpoint.

    Returns:
        Path to the merged model directory (caller should clean up).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Merging LoRA adapter: {checkpoint_path}")
    logger.info(f"Base model: {base_model_name}")

    # Load base model in bf16 (no quantization for merge)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # merge on CPU to save GPU memory
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )

    # Load and merge LoRA
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload()

    # Save merged model to temp directory
    merged_dir = tempfile.mkdtemp(prefix="r3af_merged_")
    logger.info(f"Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    return merged_dir


def main():
    parser = argparse.ArgumentParser(
        description="R3AF Evaluation (vLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to PEFT/LoRA checkpoint (triggers merge)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (overrides config)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    config = R3AFConfig.from_yaml(args.config)
    set_seed(config.seed)

    # Determine model path
    merged_dir = None
    model_path = config.model_name

    if args.checkpoint:
        merged_dir = merge_lora_checkpoint(config.model_name, args.checkpoint)
        model_path = merged_dir

    try:
        # Override model name to point at merged model (if applicable)
        config.model_name = model_path

        # Load model via vLLM
        vllm_model = VLLMModel(
            config, tensor_parallel_size=args.tensor_parallel_size
        )

        # Determine datasets
        datasets = (
            [args.dataset]
            if args.dataset
            else config.eval_datasets or config.datasets
        )

        all_results = {}
        for ds_name in datasets:
            evaluator = VLLMEvaluator(config, vllm_model)
            metrics = evaluator.evaluate(
                dataset_name=ds_name,
                split=args.split,
                max_samples=args.max_samples or config.eval_max_samples,
            )
            all_results[ds_name] = metrics.summary()
            logger.info(f"\n{ds_name} results: {metrics.summary()}")

        # Save results
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            print(json.dumps(all_results, indent=2))

    finally:
        # Clean up merged model temp directory
        if merged_dir and os.path.isdir(merged_dir):
            logger.info(f"Cleaning up merged model: {merged_dir}")
            shutil.rmtree(merged_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
