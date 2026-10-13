#!/usr/bin/env python3
"""Merge LoRA adapters into a base model and save the merged checkpoint.

vLLM cannot load PEFT adapters directly, so this script merges LoRA
weights into the base model and saves a standalone HuggingFace checkpoint
that vLLM can load.

Usage:
    python scripts/merge_lora.py \
        --base_model /path/to/Qwen2.5-7B-Instruct \
        --adapter outputs/r3af/phase1_full/checkpoint-final \
        --output merged_model/
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base_model", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to PEFT/LoRA adapter checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for merged model")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype")
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    logger.info(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="cpu",
    )

    logger.info(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )

    logger.info(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    logger.info("Done! Merged model saved successfully.")
    logger.info(f"Load in vLLM with: LLM(model='{args.output}', dtype='{args.dtype}')")


if __name__ == "__main__":
    main()
