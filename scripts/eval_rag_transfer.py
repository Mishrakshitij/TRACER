#!/usr/bin/env python3
"""RAG transfer evaluation: test existing best checkpoints with RAG framing.

Evaluates whether adversarial robustness transfers from "user follow-up"
framing to "retrieved document" (RAG) framing.

Tests the top checkpoints from each experiment with attack_framing="rag".

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/eval_rag_transfer.py --gpu_id 7
"""

import argparse
import json
import os
import subprocess
import sys
import time
import functools

import torch

print = functools.partial(print, flush=True)

# Best checkpoints to evaluate (experiment_dir, checkpoint_name, label)
BEST_CHECKPOINTS = [
    # GRPO+pinj step-1750 (only local checkpoint available for pinj experiments)
    (
        "outputs/r3af/phase1_grpo_pinj",
        "step-1750",
        "grpo_pinj_step1750",
    ),
    # GRPO+all5: best all-5 (ASR=14.6%)
    (
        "outputs/r3af/phase1_grpo_all5",
        "epoch-2",
        "grpo_all5_epoch2",
    ),
    # GRPO+all5: step-1750
    (
        "outputs/r3af/phase1_grpo_all5",
        "step-1750",
        "grpo_all5_step1750",
    ),
    # ATRC+all5: best (step-750)
    (
        "outputs/r3af/phase1_atrc_all5",
        "step-750",
        "atrc_all5_step750",
    ),
    # GDPO+all5: best (epoch-1)
    (
        "outputs/r3af/phase1_gdpo_all5",
        "epoch-1",
        "gdpo_all5_epoch1",
    ),
    # GDPO+all5: step-500
    (
        "outputs/r3af/phase1_gdpo_all5",
        "step-500",
        "gdpo_all5_step500",
    ),
]

# Config for RAG-framed evaluation
RAG_EVAL_CONFIG = "configs/experiments/eval_rag_transfer.yaml"
OUTPUT_DIR = "outputs/r3af/eval_rag_transfer"


def find_checkpoint_path(base_dir, ckpt_name):
    """Find the actual checkpoint directory."""
    # Try direct path
    direct = os.path.join(base_dir, f"checkpoint-{ckpt_name}")
    if os.path.isdir(direct):
        return direct

    # Try HF cache path
    hf_cache = os.path.join(base_dir, ".cache/huggingface/download", f"checkpoint-{ckpt_name}")
    if os.path.isdir(hf_cache):
        return hf_cache

    return None


def run_eval(config, checkpoint_path, output_path, max_samples, gpu_id):
    """Run a single evaluation as a subprocess."""
    cmd = [
        sys.executable, "scripts/evaluate_vllm.py",
        "--config", config,
        "--checkpoint", checkpoint_path,
        "--output", output_path,
        "--max_samples", str(max_samples),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200)

    if result.returncode != 0:
        for line in (result.stderr or "").split("\n"):
            if "Error" in line or "error" in line or "OOM" in line:
                print(f"  ERROR: {line.strip()}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=7)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--hold_gb", type=float, default=20)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"RAG Transfer Evaluation")
    print(f"  GPU: {args.gpu_id}, samples: {args.max_samples}")
    print(f"  Checkpoints: {len(BEST_CHECKPOINTS)}")
    print()

    completed = 0
    skipped = 0
    failed = 0

    for base_dir, ckpt_name, label in BEST_CHECKPOINTS:
        out_path = os.path.join(OUTPUT_DIR, f"rag_transfer_{label}.json")

        if os.path.exists(out_path):
            print(f"SKIP {label} (already exists)")
            skipped += 1
            continue

        ckpt_path = find_checkpoint_path(base_dir, ckpt_name)
        if ckpt_path is None:
            print(f"SKIP {label} (checkpoint not found in {base_dir})")
            skipped += 1
            continue

        print(f"\n=== {label} ===")
        print(f"  Checkpoint: {ckpt_path}")

        success = run_eval(
            RAG_EVAL_CONFIG, ckpt_path, out_path,
            args.max_samples, args.gpu_id
        )

        if success:
            completed += 1
            # Print results
            if os.path.exists(out_path):
                with open(out_path) as f:
                    results = json.load(f)
                if "gsm8k" in results:
                    r = results["gsm8k"]
                    print(f"  clean={r.get('clean_accuracy',0):.1%}, "
                          f"robust={r.get('robust_accuracy',0):.1%}, "
                          f"ASR={r.get('ASR',0):.1%}")
        else:
            failed += 1

        time.sleep(3)
        torch.cuda.empty_cache()

    print(f"\n=== DONE: {completed} completed, {skipped} skipped, {failed} failed ===")

    # Print summary table
    print(f"\n{'Label':<30} {'Clean':>8} {'Robust':>8} {'ASR':>8}")
    print("-" * 56)
    for _, _, label in BEST_CHECKPOINTS:
        out_path = os.path.join(OUTPUT_DIR, f"rag_transfer_{label}.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                results = json.load(f)
            r = results.get("gsm8k", results)
            print(f"{label:<30} {r.get('clean_accuracy',0):>7.1%} "
                  f"{r.get('robust_accuracy',0):>7.1%} "
                  f"{r.get('ASR',r.get('attack_success_rate',0)):>7.1%}")


if __name__ == "__main__":
    main()
