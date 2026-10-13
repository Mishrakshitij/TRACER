#!/usr/bin/env python3
"""Ablation evaluation to isolate framing gap components.

Tests whether the framing gap comes from the document wrapper, the system
prompt, or both.  Three ablation modes:

  "full"         -- Normal eval: system prompt AND wrapper match the framing
                    (e.g. RAG system prompt + RAG document wrapper).
  "prompt_only"  -- RAG system prompt but NO document wrapper.  The attack
                    text is delivered as a plain follow-up (followup framing
                    in the AttackMixer), while the system prompt still
                    primes the model for retrieved-document reasoning.
  "wrapper_only" -- Follow-up system prompt but WITH RAG document wrapper.
                    The attack text is wrapped in [Retrieved Document] tags
                    (RAG framing in the AttackMixer), but the system prompt
                    does NOT mention retrieved documents.

This lets us measure:
  delta(full - prompt_only)  = marginal effect of the wrapper
  delta(full - wrapper_only) = marginal effect of the system prompt
  interaction                = full - prompt_only - wrapper_only + baseline

Usage:
    python scripts/eval_ablation.py \
        --config configs/experiments/eval_gsm8k_rag.yaml \
        --ablation_mode prompt_only \
        --dataset gsm8k --split test --max_samples 200 \
        --output outputs/ablation/prompt_only.json
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
from r3af.data import prompt_templates
from r3af.evaluation import vllm_evaluator as vllm_eval_module
from r3af.utils.logging import setup_logging
from r3af.utils.seed import set_seed

logger = logging.getLogger(__name__)

ABLATION_MODES = ("full", "prompt_only", "wrapper_only")


def merge_lora_checkpoint(base_model_name: str, checkpoint_path: str) -> str:
    """Merge LoRA adapters into the base model and save to a temp directory."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Merging LoRA adapter: {checkpoint_path}")
    logger.info(f"Base model: {base_model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload()

    merged_dir = tempfile.mkdtemp(prefix="r3af_ablation_merged_")
    logger.info(f"Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    return merged_dir


def apply_ablation_patches(ablation_mode: str, framing: str):
    """Monkey-patch prompt_templates functions based on ablation mode.

    The framing arg is the framing from the config (e.g. "rag", "tool_output").

    Returns:
        (restore_fn, effective_attack_framing) -- call restore_fn() when done.
        effective_attack_framing is what should be set on config.attack_framing
        so the AttackMixer uses the right wrapper.
    """
    # Save originals
    orig_get_system_prompt = prompt_templates.get_system_prompt
    orig_get_post_attack_prompt = prompt_templates.get_post_attack_prompt

    if ablation_mode == "full":
        # No patches needed -- both system prompt and wrapper use `framing`
        effective_attack_framing = framing
        return (lambda: None), effective_attack_framing

    elif ablation_mode == "prompt_only":
        # System prompt: use the framing-specific one (e.g. RAG)
        # Wrapper: use followup (passthrough -- no document wrapper)
        # We patch get_system_prompt to always return the framing-specific
        # prompt, regardless of what the evaluator asks for.  The evaluator
        # will ask for "followup" (since we set attack_framing="followup"),
        # but we want the RAG system prompt.
        framing_system_prompt = orig_get_system_prompt(framing)
        framing_post_attack_prompt = orig_get_post_attack_prompt(framing)

        def patched_get_system_prompt(f="followup"):
            return framing_system_prompt

        def patched_get_post_attack_prompt(f="followup"):
            return framing_post_attack_prompt

        prompt_templates.get_system_prompt = patched_get_system_prompt
        prompt_templates.get_post_attack_prompt = patched_get_post_attack_prompt
        vllm_eval_module.get_system_prompt = patched_get_system_prompt
        vllm_eval_module.get_post_attack_prompt = patched_get_post_attack_prompt

        effective_attack_framing = "followup"

    elif ablation_mode == "wrapper_only":
        # System prompt: use followup (generic, no mention of retrieved docs)
        # Wrapper: use the framing-specific one (e.g. RAG document wrapper)
        followup_system_prompt = orig_get_system_prompt("followup")
        followup_post_attack_prompt = orig_get_post_attack_prompt("followup")

        def patched_get_system_prompt(f="followup"):
            return followup_system_prompt

        def patched_get_post_attack_prompt(f="followup"):
            return followup_post_attack_prompt

        prompt_templates.get_system_prompt = patched_get_system_prompt
        prompt_templates.get_post_attack_prompt = patched_get_post_attack_prompt
        vllm_eval_module.get_system_prompt = patched_get_system_prompt
        vllm_eval_module.get_post_attack_prompt = patched_get_post_attack_prompt

        effective_attack_framing = framing

    else:
        raise ValueError(f"Unknown ablation mode: {ablation_mode}")

    def restore():
        prompt_templates.get_system_prompt = orig_get_system_prompt
        prompt_templates.get_post_attack_prompt = orig_get_post_attack_prompt
        vllm_eval_module.get_system_prompt = orig_get_system_prompt
        vllm_eval_module.get_post_attack_prompt = orig_get_post_attack_prompt

    return restore, effective_attack_framing


def main():
    parser = argparse.ArgumentParser(
        description="R3AF Framing-Gap Ablation Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
    parser.add_argument("--ablation_mode", type=str, required=True,
                        choices=ABLATION_MODES,
                        help="Ablation mode: full, prompt_only, or wrapper_only")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    config = R3AFConfig.from_yaml(args.config)
    set_seed(config.seed)

    # The framing that the config originally specifies (e.g. "rag")
    original_framing = getattr(config, "attack_framing", "followup")
    logger.info(
        f"Ablation mode: {args.ablation_mode} | "
        f"Config framing: {original_framing}"
    )

    # Apply ablation patches
    restore_fn, effective_attack_framing = apply_ablation_patches(
        args.ablation_mode, original_framing
    )

    # Override config.attack_framing so AttackMixer uses the right wrapper
    config.attack_framing = effective_attack_framing

    logger.info(
        f"Effective attack_framing (wrapper): {effective_attack_framing} | "
        f"System prompt source: "
        f"{'framing-specific' if args.ablation_mode in ('full', 'prompt_only') else 'followup'}"
    )

    # Merge LoRA if checkpoint provided
    merged_dir = None
    model_path = config.model_name

    if args.checkpoint:
        merged_dir = merge_lora_checkpoint(config.model_name, args.checkpoint)
        model_path = merged_dir

    try:
        config.model_name = model_path

        vllm_model = VLLMModel(
            config, tensor_parallel_size=args.tensor_parallel_size
        )

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
            summary = metrics.summary()
            # Tag with ablation metadata
            summary["ablation_mode"] = args.ablation_mode
            summary["original_framing"] = original_framing
            summary["effective_wrapper_framing"] = effective_attack_framing
            summary["system_prompt_source"] = (
                original_framing
                if args.ablation_mode in ("full", "prompt_only")
                else "followup"
            )
            summary["post_attack_prompt_source"] = (
                original_framing
                if args.ablation_mode in ("full", "prompt_only")
                else "followup"
            )
            all_results[ds_name] = summary
            logger.info(f"\n{ds_name} [{args.ablation_mode}] results: {summary}")

        # Save results
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            print(json.dumps(all_results, indent=2))

    finally:
        restore_fn()
        if merged_dir and os.path.isdir(merged_dir):
            logger.info(f"Cleaning up merged model: {merged_dir}")
            shutil.rmtree(merged_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
