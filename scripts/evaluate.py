#!/usr/bin/env python3
"""R3AF evaluation script.

Usage:
    python scripts/evaluate.py --config configs/phase1_scripted.yaml --checkpoint outputs/r3af/checkpoint-final
"""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r3af.models.config import R3AFConfig
from r3af.models.base import R3AFModel
from r3af.evaluation.evaluator import R3AFEvaluator
from r3af.utils.logging import setup_logging
from r3af.utils.seed import set_seed

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="R3AF Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    config = R3AFConfig.from_yaml(args.config)
    set_seed(config.seed)

    model = R3AFModel(config)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)

    datasets = [args.dataset] if args.dataset else config.eval_datasets or config.datasets

    all_results = {}
    for ds_name in datasets:
        evaluator = R3AFEvaluator(config, model)
        metrics = evaluator.evaluate(
            dataset_name=ds_name,
            split=args.split,
            max_samples=args.max_samples or config.eval_max_samples,
        )
        all_results[ds_name] = metrics.summary()
        logger.info(f"\n{ds_name} results: {metrics.summary()}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
