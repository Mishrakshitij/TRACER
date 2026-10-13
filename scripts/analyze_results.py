#!/usr/bin/env python3
"""Analyze and compare R3AF evaluation results.

Usage:
    python scripts/analyze_results.py --results results/*.json
"""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r3af.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Analyze R3AF Results")
    parser.add_argument("--results", nargs="+", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    setup_logging()

    all_results = {}
    for path in args.results:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            all_results[name] = json.load(f)

    # Print comparison table
    metrics_keys = [
        "robust_accuracy", "clean_accuracy", "ASR",
        "beneficial_revision_rate", "over_stability_error",
        "length_inflation_mean",
    ]

    header = f"{'Method':<25}" + "".join(f"{k:<22}" for k in metrics_keys)
    logger.info(header)
    logger.info("-" * len(header))

    for name, results in all_results.items():
        for ds_name, metrics in results.items():
            row_name = f"{name}/{ds_name}"
            row = f"{row_name:<25}"
            for key in metrics_keys:
                val = metrics.get(key, 0.0)
                row += f"{val:<22.4f}"
            logger.info(row)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
