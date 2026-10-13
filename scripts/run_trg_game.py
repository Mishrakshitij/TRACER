#!/usr/bin/env python3
"""Standalone TRG game runner for debugging.

Usage:
    python scripts/run_trg_game.py --config configs/phase1_scripted.yaml --problem "What is 2+2?"
"""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r3af.models.config import R3AFConfig
from r3af.models.base import R3AFModel
from r3af.environment.trg_game import TRGGame
from r3af.environment.budget import AttackBudget
from r3af.attacks.attack_mixer import AttackMixer
from r3af.parsing.trace_parser import TraceParser
from r3af.data.prompt_templates import RETAINER_SYSTEM_PROMPT
from r3af.utils.logging import setup_logging
from r3af.utils.seed import set_seed

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run TRG Game")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--problem", type=str, default=None)
    parser.add_argument("--ground_truth", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--num_games", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    setup_logging()

    config = R3AFConfig.from_yaml(args.config)
    set_seed(config.seed)

    model = R3AFModel(config)
    attack_mixer = AttackMixer(config)
    trace_parser = TraceParser()

    game = TRGGame(
        retainer_model=model,
        attack_fn=attack_mixer,
        trace_parser=trace_parser,
        budget=AttackBudget(max_tokens=config.max_attack_tokens),
        config=config,
    )

    if args.problem:
        problems = [args.problem]
        ground_truths = [args.ground_truth or "unknown"]
    else:
        from r3af.tasks.base import TaskRegistry
        task = TaskRegistry.get(args.dataset)
        examples = task.load_data(split="test", max_samples=args.num_games)
        problems = [e.query for e in examples]
        ground_truths = [e.ground_truth for e in examples]

    for i, (prob, gt) in enumerate(zip(problems, ground_truths)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Game {i+1}")
        logger.info(f"Problem: {prob[:200]}...")
        logger.info(f"Ground truth: {gt}")

        transcript = game.play_game(
            problem=prob,
            ground_truth=gt,
            dataset=args.dataset,
            system_prompt=RETAINER_SYSTEM_PROMPT,
        )

        for turn in transcript.turns:
            logger.info(f"\n[{turn.role.value}] {turn.content[:500]}")
            if turn.answer:
                logger.info(f"  Answer: {turn.answer}")

        logger.info(f"\nClean answer: {transcript.clean_answer}")
        logger.info(f"Post-attack answer: {transcript.post_attack_answer}")

    if args.output:
        logger.info(f"Results would be saved to {args.output}")


if __name__ == "__main__":
    main()
