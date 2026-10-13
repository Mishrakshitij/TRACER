#!/usr/bin/env python3
"""R3AF training script.

Usage:
    python scripts/train.py --config configs/phase1_scripted.yaml
    python scripts/train.py --config configs/phase1_scripted.yaml --max_steps 5 --batch_size 1
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r3af.models.config import R3AFConfig
from r3af.utils.logging import setup_logging
from r3af.utils.seed import set_seed

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="R3AF Training")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--phase", type=int, default=None, help="Override training phase")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--group_size", type=int, default=None, help="Override group size")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode")
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level")
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # Load config
    config = R3AFConfig.from_yaml(args.config)

    # Apply overrides
    if args.phase is not None:
        config.phase = args.phase
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.group_size is not None:
        config.group_size = args.group_size
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.dry_run:
        config.dry_run = True

    set_seed(config.seed)

    logger.info(f"Training phase: {config.phase}")
    logger.info(f"Config: {config}")

    os.makedirs(config.output_dir, exist_ok=True)
    config.to_yaml(os.path.join(config.output_dir, "config.yaml"))

    # Select trainer based on phase
    if config.phase == 0:
        from r3af.trainers.sft_trainer import SFTTrainer
        trainer = SFTTrainer(config)
        trainer.train()

    elif config.phase in (1, 3):
        from r3af.trainers.retainer_trainer import RetainerTrainer
        trainer = RetainerTrainer(config)
        trainer.train()

    elif config.phase == 2:
        from r3af.trainers.attacker_trainer import AttackerTrainer
        from r3af.models.base import R3AFModel
        from r3af.models.attacker_model import AttackerModel

        retainer = R3AFModel(config)
        attacker = AttackerModel(config)
        trainer = AttackerTrainer(config, retainer, attacker)

        from r3af.tasks.base import TaskRegistry
        task = TaskRegistry.get(config.datasets[0])
        examples = task.load_data(split="train", max_samples=config.max_train_samples)
        train_data = [
            {"query": e.query, "ground_truth": e.ground_truth, "dataset": e.dataset}
            for e in examples
        ]
        trainer.train(train_data)

    elif config.phase == 4:
        # SDPO phase: uses combined loss with lambda_sdpo > 0
        from r3af.trainers.retainer_trainer import RetainerTrainer
        config.lambda_sdpo = config.lambda_sdpo or 0.1
        trainer = RetainerTrainer(config)
        trainer.train()

    else:
        raise ValueError(f"Unknown phase: {config.phase}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
