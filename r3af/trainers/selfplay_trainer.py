"""Phase 3: Self-play orchestrator.

Alternates between retainer and attacker training, maintaining an attacker
league pool for diversity.
"""

import os
import logging
from typing import Optional, List, Dict

from ..models.config import R3AFConfig
from ..models.base import R3AFModel
from ..models.attacker_model import AttackerModel
from .retainer_trainer import RetainerTrainer
from .attacker_trainer import AttackerTrainer
from ..attacks.learned_attacker import LearnedAttacker
from ..attacks.attack_mixer import AttackMixer

logger = logging.getLogger(__name__)


class SelfPlayTrainer:
    """Phase 3: Self-play orchestrator with attacker league."""

    def __init__(
        self,
        config: R3AFConfig,
        retainer_model: Optional[R3AFModel] = None,
        attacker_model: Optional[AttackerModel] = None,
    ):
        self.config = config

        self.retainer_model = retainer_model or R3AFModel(config)
        self.attacker_model = attacker_model or AttackerModel(config)

        # Attacker league: pool of past attacker checkpoints
        self.attacker_league: List[str] = []

    def train(
        self,
        num_rounds: int = 5,
        retainer_steps_per_round: int = 100,
        attacker_steps_per_round: int = 50,
        train_data: list = None,
    ):
        """Run self-play training loop."""
        logger.info(f"Starting self-play: {num_rounds} rounds")

        for round_idx in range(num_rounds):
            logger.info(f"=== Self-play Round {round_idx + 1}/{num_rounds} ===")

            # Phase A: Train attacker against current retainer
            logger.info("Training attacker...")
            attacker_config = self.config
            attacker_config.max_steps = attacker_steps_per_round

            attacker_trainer = AttackerTrainer(
                config=attacker_config,
                retainer_model=self.retainer_model,
                attacker_model=self.attacker_model,
            )
            if train_data:
                attacker_trainer.train(train_data, num_epochs=1)

            # Save attacker checkpoint to league
            ckpt_path = os.path.join(
                self.config.output_dir,
                f"attacker-league-round-{round_idx + 1}",
            )
            self.attacker_model.save_checkpoint(ckpt_path)
            self.attacker_league.append(ckpt_path)

            # Phase B: Train retainer against attacker league
            logger.info("Training retainer...")
            # Update attack mixer to include learned attacker
            learned_attacker = LearnedAttacker(self.attacker_model)

            retainer_config = self.config
            retainer_config.max_steps = retainer_steps_per_round

            retainer_trainer = RetainerTrainer(
                config=retainer_config,
                model=self.retainer_model,
            )
            # Override the TRG game's attack function to use learned attacker
            retainer_trainer.trg_game.attack_fn = (
                lambda t: (learned_attacker.generate(t), {"attack_name": "learned", "is_valid_evidence": False})
            )
            retainer_trainer.train()

            # Save retainer checkpoint
            ckpt_path = os.path.join(
                self.config.output_dir,
                f"retainer-round-{round_idx + 1}",
            )
            self.retainer_model.save_checkpoint(ckpt_path)

        logger.info("Self-play training complete!")
