"""Tests for R3AF trainers (structural tests only, no model loading)."""

import pytest

from r3af.models.config import R3AFConfig


class TestTrainerImports:
    def test_sft_trainer_importable(self):
        from r3af.trainers.sft_trainer import SFTTrainer
        assert SFTTrainer is not None

    def test_retainer_trainer_importable(self):
        from r3af.trainers.retainer_trainer import RetainerTrainer
        assert RetainerTrainer is not None

    def test_attacker_trainer_importable(self):
        from r3af.trainers.attacker_trainer import AttackerTrainer
        assert AttackerTrainer is not None

    def test_selfplay_trainer_importable(self):
        from r3af.trainers.selfplay_trainer import SelfPlayTrainer
        assert SelfPlayTrainer is not None
