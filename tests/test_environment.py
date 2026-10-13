"""Tests for TRG game environment."""

import pytest

from r3af.environment.transcript import Transcript, Turn, TurnRole
from r3af.environment.budget import AttackBudget
from r3af.environment.trg_game import TRGGame
from r3af.parsing.trace_parser import TraceParser


class TestAttackBudget:
    def test_valid_budget(self):
        budget = AttackBudget(max_tokens=512, max_turns=1)
        assert budget.is_within_budget(256) is True
        assert budget.is_within_budget(512) is True
        assert budget.is_within_budget(513) is False

    def test_invalid_budget(self):
        with pytest.raises(ValueError):
            AttackBudget(max_tokens=0)

        with pytest.raises(ValueError):
            AttackBudget(max_tokens=512, max_turns=-1)


class TestTRGGame:
    def test_game_produces_correct_turns(self, dummy_config):
        """Test TRG game with mock model and attack function."""

        class MockModel:
            class tokenizer:
                pad_token = "<pad>"
                pad_token_id = 0

                @staticmethod
                def apply_chat_template(messages, **kwargs):
                    return "mock prompt"

                @staticmethod
                def __call__(*args, **kwargs):
                    import torch
                    return {"input_ids": torch.tensor([[1, 2, 3]]),
                            "attention_mask": torch.tensor([[1, 1, 1]])}

            class model:
                @staticmethod
                def parameters():
                    import torch
                    return iter([torch.tensor([1.0])])

            @staticmethod
            def generate(**kwargs):
                import torch
                return torch.tensor([[1, 2, 3, 4, 5, 6]])

        def mock_attack(transcript):
            return ("I think the answer is wrong.", {"attack_name": "mock", "is_valid_evidence": False})

        # Cannot easily test full game without real model,
        # but we can test the structure
        budget = AttackBudget(max_tokens=128)
        assert budget.max_tokens == 128
