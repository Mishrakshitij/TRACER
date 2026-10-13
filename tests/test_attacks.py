"""Tests for R3AF attack modules."""

import pytest

from r3af.attacks.reasoning_rewrite import ReasoningRewriteAttack
from r3af.attacks.preemptive_answer import PreemptiveAnswerAttack
from r3af.attacks.prompt_injection import PromptInjectionAttack
from r3af.attacks.long_context_padding import LongContextPaddingAttack
from r3af.attacks.benign_followup import BenignFollowupAttack
from r3af.attacks.attack_mixer import AttackMixer
from r3af.attacks.base_attack import AttackRegistry


class TestReasoningRewrite:
    def test_generates_nonempty(self, mock_transcript):
        attack = ReasoningRewriteAttack()
        text = attack.generate(mock_transcript)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_is_not_valid_evidence(self):
        attack = ReasoningRewriteAttack()
        assert attack.is_valid_evidence() is False


class TestPreemptiveAnswer:
    def test_generates_nonempty(self, mock_transcript):
        attack = PreemptiveAnswerAttack()
        text = attack.generate(mock_transcript)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_is_not_valid_evidence(self):
        attack = PreemptiveAnswerAttack()
        assert attack.is_valid_evidence() is False


class TestPromptInjection:
    def test_generates_nonempty(self, mock_transcript):
        attack = PromptInjectionAttack()
        text = attack.generate(mock_transcript)
        assert isinstance(text, str)
        assert len(text) > 0


class TestLongContextPadding:
    def test_generates_nonempty(self, mock_transcript):
        attack = LongContextPaddingAttack(num_examples=3)
        text = attack.generate(mock_transcript)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_respects_max_tokens(self):
        attack = LongContextPaddingAttack(num_examples=3, max_tokens=50)
        from r3af.environment.transcript import Transcript
        t = Transcript(ground_truth="5")
        text = attack.generate(t)
        words = text.split()
        assert len(words) <= 55  # Some slack


class TestBenignFollowup:
    def test_correct_answer_gives_clarification(self, mock_transcript):
        attack = BenignFollowupAttack()
        text = attack.generate(mock_transcript)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_is_valid_evidence(self):
        attack = BenignFollowupAttack()
        assert attack.is_valid_evidence() is True


class TestAttackMixer:
    def test_mixer_returns_tuple(self, mock_transcript, dummy_config):
        mixer = AttackMixer(dummy_config)
        text, meta = mixer(mock_transcript)
        assert isinstance(text, str)
        assert isinstance(meta, dict)
        assert "attack_name" in meta
        assert "is_valid_evidence" in meta

    def test_mixer_includes_benign(self, mock_transcript, dummy_config):
        """Benign should appear at benign_ratio frequency over many samples."""
        mixer = AttackMixer(dummy_config)
        benign_count = 0
        n = 100
        for _ in range(n):
            _, meta = mixer(mock_transcript)
            if meta["attack_name"] == "benign_followup":
                benign_count += 1
        # Should be roughly 15% (but allow wide range)
        assert benign_count > 0


class TestAttackRegistry:
    def test_registered_attacks(self):
        attacks = AttackRegistry.list_attacks()
        assert "reasoning_rewrite" in attacks
        assert "preemptive_answer" in attacks
        assert "prompt_injection" in attacks
        assert "long_context_padding" in attacks
        assert "benign_followup" in attacks
