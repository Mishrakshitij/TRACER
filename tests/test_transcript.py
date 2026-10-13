"""Tests for Transcript dataclass."""

import pytest

from r3af.environment.transcript import Transcript, Turn, TurnRole


class TestTranscript:
    def test_clean_turn(self, mock_transcript):
        clean = mock_transcript.clean_turn
        assert clean is not None
        assert clean.role == TurnRole.RETAINER
        assert clean.answer == "5"

    def test_attack_turn(self, mock_transcript):
        attack = mock_transcript.attack_turn
        assert attack is not None
        assert attack.role == TurnRole.ATTACKER
        assert "7" in attack.content

    def test_post_attack_turn(self, mock_transcript):
        post = mock_transcript.post_attack_turn
        assert post is not None
        assert post.role == TurnRole.RETAINER
        assert post.answer == "5"

    def test_clean_answer(self, mock_transcript):
        assert mock_transcript.clean_answer == "5"

    def test_post_attack_answer(self, mock_transcript):
        assert mock_transcript.post_attack_answer == "5"

    def test_to_messages(self, mock_transcript):
        messages = mock_transcript.to_messages()
        assert len(messages) > 0
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "assistant" in roles
        assert "user" in roles

    def test_flipped_transcript(self, mock_transcript_flipped):
        assert mock_transcript_flipped.clean_answer == "50"
        assert mock_transcript_flipped.post_attack_answer == "55"
