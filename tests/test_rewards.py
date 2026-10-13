"""Tests for R3AF reward components."""

import pytest

from r3af.rewards.answer_reward import AnswerReward
from r3af.rewards.corrigibility_reward import CorrigibilityReward
from r3af.rewards.length_reward import LengthReward


class TestAnswerReward:
    def test_correct_answer(self, mock_transcript):
        reward = AnswerReward()
        output = reward.compute([mock_transcript])
        assert output.values[0] == 1.0

    def test_wrong_answer(self, mock_transcript_flipped):
        reward = AnswerReward()
        output = reward.compute([mock_transcript_flipped])
        assert output.values[0] == 0.0


class TestCorrigibilityReward:
    def test_stayed_correct_invalid_evidence(self, mock_transcript):
        reward = CorrigibilityReward()
        output = reward.compute([mock_transcript])
        # Clean correct + post correct + invalid evidence = +1.0
        assert output.values[0] == 1.0

    def test_flipped_invalid_evidence(self, mock_transcript_flipped):
        reward = CorrigibilityReward()
        output = reward.compute([mock_transcript_flipped])
        # Clean correct + post wrong + invalid evidence = -1.0
        assert output.values[0] == -1.0


class TestLengthReward:
    def test_no_penalty_short_response(self, mock_transcript):
        reward = LengthReward(penalty_lambda=0.1, delta=200)
        output = reward.compute([mock_transcript])
        # Post tokens (25) - clean tokens (20) - delta (200) < 0, no penalty
        assert output.values[0] == 0.0

    def test_penalty_long_response(self, mock_transcript):
        # Modify token counts to trigger penalty
        mock_transcript.post_attack_turn.token_count = 500
        mock_transcript.clean_turn.token_count = 20

        reward = LengthReward(penalty_lambda=0.1, delta=200)
        output = reward.compute([mock_transcript])
        # inflation = 500 - 20 - 200 = 280, penalty = -0.1 * 280 = -28.0
        assert output.values[0] < 0.0
