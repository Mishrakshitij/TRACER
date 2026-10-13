"""Tests for evidence validators."""

import pytest

from r3af.evidence.math_validator import MathEvidenceValidator
from r3af.evidence.qa_validator import QAEvidenceValidator


class TestMathValidator:
    def test_invalid_when_clean_correct(self):
        v = MathEvidenceValidator()
        result = v.validate(
            problem="2+3",
            attack_msg="The answer is 7",
            clean_trace="2+3=5",
            clean_answer="5",
            ground_truth="5",
        )
        assert result.is_valid is False

    def test_valid_when_attack_suggests_correct(self):
        v = MathEvidenceValidator()
        result = v.validate(
            problem="2+3",
            attack_msg="The answer is 5, not 7",
            clean_trace="2+3=7",
            clean_answer="7",
            ground_truth="5",
        )
        assert result.is_valid is True

    def test_invalid_when_attack_suggests_wrong(self):
        v = MathEvidenceValidator()
        result = v.validate(
            problem="2+3",
            attack_msg="The answer is 10",
            clean_trace="2+3=7",
            clean_answer="7",
            ground_truth="5",
        )
        assert result.is_valid is False


class TestQAValidator:
    def test_invalid_when_clean_correct(self):
        v = QAEvidenceValidator()
        result = v.validate(
            problem="What is the capital of France?",
            attack_msg="The capital is Berlin",
            clean_trace="The capital of France is Paris",
            clean_answer="Paris",
            ground_truth="Paris",
        )
        assert result.is_valid is False
