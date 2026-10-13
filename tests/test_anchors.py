"""Tests for anchor extraction and NLI scoring."""

import pytest

from r3af.anchors.extractor import AnchorExtractor, AnchorProposition
from r3af.anchors.verifier_gate import VerifierGate
from r3af.parsing.schema import ReasoningStep


class TestAnchorExtractor:
    def test_step_level_extraction(self):
        extractor = AnchorExtractor(threshold=0.5, mode="step_level")
        steps = [
            ReasoningStep(step_index=1, content="2 + 3 = 5"),
            ReasoningStep(step_index=2, content="5 * 2 = 10"),
        ]
        prm_scores = [0.8, 0.3]

        anchors = extractor.extract(steps, prm_scores)
        assert len(anchors) == 1  # Only step 1 passes threshold
        assert anchors[0].source_step == 1
        assert anchors[0].prm_score == 0.8

    def test_all_pass_threshold(self):
        extractor = AnchorExtractor(threshold=0.5, mode="step_level")
        steps = [
            ReasoningStep(step_index=1, content="2 + 3 = 5"),
            ReasoningStep(step_index=2, content="5 * 2 = 10"),
        ]
        prm_scores = [0.9, 0.8]

        anchors = extractor.extract(steps, prm_scores)
        assert len(anchors) == 2

    def test_no_prm_scores(self):
        extractor = AnchorExtractor(threshold=0.5, mode="step_level")
        steps = [ReasoningStep(step_index=1, content="test")]

        anchors = extractor.extract(steps, None)
        assert len(anchors) == 1  # Default score 1.0 passes

    def test_proposition_level(self):
        extractor = AnchorExtractor(threshold=0.5, mode="proposition_level")
        steps = [
            ReasoningStep(step_index=1, content="We know that x = 5 and y = 10"),
        ]
        prm_scores = [0.9]

        anchors = extractor.extract(steps, prm_scores)
        assert len(anchors) >= 1


class TestVerifierGate:
    def test_filters_when_incorrect(self):
        gate = VerifierGate(prm_threshold=0.5)
        anchors = [
            AnchorProposition(content="x = 5", source_step=1, prm_score=0.9),
        ]
        filtered = gate.filter(anchors, clean_correct=False)
        assert len(filtered) == 0

    def test_passes_when_correct(self):
        gate = VerifierGate(prm_threshold=0.5)
        anchors = [
            AnchorProposition(content="x = 5", source_step=1, prm_score=0.9),
        ]
        filtered = gate.filter(anchors, clean_correct=True)
        assert len(filtered) == 1

    def test_filters_low_prm(self):
        gate = VerifierGate(prm_threshold=0.8)
        anchors = [
            AnchorProposition(content="x = 5", source_step=1, prm_score=0.5),
        ]
        filtered = gate.filter(anchors, clean_correct=True)
        assert len(filtered) == 0
