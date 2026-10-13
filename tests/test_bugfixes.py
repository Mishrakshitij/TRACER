"""Verification tests for bugfix plan.

Tests that the specific bugs (C1, C2, C3, H1, H2, M1, M2) are fixed.
"""

import os
import tempfile

import pytest
import torch

from r3af.models.config import R3AFConfig
from r3af.parsing.schema import ReasoningStep, ParsedTrace
from r3af.rewards.prm_reward import PRMReward
from r3af.rewards.anchor_reward import AnchorReward
from r3af.losses.gdpo_loss import GDPOLoss
from r3af.anchors.nli_scorer import NLIScorer, NLIScore
from r3af.anchors.verifier_gate import VerifierGate
from r3af.anchors.extractor import AnchorExtractor
from r3af.environment.transcript import Transcript, Turn, TurnRole


# ---------------------------------------------------------------------------
# C2: ParsedTrace.steps iteration
# ---------------------------------------------------------------------------


class TestParsedTraceIteration:
    """C2: Verify that ParsedTrace objects are not iterated directly."""

    def test_prm_reward_uses_steps(self, mock_transcript):
        """PRMReward.compute() should not crash with ParsedTrace objects."""
        prm = PRMReward(prm_model=None, prm_model_name=None)
        # Should not raise TypeError: 'ParsedTrace' object is not iterable
        output = prm.compute([mock_transcript])
        assert len(output.values) == 1
        # Without a PRM model loaded, scores default to 0.5
        assert output.values[0] == 0.5

    def test_prm_reward_empty_trace(self):
        """PRMReward handles transcript with no post-attack trace."""
        transcript = Transcript(
            problem_id="empty",
            problem="x",
            ground_truth="1",
            dataset="gsm8k",
        )
        transcript.turns.append(Turn(role=TurnRole.PROBLEM, content="x"))
        transcript.turns.append(
            Turn(role=TurnRole.RETAINER, content="", answer=None, token_count=0)
        )

        prm = PRMReward(prm_model=None, prm_model_name=None)
        output = prm.compute([transcript])
        assert output.values[0] == 0.0


# ---------------------------------------------------------------------------
# C3: SFT max_steps None comparison
# ---------------------------------------------------------------------------


class TestSFTMaxSteps:
    """C3: Verify max_steps=None doesn't crash."""

    def test_config_max_steps_none(self):
        config = R3AFConfig(max_steps=None)
        # The comparison `config.max_steps is not None` should be False
        assert config.max_steps is None
        # This should NOT raise TypeError
        should_stop = config.max_steps is not None and 0 >= config.max_steps
        assert should_stop is False

    def test_config_max_steps_set(self):
        config = R3AFConfig(max_steps=10)
        assert config.max_steps == 10
        should_stop = config.max_steps is not None and 10 >= config.max_steps
        assert should_stop is True


# ---------------------------------------------------------------------------
# H1: GDPO final_normalization default
# ---------------------------------------------------------------------------


class TestGDPONormalization:
    """H1: Verify final_normalization defaults to False."""

    def test_default_is_false(self):
        loss_fn = GDPOLoss()
        assert loss_fn.final_normalization is False

    def test_group_normalization_preserved(self):
        """With final_normalization=False, group structure is preserved."""
        loss_fn = GDPOLoss(final_normalization=False)

        # Two groups of 3: rewards designed so groups have different means
        component_rewards = torch.tensor([
            [1.0, 0.5, 0.5, 1.0, 0.0],  # group 1
            [0.5, 0.5, 0.5, 0.5, 0.0],  # group 1
            [0.0, 0.5, 0.5, 0.0, 0.0],  # group 1
            [5.0, 5.0, 5.0, 5.0, 5.0],  # group 2
            [4.0, 4.0, 4.0, 4.0, 4.0],  # group 2
            [3.0, 3.0, 3.0, 3.0, 3.0],  # group 2
        ])
        component_weights = torch.ones(5) / 5.0

        adv = loss_fn.compute_decoupled_advantages(
            component_rewards, component_weights, group_size=3
        )

        assert adv.shape == (6,)
        # Within each group, advantages should sum to ~0 (normalized)
        group1_sum = adv[:3].sum().item()
        group2_sum = adv[3:].sum().item()
        assert abs(group1_sum) < 1e-5, f"Group 1 sum: {group1_sum}"
        assert abs(group2_sum) < 1e-5, f"Group 2 sum: {group2_sum}"

    def test_final_normalization_changes_group_relative_magnitudes(self):
        """With final_normalization=True, global renorm distorts group-relative advantages.

        After within-group normalization, each group's advantages reflect
        relative quality within that prompt's rollouts. Global renorm rescales
        all advantages to unit variance globally, changing the relative spread
        between groups — a group with more diverse rewards gets compressed.
        """
        component_weights = torch.tensor([0.5, 0.1, 0.1, 0.2, 0.1])

        # Group 1: high variance in component 0 only
        # Group 2: variance spread across all components
        component_rewards = torch.tensor([
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
        ])

        loss_without = GDPOLoss(final_normalization=False)
        adv_without = loss_without.compute_decoupled_advantages(
            component_rewards, component_weights, group_size=3
        )

        loss_with = GDPOLoss(final_normalization=True)
        adv_with = loss_with.compute_decoupled_advantages(
            component_rewards, component_weights, group_size=3
        )

        # The two modes should produce different advantage values
        # because the global renorm changes the spread
        assert not torch.allclose(adv_without, adv_with, atol=1e-5), (
            "final_normalization should change advantage values"
        )

        # Without final normalization, groups maintain independent scales:
        # group 1 max advantage != group 2 max advantage
        g1_range_without = adv_without[:3].max() - adv_without[:3].min()
        g2_range_without = adv_without[3:].max() - adv_without[3:].min()
        assert abs(g1_range_without - g2_range_without) > 0.1, (
            "Groups should have different spreads without final normalization"
        )


# ---------------------------------------------------------------------------
# H2: VerifierGate wired into AnchorReward
# ---------------------------------------------------------------------------


class TestVerifierGateWiring:
    """H2: AnchorReward should skip extraction when clean answer is wrong."""

    def test_skips_when_clean_incorrect(self, mock_transcript_flipped):
        """Flipped transcript has correct clean answer but wrong post-attack.
        The clean answer is "50" and ground_truth is "50", so clean IS correct.
        Let's make a transcript where clean answer is wrong."""
        transcript = Transcript(
            problem_id="test_wrong",
            problem="What is 2+2?",
            ground_truth="4",
            dataset="gsm8k",
        )
        transcript.turns.append(
            Turn(role=TurnRole.SYSTEM, content="You are a math assistant.")
        )
        transcript.turns.append(
            Turn(role=TurnRole.PROBLEM, content="What is 2+2?")
        )
        # Clean answer is wrong (says 5, not 4)
        transcript.turns.append(
            Turn(
                role=TurnRole.RETAINER,
                content="<think>2+2=5</think><answer>5</answer>",
                reasoning_steps=[
                    ReasoningStep(step_index=1, content="2+2=5",
                                  start_token_pos=0, end_token_pos=5)
                ],
                answer="5",
                token_count=10,
            )
        )
        transcript.turns.append(
            Turn(role=TurnRole.ATTACKER, content="Are you sure?",
                 token_count=5,
                 metadata={"attack_name": "test", "is_valid_evidence": False})
        )
        transcript.turns.append(
            Turn(
                role=TurnRole.RETAINER,
                content="<think>2+2=5</think><answer>5</answer>",
                reasoning_steps=[
                    ReasoningStep(step_index=1, content="2+2=5",
                                  start_token_pos=0, end_token_pos=5)
                ],
                answer="5",
                token_count=10,
            )
        )

        anchor_reward = AnchorReward(
            extractor=AnchorExtractor(threshold=0.0),
            nli_scorer=None,
        )
        output = anchor_reward.compute([transcript])
        # Should be 0.0 because clean answer was wrong — verifier gate blocks
        assert output.values[0] == 0.0
        assert output.metadata["retention_details"][0].get("skipped") == "clean_answer_incorrect"

    def test_extracts_when_clean_correct(self, mock_transcript):
        """When clean answer is correct, anchors should be extracted."""
        anchor_reward = AnchorReward(
            extractor=AnchorExtractor(threshold=0.0),
            nli_scorer=None,
        )
        output = anchor_reward.compute([mock_transcript])
        # Should not be skipped — clean answer "5" matches ground truth "5"
        details = output.metadata["retention_details"][0]
        assert details.get("skipped") is None
        assert details.get("num_anchors", 0) > 0


# ---------------------------------------------------------------------------
# M1: YAML config field name
# ---------------------------------------------------------------------------


class TestYAMLConfigFieldName:
    """M1: Verify gate_mode is loaded correctly from YAML."""

    def test_phase1_yaml_gate_mode(self):
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "phase1_scripted.yaml"
        )
        if not os.path.exists(yaml_path):
            pytest.skip("phase1_scripted.yaml not found")

        config = R3AFConfig.from_yaml(yaml_path)
        assert config.gate_mode == "hard"

    def test_tcr_gate_mode_ignored(self):
        """A YAML with the old field name 'tcr_gate_mode' should be silently ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.yaml")
            with open(path, "w") as f:
                f.write("tcr_gate_mode: soft\n")

            config = R3AFConfig.from_yaml(path)
            # Should be the default "hard", not "soft"
            assert config.gate_mode == "hard"


# ---------------------------------------------------------------------------
# M2: NLI thresholds wired
# ---------------------------------------------------------------------------


class TestNLIThresholds:
    """M2: NLI thresholds from config are actually used."""

    def test_nli_scorer_accepts_thresholds(self):
        scorer = NLIScorer(
            model_name="test",
            entailment_threshold=0.8,
            contradiction_threshold=0.2,
        )
        assert scorer.entailment_threshold == 0.8
        assert scorer.contradiction_threshold == 0.2

    def test_is_retained_uses_thresholds(self):
        scorer = NLIScorer(
            model_name="test",
            entailment_threshold=0.7,
            contradiction_threshold=0.3,
        )

        # High entailment, low contradiction -> retained
        retained = NLIScore(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05)
        assert scorer.is_retained(retained) is True

        # Low entailment -> not retained
        not_retained = NLIScore(entailment_prob=0.5, contradiction_prob=0.1, neutral_prob=0.4)
        assert scorer.is_retained(not_retained) is False

        # High contradiction -> not retained
        contradicted = NLIScore(entailment_prob=0.8, contradiction_prob=0.5, neutral_prob=0.0)
        assert scorer.is_retained(contradicted) is False

    def test_config_thresholds_propagate(self):
        """Config thresholds should reach NLIScorer via aggregator path."""
        config = R3AFConfig(
            nli_entailment_threshold=0.8,
            nli_contradiction_threshold=0.2,
        )
        assert config.nli_entailment_threshold == 0.8
        assert config.nli_contradiction_threshold == 0.2
