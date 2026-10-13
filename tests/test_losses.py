"""Tests for R3AF loss functions."""

import pytest
import torch

from r3af.losses.grpo_loss import GRPOLoss
from r3af.losses.gdpo_loss import GDPOLoss
from r3af.losses.tcr_kl_loss import TCRKLLoss
from r3af.losses.sdpo_loss import SDPOLoss
from r3af.losses.combined_loss import CombinedR3AFLoss
from r3af.losses import LOSS_REGISTRY


class TestGRPOLoss:
    def test_basic_computation(self, random_log_probs, response_mask):
        loss_fn = GRPOLoss(beta=0.04, epsilon=0.2)
        batch_size = random_log_probs.shape[0]

        output = loss_fn.compute_loss(
            log_probs=random_log_probs,
            old_log_probs=random_log_probs.detach(),
            ref_log_probs=random_log_probs.detach(),
            advantages=torch.randn(batch_size),
            mask=response_mask,
        )

        assert output.total_loss.shape == ()
        assert output.policy_loss.shape == ()
        assert output.kl_loss.shape == ()
        assert "clip_fraction" in output.metrics
        assert "approx_kl" in output.metrics

    def test_zero_mask(self):
        loss_fn = GRPOLoss()
        batch_size, seq_len = 2, 16
        log_probs = torch.randn(batch_size, seq_len)
        mask = torch.zeros(batch_size, seq_len)

        output = loss_fn.compute_loss(
            log_probs=log_probs,
            old_log_probs=log_probs.detach(),
            ref_log_probs=log_probs.detach(),
            advantages=torch.randn(batch_size),
            mask=mask,
        )

        assert output.total_loss.item() == 0.0


class TestGDPOLoss:
    def test_with_component_rewards(self, random_log_probs, response_mask):
        loss_fn = GDPOLoss(beta=0.04, epsilon=0.2)
        batch_size = random_log_probs.shape[0]

        component_rewards = torch.randn(batch_size, 5)
        component_weights = torch.tensor([0.25, 0.20, 0.20, 0.25, 0.10])

        output = loss_fn.compute_loss(
            log_probs=random_log_probs,
            old_log_probs=random_log_probs.detach(),
            ref_log_probs=random_log_probs.detach(),
            advantages=torch.randn(batch_size),
            mask=response_mask,
            component_rewards=component_rewards,
            component_weights=component_weights,
            group_size=2,
        )

        assert output.total_loss.shape == ()
        assert "component_0_mean_adv" in output.metrics

    def test_decoupled_advantages(self):
        loss_fn = GDPOLoss()
        batch_size, num_components = 8, 5

        component_rewards = torch.randn(batch_size, num_components)
        component_weights = torch.ones(num_components) / num_components

        adv = loss_fn.compute_decoupled_advantages(
            component_rewards, component_weights, group_size=4
        )

        assert adv.shape == (batch_size,)


class TestTCRKLLoss:
    def test_zero_when_identical(self, random_logits):
        loss_fn = TCRKLLoss(temperature=1.0)
        batch_size, seq_len = random_logits.shape[0], random_logits.shape[1]

        mask = torch.ones(batch_size, seq_len)
        gate = torch.ones(batch_size)

        output = loss_fn.compute_loss(
            clean_logits=random_logits,
            attacked_logits=random_logits.clone(),
            committed_mask=mask,
            gate=gate,
        )

        assert output.total_loss.item() < 1e-5

    def test_positive_when_different(self, random_logits):
        loss_fn = TCRKLLoss(temperature=1.0)
        batch_size, seq_len = random_logits.shape[0], random_logits.shape[1]

        different_logits = random_logits + torch.randn_like(random_logits) * 5.0
        mask = torch.ones(batch_size, seq_len)
        gate = torch.ones(batch_size)

        output = loss_fn.compute_loss(
            clean_logits=random_logits,
            attacked_logits=different_logits,
            committed_mask=mask,
            gate=gate,
        )

        assert output.total_loss.item() > 0.0

    def test_gate_masking(self, random_logits):
        loss_fn = TCRKLLoss(temperature=1.0)
        batch_size, seq_len = random_logits.shape[0], random_logits.shape[1]

        different_logits = random_logits + torch.randn_like(random_logits) * 5.0
        mask = torch.ones(batch_size, seq_len)
        gate = torch.zeros(batch_size)  # All gated off

        output = loss_fn.compute_loss(
            clean_logits=random_logits,
            attacked_logits=different_logits,
            committed_mask=mask,
            gate=gate,
        )

        assert output.total_loss.item() < 1e-5

    def test_gradient_flow(self, random_logits):
        """Verify gradients flow through attacked_logits but not clean_logits."""
        loss_fn = TCRKLLoss(temperature=1.0)
        batch_size, seq_len = random_logits.shape[0], random_logits.shape[1]

        clean = random_logits.detach().requires_grad_(True)
        attacked = random_logits.clone().requires_grad_(True)
        mask = torch.ones(batch_size, seq_len)
        gate = torch.ones(batch_size)

        output = loss_fn.compute_loss(
            clean_logits=clean,
            attacked_logits=attacked,
            committed_mask=mask,
            gate=gate,
        )

        output.total_loss.backward()

        # Clean should NOT have gradients (detached inside loss)
        assert clean.grad is None or clean.grad.abs().sum() == 0
        # Attacked SHOULD have gradients
        assert attacked.grad is not None
        assert attacked.grad.abs().sum() > 0

    def test_build_committed_mask(self):
        mask = TCRKLLoss.build_committed_mask(
            anchor_positions=[[(2, 5), (8, 12)], [(3, 7)]],
            seq_len=16,
            device=torch.device("cpu"),
            batch_size=2,
        )

        assert mask.shape == (2, 16)
        assert mask[0, 2:5].sum() == 3
        assert mask[0, 8:12].sum() == 4
        assert mask[1, 3:7].sum() == 4
        assert mask[0, 0:2].sum() == 0

    def test_build_gate(self):
        clean_correct = torch.tensor([1.0, 1.0, 0.0, 0.0])
        evidence_invalid = torch.tensor([1.0, 0.0, 1.0, 0.0])

        gate = TCRKLLoss.build_gate(clean_correct, evidence_invalid, mode="hard")

        assert gate[0].item() == 1.0  # correct AND invalid
        assert gate[1].item() == 0.0  # correct but valid evidence
        assert gate[2].item() == 0.0  # wrong
        assert gate[3].item() == 0.0  # wrong


class TestSDPOLoss:
    def test_zero_when_identical(self, random_logits, response_mask):
        loss_fn = SDPOLoss(temperature=1.0)

        output = loss_fn.compute_loss(
            base_logits=random_logits.clone(),
            feedback_logits=random_logits,
            response_mask=response_mask,
        )

        assert output.total_loss.item() < 1e-5

    def test_positive_when_different(self, random_logits, response_mask):
        loss_fn = SDPOLoss(temperature=1.0)
        different = random_logits + torch.randn_like(random_logits) * 5.0

        output = loss_fn.compute_loss(
            base_logits=different,
            feedback_logits=random_logits,
            response_mask=response_mask,
        )

        assert output.total_loss.item() > 0.0


class TestCombinedLoss:
    def test_combined_with_all_components(self, random_log_probs, random_logits, response_mask):
        loss_fn = CombinedR3AFLoss(
            beta=0.04,
            epsilon=0.2,
            lambda_tcr=0.1,
            lambda_sdpo=0.1,
        )
        batch_size = random_log_probs.shape[0]
        seq_len = random_log_probs.shape[1]

        output = loss_fn.compute_loss(
            log_probs=random_log_probs,
            old_log_probs=random_log_probs.detach(),
            ref_log_probs=random_log_probs.detach(),
            advantages=torch.randn(batch_size),
            mask=response_mask,
            component_rewards=torch.randn(batch_size, 5),
            component_weights=torch.tensor([0.25, 0.20, 0.20, 0.25, 0.10]),
            group_size=2,
            clean_logits=random_logits,
            attacked_logits=random_logits + torch.randn_like(random_logits) * 0.1,
            committed_mask=torch.ones(batch_size, seq_len),
            tcr_gate=torch.ones(batch_size),
            base_logits=random_logits,
            feedback_logits=random_logits,
            response_mask=response_mask,
        )

        assert output.total_loss.shape == ()
        assert "policy_loss" in output.metrics
        assert "reg_loss" in output.metrics
        assert "sdpo_loss" in output.metrics

    def test_combined_gdpo_only(self, random_log_probs, response_mask):
        """When TCR and SDPO disabled, should equal GDPO."""
        loss_fn = CombinedR3AFLoss(
            lambda_tcr=0.0,
            lambda_sdpo=0.0,
        )
        batch_size = random_log_probs.shape[0]

        output = loss_fn.compute_loss(
            log_probs=random_log_probs,
            old_log_probs=random_log_probs.detach(),
            ref_log_probs=random_log_probs.detach(),
            advantages=torch.randn(batch_size),
            mask=response_mask,
        )

        assert output.total_loss.shape == ()


class TestLossRegistry:
    def test_registry_has_all_losses(self):
        assert "grpo" in LOSS_REGISTRY
        assert "gdpo" in LOSS_REGISTRY
        assert "tcr_kl" in LOSS_REGISTRY
        assert "sdpo" in LOSS_REGISTRY
        assert "combined" in LOSS_REGISTRY
