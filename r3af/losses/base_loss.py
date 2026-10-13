import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class LossOutput:
    """Output from loss computation."""

    total_loss: torch.Tensor  # Scalar loss for backward
    policy_loss: torch.Tensor  # Policy gradient loss component
    kl_loss: torch.Tensor  # KL divergence penalty component
    metrics: Dict[str, float] = field(default_factory=dict)


class BasePolicyLoss(ABC, nn.Module):
    """Base class for policy gradient losses (adapted from cd_e2)."""

    def __init__(
        self,
        beta: float = 0.04,
        epsilon: float = 0.2,
        importance_sampling_level: str = "token",
        kl_clip: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.importance_sampling_level = importance_sampling_level
        self.kl_clip = kl_clip

    def compute_importance_ratio(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance sampling ratio pi/pi_old."""
        if self.importance_sampling_level == "token":
            return torch.exp(log_probs - old_log_probs)
        else:
            seq_log_ratio = ((log_probs - old_log_probs) * mask).sum(
                dim=-1
            ) / mask.sum(dim=-1).clamp(min=1)
            return torch.exp(seq_log_ratio).unsqueeze(-1).expand_as(log_probs)

    def compute_kl_divergence(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Non-negative KL divergence via Schulman's approximation.

        Uses KL ≈ (r - 1) - log(r) where r = pi/pi_ref = exp(log_pi - log_pi_ref).
        This is always >= 0, preventing the KL penalty from becoming a reward
        that incentivizes policy divergence.

        Also reports the signed approximate KL (log_pi - log_pi_ref) as a
        diagnostic metric via the approx_kl field in loss outputs.
        """
        log_ratio = log_probs - ref_log_probs
        ratio = torch.exp(log_ratio)
        # Schulman's non-negative KL estimator: (r - 1) - log(r) >= 0
        kl_per_token = (ratio - 1) - log_ratio
        kl_masked = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
        return kl_masked.clamp(max=self.kl_clip)

    def normalize_advantages(
        self,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Normalize advantages to zero mean, unit variance."""
        if advantages.numel() <= 1:
            return advantages
        mean = advantages.mean()
        std = advantages.std().clamp(min=eps)
        return (advantages - mean) / std

    def clip_ratio(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple:
        """PPO clipping. Returns (clipped_loss, clip_fraction)."""
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.epsilon, 1.0 + self.epsilon
        )

        if advantages.dim() == 1:
            advantages_expanded = advantages.unsqueeze(-1).expand_as(ratio)
        else:
            advantages_expanded = advantages

        surr1 = ratio * advantages_expanded
        surr2 = clipped_ratio * advantages_expanded

        policy_loss = torch.where(
            advantages_expanded >= 0,
            torch.min(surr1, surr2),
            torch.max(surr1, surr2),
        )

        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.epsilon).float()
            clip_fraction = (clip_fraction * mask).sum() / mask.sum().clamp(
                min=1
            )

        return policy_loss, clip_fraction

    @abstractmethod
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        ...
