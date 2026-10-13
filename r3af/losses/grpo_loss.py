import torch
from typing import Dict

from .base_loss import BasePolicyLoss, LossOutput


class GRPOLoss(BasePolicyLoss):
    """Group Relative Policy Optimization loss (adapted from cd_e2).

    Takes pre-computed scalar advantages [batch] (combined reward, already group-normalized).
    Computes token-level importance ratios.
    Applies PPO clipping.
    Adds KL divergence penalty: beta * KL(pi || pi_ref).
    """

    def __init__(
        self,
        beta: float = 0.04,
        epsilon: float = 0.2,
        importance_sampling_level: str = "token",
        normalize_advantages: bool = True,
        kl_clip: float = 0.2,
    ):
        super().__init__(
            beta=beta,
            epsilon=epsilon,
            importance_sampling_level=importance_sampling_level,
            kl_clip=kl_clip,
        )
        self._normalize_advantages = normalize_advantages

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        mask_sum = mask.sum()
        if mask_sum == 0:
            zero = torch.tensor(0.0, device=log_probs.device, dtype=log_probs.dtype)
            return LossOutput(
                total_loss=zero,
                policy_loss=zero,
                kl_loss=zero,
                metrics={
                    "clip_fraction": 0.0,
                    "approx_kl": 0.0,
                    "mean_advantage": 0.0,
                    "mean_ratio": 0.0,
                },
            )

        if self._normalize_advantages:
            adv = self.normalize_advantages(advantages)
        else:
            adv = advantages

        ratio = self.compute_importance_ratio(log_probs, old_log_probs, mask)
        clipped_surrogate, clip_fraction = self.clip_ratio(ratio, adv, mask)

        policy_loss = -(clipped_surrogate * mask).sum() / mask_sum.clamp(min=1)

        kl = self.compute_kl_divergence(log_probs, ref_log_probs, mask)
        kl_loss = self.beta * kl

        total_loss = policy_loss + kl_loss

        with torch.no_grad():
            # Signed KL (log_pi - log_pi_ref): diagnostic for drift direction
            kl_signed = ((log_probs - ref_log_probs) * mask).sum() / mask.sum().clamp(min=1)
            approx_kl = kl_signed.item()
            mean_advantage = advantages.mean().item()
            mean_ratio = ((ratio * mask).sum() / mask_sum.clamp(min=1)).item()

        metrics: Dict[str, float] = {
            "clip_fraction": clip_fraction.item()
            if isinstance(clip_fraction, torch.Tensor)
            else clip_fraction,
            "approx_kl": approx_kl,
            "kl_penalty": kl.item(),  # non-negative KL used in loss
            "mean_advantage": mean_advantage,
            "mean_ratio": mean_ratio,
        }

        return LossOutput(
            total_loss=total_loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            metrics=metrics,
        )
