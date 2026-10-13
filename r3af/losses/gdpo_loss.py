import torch
from typing import Optional, Dict

from .base_loss import BasePolicyLoss, LossOutput


class GDPOLoss(BasePolicyLoss):
    """Group Decoupled Policy Optimization loss (adapted from cd_e2).

    Per-component normalization for R3AF's 5 reward components:
    R_prm, R_anchor, R_corrigible, R_ans, R_len.
    """

    def __init__(
        self,
        beta: float = 0.04,
        epsilon: float = 0.2,
        importance_sampling_level: str = "token",
        normalize_advantages: bool = True,
        final_normalization: bool = False,
        default_num_components: int = 5,
        kl_clip: float = 0.2,
    ):
        super().__init__(
            beta=beta,
            epsilon=epsilon,
            importance_sampling_level=importance_sampling_level,
            kl_clip=kl_clip,
        )
        self._normalize_advantages = normalize_advantages
        self.final_normalization = final_normalization
        self.default_num_components = default_num_components

    def compute_decoupled_advantages(
        self,
        component_rewards: torch.Tensor,
        component_weights: torch.Tensor,
        group_size: Optional[int] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Normalize each reward component independently, then weighted sum."""
        batch_size, num_components = component_rewards.shape
        component_weights = component_weights.to(
            device=component_rewards.device, dtype=component_rewards.dtype
        )

        if group_size is not None and group_size > 0 and group_size < batch_size:
            num_full_groups = batch_size // group_size
            remainder = batch_size % group_size
            normalized = torch.zeros_like(component_rewards)

            for g in range(num_full_groups):
                start = g * group_size
                end = start + group_size
                group_slice = component_rewards[start:end]

                if group_size <= 1:
                    normalized[start:end] = group_slice - group_slice.mean(
                        dim=0, keepdim=True
                    )
                else:
                    group_mean = group_slice.mean(dim=0, keepdim=True)
                    group_std = group_slice.std(dim=0, keepdim=True).clamp(min=eps)
                    normalized[start:end] = (group_slice - group_mean) / group_std

            if remainder > 0:
                start = num_full_groups * group_size
                group_slice = component_rewards[start:]
                if remainder <= 1:
                    normalized[start:] = group_slice - group_slice.mean(
                        dim=0, keepdim=True
                    )
                else:
                    group_mean = group_slice.mean(dim=0, keepdim=True)
                    group_std = group_slice.std(dim=0, keepdim=True).clamp(min=eps)
                    normalized[start:] = (group_slice - group_mean) / group_std
        else:
            if batch_size <= 1:
                normalized = component_rewards - component_rewards.mean(
                    dim=0, keepdim=True
                )
            else:
                comp_mean = component_rewards.mean(dim=0, keepdim=True)
                comp_std = component_rewards.std(dim=0, keepdim=True).clamp(min=eps)
                normalized = (component_rewards - comp_mean) / comp_std

        advantages = (normalized * component_weights.unsqueeze(0)).sum(dim=-1)

        if self.final_normalization and advantages.numel() > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std().clamp(min=eps)
            advantages = (advantages - adv_mean) / adv_std

        return advantages

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

        component_rewards: Optional[torch.Tensor] = kwargs.get(
            "component_rewards", None
        )
        component_weights: Optional[torch.Tensor] = kwargs.get(
            "component_weights", None
        )
        group_size: Optional[int] = kwargs.get("group_size", None)

        per_component_metrics: Dict[str, float] = {}

        if component_rewards is not None and component_weights is not None:
            adv = self.compute_decoupled_advantages(
                component_rewards=component_rewards,
                component_weights=component_weights,
                group_size=group_size,
            )

            with torch.no_grad():
                batch_size, num_components = component_rewards.shape
                if batch_size > 1:
                    comp_mean = component_rewards.mean(dim=0)
                    comp_std = component_rewards.std(dim=0).clamp(min=1e-8)
                    normed = (
                        component_rewards - comp_mean.unsqueeze(0)
                    ) / comp_std.unsqueeze(0)
                else:
                    normed = component_rewards - component_rewards.mean(
                        dim=0, keepdim=True
                    )

                for k in range(num_components):
                    per_component_metrics[f"component_{k}_mean_adv"] = (
                        normed[:, k].mean().item()
                    )
                    per_component_metrics[f"component_{k}_raw_mean"] = (
                        component_rewards[:, k].mean().item()
                    )
        else:
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
            mean_advantage = adv.mean().item()
            mean_ratio = ((ratio * mask).sum() / mask_sum.clamp(min=1)).item()

        metrics: Dict[str, float] = {
            "clip_fraction": clip_fraction.item()
            if isinstance(clip_fraction, torch.Tensor)
            else clip_fraction,
            "approx_kl": approx_kl,
            "kl_penalty": kl.item(),  # non-negative KL used in loss
            "mean_advantage": mean_advantage,
            "mean_ratio": mean_ratio,
            **per_component_metrics,
        }

        return LossOutput(
            total_loss=total_loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            metrics=metrics,
        )
