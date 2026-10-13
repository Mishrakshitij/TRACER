import torch
import torch.nn as nn
from typing import Optional, Dict

from .base_loss import BasePolicyLoss, LossOutput
from .grpo_loss import GRPOLoss
from .gdpo_loss import GDPOLoss
from .tcr_kl_loss import TCRKLLoss
from .atr_loss import ATRLoss
from .atrc_loss import ATRCLoss
from .sdpo_loss import SDPOLoss


class CombinedR3AFLoss(BasePolicyLoss):
    """Combined R3AF loss with configurable policy and regularization components.

    L_total = L_policy + lambda_reg * L_reg + lambda_sdpo * L_SDPO

    Policy loss: GRPO or GDPO (configurable via policy_loss_type).
    Regularization: none, TCR-KL, ATR, or ATRC (configurable via reg_loss_type).
    """

    def __init__(
        self,
        beta: float = 0.04,
        epsilon: float = 0.2,
        policy_loss_type: str = "grpo",
        reg_loss_type: str = "none",
        lambda_reg: float = 0.1,
        lambda_tcr: float = 0.0,
        lambda_sdpo: float = 0.0,
        tcr_temperature: float = 1.0,
        tcr_gate_mode: str = "hard",
        atrc_alpha: float = 0.5,
        importance_sampling_level: str = "token",
        kl_clip: float = 0.2,
    ):
        super().__init__(
            beta=beta,
            epsilon=epsilon,
            importance_sampling_level=importance_sampling_level,
            kl_clip=kl_clip,
        )

        # Backward compat: if lambda_tcr > 0 and reg_loss_type is "none",
        # infer that user wants TCR with the old config style
        if reg_loss_type == "none" and lambda_tcr > 0:
            reg_loss_type = "tcr_kl"
            lambda_reg = lambda_tcr

        self.policy_loss_type = policy_loss_type
        self.reg_loss_type = reg_loss_type
        self.lambda_reg = lambda_reg
        self.lambda_tcr = lambda_tcr  # keep for backward compat
        self.lambda_sdpo = lambda_sdpo

        # Policy loss component
        if policy_loss_type == "grpo":
            self.policy_loss_fn = GRPOLoss(
                beta=beta,
                epsilon=epsilon,
                importance_sampling_level=importance_sampling_level,
                kl_clip=kl_clip,
            )
        elif policy_loss_type == "gdpo":
            self.policy_loss_fn = GDPOLoss(
                beta=beta,
                epsilon=epsilon,
                importance_sampling_level=importance_sampling_level,
                kl_clip=kl_clip,
            )
        else:
            raise ValueError(f"Unknown policy_loss_type: {policy_loss_type}")

        # Regularization loss component
        self.reg_loss_fn: Optional[nn.Module] = None
        if reg_loss_type == "tcr_kl":
            self.reg_loss_fn = TCRKLLoss(
                temperature=tcr_temperature,
                gate_mode=tcr_gate_mode,
            )
        elif reg_loss_type == "atr":
            self.reg_loss_fn = ATRLoss(gate_mode=tcr_gate_mode)
        elif reg_loss_type == "atrc":
            self.reg_loss_fn = ATRCLoss(
                alpha=atrc_alpha,
                temperature=tcr_temperature,
                gate_mode=tcr_gate_mode,
            )
        elif reg_loss_type != "none":
            raise ValueError(f"Unknown reg_loss_type: {reg_loss_type}")

        # SDPO
        self.sdpo_loss = SDPOLoss(temperature=1.0)

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        """Compute the combined R3AF loss.

        Expected kwargs:
            component_rewards: [batch, 5] for GDPO
            component_weights: [5] for GDPO
            group_size: int for GDPO
            clean_logits: [batch, resp_len, vocab] for TCR-KL / ATRC
            attacked_logits: [batch, resp_len, vocab] for TCR-KL / ATR / ATRC
            committed_mask: [batch, resp_len] for TCR-KL / ATR / ATRC
            tcr_gate: [batch] for TCR-KL / ATR / ATRC
            clean_token_ids: [batch, resp_len] for ATR / ATRC
            base_logits: [batch, seq_len, vocab] for SDPO
            feedback_logits: [batch, seq_len, vocab] for SDPO
            response_mask: [batch, seq_len] for SDPO
        """
        # 1. Policy loss
        policy_output = self.policy_loss_fn.compute_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            mask=mask,
            component_rewards=kwargs.get("component_rewards"),
            component_weights=kwargs.get("component_weights"),
            group_size=kwargs.get("group_size"),
        )

        total_loss = policy_output.total_loss
        all_metrics = {
            f"{self.policy_loss_type}/{k}": v
            for k, v in policy_output.metrics.items()
        }

        # 2. Regularization loss (if enabled and inputs provided)
        reg_total = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)
        if self.reg_loss_fn is not None and self.lambda_reg > 0:
            attacked_logits = kwargs.get("attacked_logits")
            committed_mask = kwargs.get("committed_mask")
            tcr_gate = kwargs.get("tcr_gate")

            if (
                attacked_logits is not None
                and committed_mask is not None
                and tcr_gate is not None
            ):
                if self.reg_loss_type == "tcr_kl":
                    clean_logits = kwargs.get("clean_logits")
                    if clean_logits is not None:
                        reg_output = self.reg_loss_fn.compute_loss(
                            clean_logits=clean_logits,
                            attacked_logits=attacked_logits,
                            committed_mask=committed_mask,
                            gate=tcr_gate,
                        )
                        reg_total = reg_output.total_loss
                        all_metrics.update(
                            {f"tcr/{k}": v for k, v in reg_output.metrics.items()}
                        )

                elif self.reg_loss_type == "atr":
                    clean_token_ids = kwargs.get("clean_token_ids")
                    if clean_token_ids is not None:
                        reg_output = self.reg_loss_fn.compute_loss(
                            attacked_logits=attacked_logits,
                            clean_token_ids=clean_token_ids,
                            committed_mask=committed_mask,
                            gate=tcr_gate,
                        )
                        reg_total = reg_output.total_loss
                        all_metrics.update(
                            {f"atr/{k}": v for k, v in reg_output.metrics.items()}
                        )

                elif self.reg_loss_type == "atrc":
                    clean_logits = kwargs.get("clean_logits")
                    clean_token_ids = kwargs.get("clean_token_ids")
                    if clean_logits is not None and clean_token_ids is not None:
                        reg_output = self.reg_loss_fn.compute_loss(
                            clean_logits=clean_logits,
                            attacked_logits=attacked_logits,
                            clean_token_ids=clean_token_ids,
                            committed_mask=committed_mask,
                            gate=tcr_gate,
                        )
                        reg_total = reg_output.total_loss
                        all_metrics.update(
                            {f"atrc/{k}": v for k, v in reg_output.metrics.items()}
                        )

                total_loss = total_loss + self.lambda_reg * reg_total

        # 3. SDPO loss (if enabled and inputs provided)
        sdpo_total = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)
        if self.lambda_sdpo > 0:
            base_logits = kwargs.get("base_logits")
            feedback_logits = kwargs.get("feedback_logits")
            response_mask = kwargs.get("response_mask", mask)

            if base_logits is not None and feedback_logits is not None:
                sdpo_output = self.sdpo_loss.compute_loss(
                    base_logits=base_logits,
                    feedback_logits=feedback_logits,
                    response_mask=response_mask,
                )
                sdpo_total = sdpo_output.total_loss
                total_loss = total_loss + self.lambda_sdpo * sdpo_total
                all_metrics.update(
                    {f"sdpo/{k}": v for k, v in sdpo_output.metrics.items()}
                )

        all_metrics["total_loss"] = total_loss.item()
        all_metrics["policy_loss"] = policy_output.total_loss.item()
        all_metrics["reg_loss"] = reg_total.item()
        all_metrics["sdpo_loss"] = sdpo_total.item()

        return LossOutput(
            total_loss=total_loss,
            policy_loss=policy_output.policy_loss,
            kl_loss=policy_output.kl_loss + reg_total + sdpo_total,
            metrics=all_metrics,
        )
