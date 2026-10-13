import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .base_loss import LossOutput


class ATRCLoss(nn.Module):
    """Adversarial Trace Recovery + Consistency (combined ATR + TCR).

    Formula:
        L_ATRC(theta) = gate * (1/|M|) * sum_{t in M}
            [alpha * KL(stopgrad(clean_t) || attacked_t)
             + (1 - alpha) * (-log P_attacked(y_clean_t))]

    alpha=1.0 -> pure TCR (KL divergence on full distribution)
    alpha=0.0 -> pure ATR (cross-entropy on specific clean token)

    Gradients flow through attacked_logits but NOT through clean_logits
    or clean_token_ids.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 1.0,
        gate_mode: str = "hard",
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.gate_mode = gate_mode

    def compute_loss(
        self,
        clean_logits: torch.Tensor,       # [batch, resp_len, vocab] -- will be detached
        attacked_logits: torch.Tensor,     # [batch, resp_len, vocab] -- has gradients
        clean_token_ids: torch.Tensor,     # [batch, resp_len] -- target tokens
        committed_mask: torch.Tensor,      # [batch, resp_len]
        gate: torch.Tensor,                # [batch]
    ) -> LossOutput:
        """Compute ATRC loss.

        Args:
            clean_logits: Logits from model under clean context. Will be detached.
            attacked_logits: Logits from model under attacked context. Has gradients.
            clean_token_ids: Token IDs from the clean response (targets).
            committed_mask: Binary mask for anchor token positions.
            gate: Per-sample gate values.

        Returns:
            LossOutput with ATRC loss.
        """
        clean_logits = clean_logits.detach()
        batch_size, resp_len, vocab_size = clean_logits.shape
        T = self.temperature

        # Allocate per-position combined loss
        combined_per_pos = torch.zeros(
            batch_size, resp_len,
            device=clean_logits.device, dtype=clean_logits.dtype,
        )

        # --- KL component (chunked for memory, same as TCR) ---
        if self.alpha > 0:
            chunk_size = 64
            kl_per_pos = torch.zeros(
                batch_size, resp_len,
                device=clean_logits.device, dtype=clean_logits.dtype,
            )
            for start in range(0, resp_len, chunk_size):
                end = min(start + chunk_size, resp_len)
                chunk_mask = committed_mask[:, start:end]
                if chunk_mask.sum() == 0:
                    continue
                clean_chunk = F.softmax(clean_logits[:, start:end, :] / T, dim=-1)
                attacked_chunk = F.log_softmax(
                    attacked_logits[:, start:end, :] / T, dim=-1
                )
                kl_per_pos[:, start:end] = F.kl_div(
                    attacked_chunk, clean_chunk, reduction="none"
                ).sum(dim=-1)

            combined_per_pos = combined_per_pos + self.alpha * kl_per_pos

        # --- NLL component ---
        if self.alpha < 1:
            log_probs = F.log_softmax(attacked_logits, dim=-1)
            target_log_probs = log_probs.gather(
                -1, clean_token_ids.unsqueeze(-1)
            ).squeeze(-1)  # [batch, resp_len]
            nll_per_pos = -target_log_probs
            combined_per_pos = combined_per_pos + (1 - self.alpha) * nll_per_pos

        # Mask to committed spans
        masked = combined_per_pos * committed_mask  # [batch, resp_len]

        # Mean over committed positions per sample
        mask_count = committed_mask.sum(dim=-1).clamp(min=1)  # [batch]
        per_sample = masked.sum(dim=-1) / mask_count  # [batch]

        # Apply gate and mean over batch
        gated = per_sample * gate  # [batch]
        loss = gated.mean()

        # T^2 scaling for KL component (knowledge distillation convention)
        if self.alpha > 0:
            loss = loss * (self.temperature ** 2)

        with torch.no_grad():
            metrics: Dict[str, float] = {
                "atrc_loss_raw": per_sample.mean().item(),
                "atrc_loss_gated": gated.mean().item(),
                "atrc_gate_mean": gate.mean().item(),
                "atrc_alpha": self.alpha,
                "atrc_committed_frac": (
                    committed_mask.sum() / max(committed_mask.numel(), 1)
                ).item(),
            }

        zero = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        return LossOutput(
            total_loss=loss,
            policy_loss=zero,
            kl_loss=loss,
            metrics=metrics,
        )
