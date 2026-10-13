import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .base_loss import LossOutput


class ATRLoss(nn.Module):
    """Adversarial Trace Recovery loss.

    Formula:
        L_ATR(theta) = gate * (1/|M|) * sum_{t in M} [-log P_attacked(y_clean_t)]

    Maximizes probability of clean response tokens under attacked context.
    More targeted than TCR (cross-entropy on specific token vs KL on full
    distribution) and more memory-efficient (gather vs full softmax KL).

    Gradients flow through attacked_logits but NOT through clean_token_ids.
    """

    def __init__(self, gate_mode: str = "hard"):
        super().__init__()
        self.gate_mode = gate_mode

    def compute_loss(
        self,
        attacked_logits: torch.Tensor,    # [batch, resp_len, vocab] -- has gradients
        clean_token_ids: torch.Tensor,     # [batch, resp_len] -- target token IDs (no grad)
        committed_mask: torch.Tensor,      # [batch, resp_len] -- anchor positions
        gate: torch.Tensor,                # [batch] -- per-sample gate
    ) -> LossOutput:
        """Compute ATR loss.

        Args:
            attacked_logits: Logits from model under attacked context. Has gradients.
            clean_token_ids: Token IDs from the clean response (targets).
            committed_mask: Binary mask for anchor token positions.
            gate: Per-sample gate values.

        Returns:
            LossOutput with ATR loss.
        """
        # Cross-entropy per position (memory-efficient: no full softmax materialized)
        log_probs = F.log_softmax(attacked_logits, dim=-1)

        # Gather log prob of clean token at each position
        target_log_probs = log_probs.gather(
            -1, clean_token_ids.unsqueeze(-1)
        ).squeeze(-1)  # [batch, resp_len]

        # Negate: minimize negative log prob = maximize log prob
        nll_per_position = -target_log_probs  # [batch, resp_len]

        # Mask to committed spans
        masked_nll = nll_per_position * committed_mask  # [batch, resp_len]

        # Mean over committed positions per sample
        mask_count = committed_mask.sum(dim=-1).clamp(min=1)  # [batch]
        per_sample_nll = masked_nll.sum(dim=-1) / mask_count  # [batch]

        # Apply gate and mean over batch
        gated_nll = per_sample_nll * gate  # [batch]
        loss = gated_nll.mean()

        with torch.no_grad():
            metrics: Dict[str, float] = {
                "atr_nll_raw": per_sample_nll.mean().item(),
                "atr_nll_gated": gated_nll.mean().item(),
                "atr_gate_mean": gate.mean().item(),
                "atr_committed_frac": (
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
