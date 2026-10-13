import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import field
from typing import Optional, Dict

from .base_loss import LossOutput


class TCRKLLoss(nn.Module):
    """Trace Consistency Regularization via KL divergence (novel contribution).

    Formula:
        L_TCR(theta) = g * (1/|M|) * sum_{t in M} KL(stopgrad(p^0_t) || p^m_t)
    where:
        p^0_t = softmax(clean_logits_t / T)  -- distribution under clean context (detached)
        p^m_t = softmax(attacked_logits_t / T)  -- distribution under attacked context (has gradients)
        M = committed span mask (anchor token positions)
        g = 1[clean_correct AND evidence_invalid] (hard gate) or soft product

    This loss encourages the model to maintain the same token-level predictions
    on anchor spans even when the context is adversarially modified.

    Gradients flow through attacked_logits but NOT through clean_logits.
    """

    def __init__(self, temperature: float = 1.0, gate_mode: str = "hard"):
        super().__init__()
        self.temperature = temperature
        self.gate_mode = gate_mode

    def compute_loss(
        self,
        clean_logits: torch.Tensor,   # [batch, seq_len, vocab] -- will be detached
        attacked_logits: torch.Tensor, # [batch, seq_len, vocab] -- has gradients
        committed_mask: torch.Tensor,  # [batch, seq_len] -- 1 for anchor positions
        gate: torch.Tensor,            # [batch] -- per-sample gate
    ) -> LossOutput:
        """Compute TCR-KL loss.

        Args:
            clean_logits: Logits from model under clean context. Will be detached.
            attacked_logits: Logits from model under attacked context. Has gradients.
            committed_mask: Binary mask for anchor token positions.
            gate: Per-sample gate values.

        Returns:
            LossOutput with TCR-KL loss.
        """
        # Detach clean logits (teacher)
        clean_logits = clean_logits.detach()

        # Memory-efficient chunked KL computation.
        # Full softmax over vocab (151k) is ~4.6GB per tensor — OOM on A6000.
        # Instead, chunk along seq_len and skip positions outside committed_mask.
        batch_size, seq_len, vocab_size = clean_logits.shape
        kl_per_position = torch.zeros(
            batch_size, seq_len, device=clean_logits.device, dtype=clean_logits.dtype
        )
        chunk_size = 64  # process 64 positions at a time (~150MB peak per chunk)
        T = self.temperature

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_mask = committed_mask[:, start:end]
            if chunk_mask.sum() == 0:
                continue  # no anchor positions — skip entirely

            clean_chunk = F.softmax(clean_logits[:, start:end, :] / T, dim=-1)
            attacked_chunk = F.log_softmax(
                attacked_logits[:, start:end, :] / T, dim=-1
            )
            kl_per_position[:, start:end] = F.kl_div(
                attacked_chunk, clean_chunk, reduction="none"
            ).sum(dim=-1)

        # Mask to committed spans
        masked_kl = kl_per_position * committed_mask  # [batch, seq_len]

        # Mean over committed positions per sample
        mask_count = committed_mask.sum(dim=-1).clamp(min=1)  # [batch]
        per_sample_kl = masked_kl.sum(dim=-1) / mask_count  # [batch]

        # Apply gate
        gated_kl = per_sample_kl * gate  # [batch]

        # Mean over batch
        loss = gated_kl.mean()

        # Temperature scaling factor (as in knowledge distillation)
        loss = loss * (self.temperature ** 2)

        with torch.no_grad():
            metrics = {
                "tcr_kl_raw": per_sample_kl.mean().item(),
                "tcr_kl_gated": gated_kl.mean().item(),
                "tcr_gate_mean": gate.mean().item(),
                "tcr_committed_frac": (
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

    @staticmethod
    def build_committed_mask(
        anchor_positions: list,
        seq_len: int,
        device: torch.device,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Build committed span mask from anchor token positions.

        Args:
            anchor_positions: List of lists of (start, end) tuples per sample.
            seq_len: Sequence length.
            device: Target device.
            batch_size: Batch size.

        Returns:
            Binary mask [batch_size, seq_len].
        """
        mask = torch.zeros(batch_size, seq_len, device=device)

        for i, positions in enumerate(anchor_positions):
            if i >= batch_size:
                break
            for start, end in positions:
                start = max(0, min(start, seq_len))
                end = max(0, min(end, seq_len))
                if start < end:
                    mask[i, start:end] = 1.0

        return mask

    @staticmethod
    def build_gate(
        clean_correct: torch.Tensor,    # [batch] bool/float
        evidence_invalid: torch.Tensor,  # [batch] bool/float
        mode: str = "hard",
    ) -> torch.Tensor:
        """Build per-sample gate.

        Hard gate: g = 1[clean_correct AND evidence_invalid]
        Soft gate: g = clean_correct_prob * evidence_invalid_prob

        Args:
            clean_correct: Whether clean answer was correct. [batch]
            evidence_invalid: Whether attack evidence is invalid. [batch]
            mode: "hard" or "soft".

        Returns:
            Gate values [batch].
        """
        clean_correct = clean_correct.float()
        evidence_invalid = evidence_invalid.float()

        if mode == "hard":
            return (clean_correct * evidence_invalid).float()
        else:
            return clean_correct * evidence_invalid
