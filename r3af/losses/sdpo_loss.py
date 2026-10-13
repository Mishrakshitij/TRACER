import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .base_loss import LossOutput


class SDPOLoss(nn.Module):
    """Self-Distillation Policy Optimization (novel contribution).

    Formula:
        L_SDPO(theta) = (1/|R|) * sum_{t in R} KL(stopgrad(pi_f_t) || pi_base_t)
    where:
        pi_f_t = softmax(feedback_logits_t / T)  -- teacher (feedback-conditioned, detached)
        pi_base_t = softmax(base_logits_t / T)   -- student (base policy, has gradients)
        R = response token positions

    The feedback can be compiler errors, test output, judge critique, etc.
    appended to the context to form the "teacher" distribution.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def compute_loss(
        self,
        base_logits: torch.Tensor,      # [batch, seq_len, vocab] -- has gradients
        feedback_logits: torch.Tensor,   # [batch, seq_len, vocab] -- will be detached
        response_mask: torch.Tensor,     # [batch, seq_len]
    ) -> LossOutput:
        """Compute SDPO loss.

        Args:
            base_logits: Logits from base policy (student). Has gradients.
            feedback_logits: Logits from feedback-conditioned policy (teacher). Detached.
            response_mask: Binary mask for response token positions.

        Returns:
            LossOutput with SDPO loss.
        """
        # Detach teacher (feedback-conditioned) logits
        feedback_logits = feedback_logits.detach()

        # Apply temperature
        teacher_probs = F.softmax(feedback_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(
            base_logits / self.temperature, dim=-1
        )

        # KL(teacher || student) per position
        kl_per_position = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)  # [batch, seq_len]

        # Mask to response positions
        masked_kl = kl_per_position * response_mask

        # Mean over response positions
        mask_count = response_mask.sum().clamp(min=1)
        loss = masked_kl.sum() / mask_count

        # Temperature scaling
        loss = loss * (self.temperature ** 2)

        with torch.no_grad():
            metrics: Dict[str, float] = {
                "sdpo_kl": loss.item(),
                "sdpo_response_tokens": response_mask.sum().item(),
            }

        zero = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        return LossOutput(
            total_loss=loss,
            policy_loss=zero,
            kl_loss=loss,
            metrics=metrics,
        )
