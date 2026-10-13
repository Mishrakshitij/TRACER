from .base_loss import BasePolicyLoss, LossOutput
from .grpo_loss import GRPOLoss
from .gdpo_loss import GDPOLoss
from .tcr_kl_loss import TCRKLLoss
from .atr_loss import ATRLoss
from .atrc_loss import ATRCLoss
from .sdpo_loss import SDPOLoss
from .combined_loss import CombinedR3AFLoss

LOSS_REGISTRY = {
    "grpo": GRPOLoss,
    "gdpo": GDPOLoss,
    "tcr_kl": TCRKLLoss,
    "atr": ATRLoss,
    "atrc": ATRCLoss,
    "sdpo": SDPOLoss,
    "combined": CombinedR3AFLoss,
}

__all__ = [
    "BasePolicyLoss",
    "LossOutput",
    "GRPOLoss",
    "GDPOLoss",
    "TCRKLLoss",
    "ATRLoss",
    "ATRCLoss",
    "SDPOLoss",
    "CombinedR3AFLoss",
    "LOSS_REGISTRY",
]
