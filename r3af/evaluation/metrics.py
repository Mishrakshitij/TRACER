from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class R3AFMetrics:
    """Comprehensive evaluation metrics for R3AF."""

    # Core metrics
    robust_accuracy: float = 0.0
    clean_accuracy: float = 0.0
    attack_success_rate: float = 0.0  # ASR

    # Corrigibility metrics
    beneficial_revision_rate: float = 0.0
    over_stability_error: float = 0.0

    # Anchor metrics
    anchor_entailment_rate: float = 0.0
    anchor_contradiction_rate: float = 0.0
    unjustified_drift_score: float = 0.0

    # PRM metrics
    prm_score_clean: float = 0.0
    prm_score_post_attack: float = 0.0

    # Length metrics
    length_inflation_mean: float = 0.0
    length_inflation_p90: float = 0.0
    length_inflation_p95: float = 0.0

    # Per-attack breakdown
    per_attack_metrics: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )

    # Sample counts
    num_samples: int = 0
    num_clean_correct: int = 0
    num_post_correct: int = 0

    def summary(self) -> dict:
        """Return a dict summary of key metrics including per-attack breakdown."""
        result = {
            "robust_accuracy": self.robust_accuracy,
            "clean_accuracy": self.clean_accuracy,
            "ASR": self.attack_success_rate,
            "beneficial_revision_rate": self.beneficial_revision_rate,
            "over_stability_error": self.over_stability_error,
            "anchor_entailment_rate": self.anchor_entailment_rate,
            "anchor_contradiction_rate": self.anchor_contradiction_rate,
            "prm_score_clean": self.prm_score_clean,
            "prm_score_post_attack": self.prm_score_post_attack,
            "length_inflation_mean": self.length_inflation_mean,
            "num_samples": self.num_samples,
        }
        if self.per_attack_metrics:
            result["per_attack_metrics"] = self.per_attack_metrics
        return result
