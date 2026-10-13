import logging
from typing import List, Dict, Optional
import numpy as np

from ..models.config import R3AFConfig
from ..models.base import R3AFModel
from .evaluator import R3AFEvaluator
from .metrics import R3AFMetrics

logger = logging.getLogger(__name__)


class RobustnessCurveAnalyzer:
    """Sweep attack budgets and compute robustness curves.

    Evaluates at multiple attack budget levels [64, 128, 256, 512, 1024]
    tokens and computes AUC of robustness metrics.
    """

    DEFAULT_BUDGETS = [64, 128, 256, 512, 1024]

    def __init__(self, config: R3AFConfig, model: R3AFModel):
        self.config = config
        self.model = model

    def sweep(
        self,
        dataset_name: str = "gsm8k",
        split: str = "test",
        max_samples: Optional[int] = None,
        budgets: Optional[List[int]] = None,
    ) -> Dict[str, List]:
        """Sweep attack budgets and collect metrics.

        Returns:
            Dict with 'budgets', 'robust_accuracy', 'asr', etc.
        """
        if budgets is None:
            budgets = self.DEFAULT_BUDGETS

        results = {
            "budgets": budgets,
            "robust_accuracy": [],
            "asr": [],
            "beneficial_revision_rate": [],
            "length_inflation_mean": [],
        }

        for budget in budgets:
            logger.info(f"Evaluating with attack budget = {budget} tokens")

            config_copy = R3AFConfig(**{
                k: getattr(self.config, k)
                for k in self.config.__dataclass_fields__
            })
            config_copy.max_attack_tokens = budget

            evaluator = R3AFEvaluator(config_copy, self.model)
            metrics = evaluator.evaluate(
                dataset_name=dataset_name,
                split=split,
                max_samples=max_samples,
            )

            results["robust_accuracy"].append(metrics.robust_accuracy)
            results["asr"].append(metrics.attack_success_rate)
            results["beneficial_revision_rate"].append(
                metrics.beneficial_revision_rate
            )
            results["length_inflation_mean"].append(
                metrics.length_inflation_mean
            )

        return results

    def compute_auc(self, results: Dict[str, List]) -> Dict[str, float]:
        """Compute area under curve for each metric."""
        budgets = np.array(results["budgets"], dtype=float)
        auc_results = {}

        for key in ["robust_accuracy", "asr", "beneficial_revision_rate"]:
            if key in results and results[key]:
                values = np.array(results[key], dtype=float)
                auc = float(np.trapz(values, budgets))
                # Normalize by budget range
                budget_range = budgets[-1] - budgets[0]
                if budget_range > 0:
                    auc /= budget_range
                auc_results[f"{key}_auc"] = auc

        return auc_results
