import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class NLIScore:
    """NLI score for an anchor proposition against post-attack text."""

    entailment_prob: float = 0.0
    contradiction_prob: float = 0.0
    neutral_prob: float = 0.0

    @property
    def retention_score(self) -> float:
        """Entailment - contradiction: positive means retained."""
        return self.entailment_prob - self.contradiction_prob


class NLIScorer:
    """NLI entailment/contradiction scoring using DeBERTa-v3-large.

    Scores anchor propositions against post-attack text to determine
    if anchors were retained (entailed) or contradicted.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        device: str = None,
        entailment_threshold: float = 0.7,
        contradiction_threshold: float = 0.3,
    ):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = device
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

    def _load_model(self):
        """Lazy load the NLI model."""
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading NLI model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = self._model.to(self._device)
        self._model.eval()

    def score(
        self,
        anchors: list,
        post_attack_text: str,
        batch_size: int = 16,
    ) -> List[NLIScore]:
        """Score anchors against post-attack text using NLI.

        Args:
            anchors: List of AnchorProposition objects.
            post_attack_text: The post-attack response text.
            batch_size: Batch size for NLI inference.

        Returns:
            List of NLIScore objects, one per anchor.
        """
        if not anchors:
            return []

        self._load_model()

        premises = [post_attack_text] * len(anchors)
        hypotheses = [a.content for a in anchors]

        all_scores = []

        for i in range(0, len(anchors), batch_size):
            batch_premises = premises[i : i + batch_size]
            batch_hypotheses = hypotheses[i : i + batch_size]

            inputs = self._tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # DeBERTa NLI: [contradiction, neutral, entailment]
            for j in range(probs.shape[0]):
                all_scores.append(
                    NLIScore(
                        contradiction_prob=probs[j, 0].item(),
                        neutral_prob=probs[j, 1].item(),
                        entailment_prob=probs[j, 2].item(),
                    )
                )

        return all_scores

    def is_retained(self, score: NLIScore) -> bool:
        """Check if an anchor is retained based on configured thresholds."""
        return (
            score.entailment_prob >= self.entailment_threshold
            and score.contradiction_prob <= self.contradiction_threshold
        )

    def compute_anchor_retention(
        self,
        anchors: list,
        post_attack_text: str,
        batch_size: int = 16,
    ) -> float:
        """Compute mean anchor retention score.

        Returns:
            Mean (entailment - contradiction) across all anchors.
        """
        scores = self.score(anchors, post_attack_text, batch_size)
        if not scores:
            return 0.0
        return sum(s.retention_score for s in scores) / len(scores)
