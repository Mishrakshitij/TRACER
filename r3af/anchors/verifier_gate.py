from typing import List, Optional

from .extractor import AnchorProposition


class VerifierGate:
    """Filter anchors by PRM threshold AND answer correctness.

    Only retains anchors when:
    1. PRM score >= threshold
    2. The clean answer was correct
    """

    def __init__(self, prm_threshold: float = 0.7):
        self.prm_threshold = prm_threshold

    def filter(
        self,
        anchors: List[AnchorProposition],
        clean_correct: bool,
        prm_scores: Optional[List[float]] = None,
    ) -> List[AnchorProposition]:
        """Filter anchors through the verifier gate.

        Args:
            anchors: List of candidate anchor propositions.
            clean_correct: Whether the clean answer was correct.
            prm_scores: Optional override PRM scores.

        Returns:
            Filtered list of anchors that pass the gate.
        """
        if not clean_correct:
            return []

        filtered = []
        for i, anchor in enumerate(anchors):
            score = prm_scores[i] if prm_scores and i < len(prm_scores) else anchor.prm_score
            if score >= self.prm_threshold:
                filtered.append(anchor)

        return filtered
