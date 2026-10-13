import re
from dataclasses import dataclass
from typing import List, Optional

from ..parsing.schema import ReasoningStep


@dataclass
class AnchorProposition:
    """An anchor proposition extracted from a reasoning trace."""

    content: str
    source_step: int
    prm_score: float = 0.0


class AnchorExtractor:
    """Extract anchor propositions from clean trace + PRM scores.

    Two modes:
    - step_level: whole step = anchor (if PRM >= threshold)
    - proposition_level: regex extracts equalities/facts from steps
    """

    def __init__(self, threshold: float = 0.7, mode: str = "step_level"):
        self.threshold = threshold
        self.mode = mode

    def extract(
        self,
        steps: List[ReasoningStep],
        prm_scores: Optional[List[float]] = None,
    ) -> List[AnchorProposition]:
        """Extract anchors from reasoning steps.

        Args:
            steps: List of reasoning steps from the clean trace.
            prm_scores: PRM scores per step. If None, all steps are included.

        Returns:
            List of AnchorProposition objects that pass the threshold.
        """
        if prm_scores is None:
            prm_scores = [1.0] * len(steps)

        # Ensure scores match steps
        if len(prm_scores) < len(steps):
            prm_scores = prm_scores + [0.0] * (len(steps) - len(prm_scores))

        anchors = []
        for step, score in zip(steps, prm_scores):
            if score < self.threshold:
                continue

            if self.mode == "step_level":
                anchors.append(
                    AnchorProposition(
                        content=step.content,
                        source_step=step.step_index,
                        prm_score=score,
                    )
                )
            elif self.mode == "proposition_level":
                propositions = self._extract_propositions(step)
                for prop in propositions:
                    anchors.append(
                        AnchorProposition(
                            content=prop,
                            source_step=step.step_index,
                            prm_score=score,
                        )
                    )

        return anchors

    def _extract_propositions(self, step: ReasoningStep) -> List[str]:
        """Extract individual propositions (equalities, facts) from a step."""
        propositions = []
        content = step.content

        # Extract equalities: "X = Y", "X equals Y"
        eq_patterns = [
            r"[\w\s\+\-\*\/\(\)]+\s*=\s*[\w\s\+\-\*\/\(\)\.]+",
            r"[\w\s]+\s+equals?\s+[\w\s\+\-\*\/\.]+",
            r"[\w\s]+\s+is\s+[\w\s\+\-\*\/\.]+",
        ]

        for pattern in eq_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                match = match.strip()
                if len(match) > 5 and len(match) < 200:
                    propositions.append(match)

        # If no propositions found, use the whole step content
        if not propositions:
            propositions.append(content)

        return propositions
