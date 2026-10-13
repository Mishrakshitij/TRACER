from typing import Optional

from .base_reward import BaseRewardComponent, RewardOutput
from ..anchors.extractor import AnchorExtractor
from ..anchors.nli_scorer import NLIScorer
from ..anchors.verifier_gate import VerifierGate
from ..verifiers import get_verifier


class AnchorReward(BaseRewardComponent):
    """R_anchor = (1/|A|) * sum_k [P_entail(p_k|tau') - P_contra(p_k|tau')]

    Measures how well anchor propositions from the clean trace are retained
    in the post-attack trace, using NLI scoring.
    """

    def __init__(
        self,
        extractor: Optional[AnchorExtractor] = None,
        nli_scorer: Optional[NLIScorer] = None,
        verifier_gate: Optional[VerifierGate] = None,
    ):
        self.extractor = extractor or AnchorExtractor()
        self.nli_scorer = nli_scorer
        self.verifier_gate = verifier_gate or VerifierGate()

    def compute(self, transcripts, **kwargs) -> RewardOutput:
        prm_scores_list = kwargs.get("prm_scores_list", None)
        values = []
        retention_details = []

        for i, transcript in enumerate(transcripts):
            clean_trace = transcript.clean_trace
            post_attack_turn = transcript.post_attack_turn

            if not clean_trace or post_attack_turn is None:
                values.append(0.0)
                retention_details.append({})
                continue

            # Check verifier gate: only extract anchors if clean answer was correct
            verifier = get_verifier(transcript.dataset)
            clean_correct = verifier.verify(
                str(transcript.clean_answer) if transcript.clean_answer else "",
                transcript.ground_truth,
            )
            if not clean_correct:
                values.append(0.0)
                retention_details.append({"skipped": "clean_answer_incorrect"})
                continue

            # Get PRM scores for anchor extraction
            prm_scores = prm_scores_list[i] if prm_scores_list else None

            # Extract anchors from clean trace steps
            anchors = self.extractor.extract(clean_trace.steps, prm_scores)

            # Apply verifier gate filtering
            anchors = self.verifier_gate.filter(anchors, clean_correct, prm_scores)

            if not anchors:
                values.append(0.0)
                retention_details.append({"num_anchors": 0})
                continue

            # Score anchors against post-attack text
            post_text = post_attack_turn.content
            if self.nli_scorer is not None:
                nli_scores = self.nli_scorer.score(anchors, post_text)
                retention = sum(s.retention_score for s in nli_scores) / len(
                    nli_scores
                )
                values.append(retention)
                retention_details.append(
                    {
                        "num_anchors": len(anchors),
                        "mean_entailment": sum(
                            s.entailment_prob for s in nli_scores
                        )
                        / len(nli_scores),
                        "mean_contradiction": sum(
                            s.contradiction_prob for s in nli_scores
                        )
                        / len(nli_scores),
                    }
                )
            else:
                # Without NLI, use simple text overlap heuristic
                retention = self._text_overlap_heuristic(anchors, post_text)
                values.append(retention)
                retention_details.append(
                    {"num_anchors": len(anchors), "method": "text_overlap"}
                )

        return RewardOutput(
            values=values,
            metadata={"retention_details": retention_details},
        )

    def _text_overlap_heuristic(self, anchors, post_text: str) -> float:
        """Simple text overlap heuristic when NLI is not available."""
        if not anchors:
            return 0.0

        post_lower = post_text.lower()
        retained = 0
        for anchor in anchors:
            # Check if key phrases from anchor appear in post-attack text
            words = anchor.content.lower().split()
            key_phrases = [
                " ".join(words[i : i + 3])
                for i in range(0, len(words) - 2)
            ]
            if any(phrase in post_lower for phrase in key_phrases):
                retained += 1
            elif len(words) <= 3 and anchor.content.lower() in post_lower:
                retained += 1

        return retained / max(len(anchors), 1)
