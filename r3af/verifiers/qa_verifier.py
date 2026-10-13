import re
import string
from collections import Counter
from typing import Optional

from .base_verifier import BaseVerifier


class QAVerifier(BaseVerifier):
    """QA F1/EM verifier for HotpotQA and similar tasks.

    For datasets with multiple valid answers (NQ, TriviaQA), pass
    gold as a pipe-separated string: "answer1|answer2|answer3".
    The verifier will check against all alternatives.
    """

    def __init__(self, f1_threshold: float = 0.5):
        self.f1_threshold = f1_threshold

    def verify(self, pred: str, gold: str) -> bool:
        # Support multiple gold answers separated by pipe
        gold_answers = gold.split("|") if "|" in gold else [gold]
        return any(self._verify_single(pred, g) for g in gold_answers)

    def _verify_single(self, pred: str, gold: str) -> bool:
        if self._exact_match(pred, gold):
            return True
        return self._f1_score(pred, gold) >= self.f1_threshold

    def extract_answer(self, text: str) -> Optional[str]:
        match = re.search(
            r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return text.strip().split("\n")[-1].strip() if text else None

    def _exact_match(self, pred: str, gold: str) -> bool:
        return self._normalize(pred) == self._normalize(gold)

    def _f1_score(self, pred: str, gold: str) -> float:
        pred_tokens = self._normalize(pred).split()
        gold_tokens = self._normalize(gold).split()

        if not pred_tokens or not gold_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _normalize(self, s: str) -> str:
        if s is None:
            return ""
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(c for c in s if c not in string.punctuation)
        return " ".join(s.split())
