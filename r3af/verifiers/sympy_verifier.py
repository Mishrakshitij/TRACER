import re
import logging
from typing import Optional

from .base_verifier import BaseVerifier
from .numeric_verifier import NumericVerifier

logger = logging.getLogger(__name__)


class SympyVerifier(BaseVerifier):
    """Symbolic math verification via sympy."""

    def __init__(self):
        self._numeric = NumericVerifier()

    def verify(self, pred: str, gold: str) -> bool:
        # Try numeric first
        if self._numeric.verify(pred, gold):
            return True

        # Try sympy symbolic equivalence
        try:
            import sympy

            pred_expr = sympy.sympify(self._clean_latex(pred))
            gold_expr = sympy.sympify(self._clean_latex(gold))

            if sympy.simplify(pred_expr - gold_expr) == 0:
                return True

            # Try numerical evaluation
            pred_val = complex(pred_expr.evalf())
            gold_val = complex(gold_expr.evalf())
            if abs(pred_val - gold_val) < 1e-6:
                return True

        except Exception:
            pass

        return self._normalize(pred) == self._normalize(gold)

    def extract_answer(self, text: str) -> Optional[str]:
        return self._numeric.extract_answer(text)

    def _clean_latex(self, s: str) -> str:
        """Clean LaTeX formatting for sympy parsing."""
        if s is None:
            return "0"
        s = s.strip()
        s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", s)
        s = s.replace("\\pi", "pi")
        s = s.replace("\\cdot", "*")
        s = s.replace("\\times", "*")
        s = s.replace("\\div", "/")
        s = re.sub(r"\\[a-zA-Z]+", "", s)
        s = s.replace("{", "(").replace("}", ")")
        return s

    def _normalize(self, s: str) -> str:
        if s is None:
            return ""
        return s.strip().lower().replace(" ", "")
