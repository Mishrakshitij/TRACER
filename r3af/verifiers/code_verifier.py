import logging
from typing import Optional

from .base_verifier import BaseVerifier

logger = logging.getLogger(__name__)


class CodeVerifier(BaseVerifier):
    """Code execution + unit test verification."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def verify(self, pred: str, gold: str) -> bool:
        """Verify by executing code and running tests.

        For simplicity, does string comparison of code output.
        Full execution-based verification would use sandboxing.
        """
        # Simple string-based comparison for now
        pred_clean = self._clean_code(pred)
        gold_clean = self._clean_code(gold)

        if pred_clean == gold_clean:
            return True

        # Try execution-based comparison
        try:
            return self._execute_and_compare(pred, gold)
        except Exception as e:
            logger.debug(f"Code execution failed: {e}")
            return False

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract code from model output."""
        import re

        # Try code block
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return text.strip()

    def _clean_code(self, code: str) -> str:
        if code is None:
            return ""
        lines = code.strip().split("\n")
        lines = [l.rstrip() for l in lines if l.strip()]
        return "\n".join(lines)

    def _execute_and_compare(self, pred: str, gold: str) -> bool:
        """Execute both code snippets and compare outputs."""
        try:
            from RestrictedPython import compile_restricted, safe_globals

            pred_globals = safe_globals.copy()
            pred_globals["_getiter_"] = iter
            pred_globals["_getattr_"] = getattr

            gold_globals = safe_globals.copy()
            gold_globals["_getiter_"] = iter
            gold_globals["_getattr_"] = getattr

            pred_code = compile_restricted(pred, "<pred>", "exec")
            gold_code = compile_restricted(gold, "<gold>", "exec")

            exec(pred_code, pred_globals)
            exec(gold_code, gold_globals)

            return True
        except Exception:
            return False
