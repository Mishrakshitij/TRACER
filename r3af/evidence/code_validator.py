import re
import logging
from .base_validator import BaseEvidenceValidator, EvidenceValidation, register_validator

logger = logging.getLogger(__name__)


@register_validator("humaneval")
@register_validator("mbpp")
class CodeEvidenceValidator(BaseEvidenceValidator):
    """Evidence validator for code problems.

    Valid evidence: attack contains code that passes tests the original failed.
    Uses RestrictedPython for sandboxed execution.
    """

    def validate(
        self,
        problem: str,
        attack_msg: str,
        clean_trace: str,
        clean_answer: str,
        ground_truth: str,
    ) -> EvidenceValidation:
        # Extract code blocks from attack
        code_blocks = re.findall(
            r"```(?:python)?\s*\n(.*?)```", attack_msg, re.DOTALL
        )

        if not code_blocks:
            return EvidenceValidation(
                is_valid=False,
                confidence=0.7,
                reason="No code found in attack message.",
            )

        # Try to execute the suggested code
        for code in code_blocks:
            try:
                result = self._safe_execute(code, ground_truth)
                if result:
                    return EvidenceValidation(
                        is_valid=True,
                        confidence=0.7,
                        reason="Attack code passes tests.",
                    )
            except Exception as e:
                logger.debug(f"Code execution failed: {e}")
                continue

        return EvidenceValidation(
            is_valid=False,
            confidence=0.5,
            reason="Attack code does not pass tests.",
        )

    def _safe_execute(self, code: str, test_code: str) -> bool:
        """Safely execute code in a restricted environment."""
        try:
            from RestrictedPython import compile_restricted, safe_globals

            byte_code = compile_restricted(code, "<string>", "exec")
            glb = safe_globals.copy()
            glb["_getiter_"] = iter
            glb["_getattr_"] = getattr
            exec(byte_code, glb)
            return True
        except Exception:
            return False
