from .base_verifier import BaseVerifier
from .numeric_verifier import NumericVerifier
from .sympy_verifier import SympyVerifier
from .code_verifier import CodeVerifier
from .qa_verifier import QAVerifier

_VERIFIER_REGISTRY = {
    "gsm8k": NumericVerifier,
    "math": SympyVerifier,
    "hotpotqa": QAVerifier,
    "humaneval": CodeVerifier,
    "mbpp": CodeVerifier,
    "nq": QAVerifier,
    "triviaqa": QAVerifier,
}


def get_verifier(dataset: str) -> BaseVerifier:
    """Get the appropriate verifier for a dataset."""
    cls = _VERIFIER_REGISTRY.get(dataset, NumericVerifier)
    return cls()


__all__ = [
    "BaseVerifier",
    "NumericVerifier",
    "SympyVerifier",
    "CodeVerifier",
    "QAVerifier",
    "get_verifier",
]
