"""No-training diagnostics for polynomial activation substitutions."""

from .activations import HerPN, PolynomialActivation
from .workflow import AnalysisConfig, analyze, replace_activations

__all__ = [
    "AnalysisConfig",
    "HerPN",
    "PolynomialActivation",
    "analyze",
    "replace_activations",
]
