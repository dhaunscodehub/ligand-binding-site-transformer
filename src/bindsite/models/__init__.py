"""Models: the cross-attention transformer and the baselines it must beat."""

from .baselines import BASELINE_MODELS, fit_baseline

__all__ = ["BASELINE_MODELS", "fit_baseline"]
