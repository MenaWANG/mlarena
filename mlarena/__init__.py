"""
MLArena - A comprehensive ML pipeline wrapper for scikit-learn compatible models.

This package provides:
- PreProcessor: Advanced data preprocessing with feature analysis and smart encoding
- MLPipeline: End-to-end ML pipeline with model training, evaluation, and deployment
"""

try:
    from importlib.metadata import version

    __version__ = version("mlarena")
except ImportError:
    __version__ = "unknown"

from . import utils
from .preprocessor import PreProcessor

__all__ = ["PreProcessor", "MLPipeline", "utils"]


def __getattr__(name):
    if name == "MLPipeline":
        from .pipeline import MLPipeline

        return MLPipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
