"""
MLArena - A comprehensive ML pipeline wrapper for scikit-learn compatible models.

This package provides:
- PreProcessor: Advanced data preprocessing with feature analysis and smart encoding
- MLPipeline: End-to-end ML pipeline with model training, evaluation, and deployment
- ML_PIPELINE: (Deprecated) Use MLPipeline instead
"""

try:
    from importlib.metadata import version

    __version__ = version("mlarena")
except ImportError:
    __version__ = "0.3.1"

from . import utils
from .pipeline import ML_PIPELINE, MLPipeline
from .preprocessor import PreProcessor

__all__ = ["PreProcessor", "MLPipeline", "ML_PIPELINE", "utils"]
