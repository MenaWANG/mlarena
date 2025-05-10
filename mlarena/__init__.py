"""
MLArena - A comprehensive ML pipeline wrapper for scikit-learn compatible models.

This package provides:
- PreProcessor: Advanced data preprocessing with feature analysis and smart encoding
- ML_PIPELINE: End-to-end ML pipeline with model training, evaluation, and deployment
"""

try:
    from importlib.metadata import version

    __version__ = version("mlarena")
except ImportError:
    __version__ = "0.2.5"

from . import utils
from .pipeline import ML_PIPELINE
from .preprocessor import PreProcessor

__all__ = ["PreProcessor", "ML_PIPELINE", "utils"]
