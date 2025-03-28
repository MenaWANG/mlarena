"""
MLArena - A comprehensive ML pipeline wrapper for scikit-learn compatible models.

This package provides:
- PreProcessor: Advanced data preprocessing with feature analysis and smart encoding
- ML_PIPELINE: End-to-end ML pipeline with model training, evaluation, and deployment
"""

from .pipeline import ML_PIPELINE
from .preprocessor import PreProcessor

__version__ = "0.1.0"
__all__ = ["PreProcessor", "ML_PIPELINE"]
