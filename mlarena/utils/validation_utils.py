"""Shared structural validation helpers for MLArena inputs."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_consistent_length, column_or_1d


def validate_1d_array(values: ArrayLike, *, name: str) -> np.ndarray:
    """Validate and convert an array-like input to a one-dimensional array."""
    try:
        return column_or_1d(values, warn=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional array-like input.") from exc


def validate_consistent_length(*arrays: Any) -> None:
    """Validate that all provided array-like inputs have equal lengths."""
    check_consistent_length(*arrays)
