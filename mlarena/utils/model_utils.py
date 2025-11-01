"""
Model utilities for MLArena package.

This module contains helper functions for model-related operations
that are shared across different components of the package.
"""

from typing import Optional, Union

from sklearn.model_selection import KFold, StratifiedKFold

__all__ = ["get_cv_strategy"]


def get_cv_strategy(
    cv: int | object = 5,
    task: str = "classification",
    random_state: int = 42,
):
    """
    Returns a cross-validation splitter object based on user input.

    Parameters
    ----------
    cv : int or CV splitter object, default=5
        - If int: number of folds for default CV strategy (StratifiedKFold or KFold).
        - If object with `.split()`: used directly as the CV splitter.
    task : str, default="classification"
        "classification" or "regression" — used to choose default CV type.
    random_state : int, default=42
        Ensures reproducibility for default splitters.

    Returns
    -------
    splitter : object
        Configured CV splitter ready for use in model selection or feature selection.

    Examples
    --------
    >>> get_cv_strategy()
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    >>> from sklearn.model_selection import GroupKFold
    >>> get_cv_strategy(GroupKFold(n_splits=4))
    GroupKFold(n_splits=4)
    """
    # If user provides a splitter object, use it directly
    if hasattr(cv, "split"):
        return cv

    # Otherwise, assume integer and use default CV logic
    if isinstance(cv, int):
        if task == "classification":
            return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        elif task == "regression":
            return KFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            raise ValueError("task must be 'classification' or 'regression'")
    else:
        raise ValueError("cv must be either an int or a CV splitter object.")
