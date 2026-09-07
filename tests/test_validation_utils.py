import numpy as np
import pandas as pd
import pytest

from mlarena.utils.validation_utils import (
    validate_1d_array,
    validate_consistent_length,
)


@pytest.mark.parametrize(
    "values",
    [pd.Series([0, 1, 0]), np.array([0, 1, 0]), [0, 1, 0]],
)
def test_validate_1d_array_normalizes_supported_inputs(values):
    result = validate_1d_array(values, name="y")

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([0, 1, 0]))


def test_validate_1d_array_rejects_multioutput_input():
    with pytest.raises(ValueError, match="y must be a one-dimensional"):
        validate_1d_array(np.ones((3, 2)), name="y")


def test_validate_consistent_length_uses_sklearn_validation():
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        validate_consistent_length(np.array([0, 1]), np.array([0]))
