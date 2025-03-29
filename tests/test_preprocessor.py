"""
Tests for the PreProcessor class.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from mlarena.preprocessor import PreProcessor


def test_preprocessor_initialization():
    """Test PreProcessor initialization with different parameters."""
    print("\nTesting PreProcessor initialization...")
    preprocessor = PreProcessor()
    assert preprocessor.num_impute_strategy == "median"
    assert preprocessor.cat_impute_strategy == "most_frequent"
    assert preprocessor.target_encode_cols is None
    assert preprocessor.target_encode_smooth == "auto"
    print("✓ PreProcessor initialization test passed")


def test_preprocessor_fit_transform():
    """Test PreProcessor fit_transform method."""
    print("\nTesting PreProcessor fit_transform method...")
    # Create sample data
    print("Generating sample data...")
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3)
    X = pd.DataFrame(
        X, columns=["feature1", "feature2", "feature3", "feature4", "feature5"]
    )

    # Add some categorical features
    print("Adding categorical features...")
    X["cat_feature1"] = np.random.choice(["A", "B", "C"], size=100)
    X["cat_feature2"] = np.random.choice(["X", "Y", "Z"], size=100)

    # Add some missing values
    print("Adding missing values...")
    X.iloc[0, 0] = np.nan
    X.iloc[1, 1] = np.nan
    X.iloc[2, 2] = np.nan

    print("Initializing and fitting PreProcessor...")
    preprocessor = PreProcessor()
    X_transformed = preprocessor.fit_transform(X, y)

    # Check if the transformed data has the expected shape
    print("Verifying transformed data shape...")
    assert X_transformed.shape[0] == X.shape[0]
    assert (
        X_transformed.shape[1] >= X.shape[1]
    )  # Should be larger due to one-hot encoding

    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print("✓ PreProcessor fit_transform test passed")


def test_preprocessor_transform():
    """Test PreProcessor transform method on new data."""
    print("\nTesting PreProcessor transform method...")
    # Create training data
    print("Generating training data...")
    X_train, y_train = make_classification(n_samples=100, n_features=5, n_informative=3)
    X_train = pd.DataFrame(
        X_train, columns=["feature1", "feature2", "feature3", "feature4", "feature5"]
    )
    X_train["cat_feature"] = np.random.choice(["A", "B", "C"], size=100)

    # Create test data
    print("Generating test data...")
    X_test = pd.DataFrame(
        np.random.randn(50, 5),
        columns=["feature1", "feature2", "feature3", "feature4", "feature5"],
    )
    X_test["cat_feature"] = np.random.choice(["A", "B", "C"], size=50)

    print("Fitting PreProcessor on training data...")
    preprocessor = PreProcessor()
    preprocessor.fit(X_train, y_train)

    print("Transforming test data...")
    X_test_transformed = preprocessor.transform(X_test)

    # Check if the transformed test data has the expected shape
    print("Verifying transformed test data shape...")
    assert X_test_transformed.shape[0] == X_test.shape[0]
    assert (
        X_test_transformed.shape[1] >= X_test.shape[1]
    )  # Should be larger due to one-hot encoding

    print(f"Original test shape: {X_test.shape}")
    print(f"Transformed test shape: {X_test_transformed.shape}")
    print("✓ PreProcessor transform test passed")


if __name__ == "__main__":
    print("Starting PreProcessor tests...")
    test_preprocessor_initialization()
    test_preprocessor_fit_transform()
    test_preprocessor_transform()
    print("\nAll PreProcessor tests completed successfully!")
