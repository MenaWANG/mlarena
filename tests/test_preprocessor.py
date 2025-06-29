"""
Tests for the PreProcessor class.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlarena.preprocessor import PreProcessor


def test_preprocessor_initialization():
    """Test PreProcessor initialization with different parameters."""
    print("\nTesting PreProcessor initialization...")
    preprocessor = PreProcessor()
    assert preprocessor.num_impute_strategy == "median"
    assert preprocessor.cat_impute_strategy == "most_frequent"
    assert preprocessor.target_encode_cols is None
    assert preprocessor.target_encode_smooth == "auto"
    assert preprocessor.sanitize_feature_names == True

    # Test with different parameter values
    preprocessor_custom = PreProcessor(
        num_impute_strategy="mean",
        cat_impute_strategy="constant",
        sanitize_feature_names=False,
    )
    assert preprocessor_custom.num_impute_strategy == "mean"
    assert preprocessor_custom.cat_impute_strategy == "constant"
    assert preprocessor_custom.sanitize_feature_names == False

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
    _ = preprocessor.fit_transform(X_train, y_train)

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


def test_preprocessor_encoding():
    """Test PreProcessor encoding for both categorical and target encoding."""
    print("\nTesting PreProcessor encoding...")

    # Create sample data with different feature types
    X = pd.DataFrame(
        {
            "numeric1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "numeric2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "binary_cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "multi_cat": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z", "X"],
            "target_encode": [
                "C1",
                "C2",
                "C1",
                "C2",
                "C1",
                "C2",
                "C1",
                "C2",
                "C1",
                "C2",
            ],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Test with different drop strategies
    for drop_strategy in ["if_binary", "first", None]:
        print(f"\nTesting with drop='{drop_strategy}'")

        # Initialize preprocessor with target encoding
        preprocessor = PreProcessor(
            target_encode_cols=["target_encode"],
            drop=drop_strategy,
            target_encode_smooth=0.1,
        )

        # Fit and transform
        X_transformed = preprocessor.fit_transform(X, y)

        # Check transformed data
        print(f"Transformed shape: {X_transformed.shape}")
        print(f"Transformed columns: {X_transformed.columns.tolist()}")

        # Verify numeric features are preserved
        assert "numeric1" in X_transformed.columns
        assert "numeric2" in X_transformed.columns

        # Verify binary categorical encoding
        if drop_strategy == "if_binary":
            assert "binary_cat_B" in X_transformed.columns
            assert "binary_cat_A" not in X_transformed.columns
        elif drop_strategy == "first":
            assert "binary_cat_B" in X_transformed.columns
            assert "binary_cat_A" not in X_transformed.columns
        else:  # drop=None
            assert "binary_cat_A" in X_transformed.columns
            assert "binary_cat_B" in X_transformed.columns

        # Verify multi-category encoding
        if drop_strategy == "first":
            assert "multi_cat_Y" in X_transformed.columns
            assert "multi_cat_Z" in X_transformed.columns
            assert "multi_cat_X" not in X_transformed.columns
        else:
            assert "multi_cat_X" in X_transformed.columns
            assert "multi_cat_Y" in X_transformed.columns
            assert "multi_cat_Z" in X_transformed.columns

        # Verify target encoding
        assert "target_encode" in X_transformed.columns
        assert X_transformed["target_encode"].dtype in [np.float64, np.float32]

        # Test transform on new data
        X_new = pd.DataFrame(
            {
                "numeric1": [6, 7],
                "numeric2": [0, 1],
                "binary_cat": ["A", "B"],
                "multi_cat": ["X", "Z"],
                "target_encode": ["C1", "C2"],
            }
        )

        X_new_transformed = preprocessor.transform(X_new)
        assert X_new_transformed.shape[0] == 2
        assert set(X_transformed.columns) == set(X_new_transformed.columns)

        print(f"✓ Test passed for drop='{drop_strategy}'")

    print("✓ All encoding tests passed")


def test_sanitize_feature_names():
    """Test PreProcessor feature name sanitization functionality."""
    print("\nTesting PreProcessor sanitize_feature_names functionality...")

    # Create minimal sample data with problematic categorical values
    X = pd.DataFrame(
        {
            "numeric1": [1, 2, 3, 4],
            "age_cat": ["18-25", "60+", "18-25", "60+"],  # Test dash and plus
            "category": ["A&B", "X|Y", "Type/1", "Grade*"],  # Test &, |, /, *
            "edge_cases": [
                "Price@$100",
                "Model#123",
                "Type~Special",
                "Brand^2",
            ],  # Test @, #, ~, ^
        }
    )
    y = pd.Series([0, 1, 0, 1])

    # Test with sanitization enabled (default)
    print("\nTesting with sanitization enabled...")
    preprocessor_sanitized = PreProcessor(sanitize_feature_names=True, drop=None)
    X_transformed_sanitized = preprocessor_sanitized.fit_transform(X, y)

    # Check key sanitizations with simple assertions
    feature_names_sanitized = X_transformed_sanitized.columns.tolist()
    sanitized_str = " ".join(feature_names_sanitized)

    # Verify key replacements
    assert "_18_25" in sanitized_str, "Dash should be replaced with underscore"
    assert "_60_plus" in sanitized_str, "Plus should be replaced with _plus"
    assert "_A_B" in sanitized_str, "& should be replaced with underscore"
    assert "_X_Y" in sanitized_str, "| should be replaced with underscore"
    assert "_Type_1" in sanitized_str, "/ should be replaced with underscore"
    assert "_Grade" in sanitized_str, "* should be replaced with underscore"

    # Check that problematic characters are gone
    for char in "@#~^":
        assert (
            char not in sanitized_str
        ), f"Special character {char} should be sanitized"

    # Test with sanitization disabled
    print("\nTesting with sanitization disabled...")
    preprocessor_raw = PreProcessor(sanitize_feature_names=False, drop=None)
    X_transformed_raw = preprocessor_raw.fit_transform(X, y)

    # Check that original characters are preserved
    feature_names_raw = X_transformed_raw.columns.tolist()
    raw_str = " ".join(feature_names_raw)

    for char in "-+&|/*":
        assert (
            char in raw_str
        ), f"Character {char} should be preserved when sanitization is disabled"

    # Test return types
    assert isinstance(
        preprocessor_sanitized._get_onehot_col_names(), list
    ), "Should return list"
    assert isinstance(
        preprocessor_raw._get_onehot_col_names(), list
    ), "Should return list"

    print("✓ All sanitize_feature_names tests passed")


def test_sanitize_feature_names_static_method():
    """Test the static _sanitize_feature_names method directly."""
    print("\nTesting _sanitize_feature_names static method...")

    # Test key transformation cases
    test_cases = [
        ("feature_18-25", "feature_18_25"),  # dash
        ("income_<$30K", "income_lt_30K"),  # < at start
        ("range_18<25", "range_18_lt_25"),  # < between values
        ("product_Basic*", "product_Basic"),  # * → underscore
        ("category_A&B", "category_A_B"),  # & → underscore
        ("type_X|Y", "type_X_Y"),  # | → underscore
        ("path_Type/1", "path_Type_1"),  # / → underscore
        ("percent_50%", "percent_50_pct"),  # % (semantic)
        ("range_60+", "range_60_plus"),  # + (semantic)
        ("score_X=Y", "score_X_eq_Y"),  # = between values
        ("special_@#$", "special"),  # multiple special chars
        ("multiple__under___scores", "multiple_under_scores"),  # cleanup
        ("__leading_trailing__", "leading_trailing"),  # strip
    ]

    for original, expected in test_cases:
        result = PreProcessor._sanitize_feature_names([original])
        assert (
            result[0] == expected
        ), f"Expected '{expected}', got '{result[0]}' for input '{original}'"

    # Test with numpy array input (simulating sklearn output)
    numpy_input = np.array(["feature_A+B", "feature_X*Y"])
    result = PreProcessor._sanitize_feature_names(numpy_input)
    expected_result = ["feature_A_plusB", "feature_X_Y"]
    assert result == expected_result, f"Expected {expected_result}, got {result}"

    print("✓ Static method sanitization tests passed")


if __name__ == "__main__":
    print("Starting PreProcessor tests...")
    test_preprocessor_initialization()
    test_preprocessor_fit_transform()
    test_preprocessor_transform()
    test_preprocessor_encoding()
    test_sanitize_feature_names()
    test_sanitize_feature_names_static_method()
    print("\nAll PreProcessor tests completed successfully!")
