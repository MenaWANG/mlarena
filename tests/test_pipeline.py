"""
Test script for MLPipeline class.
"""

# Standard library imports
from typing import Any

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Local imports
from mlarena import MLPipeline, PreProcessor

matplotlib.use("Agg")  # Use non-interactive backend for testing


@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close matplotlib figures after each test."""
    yield
    plt.close("all")


# Custom model class without verbose parameter for testing
class SimpleModelNoVerbose(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        preds = np.zeros((len(X), 2))
        preds[:, 0] = 1
        return preds


def test_supports_verbose():
    print("\nTesting verbose parameter support detection...")

    # Test with RandomForestClassifier (has verbose)
    assert (
        MLPipeline._supports_verbose(RandomForestClassifier) is True
    ), "Failed to detect verbose parameter in RandomForestClassifier"

    # Test with custom model (no verbose)
    assert (
        MLPipeline._supports_verbose(SimpleModelNoVerbose) is False
    ), "Incorrectly detected verbose parameter in SimpleModelNoVerbose"

    # Test with DecisionTreeRegressor (no verbose)
    assert (
        MLPipeline._supports_verbose(DecisionTreeRegressor) is False
    ), "Incorrectly detected verbose parameter in DecisionTreeRegressor"

    print("Verbose parameter support detection tests passed.")


@pytest.mark.parametrize("container", [np.asarray, list, pd.Series])
@pytest.mark.parametrize("method", ["cv", "bootstrap"])
def test_threshold_analysis_accepts_array_like_inputs(container, method):
    y_true = container([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_proba = container([0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 0.6, 0.2, 0.8])

    results = MLPipeline.threshold_analysis(
        y_true,
        y_pred_proba,
        method=method,
        cv_splits=2,
        bootstrap_iterations=5,
    )

    assert 0 <= results["optimal_threshold"] <= 1


def test_threshold_analysis_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        MLPipeline.threshold_analysis([0, 1], [0.1])


@pytest.mark.parametrize(
    ("model", "dataset_factory"),
    [
        (LogisticRegression(), make_classification),
        (LinearRegression(), make_regression),
    ],
)
def test_evaluate_accepts_list_target(model, dataset_factory):
    X, y = dataset_factory(n_samples=40, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
    pipeline = MLPipeline(model=model)
    pipeline.fit(X, pd.Series(y))

    results = pipeline.evaluate(X, y.tolist(), verbose=False, visualize=False)

    assert isinstance(results, dict)


def test_evaluate_rejects_target_with_inconsistent_length():
    X, y = make_regression(n_samples=20, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
    pipeline = MLPipeline(model=LinearRegression())
    pipeline.fit(X, pd.Series(y))

    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        pipeline.evaluate(X, y[:-1], verbose=False, visualize=False)


def test_tune_accepts_numpy_target():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])

    results = MLPipeline.tune(
        X=X,
        y=y,
        algorithm=DecisionTreeClassifier,
        preprocessor=None,
        param_ranges={"max_depth": (2, 3)},
        max_evals=1,
        cv=2,
        early_stopping=None,
        visualize=False,
        show_progress_bar=False,
    )

    assert isinstance(results["best_pipeline"], MLPipeline)


def test_tune_with_verbose_support():
    print("\nTesting tuning with verbose-supporting algorithm...")

    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
    y = pd.Series(y)

    # Define parameter ranges
    param_ranges = {"n_estimators": (10, 20), "max_depth": (2, 4)}

    # Run tuning with RandomForestClassifier (supports verbose)
    results = MLPipeline.tune(
        X=X,
        y=y,
        algorithm=RandomForestClassifier,
        preprocessor=PreProcessor(),
        param_ranges=param_ranges,
        max_evals=2,  # Small number for testing
        verbose=1,
        visualize=False,
        log_best_model=False,
    )

    assert (
        "verbose" in results["best_params"]
    ), "verbose parameter not found in best_params for verbose-supporting algorithm"
    assert (
        results["best_params"]["verbose"] == 1
    ), "incorrect verbose value in best_params"

    print("Tuning with verbose-supporting algorithm tests passed.")


def test_tune_without_verbose_support():
    print("\nTesting tuning with non-verbose-supporting algorithms...")

    # Test SimpleModelNoVerbose with classification data
    print("Testing SimpleModelNoVerbose (classification)...")
    X_clf, y_clf = make_classification(n_samples=100, n_features=4, random_state=42)
    X_clf = pd.DataFrame(X_clf, columns=[f"feature_{i}" for i in range(4)])
    y_clf = pd.Series(y_clf)

    param_ranges = {"n_estimators": (10, 20)}

    results = MLPipeline.tune(
        X=X_clf,
        y=y_clf,
        algorithm=SimpleModelNoVerbose,
        preprocessor=PreProcessor(),
        param_ranges=param_ranges,
        max_evals=2,  # Small number for testing
        verbose=1,
        visualize=False,
        log_best_model=False,
    )

    assert (
        "verbose" not in results["best_params"]
    ), "verbose parameter found in best_params for non-verbose-supporting algorithm"

    # Test DecisionTreeRegressor with regression data
    print("Testing DecisionTreeRegressor (regression)...")
    X_reg, y_reg = make_regression(n_samples=100, n_features=4, random_state=42)
    X_reg = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(4)])
    y_reg = pd.Series(y_reg)

    param_ranges = {"max_depth": (2, 4), "min_samples_split": (2, 4)}

    results = MLPipeline.tune(
        X=X_reg,
        y=y_reg,
        algorithm=DecisionTreeRegressor,
        preprocessor=PreProcessor(),
        param_ranges=param_ranges,
        max_evals=2,  # Small number for testing
        verbose=1,
        visualize=False,
        log_best_model=False,
    )

    assert (
        "verbose" not in results["best_params"]
    ), "verbose parameter found in best_params for DecisionTreeRegressor"

    print("Tuning with non-verbose-supporting algorithms tests passed.")


def test_classification_pipeline():
    print("\nTesting Classification Pipeline...")

    # Generate sample data
    print("Generating sample classification data...")
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y)

    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Initialize and fit preprocessor
    print("Initializing and fitting preprocessor...")
    preprocessor = PreProcessor()
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    print("Preprocessing completed.")

    # Initialize and fit pipeline
    print("Initializing and fitting ML pipeline...")
    model = RandomForestClassifier(random_state=42)
    pipeline = MLPipeline(model=model, preprocessor=preprocessor)
    pipeline.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate
    print("\nEvaluating model performance...")
    results = pipeline.evaluate(X_test, y_test, verbose=True, visualize=True)
    print("\nClassification Results:", results)

    # Test model explanation
    print("\nGenerating model explanations...")
    pipeline.explain_model(X_test)
    pipeline.explain_case(1)

    # Assertions
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "auc" in results
    assert "mcc" in results
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1"] <= 1
    assert 0 <= results["auc"] <= 1
    assert -1 <= results["mcc"] <= 1


def test_regression_pipeline():
    print("\nTesting Regression Pipeline...")

    # Generate sample data
    print("Generating sample regression data...")
    X, y = make_regression(
        n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y)

    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Initialize and fit preprocessor
    print("Initializing and fitting preprocessor...")
    preprocessor = PreProcessor()
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    print("Preprocessing completed.")

    # Initialize and fit pipeline
    print("Initializing and fitting ML pipeline...")
    model = RandomForestRegressor(random_state=42)
    pipeline = MLPipeline(model=model, preprocessor=preprocessor)
    pipeline.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate
    print("\nEvaluating model performance...")
    results = pipeline.evaluate(X_test, y_test, verbose=True, visualize=True)
    print("\nRegression Results:", results)

    # Test model explanation
    print("\nGenerating model explanations...")
    pipeline.explain_model(X_test)

    # Assertions
    assert isinstance(results, dict)
    assert "rmse" in results
    assert "r2" in results
    assert "adj_r2" in results
    assert results["rmse"] >= 0
    assert -float("inf") <= results["r2"] <= 1
    assert -float("inf") <= results["adj_r2"] <= 1


def test_explain_dependence():
    """Test explain_dependence with different algorithms and datasets"""

    # Test classification algorithms
    def test_classification_model(model_class, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        # Create a small classification dataset
        X, y = make_classification(
            n_samples=100,  # Small dataset for quick testing
            n_features=4,  # Few features for clarity
            n_informative=3,
            n_redundant=1,
            random_state=42,
        )

        # Create meaningful feature names
        feature_names = ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")

        # Initialize and train model
        model = model_class(**model_kwargs)
        pipeline = MLPipeline(model=model)
        pipeline.fit(X, y)

        # Generate explanations
        pipeline.explain_model(X)

        # Test single feature plot
        pipeline.explain_dependence("Feature_1")

        # Test feature interaction plot
        pipeline.explain_dependence("Feature_1", "Feature_2")

        return True  # If we get here without errors, test passed

    # Test regression algorithms
    def test_regression_model(model_class, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        # Create a small regression dataset
        X, y = make_regression(
            n_samples=100,  # Small dataset for quick testing
            n_features=4,  # Few features for clarity
            n_informative=3,
            random_state=42,
        )

        # Create meaningful feature names
        feature_names = ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")

        # Initialize and train model
        model = model_class(**model_kwargs)
        pipeline = MLPipeline(model=model)
        pipeline.fit(X, y)

        # Generate explanations
        pipeline.explain_model(X)

        # Test single feature plot
        pipeline.explain_dependence("Feature_1")

        # Test feature interaction plot
        pipeline.explain_dependence("Feature_1", "Feature_2")

        return True  # If we get here without errors, test passed

    # Test classification models
    classification_models = [
        (RandomForestClassifier, {"n_estimators": 10, "random_state": 42}),
        (DecisionTreeClassifier, {"random_state": 42}),
        (LogisticRegression, {"random_state": 42}),
        (lgb.LGBMClassifier, {"n_estimators": 10, "random_state": 42}),
    ]

    for model_class, kwargs in classification_models:
        assert test_classification_model(model_class, kwargs)

    # Test regression models
    regression_models = [
        (RandomForestRegressor, {"n_estimators": 10, "random_state": 42}),
        (DecisionTreeRegressor, {"random_state": 42}),
        (LinearRegression, {}),
        (lgb.LGBMRegressor, {"n_estimators": 10, "random_state": 42}),
    ]

    for model_class, kwargs in regression_models:
        assert test_regression_model(model_class, kwargs)


def test_explain_dependence_plot_errors():
    """Test error handling in explain_dependence_plot"""

    # Create a small dataset
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=["Feature_1", "Feature_2", "Feature_3", "Feature_4"])
    y = pd.Series(y)

    # Initialize model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    pipeline = MLPipeline(model=model)
    pipeline.fit(X, y)

    # Test calling explain_dependence_plot before explain_model
    with pytest.warns(UserWarning, match="Please explain model first"):
        pipeline.explain_dependence("Feature_1")

    # Generate explanations
    pipeline.explain_model(X)

    # Test with non-existent feature
    with pytest.raises(ValueError, match="not found in the dataset"):
        pipeline.explain_dependence("NonExistentFeature")

    # Test with non-existent interaction feature
    with pytest.raises(ValueError, match="not found in the dataset"):
        pipeline.explain_dependence("Feature_1", "NonExistentFeature")


def test_column_order_consistency():
    """Test automatic column reordering for prediction data"""
    print("\nTesting Column Order Consistency...")

    # Test 1: Classification with reordered columns
    print("Testing classification with reordered columns...")
    X_clf, y_clf = make_classification(n_samples=100, n_features=4, random_state=42)
    feature_names = ["Feature_A", "Feature_B", "Feature_C", "Feature_D"]
    X_clf = pd.DataFrame(X_clf, columns=feature_names)
    y_clf = pd.Series(y_clf)

    # Train model
    model_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    pipeline_clf = MLPipeline(model=model_clf)
    pipeline_clf.fit(X_clf, y_clf)

    # Predict with same order
    X_test_same_order = X_clf.iloc[:10].copy()
    pred_same_order = pipeline_clf.predict(context=None, model_input=X_test_same_order)

    # Predict with different column order
    X_test_reordered = X_clf.iloc[:10][
        ["Feature_D", "Feature_A", "Feature_C", "Feature_B"]
    ].copy()
    pred_reordered = pipeline_clf.predict(context=None, model_input=X_test_reordered)

    # Predictions should be identical regardless of input column order
    np.testing.assert_array_almost_equal(
        pred_same_order,
        pred_reordered,
        decimal=10,
        err_msg="Predictions differ when columns are reordered",
    )
    print("✓ Classification predictions are consistent across column orders")

    # Test 2: Regression with reordered columns
    print("Testing regression with reordered columns...")
    X_reg, y_reg = make_regression(n_samples=100, n_features=4, random_state=42)
    X_reg = pd.DataFrame(X_reg, columns=feature_names)
    y_reg = pd.Series(y_reg)

    # Train model
    model_reg = RandomForestRegressor(n_estimators=10, random_state=42)
    pipeline_reg = MLPipeline(model=model_reg)
    pipeline_reg.fit(X_reg, y_reg)

    # Predict with same order
    X_test_reg_same = X_reg.iloc[:10].copy()
    pred_reg_same = pipeline_reg.predict(context=None, model_input=X_test_reg_same)

    # Predict with different column order
    X_test_reg_reordered = X_reg.iloc[:10][
        ["Feature_C", "Feature_D", "Feature_A", "Feature_B"]
    ].copy()
    pred_reg_reordered = pipeline_reg.predict(
        context=None, model_input=X_test_reg_reordered
    )

    # Predictions should be identical
    np.testing.assert_array_almost_equal(
        pred_reg_same,
        pred_reg_reordered,
        decimal=10,
        err_msg="Regression predictions differ when columns are reordered",
    )
    print("✓ Regression predictions are consistent across column orders")

    # Test 3: Missing column raises error
    print("Testing error handling for missing columns...")
    X_missing = X_clf.iloc[:10][["Feature_A", "Feature_B"]].copy()  # Missing C and D

    with pytest.raises(KeyError):
        pipeline_clf.predict(context=None, model_input=X_missing)
    print("✓ KeyError raised correctly for missing columns")

    # Test 4: Extra columns are ignored
    print("Testing that extra columns are ignored...")
    X_extra = X_clf.iloc[:10].copy()
    X_extra["Feature_E"] = 999  # Add extra column
    X_extra["Feature_F"] = -999  # Add another extra column

    # Reorder with extra columns
    X_extra_reordered = X_extra[
        ["Feature_F", "Feature_B", "Feature_E", "Feature_A", "Feature_D", "Feature_C"]
    ]
    pred_extra = pipeline_clf.predict(context=None, model_input=X_extra_reordered)

    # Should match original predictions (extra columns ignored)
    np.testing.assert_array_almost_equal(
        pred_same_order,
        pred_extra,
        decimal=10,
        err_msg="Predictions differ when extra columns are present",
    )
    print("✓ Extra columns are correctly ignored")

    print("All column order consistency tests passed!")


if __name__ == "__main__":
    print("Starting MLPipeline tests...")

    # Test verbose support functionality
    print("\n" + "=" * 50)
    print("Running Verbose Support Tests")
    print("=" * 50)
    test_supports_verbose()
    test_tune_with_verbose_support()
    test_tune_without_verbose_support()

    # Test classification
    print("\n" + "=" * 50)
    print("Running Classification Tests")
    print("=" * 50)
    test_classification_pipeline()

    # Test regression
    print("\n" + "=" * 50)
    print("Running Regression Tests")
    print("=" * 50)
    test_regression_pipeline()

    # Test explain_dependence
    print("\n" + "=" * 50)
    print("Running Explain Dependence Plot Tests")
    print("=" * 50)
    test_explain_dependence()
    test_explain_dependence_plot_errors()

    # Test column order consistency
    print("\n" + "=" * 50)
    print("Running Column Order Consistency Tests")
    print("=" * 50)
    test_column_order_consistency()

    print("\nAll tests completed successfully!")
