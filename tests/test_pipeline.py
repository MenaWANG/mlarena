"""
Test script for MLPipeline class.
"""

# Standard library imports
from typing import Any

import matplotlib

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Local imports
from mlarena import MLPipeline, PreProcessor

matplotlib.use("Agg")  # Use non-interactive backend for testing


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

    print("\nAll tests completed successfully!")
