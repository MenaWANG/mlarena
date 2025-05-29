"""
Test script for MLPipeline class.
"""

# Standard library imports
from typing import Any

import matplotlib

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Local imports
from mlarena import MLPipeline, PreProcessor

matplotlib.use("Agg")  # Use non-interactive backend for testing


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
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1"] <= 1
    assert 0 <= results["auc"] <= 1


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
