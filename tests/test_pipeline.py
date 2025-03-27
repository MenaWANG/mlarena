"""
Test script for ML_PIPELINE class.
"""

# Standard library imports
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Local imports
from mlarena import PreProcessor, ML_PIPELINE

def test_classification_pipeline():
    print("\nTesting Classification Pipeline...")
    
    # Generate sample data
    print("Generating sample classification data...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                             n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
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
    pipeline = ML_PIPELINE(model=model, preprocessor=preprocessor)
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
    
    return results

def test_regression_pipeline():
    print("\nTesting Regression Pipeline...")
    
    # Generate sample data
    print("Generating sample regression data...")
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, 
                          noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
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
    pipeline = ML_PIPELINE(model=model, preprocessor=preprocessor)
    pipeline.fit(X_train, y_train)
    print("Model training completed.")
    
    # Evaluate
    print("\nEvaluating model performance...")
    results = pipeline.evaluate(X_test, y_test, verbose=True, visualize=True)
    print("\nRegression Results:", results)
    
    # Test model explanation
    print("\nGenerating model explanations...")
    pipeline.explain_model(X_test)
    pipeline.explain_case(1)
    
    return results

if __name__ == "__main__":
    print("Starting ML_PIPELINE tests...")
    
    # Test classification
    print("\n" + "="*50)
    print("Running Classification Tests")
    print("="*50)
    classification_results = test_classification_pipeline()
    
    # Test regression
    print("\n" + "="*50)
    print("Running Regression Tests")
    print("="*50)
    regression_results = test_regression_pipeline()
    
    print("\nAll tests completed successfully!") 