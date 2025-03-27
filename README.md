# MLArena

An algorithm-agnostic machine learning toolkit for model training, diagnostics and optimization.

## Features

- **Comprehensive ML Pipeline**:
  - End-to-end workflow from preprocessing to deployment
  - Model-agnostic design (works with any scikit-learn compatible model)
  - Support for both classification and regression tasks
  - Early stopping and validation set support
  - MLflow integration for experiment tracking and deployment  

- **Intelligent Preprocessing**:
  - Automated feature type detection and handling
  - Smart encoding recommendations based on feature cardinality and rare category
  - Target encoding with visualization to support smoothing parameter selection
  - Missing value handling with configurable strategies
  - Feature selection recommendations with mutual information analysis

- **Advanced Model Evaluation**:
  - Comprehensive metrics for both classification and regression
  - Diagnostic visualization of model performance
  - Threshold analysis for classification tasks
  - SHAP-based model explanations (global and local)
  - Cross-validation with variance penalty

- **Hyperparameter Optimization**:
  - Bayesian optimization with Hyperopt
  - Cross-validation based tuning
  - Parallel coordinates visualization for search space analysis
  - Early stopping to prevent overfitting
  - Variance penalty to ensure stable solutions


## Installation

```bash
pip install mlarena
```

## Quick Start

```python
from mlarena import PreProcessor, ML_PIPELINE
from sklearn.ensemble import RandomForestClassifier

# Initialize the preprocessor
preprocessor = PreProcessor(
    num_impute_strategy='median',
    cat_impute_strategy='most_frequent'
)

# Initialize the pipeline
ml_pipeline = ML_PIPELINE(
    model = RandomForestClassifier(),
    preprocessor = preprocessor
)

# Train the model
ml_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = ml_pipeline.predict(X_test)

# Comprehensive Evaluation Report and Visuals
results = ml_pipeline.evaluate(X_test, y_test)

# Explain the model
ml_pipeline.explain_model(X_test)

```

## Documentation

### PreProcessor

The `PreProcessor` class handles all data preprocessing tasks:

- Filter Feature Selection
- Categorical encoding (OneHot, Target)
- Recommendation of encoding strategy
- Plot to compare target encoding smoothing parameters
- Numeric scaling
- Missing value imputation

### ML_PIPELINE

The `ML_PIPELINE` class provides a complete machine learning workflow:

- Algorithm agnostic model wrapper
- Support both classification (binary) and regression algorithms
- Model training and scoring
- Model global and local explanation
- Model evaluation with comprehensive reporting and plots
- Iterative hyperparameter tuning with diagnostic plot
- Threshold analysis and optimization for classification models


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 