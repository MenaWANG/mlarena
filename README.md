# MLArena

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/mlarena.svg)](https://badge.fury.io/py/mlarena)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![CI/CD](https://github.com/MenaWANG/mlarena/actions/workflows/mlarena.yml/badge.svg)](https://github.com/MenaWANG/mlarena/actions/workflows/mlarena.yml)

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

## Publications

Learn more about the concepts and methodologies behind MLArena through these articles:

1. [Algorithm-Agnostic Model Building with MLflow](https://medium.com/data-science/algorithm-agnostic-model-building-with-mlflow-b106a5a29535) - Published in Towards Data Science
   > A foundational guide demonstrating how to build algorithm-agnostic ML pipelines using mlflow.pyfunc. The article explores creating generic model wrappers, encapsulating preprocessing logic, and leveraging MLflow's unified model representation for seamless algorithm transitions.

2. [Explainable Generic ML Pipeline with MLflow](https://medium.com/data-science/explainable-generic-ml-pipeline-with-mlflow-2494ca1b3f96) - Published in Towards Data Science
   > An advanced implementation guide that extends the generic ML pipeline with more sophisticated preprocessing and SHAP-based model explanations. The article demonstrates how to build a production-ready pipeline that supports both classification and regression tasks, handles feature preprocessing, and provides interpretable model insights while maintaining algorithm agnosticism.

## Installation

```bash
pip install mlarena
```

## Usage Example

* For quick start with a basic example, see [examples/basic_usage.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/basic_usage.ipynb).   
* For more advanced examples, see [examples/advanced_usage.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/advanced_usage.ipynb).   

## Visual Examples:

### Model Performance Analysis

![Classification Model Performance](docs/images/model_performance_classification.png)    

![Regression Model Performance](docs/images/model_performance_regression.png)    

### Explainable ML
One liner to create global and local explaination based on shap that will work across various classification and regression algorithms.     

![Global Explanation](docs/images/global_explanation.png)    

![Local Explanation](docs/images/local_explanation.png)    

### Hyperparameter Optimization
Parallel Coordinate plot for hyperparameter search space diagnostics.    
![Hyperparameter Search Space](docs/images/parallel_coordinates.png)


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
