# MLArena

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/mlarena.svg)](https://pypi.org/project/mlarena/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![CI/CD](https://github.com/MenaWANG/mlarena/actions/workflows/mlarena.yml/badge.svg)](https://github.com/MenaWANG/mlarena/actions/workflows/mlarena.yml)


`mlarena` is an algorithm-agnostic machine learning toolkit for streamlined model training, diagnostics, and optimization. Implemented as a custom `mlflow.pyfunc` model, it ensures seamless integration with the MLflow ecosystem for robust experiment tracking, model versioning, and framework-agnostic deployment.

It blends smart automation that embeds ML best practices with comprehensive tools for expert-level customization and diagnostics. This unique combination fills the gap between manual ML development and fully automated AutoML platforms. Moreover, it comes with a suite of practical utilities for data analysis and visualizations - see our [comparison with AutoML platforms](https://github.com/MenaWANG/mlarena/blob/master/docs/comparision-autoML.md) to determine which approach best fits your needs.

## Publications

Read about the concepts and methodologies behind MLArena through these articles:

1. [Algorithm-Agnostic Model Building with MLflow](https://medium.com/data-science/algorithm-agnostic-model-building-with-mlflow-b106a5a29535) - Published in Towards Data Science
   > A foundational guide demonstrating how to build algorithm-agnostic ML pipelines using mlflow.pyfunc. The article explores creating generic model wrappers, encapsulating preprocessing logic, and leveraging MLflow's unified model representation for seamless algorithm transitions.

2. [Explainable Generic ML Pipeline with MLflow](https://medium.com/data-science/explainable-generic-ml-pipeline-with-mlflow-2494ca1b3f96) - Published in Towards Data Science
   > An advanced implementation guide that extends the generic ML pipeline with more sophisticated preprocessing and SHAP-based model explanations. The article demonstrates how to build a production-ready pipeline that supports both classification and regression tasks, handles feature preprocessing, and provides interpretable model insights while maintaining algorithm agnosticism.

## Installation

The package is undergoing rapid development at the moment (pls see [CHANGELOG](https://github.com/MenaWANG/mlarena/blob/master/CHANGELOG.md) for details), it is therefore highly recommended to install with specific versions. For example

```bash
%pip install mlarena==0.3.4
```

If you are using the package in [Databricks ML Cluster with DBR runtime >= 16.0](https://learn.microsoft.com/en-us/azure/databricks/release-notes/runtime/16.0ml), you can install without dependencies like below:

```bash
%pip install mlarena==0.3.4 --no-deps
```
If you are using earlier DBR runtimes, simply install `optuna` in addition like below. Note: As of 2025-04-26, `optuna` is recommended by Databricks, while `hyperopt` will be [removed from Databricks ML Runtime](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/).

```bash
%pip install mlarena==0.3.4 --no-deps
%pip install optuna==3.6.1
```

## Usage Example

* For quick start with a basic example, see [1.basic_usage.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/1.basic_usage.ipynb).   
* For more advanced examples on model optimization, see [2.advanced_usage.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/2.advanced_usage.ipynb).   
* For visualization utilities, see [3.utils_plot.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/3.utils_plot.ipynb).
* For data cleaning and manipulation utilities, see [3.utils_data.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/3.utils_data.ipynb).
* For input/output utilities, see [3.utils_io.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/3.utils_io.ipynb)
* For handling common challenges in machine learning, see [4.ml_discussions.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/4.ml_discussions.ipynb).

## Visual Examples:

### Quick Start
Train and evaluate models quickly with `mlarena`'s default preprocessing pipeline, comprehensive reporting, and model explainability. The framework handles the complexities behind the scenes, allowing you to focus on insights rather than boilerplate code. See [1.basic_usage.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/1.basic_usage.ipynb) for complete examples.
<br>

| Category | Classification Metrics & Plots | Regression Metrics & Plots |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Metrics | **Evaluation Parameters**<br>• Threshold (classification cutoff)<br>• Beta (F-beta weight parameter)<br><br>**Core Performance Metrics**<br>• Accuracy (overall correct predictions)<br>• Precision (true positives / predicted positives)<br>• Recall (true positives / actual positives)<br>• F1 Score (harmonic mean of Precision & Recall)<br>• Fβ Score (weighted harmonic mean, if β ≠ 1)<br>• AUC (ranking quality)<br>• Log Loss (confidence-weighted error)<br><br>**Prediction Distribution**<br>• Positive Rate (fraction of positive predictions)<br>• Base Rate (actual positive class rate) | **Error Metrics**<br>• RMSE (Root Mean Squared Error)<br>• MAE (Mean Absolute Error)<br>• Median Absolute Error<br>• NRMSE (Normalized RMSE as % of mean, std, IQR)<br>• MAPE (Mean Absolute Percentage Error, excl. zeros)<br>• SMAPE (Symmetric Mean Absolute Percentage Error)<br><br>**Goodness of Fit**<br>• R² (Coefficient of Determination)<br>• Adjusted R²<br><br>**Improvement over Baseline**<br>• RMSE Improvement over Mean Baseline (%)<br>• RMSE Improvement over Median Baseline (%) |
| Plots | • Metrics vs Threshold (Precision, Recall, Fβ, with vertical threshold line)<br>• ROC Curve<br>• Confusion Matrix (with colored overlays) | • Residual analysis (residuals vs predicted, with 95% prediction interval)<br>• Prediction error plot (actual vs predicted, with perfect prediction line and error bands) |
<br>

#### Classification Models

![Classification Model Performance](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/model_performance_classification.png)    

#### Regression Models

![Regression Model Performance](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/model_performance_regression.png)    

### Explainable ML
One liner to create global and local explanation based on SHAP that will work across various classification and regression algorithms.     

![Global Explanation](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/global_explanation.png)    

![Local Explanation](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/local_explanation.png)    

### Hyperparameter Optimization
`mlarena` offers iterative hyperparameter tuning with cross-validation for robust results and parallel coordinates visualization for search space diagnostics. See [2.advanced_usage.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/2.advanced_usage.ipynb) for more.

![Hyperparameter Search Space](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/parallel_coordinates.png)

### Plotting Utility Functions 

`mlarena` offers handy utility visualizations for data exploration. Pls see [3.utils_plot.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/3.utils_plot.ipynb) for more.

#### `plot_box_scatter` for comparing numerical distributions across categories with optional statistical testing
![plot_box_scatter](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/plot_box_scatter.png)

#### `plot_distribution_over_time` for comparing numerical distributions over time
![plot_distribution_over_time](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/plot_distribution_over_time.png)

#### `plot_distribution_over_time` color points using `point_hue`
![plot_distribution_over_time_point_hue](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/plot_distribution_over_time_point_hue.png)

#### `plot_stacked_bar_over_time` for comparing categorical distributions over time
![plot_stacked_bar_over_time](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/plot_stacked_bar_over_time.png)

#### `plot_metric_event_over_time` for timeseries trends and events
![plot_metric_event_over_time](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/plot_metric_event_over_time.png)

### Data Utilities

Some handy utilities for data validation, cleaning and manipulations. Pls see [3.utils_data.ipynb](https://github.com/MenaWANG/mlarena/blob/master/examples/3.utils_data.ipynb) for more. 

#### `is_primary_key`
![is_primary_key](https://raw.githubusercontent.com/MenaWANG/mlarena/master/docs/images/is_primary_key_demo.png)


## Features

**Algorithm Agnostic ML Pipeline**
- Unified interface for any scikit-learn compatible model
- Consistent workflow across classification and regression tasks
- Automated report generation with comprehensive metrics and visuals
- Built on `mlflow.pyfunc` for seamless MLOps integration
- Automated experiment tracking of parameters, metrics, and models
- Simplified handoff from experimentation to production via the MLflow framework

**Intelligent Preprocessing**
- Streamlined feature preprocessing with smart defaults and minimal code
- Automatic feature analysis with data-driven encoding recommendations 
- Integrated target encoding with visualization for optimal smoothing selection
- Feature filtering based on information theory metrics (mutual information)
- Intelligent feature name sanitization to prevent pipeline failure
- Handles the full preprocessing pipeline from missing values to feature encoding
- Seamless integration with scikit-learn and MLflow for production deployment


**Model Optimization**
- Efficient hyperparameter tuning with Optuna's TPE sampler
- Smart early stopping with patient pruning to save computation resources
  - Configurable early stopping parameter
  - Startup trials before pruning begins
  - Warmup steps per trial
- Cross-validation with variance penalty to prevent overfitting
- Parallel coordinates visualization for search history tracking and parameter space diagnostics
- Automated threshold optimization with business-focused F-beta scoring
  - Cross-validation or bootstrap methods
  - Configurable beta parameter for precision/recall trade-off
  - Confidence intervals for bootstrap method
- Flexible metric selection for optimization
  - Classification: AUC (default), F1, accuracy, log_loss
  - Regression: RMSE (default), MAE, median_ae, SMAPE, NRMSE (mean/std/IQR)

**Performance Analysis**
- Comprehensive metric tracking
  - Classification: AUC, F1, Fbeta, precision, recall, accuracy, log_loss, positive_rate, base_rate
  - Regression: RMSE, MAE, median_ae, R2, adjusted R2, MAPE, SMAPE, NRMSE (mean/std/IQR), improvement over mean/median baselines
- Performance visualization
  - Classification: 
    - Metrics vs Threshold plot (precision, recall, F-beta)
    - ROC curve with AUC score
    - Confusion matrix with color-coded cells
  - Regression:
    - Residual analysis with 95% prediction intervals
    - Prediction error plot with perfect prediction line and error bands
- Model interpretability
  - Global feature importance
  - Local prediction explanations

**Utils**
- Advanced plotting utilities
  - Box plots with scatter overlay for detailed distribution analysis
  - Time series metrics visualization with optional event markers
  - Stacked bar for categorical distributions over time with flexible aggregation
  - Numeric distribution tracking over time with flexible aggregation
- Data manipulation tools
  - Standardized dollar amount cleaning for financial analysis
  - Value counts with percentage calculation for categorical analysis
  - Smart date column transformation with flexible format handling
  - Schema and data quality utilities
    - Primary key validation with detailed diagnostics
    - Alphabetically sorted schema display
    - Safe column selection with case sensitivity options
    - Automatic removal of fully null columns
- I/O utilities
  - `save_object`: Store Python objects to disk with customizable options
    - Support for pickle and joblib backends
    - Optional date stamping in filenames
    - Compression support for joblib backend
  - `load_object`: Retrieve Python objects with automatic backend detection
    - Seamless loading regardless of storage format
    - Direct compatibility with paths returned by `save_object`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
