# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 

### Added
- Auto-detection of task type (classification/regression) in `tune` function
  - Automatically detects whether the algorithm is a classifier or regressor based on its capabilities
  - Removes the need for users to explicitly specify task type
  - Maintains backward compatibility by allowing manual task specification

## [0.2.0] - 2025-04-26

## Added
- Switched hyperparameter tuning framework from `hyperopt` to `Optuna`
  - Support for parallel coordinate plots with consistent colorscales (red for higher score)
  - Implemented patient pruning for early stopping in hyperparameter search
  - Utilize n_startup_trials to delay pruning until sufficient baseline trials are completed
  - Added disable_optuna_logging parameter to control Optuna's verbosity
  - Note that `hyperopt` will be [removed from the Databricks DBR ML Runtime](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/hyperopt-concepts#:~:text=The%20open%2Dsource%20version%20of,Hyperopt%20distributed%20hyperparameter%20tuning%20functionality.)
- Add `plot_medical_timeseries` function to plot_utils
  - Support 1-2 metrics with medical standard colors (black/red)
  - Support treatment markers and min/max annotations
  - Display alternating year backgrounds
- Add `plot_stacked_bar_over_time` function to plot_utils
  - Display time-based categorical distributions
  - Support percentage and count visualization
  - Include custom color palettes and label mapping
  

## [0.1.9] - 2025-04-19

## Added
- Doc: Add example notebook `3.utils_plot.ipynb` demonstrating plot utilities
- Doc: Add `4.ml_discussions.ipynb` explaining MLflow integration challenges and solutions for Pandas category dtype.
- Add model signature to `_log_model` method, improving MLflow logging in both `evaluate` and `tune` methods

## Fixed
- update naming of `boxplot_scatter_overlay` parameters to be consistent with seaborn and matplotlib style
- Add automatic type conversion for pandas category and integer columns in MLflow models
- Improve MLflow model loading compatibility with proper handling of pandas category dtypes
- Fix SHAP visualization warnings by using explicit random number generator


## [0.1.8] - 2025-04-14

## Added
- Add MLflow model logging functionality:
  - Add `_log_model` method to ML_PIPELINE class
  - Add optional `log_model` parameter to `evaluate` method 
  - Add optional `log_best_model` parameter to `tune` method
  - Return model logging info in evaluation results when logging enabled
  - Update basic and advanced usage examples to demonstrate MLflow integration

- Add utility functions:
  - New utils module with initial plotting utilities (`boxplot_scatter_overlay`)
  - Modular structure for utility functions with clean import paths

## Fixed
- tests/test_pipeline.py: Handle SHAP visualization errors
- test/test_preprocessor: Add testing for all feature types
- leverage `drop` parameter and "if_binary" option in sklearn OneHotEncoder rather than rewrite the logic


## [0.1.7] - 2025-04-08

### Added
- New `drop_first` (boolean) parameter in PreProcessor to fine tune one-hot encoding
- Intelligent handling of binary vs multi-category features in one-hot encoding
  - Binary features always use `drop="first"` 
  - Multi-category features behavior controlled by the `drop_first` parameter
- Update get_transformed_cat_cols to handle dropped categories and ColumnTransformer

### Fixed
- Replace fit with fit_transform to leverage TargetEncoder's cross-fitting ability

## [0.1.6] - 2025-04-03

### Fixed
- The `tune` methods only need to produce comprehensive reports automatically for the best run


## [0.1.5] - 2025-04-02

### Changed
- Updated core dependencies for better compatibility with Databricks ML Runtime 15.2
- Dynamic package versioning 

## [0.1.4] - 2025-03-30

### Changed
- Improved documentation clarity
  - Updated Python version badge to explicitly show supported versions (3.10, 3.11, 3.12)

### Fixed
- Fixed PyPI badge reliability by switching to badge.fury.io

## [0.1.3] - 2025-03-30

### Changed
- Updated README.md with improved documentation and images
- Updated PyPI badge to use shields.io for better reliability

### Fixed
- None

## [0.1.2] - 2025-03-29

### Added
- Added pytest-cov for code coverage reporting
- Added comprehensive test assertions for pipeline evaluation results

### Changed
- Improved test structure with proper assertions instead of return statements
- Added warning filters in pytest.ini to suppress third-party deprecation warnings
- Improved code formatting with black (line length set to 88)

### Fixed
- None

## [0.1.1] - 2025-03-29

### Fixed
- Added `setuptools` as a dependency to fix compatibility issues with hyperopt's ATPE module
- This fix is particularly important for Python 3.12+ users where setuptools is not automatically installed

## [0.1.0] - 2025-03-27

### Added
- Initial release of MLArena
- `PreProcessor` class for data preprocessing
  - Filter Feature Selection
  - Categorical encoding (OneHot, Target)
  - Recommendation of encoding strategy
  - Plot to compare target encoding smoothing parameters
  - Numeric scaling
  - Missing value imputation
- `ML_PIPELINE` class for machine learning workflow
  - Algorithm agnostic model wrapper
  - Support both classification (binary) and regression algorithms
  - Model training and scoring
  - Model global and local explanation
  - Model evaluation with comprehensive reporting and plots
  - Iterative hyperparameter tuning with diagnostic plot
  - Threshold analysis and optimization for classification models
- Comprehensive documentation
  - Installation guide
  - Usage guide
  - API reference
  - Examples
  - Contributing guidelines
- Test suite with pytest
- Development tools configuration
  - black for code formatting
  - flake8 for linting
  - isort for import sorting
  - mypy for type checking

### Changed
- None (initial release)

### Deprecated
- None (initial release)

### Removed
- None (initial release)

### Fixed
- None (initial release)

### Security
- None (initial release) 