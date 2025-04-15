# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.9] 

## Added
- Doc: Add example notebook `3.utils_plot.ipynb` demonstrating plot utilities


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