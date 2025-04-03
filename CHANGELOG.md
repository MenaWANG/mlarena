# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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