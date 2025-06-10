# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.2] - 2025-06-10

### Added
- **Major Enhancement**: Statistical testing capabilities to plotting functions:
  - **`plot_box_scatter`** - Statistical tests for continuous outcomes across categorical groups:
    - Added support for "anova" (One-way ANOVA with η² effect size) and "kruskal" (Kruskal-Wallis with ε² effect size)
    - Automatic validation requiring at least 2 groups for testing
    - Returns descriptive statistics (count, mean, median, std) per category
  - **`plot_stacked_bar`** - Statistical tests for categorical associations:
    - Added support for "chi2" (Chi-square test with Cramér's V effect size) and "fisher" (Fisher's exact test for 2×2 tables)
    - Automatic validation ensuring Fisher's test only used with 2×2 tables
    - Returns contingency tables with totals, percentage tables, and sample size information
  - **Common features across both functions**:
    - `stat_summary` parameter: Returns relevant statistical summaries (renamed from `return_summary` for consistency)
    - `stat_test` parameter: Performs statistical tests with standardized lowercase method names
    - `stats_only` parameter: Skip plotting entirely for analysis workflows - perfect for Jupyter notebooks
    - `show_stat_test` parameter: Display test results as annotations on plots with professional formatting
      - Scientific p-value conventions (p < 0.001, p < 0.01, p < 0.05, exact values for p ≥ 0.05)
      - Customizable positioning (`stat_annotation_pos`), font size, and background styling
      - Publication-ready annotations with proper test names and effect sizes
    - Enhanced return structure: Unified `results` dictionary with optional `summary_table` and `stat_test` outputs
    - Maintains full backward compatibility for existing function calls

### Changed
- **Breaking Change**: Renamed `return_summary` parameter to `stat_summary` in `plot_box_scatter`:
  - Provides consistent naming convention with other stat-related parameters
  - Creates intuitive "stat" prefix pattern: `stat_summary`, `stat_test`, `stats_only`
  - Updated all documentation and examples to reflect new parameter name
- Enhanced statistical analysis robustness:
  - Added comprehensive input validation and warning messages
  - Improved error handling for edge cases (insufficient groups, missing data)

### Improved
- Optimized `plot_box_scatter` performance and code organization:
  - Moved statistical calculations before plotting code for `stats_only` efficiency
  - Reorganized function structure with early returns for better flow
  - Enhanced parameter validation with clear error messages
  - Improved documentation with comprehensive parameter descriptions and usage examples

## [0.3.1] - 2025-06-03

### Added
- Enhanced plotting utilities:
  - Added `plot_stacked_bar` function for visualizing categorical distributions
    - Supports percentage and count-based visualization
    - Includes custom labeling and color palette options
    - Features configurable transparency with `bar_alpha` parameter
    - Provides consistent styling with other plotting functions
- Enhanced data utility functions for DataFrame filtering:
  - Added `filter_rows_by_substring` function for filtering DataFrame rows based on substring matches in specified columns
    - Supports case-sensitive and case-insensitive matching (default: case-insensitive)
    - Handles NaN values gracefully by converting to strings
    - Includes comprehensive error handling for non-existent columns
    - Ideal for real-world data cleaning tasks like transaction categorization
  - Added `filter_columns_by_substring` function for filtering DataFrame columns based on naming patterns
    - Supports case-sensitive and case-insensitive matching (default: case-insensitive)  
    - Handles non-string column names safely
    - Returns empty DataFrame with preserved index when no columns match
    - Perfect for selecting related columns by naming conventions (e.g., all "price_*" columns)
  - Added comprehensive test coverage with realistic business scenarios
    - Transaction data filtering examples demonstrating practical use cases
    - Edge case handling and error condition testing
    - Integration with existing test suite

### Changed
- Standardized parameter naming across data utility functions for consistency:
  - Renamed `strict` parameter to `case_sensitive` in `select_existing_cols` function
  - All filtering functions now use consistent `case_sensitive` parameter naming
  - Improves API consistency and user experience across the module
  - Updated function signatures, docstrings, and examples to reflect the change

### Improved
- Enhanced test organization and reliability:
  - Added pytest markers to `test_io_utils.py` for better test categorization
    - Added `@pytest.mark.file_io` markers to all file I/O operations
    - Usage: `pytest -m "not file_io"` to skip file operations during development if preferred
  - Added information regarding anti-virus software compatibility considerations in I/O tests
  - Improved test maintainability and developer experience

## [0.3.0] - 2025-05-30

### ⚠️ Breaking Changes
- **Renamed ML_PIPELINE to MLPipeline** to follow Python naming conventions (PEP 8):
  - The old `ML_PIPELINE` class name has been renamed to `MLPipeline`
  - The old name is now deprecated but still available for backward compatibility
  - Users will see deprecation warnings when using `ML_PIPELINE` with instructions to migrate
  - See [upgrading.md](docs/upgrading.md) for migration instructions
  - Reasoning: Class names should use CapWords (PascalCase) convention per PEP 8
  - Migration Timeline
    - Current version: `ML_PIPELINE` deprecated but functional with warnings
    - From v0.4.0: `ML_PIPELINE` support will be removed
    - Recommended action: Replace `ML_PIPELINE` with `MLPipeline`

### Added
- Enhanced SHAP visualization controls in `explain_model` method:
  - Added `max_features` parameter (default=20) to control how many features are displayed in SHAP plots
  - Added `group_remaining_features` parameter (default=True) to control whether remaining features are grouped together
    - When True: shows collective impact of all remaining features beyond `max_features`
    - When False: excludes remaining features for cleaner display
    - Note: Only applies to beeswarm plots; ignored for legacy summary plots
  - Improved plot type naming for clarity:
    - `"interactive"` → `"beeswarm"` (uses modern `shap.plots.beeswarm()`)
    - `"static"` → `"summary"` (uses legacy `shap.summary_plot()`)
    - `"auto"` remains unchanged (tries beeswarm first, falls back to summary)
  - Parameter names matching official SHAP documentation
  - Provides granular control over SHAP visualization presentation and feature focus

## [0.2.10] - 2025-05-25

### Added
- Added intelligent feature name sanitization to `PreProcessor`:
  - New `sanitize_feature_names` parameter (default=True) automatically converts problematic special characters in categorical feature names into ML-pipeline-friendly alternatives
  - Smart semantic replacements preserve important meaning:
    - `+` → `_plus` (range indicators like "60+")
    - `%` → `_pct` (percentages like "50%")
    - `<`, `>`, `=` → `_lt_`, `_gt_`, `_eq_` (comparisons like "X=Y" → "X_eq_Y")
  - Clean separator handling for better readability:
    - `&`, `|`, `/`, `*`, `-` → `_` (connectors like "A&B" → "A_B")
  - Comprehensive cleanup with regex fallback for any remaining special characters
  - Prevents common production failures by converts problematic characters to safe alternatives.
  - Added comprehensive test coverage for sanitization functionality
  - Added demo for user experimentation

### Improved
- Enhanced I/O utility functions in `io_utils`:
  - Improved parameter naming for better clarity and consistency:
    - In `save_object`: Changed `path` to `directory` and `filename` to `basename`
    - In `load_object`: Changed `file_path` to `filepath`
  - Fixed variable name bug in the `save_object` function
  - Enhanced internal variable naming with `final_filepath` for better code readability
  - Updated docstrings to clearly indicate return value compatibility between functions
  - Strengthened conceptual connection between `save_object` and `load_object`

### Documentation
- Added example notebook `3.utils_io.ipynb` demonstrating I/O utilities:
  - Examples showing how to save and load different types of objects:
    - Python dictionaries using pickle backend
    - Pandas DataFrames using joblib backend with compression
    - Complete ML_PIPELINE instances for model persistence
  - Data integrity verification after loading objects

## [0.2.9] - 2025-05-19

### Added
- Enhanced `plot_box_scatter` function with better error handling
  - Added warning when `point_hue` column is not found in DataFrame
  - Added graceful fallback to single-color plot when `point_hue` is missing
- Added new tests for `plot_box_scatter`
  - Test for warning when point_hue column is missing
  - Test for fallback to single-color behavior
  - Test for correct color assignment with valid point_hue
- Added `n_top_features` parameter to `filter_feature_selection`
  - Allows selecting top N features based on mutual information scores
  - Applies after basic filtering (missing values and unique values)
  - Provides alternative to threshold-based selection
  - Includes warning when requested features exceed available features

### Fixed
- Removed redundant `subplots_adjust` call in `plot_metric_event_over_time`
  - Eliminated layout warnings by relying on `constrained_layout`
  - Improved consistency with other plotting functions
  - Better compatibility with matplotlib's automatic layout management


## [0.2.8] - 2025-05-17

### Added
- Enhanced data utility functions:
  - Added `print_schema_alphabetically` function to display DataFrame schemas in sorted order
  - Added `is_primary_key` function to verify if column(s) can serve as a primary key
    - Supports both single column and composite key validation
    - Provides detailed feedback about missing values and uniqueness
    - Practical approach to handle real-world data with missing values
  - Added `select_existing_cols` function for safe column selection
    - Supports both case-sensitive and case-insensitive matching
    - Provides verbose mode for debugging column selection
    - Handles both single column and list inputs
  - Enhanced `value_counts_with_pct` function:
    - Added support for analyzing multiple columns (value combinations)
  - Improved `drop_fully_null_cols` function:
    - Added `verbose` parameter (default=False) to control output messaging
    - Better aligned with common data science practices for optional verbosity
  - Added comprehensive test coverage for all data utility functions
  - Updated example notebook with demonstrations of new functions
- Enhanced plotting utilities:
  - Optimized `plot_box_scatter` performance with `point_hue`
    - Significantly improved computational efficiency by plotting points by hue value
    - Reduced redundant scatter operations and memory usage
    - Enhanced label formatting and positioning
  - Standardized layout management across all plotting functions using `constrained_layout`
    - Improved handling of complex layouts with multiple subplots, colorbars, and legends
    - Better automatic adjustment of spacing between plot elements
    - More robust handling of rotated labels and varying font sizes
  - Added some test coverage for plotting utilities in pytest
- [EXPERIMENTAL] Added I/O utility functions in `io_utils`:
  - Added `save_object` and `load_object` for flexible object serialization
  - Supports both pickle and joblib backends with automatic defaults
  - Robust safety measures:
    - Validates backend selection ('pickle' or 'joblib')
    - Ensures file extension matches the chosen backend
    - Verifies joblib availability before use
    - Validates file existence before loading
    - Creates directories safely if they don't exist
  - Features include:
    - Date-based versioning in filenames
    - Automatic directory creation
    - Optional compression for joblib backend
    - Backward compatibility options via pickle protocol selection
  - Added extensive test coverage with safety considerations:
    - Tests for all error conditions
    - Validation of data integrity after save/load
    - Safe cleanup of test files
    - Windows-compatible file handling


### Changed
- Standardized function naming conventions in data utilities:
  - Adopted "cols" instead of "columns" for consistency and clarity
  - Rationale:
    - Reduces potential naming collisions 
    - More concise while maintaining clarity
    - Aligns with common data science community conventions
  - Updated functions:
    - `clean_dollar_cols`
    - `transform_date_cols`
    - `drop_fully_null_cols`
  - Unchanged functions (already following convention):
    - `value_counts_with_pct`
    - `print_schema_alphabetically`
    - `is_primary_key`
- Updated corresponding test files and documentation to reflect new naming convention


## [0.2.7] - 2025-05-14

### Added
- Added progress bar display during hyperparameter optimization in `tune` method
- Added early stopping functionality to the `tune` method to stop optimization when no improvement is seen
- Simplified pruning mechanism to use only MedianPruner for clearer functionality

### Changed
- Removed `patience` parameter in favor of `early_stopping` for more intuitive optimization control
- Default `early_stopping` set to 50 trials


## [0.2.6] - 2025-05-13

### Added / Improved
- Methodology updates:
  - Updated `threshold_anlaysis` method to use bootstrap as the default method for more robust result
- Utility enhancements:
  - Added `drop_fully_null_columns` utility function to handle DataFrame display issues in Databricks
  - Added comprehensive test coverage for the new utility function
- Documentation improvements:
  - Updated metrics and plots summary tables in demo notebooks for both classification and regression tasks
- Visualization enhancements:
  - Improved legend positioning in classification metric plots to avoid overlap with threshold lines


## [0.2.5] - 2025-05-10

### Added
- Enhanced classification metrics visualization:
  - Added confusion matrix to complement existing metrics vs threshold and ROC curve plots
  - Implemented new layout with metrics vs threshold at top, ROC curve and confusion matrix at bottom
  - Color highlighting in confusion matrix to highlight True Positives and True Negatives
- Enhanced SHAP visualization reliability:
  - Updated SHAP plot implementation to use modern beeswarm with legacy fallback
  - Fixed NumPy random generator warnings while maintaining compatibility

### Changed
- Improved metrics reporting organization and readability:
  - Regression metrics now grouped into three logical sections:
    - Error Metrics (RMSE, MAE, NRMSE variants, MAPE, SMAPE)
    - Goodness of Fit (R², Adjusted R²)
    - Improvement over Baseline (vs mean, vs median)
  - Classification metrics reorganized into three sections:
    - Evaluation Parameters (threshold, beta)
    - Core Performance Metrics (accuracy, AUC, precision, recall, F-scores)
    - Prediction Distribution (positive rate, base rate)
- Consolidated color scheme across all visualizations:
  - Standardized color constants based on matplotlib's default color cycle   
    (MPL_BLUE, MPL_RED, MPL_ORANGE, MPL_GREEN, MPL_GRAY)
  - Updated regression plots to use consistent colors
- Enhanced `plot_metric_event_over_time` function for better usability:
  - Simplified metrics parameter to accept either a single string or list of strings
  - rename `metrics` parameter to be y 
  - Standardized color handling using matplotlib's default color cycle

## [0.2.4] - 2025-05-06

### Added
- Added log_loss as optimization metric in tune function
  - Supports minimization of log_loss with proper variance penalty
  - Maintains consistent visualization where red indicates better performance

### Changed
- Enhanced `plot_metric_event_over_time` function for better event label handling
  - Fixed event label positioning to ensure visibility with dual axes
  - Improved label placement using axes fraction coordinates
  - Better support for different data scales between metrics
  - Ensured event labels always appear on top of plot elements
- Enhanced regression metrics in evaluation and tuning:
  - Renamed metrics for clarity:
    - `nrmse` → `nrmse_mean` (RMSE normalized by mean)
    - `rmse_improvement` → `rmse_improvement_over_mean`
  - Added new normalization variants for RMSE:
    - `nrmse_std`: RMSE normalized by standard deviation
    - `nrmse_iqr`: RMSE normalized by interquartile range
  - Added new error metrics:
    - `mae`: Mean Absolute Error
    - `median_ae`: Median Absolute Error
    - `smape`: Symmetric Mean Absolute Percentage Error
    - `rmse_improvement_over_median`: RMSE improvement over median baseline
  - Improved MAPE calculation:
    - Now handles zero values in target variable
    - Returns nan when all target values are zero
    - Clearly indicates when zeros are excluded from calculation
  - Updated tune function to support new metrics:
    - Added support for `nrmse_iqr`, `nrmse_std`, `mae`, `smape` as optimization targets
    - All regression metrics properly handled as minimization objectives

### Documentation
- Enhanced example notebook (3.utils_plot.ipynb)
  - Added explanations for each plotting utility function
  - Improved clarity and understanding of function purposes and use cases


## [0.2.3] - 2025-05-04

### Added
- Enhanced `threshold_analysis` function with bootstrap resampling capabilities
  - Added bootstrap method as alternative to cross-validation for more robust threshold estimation
  - Introduced separate `cv_splits` and `bootstrap_iterations` parameters for better method control
  - Returns 95% confidence intervals when using bootstrap method
  - Maintains backward compatibility with existing cross-validation implementation
  - Default values optimized for each method (5 splits for CV, 100 iterations for bootstrap)

### Changed
- Renamed `plot_medical_timeseries` to `plot_metric_event_over_time` for more generic naming
  - Updated function to be domain-agnostic while preserving original functionality
  - Renamed parameters from domain-specific terms (e.g., `treatment_dates` to `event_dates`)
  - Improved event label placement and readability across different scales
  - Added customization parameters for event styling:
    - `event_line_color`, `event_line_style`, `event_line_alpha` for vertical marker lines
    - `event_label_color`, `event_label_fontsize` for label text appearance
    - `event_label_y_pos`, `event_label_x_offset` for precise label positioning
    - `event_label_background` to toggle white background behind labels
  - Fixed label positioning to work correctly with different data scales and ranges
  - Ensured event labels remain visible regardless of metric scales
  - Implemented adaptive x-axis formatting based on data timespan:
    - Automatically selects appropriate date format (hourly/daily/monthly/yearly)
    - Adjusts tick frequency for optimal readability
    - Uses proportional padding instead of fixed padding for better display
    - Provide detailed documentation of format selection thresholds


## [0.2.2] - 2025-05-03

### Added
- Added `plot_distribution_over_time` function for visualizing continuous variable distributions over time
  - Implements boxplots with scatter overlay visualization to display both central tendency and dispersion
  - Supports flexible time frequencies (minutes to years) with automatic formatting
  - Allows coloring points by category with the `point_hue` parameter for pattern identification
  - Ensures chronological ordering for accurate trend visualization
  - Provides optional summary statistics (count, mean, median, std)
  - Uses helper functions `_get_date_format_for_freq` and `_get_label_for_freq` for code reusability
  - Maintains consistent parameter naming with matplotlib/seaborn
- Enhanced `plot_box_scatter` function with point coloring capabilities
  - Added `point_hue` parameter to color points by categorical or numerical variables
  - Maintained backward compatibility with single-color visualization
- Enhanced `transform_date_cols` function with improved usability
  - Added flexible input handling for single or multiple columns
  - Implemented format customization with standard Python strftime directives
  - Added smart case handling for month abbreviations

### Changed
- Improved code quality and documentation:
  - Standardized docstring format across all functions using numpy/scipy style, with consistent parameter descriptions and return value documentation
  - Added proper type annotations to all function parameters and return values, following Python typing conventions
- Standardized parameter naming in visualization functions
  - Renamed parameters from "dot" to "point" for consistency with matplotlib/seaborn conventions (e.g., `dot_size` → `point_size`, `dot_alpha` → `point_alpha`)
  - Renamed `boxplot_scatter_overlay` to `plot_box_scatter` to follow a consistent naming pattern with other functions like `plot_distribution_over_time`
  - Renamed `date_col` to `x` in `plot_medical_timeseries` for consistency with other time-series functions
  - Reordered parameters to group related options together, improving readability and usability
  - Improved API consistency across all plotting functions for better usability


### Documentation
- Added example notebooks demonstrating new functionality
  - `3.utils_plot.ipynb`: Examples of enhanced plotting capabilities including time-based distributions with single-color and hue-based visualizations
  - `3.utils_data.ipynb`: Demonstrations of data utility functions and their applications


## [0.2.1] - 2025-04-30

### Added
- Customizable metric selection in `tune` function
  - Added `tune_metric` parameter to specify which metric to optimize
  - Defaults to 'auc' for classification and 'rmse' for regression
  - Supports multiple metrics:
    - Classification: 'auc', 'f1', 'accuracy'
    - Regression: 'rmse', 'nrmse', 'mape'
  - Maintains all other metrics in results for reference
- Auto-detection of task type (classification/regression) in `tune` function
  - Automatically detects whether the algorithm is a classifier or regressor based on its capabilities
  - Maintains backward compatibility by allowing manual task specification
- Enhanced `boxplot_scatter_overlay` function
  - Added `return_summary` parameter to optionally return summary statistics
  - Returns DataFrame with count, mean, median, and standard deviation per category
  - Maintains backward compatibility with default return of figure and axis objects
- Added time granularity control to `plot_stacked_bar_over_time`
  - New `freq` parameter to specify time aggregation frequency
  - Supports multiple time frequencies:
    - 'm' for minutes
    - 'h' for hours
    - 'D' for days
    - 'ME'/'MS' for month end/start
    - 'YE'/'YS' for year end/start
  - Automatic date formatting based on selected frequency
  - Default frequency set to 'ME' (month end)
- Improved plotting utilities consistency
  - Added descriptive default titles to all plotting functions
  - Added customizable xlabel and ylabel parameters to all plotting functions
  - Removed hardcoded styling elements to align with matplotlib practices
  - Enhanced documentation for all plotting functions
- Added some data utility functions for common data manipulation tasks
  - `clean_dollar_cols`: Clean dollar amount columns by removing '$' and commas
  - `value_counts_with_pct`: Calculate value counts with percentages
  - `transform_date_cols`: Convert string dates to datetime format
  - Added tests for data utility functions

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

### Added
- Add MLflow model logging functionality:
  - Add `_log_model` method to ML_PIPELINE class
  - Add optional `log_model` parameter to `evaluate` method
  - Add optional `log_best_model` parameter to `tune` method
  - Return model logging info in evaluation results when logging enabled
  - Update basic and advanced usage examples to demonstrate MLflow integration
- Add utility functions:
  - New utils module with initial plotting utilities (`boxplot_scatter_overlay`)
  - Modular structure for utility functions with clean import paths

### Fixed
- Handle SHAP visualization errors in tests/test_pipeline.py
- Add testing for all feature types in test/test_preprocessor
- Leverage `drop` parameter and "if_binary" option in sklearn OneHotEncoder rather than rewrite the logic

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

## [0.1.2] - 2025-03-29

### Added
- Added pytest-cov for code coverage reporting
- Added comprehensive test assertions for pipeline evaluation results

### Changed
- Improved test structure with proper assertions instead of return statements
- Added warning filters in pytest.ini to suppress third-party deprecation warnings
- Improved code formatting with black (line length set to 88)

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