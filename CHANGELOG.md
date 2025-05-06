# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.4] -

### Added
- Added log_loss as optimization metric in tune function
  - Supports minimization of log_loss with proper variance penalty
  - Maintains consistent visualization where red indicates better performance
- Enhanced `plot_metric_event_over_time` function for better event label handling
  - Fixed event label positioning to ensure visibility with dual axes
  - Improved label placement using axes fraction coordinates
  - Better support for different data scales between metrics
  - Ensured event labels always appear on top of plot elements

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

## Added
- Add MLflow model logging functionality:
  - Add `_log_model` method to ML_PIPELINE class
  - Add optional `log_model` parameter to `evaluate`