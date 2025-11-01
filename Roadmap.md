# MLArena Development Roadmap

This document outlines the planned features, improvements, and changes for future releases of the MLArena package. The roadmap represents our current priorities and direction, though specific timelines are not guaranteed. Features may be added, modified, or reprioritized based on community feedback and evolving needs.

We welcome contributions and suggestions related to these roadmap items. If you're interested in implementing any of these features or have ideas for improvements, please feel free to open an issue or submit a pull request on our GitHub repository.

## Planned Features and Improvements

- **Enhanced Cross-Validation Flexibility**:
  - Allow user to specify splitter object for cross-validation in `wrapper_feature_selection` and `tune` functions
  - Support sklearn CV splitter objects (e.g., `TimeSeriesSplit`, `GroupKFold`, `LeaveOneGroupOut`, etc.)
  - Maintain backward compatibility with existing `cv` parameter (defaults to `StratifiedKFold`/`KFold`)
  - When `cv` is provided as a splitter object, use it directly; otherwise fall back to current default behavior
  - Add examples demonstrating different CV strategies for specialized use cases (time series, grouped data, etc.)

- **Support for Fixed Parameters in Hyperparameter Tuning**:
  - Add `fixed_params` parameter to the `tune` method
  - Allow users to specify parameters that should remain constant during tuning
  - Combine fixed and tunable parameters when creating model instances
  - Maintain backward compatibility with existing usage patterns
  - Add examples demonstrating mixed fixed/tunable parameter scenarios
  - Enable common use cases like:
    - Setting algorithm-specific parameters that don't need tuning
    - Setting regularization parameters while tuning learning rates

