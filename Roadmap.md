# MLArena Development Roadmap

This document outlines the planned features, improvements, and changes for future releases of the MLArena package. The roadmap represents our current priorities and direction, though specific timelines are not guaranteed. Features may be added, modified, or reprioritized based on community feedback and evolving needs.

We welcome contributions and suggestions related to these roadmap items. If you're interested in implementing any of these features or have ideas for improvements, please feel free to open an issue or submit a pull request on our GitHub repository.

## Planned Features and Improvements

- **Enhanced Cross-Validation Flexibility**:
  - Add `cv_method` parameter to `tune` and `wrapper_feature_selection` functions
  - Support either string identifiers or sklearn CV splitter objects or both
  - Enable support for specialized CV methods like `TimeSeriesSplit`, `GroupKFold`, etc.
  - Maintain backward compatibility with existing `cv` parameter
  - Add examples demonstrating different CV strategies for specialized use cases

- **Support for Fixed Parameters in Hyperparameter Tuning**:
  - Add `fixed_params` parameter to the `tune` method
  - Allow users to specify parameters that should remain constant during tuning
  - Combine fixed and tunable parameters when creating model instances
  - Maintain backward compatibility with existing usage patterns
  - Add examples demonstrating mixed fixed/tunable parameter scenarios
  - Enable common use cases like:
    - Setting algorithm-specific parameters that don't need tuning
    - Setting regularization parameters while tuning learning rates

