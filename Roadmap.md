# MLArena Development Roadmap

This document outlines the planned features, improvements, and changes for future releases of the MLArena package. The roadmap represents our current priorities and direction, though specific timelines are not guaranteed. Features may be added, modified, or reprioritized based on community feedback and evolving needs.

We welcome contributions and suggestions related to these roadmap items. If you're interested in implementing any of these features or have ideas for improvements, please feel free to open an issue or submit a pull request on our GitHub repository.

## Planned Features and Improvements

- **Enhanced Cross-Validation Flexibility**:
  - Allow user to specify splitter object for cross-validation in 
    - [x] `wrapper_feature_selection` and 
    - [ ] `tune` functions
  - [x] Support sklearn CV splitter objects (e.g., `TimeSeriesSplit`, `GroupKFold`, `LeaveOneGroupOut`, etc.)
  - [x] Maintain backward compatibility with existing `cv` parameter (defaults to `StratifiedKFold`/`KFold`)
  - [x] When `cv` is provided as a splitter object, use it directly; otherwise fall back to current default behavior
  - [x] Add examples demonstrating different CV strategies for specialized use cases (time series, grouped data, etc.)

- **Support for Fixed Parameters in Hyperparameter Tuning**:
  - [ ] Add `fixed_params` parameter to the `tune` method
  - Allow users to specify parameters that should remain constant during tuning
  - [ ] Combine fixed and tunable parameters when creating model instances
  - [ ] Maintain backward compatibility with existing usage patterns
  - [ ] Add examples demonstrating mixed fixed/tunable parameter scenarios
  - Enable common use cases like:
    - [ ] Setting algorithm-specific parameters that don't need tuning
    - [ ] Setting regularization parameters while tuning learning rates

- **Test and support for Python 3.13**:
  - [x] Add tests for Python 3.13
  - [x] Add support for Python 3.13

- **Extend Influence Analysis to Classifiers**:

  - [x] Generalize the `calculate_cooks_d_like_influence` method to handle classification models
  - Support probability-based metrics 
    - [x] Support MSE-like metrics for probability-based metrics in 1st iteration
    - [?] Support other probability-based metrics (e.g., Jensen–Shannon divergence, L2 distance, log-loss change) for measuring prediction shift
  - Ensure compatibility with binary classifiers using `predict_proba` 
  - [x] Maintain consistent output format and aggregation with regression version
  - [x] Add examples demonstrating influence detection on classification datasets
  - [x] Add comprehensive test coverage for classification models

- **Column Order Consistency for Predictions**:
  - Some tree-based libraries (e.g. XGBoost) rely on column position rather than column names during prediction. If the column order in test data differs from training data, the model will silently use wrong feature values, leading to incorrect predictions without raising errors. To protect its users against such silent errors, MLArena can add a consistency layer to automatically align column orders, preventing these issues from underlying algorithms and enhancing robustness across all supported algorithms.
  - [x] Store training feature column order during `fit()` method
  - [x] Automatically reorder prediction data columns to match training order in `predict()` method
  - [x] Validate that all expected training columns are present in prediction data
  - [x] Raise informative error if required columns are missing
  - [x] Add tests to verify correct behavior with mismatched column orders

- **Agent-Friendliness Improvements**:
  - MLArena is increasingly used in AI-driven and agentic workflows where programmatic discoverability, type safety, and side-effect predictability matter. The following improvements make the package easier to use reliably in such contexts.
  - **Type Hints & PEP 561 Compliance**:
    - [ ] Add full type annotations to `MLPipeline.tune` method signature
    - [ ] Add full type annotations to `PreProcessor.__init__` and other under-typed public methods
    - [ ] Add a `py.typed` marker file to declare the package as typed (PEP 561)
  - **Headless / Non-Interactive Visualization**:
    - [ ] Audit all public methods that generate plots and ensure a consistent `visualize: bool = True` parameter is available
    - [ ] Ensure all plot-generating methods consistently return the `Figure` or `Axes` object (or a `dict` of them for multi-plot methods) regardless of the `visualize` flag — so callers can always capture artefacts programmatically without being forced to choose between rendering and capturing
    - [ ] Document the recommended `matplotlib.use("Agg")` pattern for headless/agent environments in the README and docstrings
  - **MLflow Side-Effect Transparency**:
    - [ ] Clearly document which methods require an active MLflow run and what happens when none is present
    - [x] Ensure `log_model=False` (default) truly produces no MLflow side effects
    - [ ] Change `MLPipeline.tune` default from `log_best_model=True` to `log_best_model=False` so tuning is safe-by-default outside active MLflow runs
    - [ ] Keep backward compatibility note in changelog/docs: users who want logging can pass `log_best_model=True`
  - **Docstring & Discoverability**:
    - [ ] Add a module-level docstring to `pipeline.py` listing all public classes and methods with one-line summaries, so agents can orient without reading the full file
    - [ ] Fix the stale `mlarena.exceptions` reference in `docs/api.rst` (module does not exist)
    - [ ] Ensure all `utils` submodule `__all__` lists are complete and accurate
- **Agent-Friendly Documentation**:
  - Structured, machine-readable documentation that helps AI agents and agent developers work with MLArena correctly. Two complementary files serve distinct audiences:
    - `llms.txt` is consumed **directly by agents** — it is the package's machine-readable entry point.
    - `AGENTS.md` is written **for developers building agentic systems** on top of MLArena — it documents the integration concerns a human needs to handle before handing control to an agent.
  - **`llms.txt`** (see [llmstxt.org](https://llmstxt.org) convention):
    - [x] Create a `/llms.txt` file at the repo root following the emerging `llms.txt` standard — a concise, markdown-formatted entry point for LLMs describing what the package does, its public API, and key usage patterns
    - [ ] Include links to the most relevant documentation pages (README, API reference, example notebooks) so agents can fetch deeper context on demand
    - [ ] Keep it maintained alongside releases so it reflects the current API
  - **`AGENTS.md`**:
    - [ ] Create an `AGENTS.md` file targeted at developers integrating MLArena into agentic pipelines, covering: recommended import patterns, known side effects (MLflow run requirements, plotting defaults), environment setup for headless use, and explicit do/don't examples




