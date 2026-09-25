# Core concepts

## Estimator and pipeline

`MLPipeline` wraps your estimator and optional preprocessor. You select the
algorithm and features; the wrapper provides training, prediction, evaluation,
tuning, and explanations. Supplying no preprocessor leaves features unchanged.

Task detection checks whether the estimator has `predict_proba`: if it does,
the pipeline treats it as a classifier; otherwise it treats it as a regressor.
The classification prediction path selects the second probability column.
Use this workflow for binary classification, not as a general multiclass API.

`fit` mutates the pipeline and does not return it. Call `fit` on a separate
line rather than chaining `MLPipeline(...).fit(...)` into an assignment.
Prediction expects a pandas DataFrame containing the original training columns;
the pipeline restores their training order.

## Preprocessing

By default, `PreProcessor` imputes numeric values with the median and scales
them. It imputes categorical values with the most frequent category and applies
one-hot encoding. Columns specified in `target_encode_cols` use target encoding
instead; fitting those columns requires `y`.

For standalone use, call `fit_transform(X_train, y_train)` followed by
`transform(X_test)`. The class currently does not implement a separate `fit`
method, so do not assume it supports every scikit-learn transformer workflow.

Split your data before fitting preprocessing or selecting features. Keep final
test data out of decisions about encoding, feature selection, hyperparameters,
and thresholds. For grouped or time-ordered data, choose splits that reflect
the intended prediction scenario.

## Hyperparameter tuning

`MLPipeline.tune` takes an estimator **class**, a preprocessor, and a dictionary
of parameter ranges. Tuples specify numeric ranges and lists specify discrete
choices. It returns a dictionary containing `best_params`, `best_pipeline`, and
the Optuna `study`, along with evaluation results.

The method makes an internal 80/20 split and performs cross-validation on the
training portion, fitting preprocessing within each fold. Its built-in split
does not provide a time-series or group-aware holdout. Keep a separate final
test set if you repeatedly use its results to guide modeling decisions.

Defaults include 500 trials, five folds, and a variance penalty of 0.1. Start
with a smaller `max_evals` when exploring. The default metric is AUC for
classification and RMSE for regression. Refer to the API for all options.

## Evaluation and thresholds

`evaluate` returns a metric dictionary. `visualize` and `verbose` default to
`True`; disable them in scripts where you only need the results.
The default classification threshold is 0.5. `beta` controls the relative weight
of recall in F-beta: larger values favor recall. Choose thresholds using
validation data and evaluate the chosen threshold once on held-out test data.

## Explanations

`explain_model` computes SHAP explanations. `explain_case` and
`explain_dependence` use the explanation state it creates, so run
`explain_model` first. SHAP computation cost and compatibility depend on the
estimator and the data size; start with a representative sample.

## MLflow

Training alone does not automatically log a model. To log during evaluation,
configure your MLflow tracking URI and experiment, then use
`evaluate(..., log_model=True)`. The returned dictionary includes `model_info`.
During tuning, use `log_best_model=True` to log the selected pipeline.

The logging helper ends the active MLflow run after attempting to log the model.
Plan run boundaries accordingly. A logged model can be loaded with
`mlflow.pyfunc.load_model(model_info.model_uri)`; the loaded model accepts
`loaded_model.predict(X)` without an explicit `context` argument.

## Utilities

Data, plotting, statistics, model diagnostics, and I/O helpers are available
independently of `MLPipeline`. Browse the [API reference](api.rst) for parameters
and returns, and the [notebooks](examples.md) for worked examples.
