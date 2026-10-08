# Upgrading MLArena

This guide helps you upgrade between major versions of MLArena that contain breaking changes.

## Table of Contents
- [Upgrading to v0.6.0: MLflow 3](#upgrading-to-v060-mlflow-3)
- [Upgrading to v0.3.0](#upgrading-to-v030)
- [Upgrading to v0.5.2](#upgrading-to-v052)
- [Future Versions](#future-versions)  
- [Need Help?](#need-help)

## Upgrading to v0.3.0

### ⚠️ Breaking Changes
- **Class renamed**: `ML_PIPELINE` → `MLPipeline`
- **why?**: Follow Python PEP 8 naming conventions (classes use CapWords)

### 🔧 Action recommended:
- Find: `ML_PIPELINE`  
- Replace: `MLPipeline`

### 📅 Timeline

- **v0.3.0**: `ML_PIPELINE` functional with deprecated warning
- **v0.4.0**: `ML_PIPELINE` support will be removed

### 📝 Example

```python
# Before (v0.2.x)
from mlarena import ML_PIPELINE
pipeline = ML_PIPELINE(model=your_model)
results = ML_PIPELINE.tune(X, y, algorithm, preprocessor, param_ranges)

# After (v0.3.0+)
from mlarena import MLPipeline
pipeline = MLPipeline(model=your_model)
results = MLPipeline.tune(X, y, algorithm, preprocessor, param_ranges)
```

## Upgrading to v0.5.2

### Breaking Changes
- **Removed deprecated class**: `ML_PIPELINE`
- **Replacement**: Use `MLPipeline`

### Action Required
- Find: `ML_PIPELINE`
- Replace: `MLPipeline`

```python
from mlarena import MLPipeline
```

## Upgrading to v0.6.0: MLflow 3

MLArena 0.6.0 requires **MLflow >=3.0.1,<4** and drops support for MLflow 2.x.
CI tests Python 3.10–3.13 with both
MLflow 3.0.1 and the latest stable 3.x release.

### Existing installations

Publishing this release does not change existing installations or the dependency
metadata of MLArena 0.5.2. If you need MLflow 2.x, pin `mlarena==0.5.2` and
retain your environment's tested MLflow version or lock file. Pinning MLArena
alone does not freeze its dependencies. Retaining an old environment does not
resolve security issues in that environment.

Upgrading MLArena can upgrade MLflow too. Unpinned installations and automated
environment rebuilds may pick up the new release. Dependency constraints that
require MLflow 2.x will conflict with the new release.

### Tracking servers and Databricks

Installing a new Python client does not upgrade a separately deployed tracking
server or its database. Before upgrading, check your server or managed runtime's
MLflow 3 support. Back up self-hosted tracking data before following the
[official MLflow upgrade guidance](https://mlflow.org/docs/latest/self-hosting/migration/).
Using `--no-deps` bypasses dependency installation; it does not make an older
Databricks runtime compatible with the new requirement.

### Model logging and loading

MLArena now calls `mlflow.pyfunc.log_model(name="ml_pipeline", ...)`.
Use the returned `model_info.model_uri` to load the model:

```python
loaded_model = mlflow.pyfunc.load_model(results["model_info"].model_uri)
predictions = loaded_model.predict(X)
```

MLflow validates the input schema before calling the pipeline. For pandas
`category` columns, pass an `object`-typed copy to the loaded model, matching
the conversion MLArena applies when logging the input example:

```python
X_serving = X.copy()
for column in X_serving.select_dtypes(include=["category"]).columns:
    X_serving[column] = X_serving[column].astype("object")
predictions = loaded_model.predict(X_serving)
```

This applies to the loaded MLflow wrapper; the original pipeline can still
accept categorical columns. Keep the original values and column names.

MLflow 3 logged models have their own model IDs and artifact locations. Do not
construct `runs:/<run_id>/ml_pipeline` for newly logged models. Existing models
keep their original URIs and saved dependency requirements; this release does
not rewrite or migrate them. Loading historical MLflow 2 models into an upgraded
environment needs separate validation with representative saved models.

The logging helper continues to end the active run after attempting to log the
model. Start a new run before another logging operation. The default remains
`log_model=False` for evaluation and `log_best_model=False` for tuning.

## Future Versions

This section will be updated with upgrade instructions for future breaking changes.

## Need Help?

- Check the [Changelog](https://github.com/MenaWANG/mlarena/blob/master/CHANGELOG.md) for detailed version notes
- Open an [issue](https://github.com/MenaWANG/mlarena/issues) if you need assistance
- Review the [API documentation](api.rst) for current method signatures
