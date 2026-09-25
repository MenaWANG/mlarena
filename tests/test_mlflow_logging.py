"""MLflow wrapper contracts and isolated log/load round trips.

These tests exercise the installed MLflow version. Run them against each supported
MLflow version to check compatibility; signature-aware mocks alone are not a
substitute for that matrix.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec, sentinel

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models.model import ModelInfo
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlarena import MLPipeline


@pytest.fixture
def regression_data():
    X = pd.DataFrame({"x1": np.arange(1, 9, dtype=float), "x2": [0.0, 1.0] * 4})
    y = 2 * X["x1"] + 0.5 * X["x2"] + 1
    return X, y


@pytest.fixture
def mlflow_calls(monkeypatch):
    """Mock side effects, but retain the installed log_model API's signature."""
    calls = SimpleNamespace(
        log_model=create_autospec(mlflow.pyfunc.log_model, spec_set=True),
        log_metrics=Mock(),
        log_params=Mock(),
        end_run=Mock(),
        infer_signature=Mock(return_value=sentinel.signature),
    )
    calls.log_model.return_value = sentinel.model_info
    monkeypatch.setattr(mlflow.pyfunc, "log_model", calls.log_model)
    monkeypatch.setattr(mlflow, "log_metrics", calls.log_metrics)
    monkeypatch.setattr(mlflow, "log_params", calls.log_params)
    monkeypatch.setattr(mlflow, "end_run", calls.end_run)
    monkeypatch.setattr("mlarena.pipeline.infer_signature", calls.infer_signature)
    return calls


def test_log_model_defaults_and_return_value(mlflow_calls):
    pipeline = MLPipeline(model=LinearRegression())

    result = pipeline._log_model()

    assert result is sentinel.model_info
    mlflow_calls.log_model.assert_called_once_with(
        artifact_path="ml_pipeline",
        python_model=pipeline,
        artifacts={},
        signature=None,
        input_example=None,
    )
    mlflow_calls.log_metrics.assert_not_called()
    mlflow_calls.log_params.assert_not_called()
    mlflow_calls.infer_signature.assert_not_called()
    mlflow_calls.end_run.assert_called_once_with()


def test_log_model_forwards_metadata_and_preserves_inputs(mlflow_calls):
    pipeline = MLPipeline(model=LinearRegression())
    sample_input = pd.DataFrame(
        {"category": pd.Categorical(["a", "b"]), "value": [1.0, 2.0]}
    )
    original = sample_input.copy(deep=True)
    sample_output = np.array([2.0, 4.0])
    metrics = {"rmse": 0.0}
    params = {"fit_intercept": True}
    artifacts = {"notes": "notes.txt"}

    result = pipeline._log_model(
        metrics=metrics,
        params=params,
        additional_artifacts=artifacts,
        sample_input=sample_input,
        sample_output=sample_output,
    )

    assert result is sentinel.model_info
    mlflow_calls.log_metrics.assert_called_once_with(metrics)
    mlflow_calls.log_params.assert_called_once_with(params)
    mlflow_calls.log_model.assert_called_once()
    kwargs = mlflow_calls.log_model.call_args.kwargs
    assert kwargs["python_model"] is pipeline
    assert kwargs["artifacts"] == artifacts
    assert kwargs["artifacts"] is not artifacts
    assert artifacts == {"notes": "notes.txt"}
    assert kwargs["signature"] is sentinel.signature
    expected = original.assign(category=original["category"].astype("object"))
    pd.testing.assert_frame_equal(kwargs["input_example"], expected)
    pd.testing.assert_frame_equal(sample_input, original)
    assert kwargs["input_example"] is not sample_input
    mlflow_calls.infer_signature.assert_called_once()
    signature_args = mlflow_calls.infer_signature.call_args.args
    assert signature_args[0] is kwargs["input_example"]
    np.testing.assert_array_equal(signature_args[1], sample_output)
    mlflow_calls.end_run.assert_called_once_with()


@pytest.mark.parametrize("has_input,has_output", [(True, False), (False, True)])
def test_log_model_requires_both_samples_to_infer_signature(
    mlflow_calls, has_input, has_output
):
    pipeline = MLPipeline(model=LinearRegression())
    sample_input = pd.DataFrame({"x": [1.0]}) if has_input else None
    sample_output = np.array([2.0]) if has_output else None

    pipeline._log_model(sample_input=sample_input, sample_output=sample_output)

    mlflow_calls.infer_signature.assert_not_called()
    assert mlflow_calls.log_model.call_args.kwargs["signature"] is None


def test_log_model_propagates_logging_error_and_ends_run(mlflow_calls):
    pipeline = MLPipeline(model=LinearRegression())
    error = RuntimeError("artifact upload failed")
    mlflow_calls.log_model.side_effect = error

    with pytest.raises(RuntimeError, match="artifact upload failed") as exc:
        pipeline._log_model()

    assert exc.value is error
    mlflow_calls.log_model.assert_called_once()
    mlflow_calls.end_run.assert_called_once_with()


@pytest.mark.parametrize("log_model", [None, False, True])
def test_evaluate_logs_only_when_opted_in(monkeypatch, regression_data, log_model):
    X, y = regression_data
    pipeline = MLPipeline(model=LinearRegression())
    pipeline.fit(X, y)
    log = Mock(return_value=sentinel.model_info)
    monkeypatch.setattr(pipeline, "_log_model", log)
    options = {} if log_model is None else {"log_model": log_model}

    result = pipeline.evaluate(X, y, visualize=False, verbose=False, **options)

    if not log_model:
        log.assert_not_called()
        assert "model_info" not in result
        return

    assert result["model_info"] is sentinel.model_info
    log.assert_called_once()
    kwargs = log.call_args.kwargs
    assert kwargs["metrics"] == {
        key: value for key, value in result.items() if key != "model_info"
    }
    assert kwargs["params"] == pipeline.model.get_params()
    pd.testing.assert_frame_equal(kwargs["sample_input"], X.iloc[:1])
    np.testing.assert_allclose(kwargs["sample_output"], pipeline.predict(None, X)[:1])


@pytest.fixture
def isolated_tracking(tmp_path, monkeypatch):
    """Use a private SQLite store and local artifacts, never the user's server."""
    assert mlflow.active_run() is None, "A run leaked into the MLflow logging tests"
    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_registry_uri = mlflow.get_registry_uri()
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    monkeypatch.setenv("MLFLOW_SKIP_PIP_REQUIREMENTS_CHECK", "true")
    for variable in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME"):
        monkeypatch.delenv(variable, raising=False)
    # Dependency discovery is not under test. Skip its subprocess/PyPI work while
    # keeping real MLflow serialization, artifact storage, and model loading.
    monkeypatch.setattr(
        mlflow.models, "infer_pip_requirements", lambda *args, **kwargs: []
    )
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=uri, registry_uri=uri)
        experiment_id = client.create_experiment(
            "logging-tests", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        yield client, experiment_id
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking_uri)
        mlflow.set_registry_uri(previous_registry_uri)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_log_model_round_trip(isolated_tracking, regression_data, tmp_path, task):
    """Catch real API/serialization/storage changes that mocks cannot detect."""
    client, experiment_id = isolated_tracking
    X, y = regression_data
    if task == "classification":
        model = LogisticRegression()
        y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
    else:
        model = LinearRegression()
    pipeline = MLPipeline(model=model)
    pipeline.fit(X, y)
    expected = pipeline.predict(None, X)
    artifact = tmp_path / "notes.txt"
    artifact.write_text("logging round-trip fixture", encoding="utf-8")
    run = mlflow.start_run(experiment_id=experiment_id)
    run_id = run.info.run_id

    model_info = pipeline._log_model(
        metrics={"test_metric": 0.25},
        params={"task": task},
        additional_artifacts={"notes": str(artifact)},
        sample_input=X.iloc[:2],
        sample_output=expected[:2],
    )

    assert isinstance(model_info, ModelInfo)
    assert model_info.run_id == run_id
    assert model_info.model_uri
    assert model_info.signature is not None
    assert mlflow.active_run() is None
    stored_run = client.get_run(run_id)
    assert stored_run.info.status == "FINISHED"
    assert stored_run.data.metrics["test_metric"] == pytest.approx(0.25)
    assert stored_run.data.params["task"] == task

    # Use the returned URI, not a hardcoded MLflow 2 run-artifact path: MLflow 3
    # models are first-class entities with their own artifact locations.
    loaded = mlflow.pyfunc.load_model(model_info.model_uri)
    np.testing.assert_allclose(loaded.predict(X), expected)
    assert isinstance(loaded.unwrap_python_model(), MLPipeline)
    local_model = mlflow.artifacts.download_artifacts(
        artifact_uri=model_info.model_uri, dst_path=str(tmp_path / "download")
    )
    assert (Path(local_model) / "artifacts" / "notes.txt").read_text(
        encoding="utf-8"
    ) == "logging round-trip fixture"
