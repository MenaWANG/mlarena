"""Execute and validate the examples in docs/quickstart.md."""

import os
from pathlib import Path

import numpy as np
from markdown_it import MarkdownIt

QUICKSTART = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"


def check_quickstart(path: Path = QUICKSTART) -> None:
    """Run the classifier and regressor examples with their shared imports."""
    os.environ["MPLBACKEND"] = "Agg"
    blocks = [
        token
        for token in MarkdownIt().parse(path.read_text(encoding="utf-8"))
        if token.type == "fence" and token.info.strip() == "python"
    ]
    assert len(blocks) == 2, "Expected classification and regression Python examples"

    namespace = {"__name__": "__main__"}
    for block, task in zip(blocks, ("classification", "regression")):
        # Preserve Markdown line numbers in tracebacks for failed examples.
        source = "\n" * (block.map[0] + 1) + block.content
        exec(compile(source, str(path), "exec"), namespace)
        assert namespace["pipeline"].task == task, f"Expected a {task} example"

        prediction_name = "probabilities" if task == "classification" else "predictions"
        predictions = np.asarray(namespace[prediction_name])
        assert predictions.shape == (len(namespace["X_test"]),), task
        assert np.isfinite(predictions).all(), f"Non-finite {task} predictions"

        metrics = namespace["metrics"]
        if task == "classification":
            assert (
                (predictions >= 0) & (predictions <= 1)
            ).all(), "Invalid probabilities"
            assert 0.9 < metrics["auc"] <= 1, "Unexpected classifier AUC"
            np.testing.assert_array_equal(
                namespace["labels"], (predictions >= 0.5).astype(int)
            )
        else:
            assert np.isfinite(metrics["rmse"]) and metrics["rmse"] > 0, "Invalid RMSE"

    print(f"Passed {len(blocks)} Quickstart examples.")


if __name__ == "__main__":
    check_quickstart()
