# Quickstart

MLArena supports Python 3.10–3.13. Start in a virtual environment and install
the package from your terminal:

```bash
python -m pip install mlarena==0.5.2
```

In a notebook, use `%pip install mlarena==0.5.2`. LightGBM is optional;
the example below uses scikit-learn and requires no dataset downloads.

## Train a binary classifier

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlarena import MLPipeline, PreProcessor

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = MLPipeline(
    model=LogisticRegression(max_iter=1000, random_state=42),
    preprocessor=PreProcessor(),
)
pipeline.fit(X_train, y_train)

metrics = pipeline.evaluate(X_test, y_test, visualize=False, verbose=False)
print(f"Test AUC: {metrics['auc']:.3f}")

probabilities = pipeline.predict(context=None, model_input=X_test)
labels = (probabilities >= 0.5).astype(int)
print(labels[:5])
```

`fit` fits the preprocessor on the training data before training the estimator.
`evaluate` and `predict` reuse that fitted preprocessor. Pass the original feature
DataFrame to both methods; do not preprocess it a second time.

For binary classification, `predict` returns probabilities for the estimator's
second class (`model.classes_[1]`). Use targets encoded as 0 and 1 for this
workflow. The `context=None` argument is part of the MLflow PythonModel interface.

Set `visualize=True` in `evaluate` to display diagnostic plots. MLflow logging is
off by default, so this example does not require an MLflow server.

## Use a regression model

With the imports above, the same workflow supports regression:

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline = MLPipeline(model=Ridge(), preprocessor=PreProcessor())
pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test, visualize=False, verbose=False)
print(f"Test RMSE: {metrics['rmse']:.3f}")
predictions = pipeline.predict(context=None, model_input=X_test)
```

Here, `predict` returns predicted target values. See the [user guide](user-guide.md)
for preprocessing, tuning, and MLflow behavior, or explore the
[example notebooks](examples.md).
