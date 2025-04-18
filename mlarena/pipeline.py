"""
Pipeline module for MLArena package.
"""

import warnings

# Standard library imports
from typing import Any

# mlarena is published on PyPI on 2025-03-27, but mlflow packages index is updated till 2025-03-04 at the moment
warnings.filterwarnings(
    "ignore",
    message=".*The following packages were not found in the public PyPI package index.*mlarena.*",
)

import matplotlib.pyplot as plt
import mlflow

# Third-party imports
import numpy as np
import pandas as pd
import shap
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from mlflow.models.signature import infer_signature
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

# Local imports
from .preprocessor import PreProcessor


class ML_PIPELINE(mlflow.pyfunc.PythonModel):
    """
    Custom ML pipeline for classification and regression.

    - Works with scikit-learn compatible models
    - Plug in custom preprocessor to handle data preprocessing
    - Manages model training and predictions
    - Provide global and local model explanation
    - Offer comprehensive evaluation report including key metrics and plots
    - Iterative hyperparameter tuning with cross-validation and optional variance penalty
    - Parallel coordinates plot for disgnosis of the yperparameter tuning search space
    - Threshold analysis to find the optimize threshold based on business preference over precision and recall
    - Compatible with MLflow tracking
    - Supports MLflow deployment

    Attributes:
        model (BaseEstimator or None): A scikit-learn compatible model instance
        preprocessor (Any or None): Data preprocessing pipeline
        config (Any or None): Optional config for model settings
        task(str): Type of ML task ('classification' or 'regression')
        n_features (int): Number of features after preprocessing
        both_class (bool): Whether SHAP values include both classes
        shap_values (shap.Explanation): SHAP values for model explanation
        X_explain (pd.DataFrame): Processed features for SHAP explanation
    """

    def __init__(self, model: BaseEstimator = None, preprocessor=None, config=None):
        """
        Initialize the ML_PIPELINE with an optional model, preprocessor, and configuration.

        Parameters:
            model (BaseEstimator, optional): A scikit-learn compatible model, such as LightGBM
                or XGBoost, for training and predictions. Defaults to None.
            preprocessor (Any, optional): A transformer or pipeline used to preprocess the input
                data. Defaults to None.
            config (Any, optional): Additional configuration settings for the model, if needed.
                Defaults to None.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.task = (
            "classification" if hasattr(self.model, "predict_proba") else "regression"
        )
        self.shap_values = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set=None,
        early_stopping_rounds=None,
    ):
        """
        Train the model using the provided training data, after applying preprocessing.

        Parameters:
            X_train (pd.DataFrame): A DataFrame containing feature columns for training.
            y_train (pd.Series): A Series containing the target variable values.
        """
        if self.preprocessor is not None:
            X_train_preprocessed = self.preprocessor.fit_transform(
                X_train.copy(), y_train.copy()
            )
        else:
            X_train_preprocessed = X_train.copy()

        self.n_features = X_train_preprocessed.shape[1]

        # Prepare fit parameters
        fit_params = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_transformed = self.preprocessor.transform(X_val)
            fit_params["eval_set"] = [(X_val_transformed, y_val)]

            if early_stopping_rounds is not None:
                fit_params["early_stopping_rounds"] = early_stopping_rounds

        self.model.fit(X_train_preprocessed, y_train, **fit_params)

    def predict(self, context: Any, model_input: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the pre-trained model, applying preprocessing to the input data.

        Parameters:
            context (Any): Optional context information provided by MLflow during the
                prediction phase.
            model_input (pd.DataFrame): The DataFrame containing input features for predictions.

        Returns:
            Any: A NumPy array or DataFrame with the predicted probabilities or output values.
        """

        if self.preprocessor is not None:
            processed_model_input = self.preprocessor.transform(model_input.copy())
        else:
            processed_model_input = model_input.copy()

        if self.task == "classification":
            prediction = self.model.predict_proba(processed_model_input)[:, 1]
        elif self.task == "regression":
            prediction = self.model.predict(processed_model_input)
        return prediction

    def explain_model(self, X, plot_size=(8, 6)):
        """
        Generate SHAP values and plots for model interpretation.

        This method:
        1. Transforms the input data using the fitted preprocessor
        2. Creates a SHAP explainer appropriate for the model type
        3. Calculates SHAP values for feature importance
        4. Generates a summary plot of feature importance

        Parameters:
            X : pd.DataFrame
                Input features to generate explanations for. Should have the same
                columns as the training data.
            plot_size: tuple, default=(12,6)
                Tuple specifying the width and height of the SHAP summary plot in inches.

        Returns: None
            The method stores the following attributes in the class:
            - self.X_explain : pd.DataFrame
                Transformed data with original numeric values for interpretation
            - self.shap_values : shap.Explanation
                SHAP values for each prediction
            - self.both_class : bool
                Whether the model outputs probabilities for both classes
        """
        if self.preprocessor is not None:
            X_transformed = self.preprocessor.transform(X.copy())
        else:
            X_transformed = X.copy()

        self.X_explain = X_transformed.copy()
        # if trained preprocessor is available, get pre-transformed values for numeric features
        # else, users see the transformed values at the moment
        if self.preprocessor is not None:
            self.X_explain[self.preprocessor.num_features] = X[
                self.preprocessor.num_features
            ]

        self.X_explain.reset_index(drop=True)
        try:
            # Attempt to create an explainer that directly supports the model
            explainer = shap.Explainer(self.model)
        except:
            # Fallback for models or shap versions where direct support may be limited
            explainer = shap.Explainer(self.model.predict, X_transformed)
        self.shap_values = explainer(X_transformed)

        # get the shape of shap values and extract accordingly
        self.both_class = len(self.shap_values.values.shape) == 3
        try:
            if self.both_class:
                rng = np.random.RandomState(42)
                shap.summary_plot(
                    self.shap_values[:, :, 1], plot_size=plot_size, rng=rng
                )
            elif self.both_class == False:
                rng = np.random.RandomState(42)
                shap.summary_plot(self.shap_values, plot_size=plot_size, rng=rng)
        except Exception as e:
            print(
                "warnings: Could not display SHAP plot. This might be due to display configuration."
            )
            print("SHAP values are still calculated and available in self.shap_values")

    def explain_case(self, n):
        """
        Generate SHAP waterfall plot for one specific case.

        - Shows feature contributions
        - Starts from base value
        - Ends at final prediction
        - Shows original feature values for better interpretability

        Parameters:
            n (int): Case index (1-based)
                     e.g., n=1 explains the first case.

        Returns:
            None: Displays SHAP waterfall plot

        Notes:
            - Requires explain_model() first
            - Shows positive class for binary tasks
        """
        if self.shap_values is None:
            print(
                """
                  Please explain model first by running
                  `explain_model()` using a selected dataset
                  """
            )
        else:
            self.shap_values.data = self.X_explain
            if self.both_class:
                shap.plots.waterfall(self.shap_values[:, :, 1][n - 1])
            elif not self.both_class:
                shap.plots.waterfall(self.shap_values[n - 1])

    def _evaluate_regression_model(self, y_true, y_pred, verbose: bool = False):
        """
        Calculate multiple regression metrics for better interpretability.

        Parameters:
            y_true: True target values
            y_pred: Predicted target values
            verbose: If True, prints detailed evaluation metrics and analysis.
                    If False, returns metrics without printing (default=False)

        Returns:
            dict: Dictionary of different metrics
        """
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        n_samples = len(y_true)
        if n_samples > self.n_features + 1:
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - self.n_features - 1)
        else:
            adj_r2 = float("nan")

        # Scale-independent metrics: mean absolute percentage error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Normalized RMSE
        nrmse = rmse / np.mean(y_true) * 100

        # Compare to baseline (using mean)
        baseline_pred = np.full_like(y_true, np.mean(y_true))
        baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
        rmse_improvement = (baseline_rmse - rmse) / baseline_rmse * 100

        if verbose:
            print("Regression Model Evaluation:")
            print("-" * 40)
            print(f"RMSE: {rmse:.3f}")
            print(f"Normalized RMSE: {nrmse:.1f}% of target mean")
            print(f"MAPE: {mape:.1f}%")
            print(f"R² Score: {r2:.3f}")
            print(f"Adjusted R² Score: {adj_r2:.3f}")
            print(f"Improvement over baseline: {rmse_improvement:.1f}%")
            if r2 - adj_r2 > 0.1:
                print(
                    "\nWarning: Large difference between R² and Adjusted R² suggests possible overfitting"
                )
                print(f"- R² dropped by {(r2 - adj_r2):.3f} after adjustment")
                print("- Consider feature selection or regularization")

        return {
            "rmse": rmse,
            "nrmse": nrmse,
            "mape": mape,
            "r2": r2,
            "adj_r2": adj_r2,
            "rmse_improvement": rmse_improvement,
        }

    def _evaluate_classification_model(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        beta: float = 1.0,
        verbose: bool = False,
    ):
        """
        Calculate classification metrics at a given threshold.

        Parameters:
            y_true: True target values
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold (default=0.5)
            beta: Beta value for F-beta score (default=1.0)
            verbose: If True, prints detailed evaluation metrics

        Returns:
            dict: Dictionary containing:
                - evaluation parameters (threshold, beta)
                - classification metrics
        """
        # Get predictions at specified threshold
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            # Evaluation parameters
            "threshold": threshold,
            "beta": beta,
            # Core metrics
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "f_beta": fbeta_score(y_true, y_pred, beta=beta),
            "auc": roc_auc_score(y_true, y_pred_proba),
            # Additional context
            "positive_rate": np.mean(y_pred),  # % of positive predictions
        }

        if verbose:
            print("Classification Metrics Report")
            print("=" * 50)
            print("\nEvaluation Parameters:")
            print(f"Threshold: {metrics['threshold']:.3f}")
            print(f"Beta:      {metrics['beta']:.3f}")
            print("\nMetrics:")
            print(f"Accuracy:  {metrics['accuracy']:.3f}")
            print(f"F1:        {metrics['f1']:.3f}")
            if beta != 1:
                print(f"F_beta:    {metrics['f_beta']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall:    {metrics['recall']:.3f}")
            print(f"Pos Rate:  {metrics['positive_rate']:.3f}")
            print("\nAUC (threshold independent):")
            print(f"AUC:   {metrics['auc']:.3f}")

        return metrics

    @staticmethod
    def _plot_classification_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        beta: float = 1.0,
    ) -> None:
        """
        Create visualization for classification metrics including ROC curve and Metrics vs Threshold.

        Parameters:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            beta: Beta value for F-beta score
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # ROC Curve (left plot)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        ax1.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend(loc="lower right")
        ax1.grid(True)

        # Metrics vs Threshold (right plot)
        thresholds = np.linspace(0, 1, 200)
        precisions = []
        recalls = []
        f_scores = []

        for t in thresholds:
            y_pred = (y_pred_proba >= t).astype(int)
            if np.sum(y_pred) > 0:
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f_beta = fbeta_score(y_true, y_pred, beta=beta)

                precisions.append(prec)
                recalls.append(rec)
                f_scores.append(f_beta)
            else:
                break

        valid_thresholds = thresholds[: len(precisions)]

        # Current metrics at threshold
        current_precision = precision_score(y_true, y_pred_proba >= threshold)
        current_recall = recall_score(y_true, y_pred_proba >= threshold)
        current_f = fbeta_score(y_true, y_pred_proba >= threshold, beta=beta)

        # Plot metrics
        ax2.plot(
            valid_thresholds, np.array(precisions) * 100, "b-", label="Precision", lw=2
        )
        ax2.plot(valid_thresholds, np.array(recalls) * 100, "r-", label="Recall", lw=2)
        ax2.plot(
            valid_thresholds,
            np.array(f_scores) * 100,
            "g--",
            label=f"F{beta:.1f} Score",
            lw=2,
        )

        # Add threshold line with metrics
        ax2.axvline(
            x=threshold,
            color="gray",
            linestyle="--",
            label=f"Threshold = {threshold:.3f}\n"
            f"Precision = {current_precision:.3f}\n"
            f"Recall = {current_recall:.3f}\n"
            f"F{beta:.1f} = {current_f:.3f}",
        )

        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Score (%)")
        ax2.set_title("Metrics vs Threshold")
        ax2.legend(loc="center right")
        ax2.grid(True)
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_regression_metrics(X_test, y_test, y_pred):
        """
        Create side-by-side diagnostic plots for regression models:
        - Left: Residual analysis (residuals vs predicted)
        - Right: Prediction error plot (actual vs predicted with error bands)

        Parameters:
            X_test: test features
            y_test: true target values
            y_pred: model predictions
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left plot: Residual analysis
        residuals = y_test - y_pred
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color="r", linestyle="--")
        ax1.set_xlabel("Predicted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Predicted")

        # Add prediction intervals (±2σ for ~95% confidence)
        std_residuals = np.std(residuals)
        ax1.fill_between(
            [y_pred.min(), y_pred.max()],
            -2 * std_residuals,
            2 * std_residuals,
            alpha=0.2,
            color="gray",
            label="95% Prediction Interval",
        )
        ax1.legend()

        # Right plot: Prediction Error Plot
        ax2.scatter(y_test, y_pred, alpha=0.5)

        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax2.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        # Add error bands (±2σ)
        sorted_indices = np.argsort(y_test)
        sorted_y_test = (
            y_test.iloc[sorted_indices]
            if hasattr(y_test, "iloc")
            else y_test[sorted_indices]
        )
        sorted_y_pred = y_pred[sorted_indices]

        ax2.fill_between(
            sorted_y_test,
            sorted_y_pred - 2 * std_residuals,
            sorted_y_pred + 2 * std_residuals,
            alpha=0.2,
            color="gray",
            label="95% Prediction Interval",
        )

        ax2.set_xlabel("Actual Values")
        ax2.set_ylabel("Predicted Values")
        ax2.set_title("Actual vs Predicted")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5,
        beta: float = 1.0,
        verbose: bool = True,
        visualize: bool = True,
        log_model: bool = False,
    ) -> dict:
        """
        Evaluate model performance using appropriate metrics.

        Parameters:
            X_test: Test features DataFrame
            y_test: True target values
            threshold: Classification threshold (default=0.5)
            beta: Beta value for F-beta score (default=1.0)
            verbose: If True, prints detailed evaluation metrics
            visualize: If True, displays relevant visualization plots
            log_model: If True, logs model to MLflow (default=False)

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.task == "classification":
            y_pred_proba = self.predict(context=None, model_input=X_test)
            metrics = self._evaluate_classification_model(
                y_test, y_pred_proba, threshold, beta, verbose=verbose
            )
            if visualize:
                ML_PIPELINE._plot_classification_metrics(
                    y_test, y_pred_proba, threshold, beta
                )
        else:  # regression
            y_pred = self.predict(context=None, model_input=X_test)
            metrics = self._evaluate_regression_model(y_test, y_pred, verbose=verbose)
            if visualize:
                self._plot_regression_metrics(X_test, y_test, y_pred)

        results = metrics.copy()
        if log_model:
            sample_input = X_test.iloc[:1] if len(X_test) > 0 else None
            sample_output = (
                y_pred_proba[:1] if self.task == "classification" else y_pred[:1]
            )
            model_info = self._log_model(
                metrics=metrics,
                params=self.model.get_params(),
                sample_input=sample_input,
                sample_output=sample_output,
            )
            results["model_info"] = model_info

        return results

    @staticmethod
    def _plot_hyperparameter_search(trials, save_path=None):
        """
        Visualize hyperparameter search results using parallel coordinates plot.

        This visualization helps with:
        - Parameter relationships: Shows how different combinations affect performance
        - Search coverage: Reveals explored parameter space and potential gaps

        Parameters:
            trials: hyperopt trials object
            save_path: optional path to save the plot
        """
        # Extract parameter names and values
        results = []
        for trial in trials.trials:
            params = trial["misc"]["vals"]
            current_params = {
                key: values[0] if values else None for key, values in params.items()
            }
            current_params["score"] = -trial["result"]["loss"]
            results.append(current_params)

        df = pd.DataFrame(results)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Normalize all columns to [0,1] for better visualization
        df_norm = df.copy()
        for col in df.columns:
            df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # Plot parameters
        for i in range(len(df)):
            color = plt.cm.RdYlBu_r(df_norm["score"].iloc[i])
            ax.plot(range(len(df.columns)), df_norm.iloc[i], color=color, alpha=0.5)

        # Customize plot
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_ylabel("Normalized Hyperparameter Values")
        ax.set_title("Hyperparameter Search Results")

        # Add colorbar with proper axes
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r)
        sm.set_array(df["score"])
        plt.colorbar(sm, ax=ax, label=f"Mean CV Score")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    @staticmethod
    def tune(
        X,
        y,
        algorithm,
        preprocessor,
        space,
        max_evals=500,
        random_state=42,
        beta=1,
        early_stopping_rounds=100,
        verbose=0,
        cv=5,
        cv_variance_penalty=0.1,
        visualize=True,
        task="classification",
        log_best_model=True,
    ):
        """
        Static method to tune hyperparameters using AUC and find optimal threshold.

        Parameters:
            X: Features
            y: Target
            algorithm: ML algorithm class (e.g., lgb.LGBMClassifier)
            preprocessor (Any or None): Data preprocessing pipeline
            space: Hyperopt parameter search space
            max_evals: Maximum number of evaluations
            random_state: Random seed for reproducibility
            beta: Beta value for F-beta score optimization (default=1.0)
                beta > 1 gives more weight to recall
                beta < 1 gives more weight to precision
            early_stopping_rounds: Stop tuning if no improvement in specified number of trials
            cv: number of splits for cross-validation
            cv_variance_penalty: Weight for penalizing high variance in cross-validation scores (default=0.1)
            visualize: If True, displays relevant visualization plots
            task: classification or regression
            log_best_model: If True, logs the best model to MLflow (default=True)

        Returns:
            dict: Contains:
                - best_params: Best hyperparameters found
                - best_pipeline: Best pipeline model
                - other metrics and results
        """

        # Split train+test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        def objective(params):
            cv_scores = []
            if task == "classification":
                kf = StratifiedKFold(
                    n_splits=cv, shuffle=True, random_state=random_state
                )
            elif task == "regression":
                kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            else:
                raise ValueError("task must be 'classification' or 'regression'")

            for fold, (train_idx, val_idx) in enumerate(
                kf.split(X_train_full, y_train_full)
            ):
                X_fold_train = X_train_full.iloc[train_idx]
                X_fold_val = X_train_full.iloc[val_idx]

                # For numpy arrays or pandas Series
                if isinstance(y_train_full, pd.Series):
                    y_fold_train = y_train_full.iloc[train_idx]
                    y_fold_val = y_train_full.iloc[val_idx]
                else:
                    y_fold_train = y_train_full[train_idx]
                    y_fold_val = y_train_full[val_idx]

                model = ML_PIPELINE(
                    model=algorithm(**params, verbose=verbose),
                    preprocessor=preprocessor,
                )

                model.fit(X_fold_train, y_fold_train)
                results = model.evaluate(
                    X_fold_val, y_fold_val, verbose=False, visualize=False
                )
                if task == "classification":
                    cv_scores.append(results["auc"])  # maximize auc
                elif task == "regression":
                    cv_scores.append(
                        results["adj_r2"]
                    )  # maximize adj_r2, backlog: consider minimize RMSE

            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            # Store CV scores for the best parameters
            # print(f'mean_auc for this round: {mean_auc}')
            if len(trials.trials) > 0:
                historical_max = max(
                    trial["result"].get("mean_score", 0) for trial in trials.trials
                )
                # print(f"highest historical mean_auc: {historical_max}")
            else:
                # print("First trial, no historical mean_auc yet")
                historical_max = 0

            if len(trials.trials) == 0 or mean_score > historical_max:
                objective.best_cv_scores = {"mean": mean_score, "std": std_score}

            # penalize high variance solutions
            score = mean_score - cv_variance_penalty * std_score

            return {
                "loss": -score,
                "status": STATUS_OK,
                "mean_score": mean_score,
                "std_score": std_score,
            }

        # Initialize storage for best CV scores
        objective.best_cv_scores = {"mean": 0, "std": 0}

        # Run optimization
        trials = Trials()

        def early_stop_fn(trials, *args):
            if len(trials.trials) < early_stopping_rounds:
                return (False, "Not enough trials")
            # score for the last early_stopping_rounds, the larger the better
            scores = [
                -trial["result"]["loss"]
                for trial in trials.trials[-early_stopping_rounds:]
            ]

            if len(scores) >= early_stopping_rounds and max(scores) == scores[0]:
                return (True, f"No improvement in last {early_stopping_rounds} trials")
            else:
                return (False, "Continuing optimization")

        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(random_state),
            early_stop_fn=early_stop_fn,
        )

        # Get best parameters and create best pipeline
        best_params = space_eval(space, best)
        # Train final model with best parameters on full training set
        final_model = ML_PIPELINE(
            model=algorithm(**best_params, verbose=verbose), preprocessor=PreProcessor()
        )
        final_model.fit(X_train_full, y_train_full)
        if task == "classification":
            y_pred_proba = final_model.predict(context=None, model_input=X_train_full)
            optimal_threshold = ML_PIPELINE.threshold_analysis(
                y_train_full, y_pred_proba, beta=beta
            )["optimal_threshold"]

            # Print results on new data
            print(
                f"Best CV AUC: {objective.best_cv_scores['mean']:.3f}({objective.best_cv_scores['std']:.3f})"
            )
            print("\nPerformance on holdout validation set:")
            final_results = final_model.evaluate(
                X_test,
                y_test,
                optimal_threshold,
                beta=beta,
                verbose=True,
                visualize=True,
            )
        elif task == "regression":
            # Print results on new data
            y_pred = final_model.predict(
                context=None, model_input=X_train_full
            )  # for logging
            print(
                f"Best CV adjusted r square: {objective.best_cv_scores['mean']:.3f}({objective.best_cv_scores['std']:.3f})"
            )
            print("\nPerformance on holdout validation set:")
            final_results = final_model.evaluate(
                X_test, y_test, verbose=True, visualize=True
            )

        print("\nHyperparameter Tuning Results")
        print("=" * 50)
        print("\nBest parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        if log_best_model:
            print("Logging the best model to MLflow")
            sample_input = X_train_full.iloc[:1] if len(X_train_full) > 0 else None
            sample_output = y_pred_proba[:1] if task == "classification" else y_pred[:1]
            model_info = final_model._log_model(
                metrics=final_results,
                params=best_params,
                sample_input=sample_input,
                sample_output=sample_output,
            )

        if visualize:
            ML_PIPELINE._plot_hyperparameter_search(trials)

        if task == "classification":
            return {
                "model_info": model_info,
                "best_params": best_params,
                "best_pipeline": final_model,
                "trials": trials,
                "beta": beta,  # beta for f_beta
                "optimal_threshold": optimal_threshold,  # optimal threshold to maximize f_beta
                "test_auc": final_results["auc"],
                "test_Fbeta": final_results["f_beta"],
                "test_precision": final_results["precision"],
                "test_recall": final_results["recall"],
                "cv_auc_mean": objective.best_cv_scores["mean"],
                "cv_auc_std": objective.best_cv_scores["std"],
            }
        elif task == "regression":
            return {
                "model_info": model_info,
                "best_params": best_params,
                "best_pipeline": final_model,
                "trials": trials,
                "test_rmse": final_results["rmse"],
                "test_nrmse": final_results["nrmse"],
                "test_mape": final_results["mape"],
                "test_r2": final_results["r2"],
                "test_adj_r2": final_results["adj_r2"],
                "test_rmse_improvement": final_results["rmse_improvement"],
                "cv_rmse_mean": objective.best_cv_scores["mean"],
                "cv_rmse_std": objective.best_cv_scores["std"],
            }

    @staticmethod
    def threshold_analysis(
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        beta: float = 1.0,
        n_splits: int = 5,
    ):
        """
        Identify the optimal threshold that maximize thresholds using cross-validation.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            beta: F-beta score parameter
            n_splits: Number of CV splits
        """
        # Initialize CV
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        thresholds_cv = []

        for train_idx, val_idx in cv.split(y_true, y_true):
            y_true_val = y_true.iloc[val_idx]
            y_pred_proba_val = y_pred_proba[val_idx]
            precisions, recalls, pr_thresholds = precision_recall_curve(
                y_true_val, y_pred_proba_val
            )
            fpr, tpr, roc_thresholds = roc_curve(y_true_val, y_pred_proba_val)
            # Find optimal threshold for this fold
            f_beta_scores = (
                (1 + beta**2)
                * (precisions * recalls)
                / (beta**2 * precisions + recalls + 1e-10)
            )
            optimal_idx = np.argmax(f_beta_scores)
            optimal_threshold = (
                pr_thresholds[optimal_idx] if len(pr_thresholds) > optimal_idx else 0.5
            )

            thresholds_cv.append(optimal_threshold)

        return {
            "optimal_threshold": np.mean(thresholds_cv),
            "threshold_std": np.std(thresholds_cv),
            "threshold_cv_values": thresholds_cv,
        }

    def _log_model(
        self,
        metrics=None,
        params=None,
        additional_artifacts=None,
        sample_input=None,
        sample_output=None,
    ):
        """
        Log model, metrics, parameters and additional artifacts to MLflow.

        Args:
            metrics (dict, optional): Metrics to log
            params (dict, optional): Parameters to log
            additional_artifacts (dict, optional): Additional artifacts to log
                e.g., {"parallel_coords_plot": plot_path}
            sample_input (pd.DataFrame, optional): Sample input for signature inference
            sample_output (np.ndarray, optional): Sample output for signature inference
        """
        # Log metrics and parameters
        if metrics:
            mlflow.log_metrics(metrics)
        if params:
            mlflow.log_params(params)

        # Add any additional artifacts
        artifacts = {}
        if additional_artifacts:
            artifacts.update(additional_artifacts)

        if sample_input is not None:
            sample_input = sample_input.copy()
            # Convert any category columns to string type for signature inference
            for col in sample_input.select_dtypes(include=["category"]).columns:
                sample_input[col] = sample_input[col].astype("object")

        signature = None
        if sample_input is not None and sample_output is not None:
            signature = infer_signature(sample_input, sample_output)

        try:
            model_info = mlflow.pyfunc.log_model(
                artifact_path="ml_pipeline",
                python_model=self,
                artifacts=artifacts,
                signature=signature,
                input_example=sample_input,
            )
            return model_info
        finally:
            mlflow.end_run()
