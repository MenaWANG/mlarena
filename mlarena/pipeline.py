"""
Pipeline module for MLArena package.
"""

import logging
import warnings

# Standard library imports
from typing import Any

from optuna.exceptions import ExperimentalWarning

# mlarena is published on PyPI on 2025-03-27, but mlflow packages index is updated till 2025-03-04 at the moment
warnings.filterwarnings(
    "ignore",
    message=".*The following packages were not found in the public PyPI package index.*mlarena.*",
)
warnings.filterwarnings("ignore", category=ExperimentalWarning)

import matplotlib.pyplot as plt
import mlflow

# Third-party imports
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from mlflow.models.signature import infer_signature
from optuna.pruners import MedianPruner
from optuna.visualization import plot_parallel_coordinate
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
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


class MLPipeline(mlflow.pyfunc.PythonModel):
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

    Attributes
    ----------
    model : BaseEstimator or None
        A scikit-learn compatible model instance.
    preprocessor : Any or None
        Data preprocessing pipeline.
    config : Any or None
        Optional config for model settings.
    task : str
        Type of ML task ('classification' or 'regression').
    n_features : int
        Number of features after preprocessing.
    both_class : bool
        Whether SHAP values include both classes.
    shap_values : shap.Explanation
        SHAP values for model explanation.
    X_explain : pd.DataFrame
        Processed features for SHAP explanation.
    """

    def __init__(self, model: BaseEstimator = None, preprocessor=None, config=None):
        """
        Initialize the MLPipeline with an optional model, preprocessor, and configuration.

        Parameters
        ----------
        model : BaseEstimator, optional
            A scikit-learn compatible model, such as LightGBM or XGBoost,
            for training and predictions.
        preprocessor : Any, optional
            A transformer or pipeline used to preprocess the input data.
        config : Any, optional
            Additional configuration settings for the model, if needed.
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

        Parameters
        ----------
        X_train : pd.DataFrame
            A DataFrame containing feature columns for training.
        y_train : pd.Series
            A Series containing the target variable values.
        eval_set : tuple, optional
            Validation set (X_val, y_val) for early stopping.
        early_stopping_rounds : int, optional
            Number of rounds without improvement before early stopping.
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

        Parameters
        ----------
        context : Any
            Optional context information provided by MLflow during the
            prediction phase.
        model_input : pd.DataFrame
            The DataFrame containing input features for predictions.

        Returns
        -------
        np.ndarray
            Model predictions (probabilities for classification, values for regression).
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

    def explain_model(
        self,
        X,
        plot_size="auto",
        plot_type="auto",
        max_features=20,
        group_remaining_features=True,
    ):
        """
        Generate SHAP values and plots for model interpretation.

        This method:
        1. Transforms the input data using the fitted preprocessor
        2. Creates a SHAP explainer appropriate for the model type
        3. Calculates SHAP values for feature importance
        4. Generates a summary plot of feature importance

        Parameters
        ----------
        X : pd.DataFrame
            Input features to explain.
        plot_size : tuple
            Size of the summary plot (width, height).
        plot_type : str
            Options: "auto", "beeswarm", or "summary"
            - "auto": tries modern beeswarm plot, falls back to legacy summary if needed
            - "beeswarm": always uses modern shap.plots.beeswarm()
            - "summary": always uses legacy shap.summary_plot()
        max_features : int, default=20
            Maximum number of features to display in the summary plot.
        group_remaining_features : bool, default=True
            Whether to group remaining features (beyond max_features) together
            and show them as a single "remaining features" entry. If False,
            remaining features are simply excluded from the plot.
            Note: Only applies to beeswarm plots; ignored for legacy summary plots.

        Returns
        -------
        None
            The method stores the following attributes in the class:
            - self.X_explain : Transformed data with original numeric values
            - self.shap_values : SHAP values for each prediction
            - self.both_class : Whether the model outputs probabilities for both classes
        """
        if self.preprocessor is not None:
            X_transformed = self.preprocessor.transform(X.copy())
        else:
            X_transformed = X.copy()

        self.X_explain = X_transformed.copy()
        if self.preprocessor is not None:
            self.X_explain[self.preprocessor.num_features] = X[
                self.preprocessor.num_features
            ]

        try:
            explainer = shap.Explainer(self.model)
        except Exception:
            explainer = shap.Explainer(self.model.predict, X_transformed)

        self.shap_values = explainer(X_transformed)
        self.both_class = len(self.shap_values.values.shape) == 3

        try:
            # Use modern beeswarm or legacy summary plots depending on plot_type
            if plot_type == "beeswarm":
                if self.both_class:
                    shap.plots.beeswarm(
                        self.shap_values[:, :, 1],
                        max_display=max_features,
                        plot_size=plot_size,
                        group_remaining_features=group_remaining_features,
                    )
                else:
                    shap.plots.beeswarm(
                        self.shap_values,
                        max_display=max_features,
                        plot_size=plot_size,
                        group_remaining_features=group_remaining_features,
                    )

            elif plot_type == "summary":  # the legacy display
                if self.both_class:
                    shap.summary_plot(
                        self.shap_values[:, :, 1],
                        plot_size=plot_size,
                        max_display=max_features,
                        show=True,
                    )
                else:
                    shap.summary_plot(
                        self.shap_values,
                        plot_size=plot_size,
                        max_display=max_features,
                        show=True,
                    )
                plt.show()

            elif plot_type == "auto":
                try:
                    if self.both_class:
                        shap.plots.beeswarm(
                            self.shap_values[:, :, 1],
                            max_display=max_features,
                            plot_size=plot_size,
                            group_remaining_features=group_remaining_features,
                        )
                    else:
                        shap.plots.beeswarm(
                            self.shap_values,
                            max_display=max_features,
                            plot_size=plot_size,
                            group_remaining_features=group_remaining_features,
                        )
                except Exception:
                    if self.both_class:
                        shap.summary_plot(
                            self.shap_values[:, :, 1],
                            plot_size=plot_size,
                            max_display=max_features,
                            show=True,
                        )
                    else:
                        shap.summary_plot(
                            self.shap_values,
                            plot_size=plot_size,
                            max_display=max_features,
                            show=True,
                        )
                    plt.show()

        except Exception as e:
            print("⚠️ Could not display SHAP summary plot.")
            print("SHAP values are still available in self.shap_values.")

    def explain_case(self, n):
        """
        Generate SHAP waterfall plot for one specific case.

        - Shows feature contributions
        - Starts from base value
        - Ends at final prediction
        - Shows original feature values for better interpretability

        Parameters
        ----------
        n : int
            Case index (1-based), e.g., n=1 explains the first case.

        Returns
        -------
        None
            Displays SHAP waterfall plot.

        Notes
        -----
        - Requires explain_model() to be called first
        - Shows positive class for binary classification tasks
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

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.
        verbose : bool, default=False
            If True, prints detailed evaluation metrics and analysis.

        Returns
        -------
        dict
            Dictionary of metrics including:
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - median_ae: Median absolute error
            - nrmse_mean: RMSE normalized by mean (%)
            - nrmse_std: RMSE normalized by standard deviation (%)
            - nrmse_iqr: RMSE normalized by interquartile range (%)
            - mape: Mean absolute percentage error (%), excluding zero values
            - smape: Symmetric mean absolute percentage error (%)
            - r2: R-squared score
            - adj_r2: Adjusted R-squared
            - rmse_improvement_over_mean: Improvement over mean baseline (%)
            - rmse_improvement_over_median: Improvement over median baseline (%)
        """
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        median_ae = median_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        n_samples = len(y_true)
        if n_samples > self.n_features + 1:
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - self.n_features - 1)
        else:
            adj_r2 = float("nan")

        # Scale-independent metrics using different normalizations
        y_true_mean = np.mean(y_true)
        y_true_std = np.std(y_true)
        y_true_median = np.median(y_true)
        q75, q25 = np.percentile(y_true, [75, 25])
        iqr = q75 - q25

        nrmse_mean = rmse / y_true_mean * 100  # Normalized by mean
        nrmse_std = rmse / y_true_std * 100  # Normalized by standard deviation
        nrmse_iqr = rmse / iqr * 100  # Normalized by interquartile range

        # MAPE (excluding observations where y_true is zero)
        non_zero = y_true != 0
        if np.any(non_zero):
            mape = (
                np.mean(
                    np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
                )
                * 100
            )
        else:
            mape = float("nan")

        # SMAPE (handles zeros gracefully)
        smape = (
            np.mean(
                2
                * np.abs(y_pred - y_true)
                / (np.abs(y_true) + np.abs(y_pred) + np.finfo(float).eps)
            )
            * 100
        )

        # Compare to different baselines
        mean_baseline_pred = np.full_like(y_true, y_true_mean)
        median_baseline_pred = np.full_like(y_true, y_true_median)

        rmse_mean_baseline = np.sqrt(mean_squared_error(y_true, mean_baseline_pred))
        rmse_median_baseline = np.sqrt(mean_squared_error(y_true, median_baseline_pred))

        rmse_improvement_over_mean = (
            (rmse_mean_baseline - rmse) / rmse_mean_baseline * 100
        )
        rmse_improvement_over_median = (
            (rmse_median_baseline - rmse) / rmse_median_baseline * 100
        )

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "median_ae": median_ae,
            "nrmse_mean": nrmse_mean,
            "nrmse_std": nrmse_std,
            "nrmse_iqr": nrmse_iqr,
            "mape": mape,
            "smape": smape,
            "r2": r2,
            "adj_r2": adj_r2,
            "rmse_improvement_over_mean": rmse_improvement_over_mean,
            "rmse_improvement_over_median": rmse_improvement_over_median,
        }

        if verbose:
            print("\n=== Regression Model Evaluation ===")

            print("\n1. Error Metrics")
            print("-" * 40)
            print(f"• RMSE:         {rmse:.3f}      (Root Mean Squared Error)")
            print(f"• MAE:          {mae:.3f}      (Mean Absolute Error)")
            print(f"• Median AE:    {median_ae:.3f}      (Median Absolute Error)")
            print(f"• NRMSE Mean:   {nrmse_mean:.1f}%      (RMSE/mean)")
            print(f"• NRMSE Std:    {nrmse_std:.1f}%      (RMSE/std)")
            print(f"• NRMSE IQR:    {nrmse_iqr:.1f}%      (RMSE/IQR)")
            print(
                f"• MAPE:         {mape:.1f}%      (Mean Abs % Error, excl. zeros)"
                if not np.isnan(mape)
                else "• MAPE:         N/A      (not available - zeros in true values)"
            )
            print(f"• SMAPE:        {smape:.1f}%      (Symmetric Mean Abs % Error)")

            print("\n2. Goodness of Fit")
            print("-" * 40)
            print(f"• R²:           {r2:.3f}      (Coefficient of Determination)")
            print(f"• Adj. R²:      {adj_r2:.3f}      (Adjusted for # of features)")

            print("\n3. Improvement over Baseline")
            print("-" * 40)
            print(
                f"• vs Mean:      {rmse_improvement_over_mean:.1f}%      (RMSE improvement)"
            )
            print(
                f"• vs Median:    {rmse_improvement_over_median:.1f}%      (RMSE improvement)"
            )

            if r2 - adj_r2 > 0.1:
                print("\n⚠️ Model Complexity Warning:")
                print("-" * 40)
                print(f"• R² dropped by {(r2 - adj_r2):.3f} after adjustment")
                print("• Consider feature selection or regularization")

        return metrics

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

        Parameters
        ----------
        y_true : pd.Series
            True target values.
        y_pred_proba : np.ndarray
            Predicted probabilities.
        threshold : float, default=0.5
            Classification threshold.
        beta : float, default=1.0
            Beta value for F-beta score.
        verbose : bool, default=False
            If True, prints detailed evaluation metrics.

        Returns
        -------
        dict
            Dictionary containing:
            - threshold: Classification threshold used
            - beta: Beta value used for F-beta
            - accuracy: Classification accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
            - f_beta: F-beta score
            - auc: Area under ROC curve
            - log_loss: Logarithmic loss
            - positive_rate: Percentage of positive predictions
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
            "log_loss": log_loss(y_true, y_pred_proba),
            # Additional context
            "positive_rate": np.mean(y_pred),  # % of positive predictions
            "base_rate": np.mean(y_true),
        }

        if verbose:
            print("\n=== Classification Model Evaluation ===")

            print("\n1. Evaluation Parameters")
            print("-" * 40)
            print(
                f"• Threshold:   {metrics['threshold']:.3f}    (Classification cutoff)"
            )
            print(f"• Beta:        {metrics['beta']:.3f}    (F-beta weight parameter)")

            print("\n2. Core Performance Metrics")
            print("-" * 40)
            print(
                f"• Accuracy:    {metrics['accuracy']:.3f}    (Overall correct predictions)"
            )
            print(f"• AUC:         {metrics['auc']:.3f}    (Ranking quality)")
            print(
                f"• Log Loss:    {metrics['log_loss']:.3f}    (Confidence-weighted error)"
            )
            print(
                f"• Precision:   {metrics['precision']:.3f}    (True positives / Predicted positives)"
            )
            print(
                f"• Recall:      {metrics['recall']:.3f}    (True positives / Actual positives)"
            )
            print(
                f"• F1 Score:    {metrics['f1']:.3f}    (Harmonic mean of Precision & Recall)"
            )
            if beta != 1:
                print(
                    f"• F{beta:.1f} Score:  {metrics['f_beta']:.3f}    (Weighted harmonic mean)"
                )

            print("\n3. Prediction Distribution")
            print("-" * 40)
            print(
                f"• Pos Rate:    {metrics['positive_rate']:.3f}    (Fraction of positive predictions)"
            )
            print(
                f"• Base Rate:   {metrics['base_rate']:.3f}    (Actual positive class rate)"
            )

        return metrics

    @staticmethod
    def _plot_classification_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        beta: float = 1.0,
    ) -> None:
        """
        Create visualization for classification metrics including:
        - Metrics vs Threshold (top)
        - ROC curve (bottom left)
        - Confusion Matrix (bottom right)

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred_proba : np.ndarray
            Predicted probabilities.
        threshold : float, default=0.5
            Classification threshold.
        beta : float, default=1.0
            Beta value for F-beta score.

        Returns
        -------
        None
            Displays plots for metrics vs threshold, ROC curve, and confusion matrix.
        """
        # Get matplotlib default colors
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Define color constants for better readability
        MPL_BLUE = colors[0]  # Precision
        MPL_ORANGE = colors[1]  # ROC curve
        MPL_GREEN = colors[2]  # F-beta score
        MPL_RED = colors[3]  # Recall
        MPL_GRAY = "#666666"  # Reference lines

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        # Top plot: Metrics vs Threshold
        ax1 = fig.add_subplot(gs[0, :])
        # Bottom left: ROC curve
        ax2 = fig.add_subplot(gs[1, 0])
        # Bottom right: Confusion matrix
        ax3 = fig.add_subplot(gs[1, 1])

        # Metrics vs Threshold (top plot)
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
        y_pred = (y_pred_proba >= threshold).astype(int)
        current_precision = precision_score(y_true, y_pred)
        current_recall = recall_score(y_true, y_pred)
        current_f = fbeta_score(y_true, y_pred, beta=beta)

        # Plot metrics
        ax1.plot(
            valid_thresholds,
            np.array(precisions) * 100,
            color=MPL_BLUE,
            label="Precision",
            lw=2,
        )
        ax1.plot(
            valid_thresholds,
            np.array(recalls) * 100,
            color=MPL_RED,
            label="Recall",
            lw=2,
        )
        ax1.plot(
            valid_thresholds,
            np.array(f_scores) * 100,
            color=MPL_GREEN,
            linestyle="--",
            label=f"F{beta:.1f} Score",
            lw=2,
        )

        # Add threshold line with metrics
        ax1.axvline(
            x=threshold,
            color=MPL_GRAY,
            linestyle="--",
            label=f"Threshold = {threshold:.3f}\n"
            f"Precision = {current_precision:.3f}\n"
            f"Recall = {current_recall:.3f}\n"
            f"F{beta:.1f} = {current_f:.3f}",
        )

        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Metrics (%)")
        ax1.set_title("Metrics vs Threshold")
        buffer = max(0.04 - 0.075 * threshold, 0.011)
        anchor_x = threshold + buffer  # handle legend position with small threshold
        ax1.legend(loc="lower left", bbox_to_anchor=(anchor_x, 0.05))
        ax1.grid(True)
        ax1.set_ylim(0, 100)

        # ROC Curve (bottom left)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        ax2.plot(
            fpr, tpr, color=MPL_ORANGE, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        ax2.plot([0, 1], [0, 1], color=MPL_GRAY, lw=2, linestyle="--", label="Random")

        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend(loc="lower right")
        ax2.grid(True)

        # Confusion Matrix (bottom right)
        cm = confusion_matrix(y_true, y_pred)

        # Create base heatmap with light colors
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            ax=ax3,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            cmap="Blues",
            cbar=False,
            alpha=0.4,
        )

        # Overlay rectangles with specific colors and opacities
        # True Negatives (0,0) - Blue with 60% opacity
        ax3.add_patch(
            plt.Rectangle((0, 0), 1, 1, fill=True, color=MPL_BLUE, alpha=0.6, zorder=2)
        )
        # True Positives (1,1) - Red with 60% opacity
        ax3.add_patch(
            plt.Rectangle((1, 1), 1, 1, fill=True, color=MPL_RED, alpha=0.6, zorder=2)
        )

        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")
        ax3.set_title("Confusion Matrix")

        # Adjust layout with specific padding
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_regression_metrics(X_test, y_test, y_pred):
        """
        Create side-by-side diagnostic plots for regression models.

        Creates two plots:
        - Left: Residual analysis (residuals vs predicted)
        - Right: Prediction error plot (actual vs predicted with error bands)

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series or np.ndarray
            True target values.
        y_pred : np.ndarray
            Model predictions.

        Returns
        -------
        None
            Displays diagnostic plots.
        """
        # Get matplotlib default colors
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Define color constants for better readability
        MPL_BLUE = colors[0]  # Main color
        MPL_RED = colors[3]  # Reference lines
        MPL_GRAY = "#666666"  # Error bands

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left plot: Residual analysis
        residuals = y_test - y_pred
        ax1.scatter(y_pred, residuals, alpha=0.5, color=MPL_BLUE)
        ax1.axhline(y=0, color=MPL_RED, linestyle="--")
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
            color=MPL_GRAY,
            label="95% Prediction Interval",
        )
        ax1.legend()

        # Right plot: Prediction Error Plot
        ax2.scatter(y_test, y_pred, alpha=0.5, color=MPL_BLUE)

        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax2.plot(
            [min_val, max_val],
            [min_val, max_val],
            color=MPL_RED,
            linestyle="--",
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
            color=MPL_GRAY,
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

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features DataFrame.
        y_test : pd.Series
            True target values.
        threshold : float, default=0.5
            Classification threshold.
        beta : float, default=1.0
            Beta value for F-beta score.
        verbose : bool, default=True
            If True, prints detailed evaluation metrics.
        visualize : bool, default=True
            If True, displays relevant visualization plots.
        log_model : bool, default=False
            If True, logs model to MLflow.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics. For classification, includes
            threshold, precision, recall, F-scores, and AUC. For regression, includes
            RMSE, R², and other regression metrics.
        """
        if self.task == "classification":
            y_pred_proba = self.predict(context=None, model_input=X_test)
            metrics = self._evaluate_classification_model(
                y_test, y_pred_proba, threshold=threshold, beta=beta, verbose=verbose
            )
            if visualize:
                MLPipeline._plot_classification_metrics(
                    y_test, y_pred_proba, threshold=threshold, beta=beta
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
    def tune(
        X,
        y,
        algorithm,
        preprocessor,
        param_ranges,
        max_evals=500,
        random_state=42,
        beta=1,
        early_stopping=50,
        n_startup_trials=5,
        n_warmup_steps=0,
        verbose=0,
        cv=5,
        cv_variance_penalty=0.1,
        visualize=True,
        task=None,
        tune_metric=None,
        log_best_model=True,
        disable_optuna_logging=True,
        configure_plotly=True,
        show_progress_bar=True,
    ):
        """
        Static method to tune hyperparameters using Optuna.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target.
        algorithm : class
            ML algorithm class (e.g., lgb.LGBMClassifier).
        preprocessor : Any or None
            Data preprocessing pipeline.
        param_ranges : dict
            Dictionary of parameter ranges for Optuna.
            e.g., {'n_estimators': (50, 500), 'max_depth': (3, 10)}
            Specify as tuple (min, max) for int/float or list for categorical.
        max_evals : int, default=500
            Maximum number of evaluations.
        random_state : int, default=42
            Random seed for reproducibility.
        beta : float, default=1.0
            Beta value for F-beta score optimization.
            beta > 1 gives more weight to recall.
            beta < 1 gives more weight to precision.
        early_stopping : int, default=50
            Number of trials without improvement before stopping the optimization process.
            If None, will run for max_evals trials.
        n_startup_trials : int, default=5
            Number of trials to run before pruning starts.
        n_warmup_steps : int, default=0
            Number of steps per trial to run before pruning.
        verbose : int, default=0
            Verbosity level.
        cv : int, default=5
            Number of splits for cross-validation.
        cv_variance_penalty : float, default=0.1
            Weight for penalizing high variance in cross-validation scores.
        visualize : bool, default=True
            If True, displays relevant visualization plots.
        task : str, optional
            Task type ('classification' or 'regression').
            If None, will be automatically detected.
        tune_metric : str, optional
            Metric to optimize during hyperparameter tuning.
            If None, defaults to 'auc' for classification and 'rmse' for regression.
            Classification metrics: 'auc', 'f1', 'accuracy', 'log_loss'
            Regression metrics: 'rmse', 'mae', 'median_ae', 'smape', 'nrmse_mean', 'nrmse_iqr', 'nrmse_std'
        log_best_model : bool, default=True
            If True, logs the best model to MLflow.
        disable_optuna_logging : bool, default=True
            If True, suppresses Optuna's verbose logging.
        configure_plotly : bool, default=True
            If True, automatically configures Plotly renderer for optimal display
            across different environments including GitHub.
        show_progress_bar : bool, default=True
            If True, displays a progress bar during optimization.

        Returns
        -------
        dict
            Dictionary containing:
            - best_params: Best hyperparameters found
            - best_pipeline: Best pipeline model
            - study: Optuna study object
            - model_info: MLflow model info (if logged)
            - Various test and CV metrics based on task type
        """
        # Configure optuna logging to suppress outputs
        if disable_optuna_logging:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Split train+test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        # Auto-detect task type if not specified
        if task is None:
            # Create a temporary instance to check the type
            temp_model = algorithm()
            if hasattr(temp_model, "predict_proba"):
                task = "classification"
            else:
                task = "regression"
            del temp_model

        # Set default metric if not specified
        if tune_metric is None:
            tune_metric = "auc" if task == "classification" else "rmse"

        def objective(trial):
            # Create parameters dict from param_ranges
            params = {}
            for param_name, param_range in param_ranges.items():
                # Handle different parameter types based on range values
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    start, end = param_range
                    if isinstance(start, int) and isinstance(end, int):
                        params[param_name] = trial.suggest_int(param_name, start, end)
                    elif isinstance(start, float) or isinstance(end, float):
                        params[param_name] = trial.suggest_float(param_name, start, end)
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_range
                    )

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

                model = MLPipeline(
                    model=algorithm(**params, verbose=verbose),
                    preprocessor=preprocessor,
                )

                model.fit(X_fold_train, y_fold_train)
                results = model.evaluate(
                    X_fold_val, y_fold_val, verbose=False, visualize=False
                )

                if task == "classification":
                    if tune_metric not in ["auc", "f1", "accuracy", "log_loss"]:
                        raise ValueError(
                            f"Unsupported metric for classification: {tune_metric}"
                        )
                    cv_scores.append(results[tune_metric])  # maximize metric
                elif task == "regression":
                    if tune_metric not in [
                        "rmse",
                        "mae",
                        "median_ae",
                        "smape",
                        "nrmse_mean",
                        "nrmse_iqr",
                        "nrmse_std",
                    ]:
                        raise ValueError(
                            f"Unsupported metric for regression: {tune_metric}"
                        )
                    cv_scores.append(results[tune_metric])  # minimize metric

            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            # Store components separately for later analysis
            if task == "classification":
                trial.set_user_attr(f"mean_{tune_metric}", mean_score)
                trial.set_user_attr(f"std_{tune_metric}", std_score)
                if tune_metric == "log_loss":
                    # For log_loss (minimizing), add penalty like regression metrics
                    score = mean_score + cv_variance_penalty * std_score
                else:
                    # For other classification metrics (maximizing), subtract penalty
                    score = mean_score - cv_variance_penalty * std_score
                trial.set_user_attr(f"penalized_{tune_metric}", score)
                # For log_loss, store negative for visualization like regression metrics
                if tune_metric == "log_loss":
                    trial.set_user_attr(f"negative_penalized_{tune_metric}", -score)
            else:  # regression
                trial.set_user_attr(f"mean_{tune_metric}", mean_score)
                trial.set_user_attr(f"std_{tune_metric}", std_score)
                # Apply penalty (minimizing)
                score = mean_score + cv_variance_penalty * std_score
                trial.set_user_attr(f"penalized_{tune_metric}", score)
                # Store negative for visualization
                trial.set_user_attr(f"negative_penalized_{tune_metric}", -score)

            # Return score based on task (maximize for classification, minimize for regression)
            return score

        # Simplify to just MedianPruner
        pruner = MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps
        )

        # Create and run study with appropriate direction: minimize for log_loss and regression tasks
        direction = (
            "minimize"
            if tune_metric == "log_loss"
            else "maximize" if task == "classification" else "minimize"
        )
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=random_state),
            pruner=pruner,  # Using MedianPruner directly
        )

        def early_stopping_callback(study, trial):
            if early_stopping is None:
                return

            current_best_trial = study.best_trial.number

            # If we've done enough trials after the best one
            if (trial.number - current_best_trial) >= early_stopping:
                study.stop()

        study.optimize(
            objective,
            n_trials=max_evals,
            callbacks=[early_stopping_callback],
            show_progress_bar=show_progress_bar,
        )

        # Get best parameters
        best_params = study.best_params

        # Train final model with best parameters on full training set
        final_model = MLPipeline(
            model=algorithm(**best_params, verbose=verbose), preprocessor=PreProcessor()
        )
        final_model.fit(X_train_full, y_train_full)

        if task == "classification":
            y_pred_proba = final_model.predict(context=None, model_input=X_train_full)
            optimal_threshold = MLPipeline.threshold_analysis(
                y_train_full, y_pred_proba, beta=beta
            )["optimal_threshold"]

            # Print results on new data
            best_trial = study.best_trial
            print(
                f"Best CV {tune_metric.upper()}: {best_trial.user_attrs[f'mean_{tune_metric}']:.3f}({best_trial.user_attrs[f'std_{tune_metric}']:.3f})"
            )
            print("\nPerformance on holdout validation set:")
            final_results = final_model.evaluate(
                X_test,
                y_test,
                threshold=optimal_threshold,
                beta=beta,
                verbose=True,
                visualize=True,
            )
        elif task == "regression":
            # Print results on new data
            y_pred = final_model.predict(
                context=None, model_input=X_train_full
            )  # for logging
            best_trial = study.best_trial
            print(
                f"Best CV {tune_metric.upper()}: {best_trial.user_attrs[f'mean_{tune_metric}']:.3f}({best_trial.user_attrs[f'std_{tune_metric}']:.3f})"
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
            try:
                print("\nHyperparameter Parallel Coordinate Plot:")
                # Use appropriate target based on task and metric
                if task == "classification":
                    if tune_metric == "log_loss":
                        target_name = f"Negative Penalized {tune_metric.upper()}"
                        target_attr = f"negative_penalized_{tune_metric}"
                    else:
                        target_name = f"Penalized {tune_metric.upper()}"
                        target_attr = f"penalized_{tune_metric}"
                else:  # regression
                    target_name = f"Negative Penalized {tune_metric.upper()}"
                    target_attr = f"negative_penalized_{tune_metric}"

                fig_parallel = plot_parallel_coordinate(
                    study,
                    target=lambda t: t.user_attrs[target_attr],
                    target_name=target_name,
                )

                # Set color scale to ensure red is for higher values
                for trace in fig_parallel.data:
                    if trace.type == "parcoords":
                        trace.line.colorscale = "RdYlBu"
                        trace.line.reversescale = True  # red represent higher values
                        trace.line.colorbar = dict(
                            title=target_name, title_side="right"
                        )
                fig_parallel.update_layout(width=1200)  # match width=15
                # Try to display the plot
                try:
                    fig_parallel.show()
                except Exception:
                    # Just print a simple message about the plotly display
                    print(
                        "Note: Plotly figures may not display in all environments. The figure will display correctly when running in an IDE like VS Code, JupyterLab, or Jupyter Notebook."
                    )

            except Exception as e:
                print("Error with parallel coordinate plot:", e)

        # Prepare return values, similar to current function
        results = {
            "best_params": best_params,
            "best_pipeline": final_model,
            "study": study,
        }

        if log_best_model:
            results["model_info"] = model_info

        if task == "classification":
            results.update(
                {
                    "beta": beta,  # beta for f_beta
                    "optimal_threshold": optimal_threshold,  # optimal threshold to maximize f_beta
                    f"test_{tune_metric}": final_results[tune_metric],
                    "test_auc": final_results["auc"],
                    "test_Fbeta": final_results["f_beta"],
                    "test_precision": final_results["precision"],
                    "test_recall": final_results["recall"],
                    "test_log_loss": final_results["log_loss"],
                    f"cv_{tune_metric}_mean": best_trial.user_attrs[
                        f"mean_{tune_metric}"
                    ],
                    f"cv_{tune_metric}_std": best_trial.user_attrs[
                        f"std_{tune_metric}"
                    ],
                }
            )
        elif task == "regression":
            results.update(
                {
                    f"test_{tune_metric}": final_results[tune_metric],
                    "test_rmse": final_results["rmse"],
                    "test_mae": final_results["mae"],
                    "test_median_ae": final_results["median_ae"],
                    "test_nrmse_mean": final_results["nrmse_mean"],
                    "test_nrmse_std": final_results["nrmse_std"],
                    "test_nrmse_iqr": final_results["nrmse_iqr"],
                    "test_mape": final_results["mape"],
                    "test_smape": final_results["smape"],
                    "test_r2": final_results["r2"],
                    "test_adj_r2": final_results["adj_r2"],
                    "test_rmse_improvement_over_mean": final_results[
                        "rmse_improvement_over_mean"
                    ],  # renamed
                    "test_rmse_improvement_over_median": final_results[
                        "rmse_improvement_over_median"
                    ],  # new
                    f"cv_{tune_metric}_mean": best_trial.user_attrs[
                        f"mean_{tune_metric}"
                    ],
                    f"cv_{tune_metric}_std": best_trial.user_attrs[
                        f"std_{tune_metric}"
                    ],
                }
            )

        return results

    @staticmethod
    def threshold_analysis(
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        beta: float = 1.0,
        method: str = "bootstrap",
        cv_splits: int = 5,
        bootstrap_iterations: int = 100,
        random_state: int = 42,
    ):
        """
        Identify the optimal threshold that maximizes F-beta score using CV or bootstrap.

        Parameters
        ----------
        y_true : pd.Series
            True labels.
        y_pred_proba : np.ndarray
            Predicted probabilities.
        beta : float, default=1.0
            F-beta score parameter.
        method : str, default='bootstrap'
            Method to use for threshold selection:
            - 'cv': Cross-validation (faster, deterministic splits)
            - 'bootstrap': Bootstrap resampling (more robust, with confidence intervals)
        cv_splits : int, default=5
            Number of folds for cross-validation when method='cv'.
        bootstrap_iterations : int, default=100
            Number of bootstrap samples when method='bootstrap'.
        random_state : int, default=42
            Random state for reproducibility.

        Returns
        -------
        dict
            Dictionary containing:
            - optimal_threshold: Mean of thresholds across iterations/splits
            - threshold_std: Standard deviation of thresholds
            - threshold_values: List of individual thresholds
            For bootstrap method, also includes:
            - ci_lower: Lower bound of 95% confidence interval
            - ci_upper: Upper bound of 95% confidence interval
        """
        np.random.seed(random_state)
        thresholds_values = []

        if method == "cv":
            # Efficient CV implementation
            cv = StratifiedKFold(
                n_splits=cv_splits, shuffle=True, random_state=random_state
            )

            for train_idx, val_idx in cv.split(y_true, y_true):
                y_true_val = y_true.iloc[val_idx]
                y_pred_proba_val = y_pred_proba[val_idx]

                precisions, recalls, pr_thresholds = precision_recall_curve(
                    y_true_val, y_pred_proba_val
                )

                f_beta_scores = (
                    (1 + beta**2)
                    * (precisions * recalls)
                    / (beta**2 * precisions + recalls + 1e-10)
                )
                optimal_idx = np.argmax(f_beta_scores)

                if optimal_idx < len(pr_thresholds):
                    optimal_threshold = pr_thresholds[optimal_idx]
                else:
                    # use 0.5 when optimal F-beta occured at the end of the precision-recall curve
                    optimal_threshold = 0.5

                thresholds_values.append(optimal_threshold)

        elif method == "bootstrap":
            n_samples = len(y_true)

            for _ in range(bootstrap_iterations):
                # Bootstrap sampling
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                y_true_boot = y_true.iloc[indices]
                y_pred_proba_boot = y_pred_proba[indices]

                precisions, recalls, pr_thresholds = precision_recall_curve(
                    y_true_boot, y_pred_proba_boot
                )

                f_beta_scores = (
                    (1 + beta**2)
                    * (precisions * recalls)
                    / (beta**2 * precisions + recalls + 1e-10)
                )
                optimal_idx = np.argmax(f_beta_scores)

                if optimal_idx < len(pr_thresholds):
                    optimal_threshold = pr_thresholds[optimal_idx]
                else:
                    optimal_threshold = 0.5

                thresholds_values.append(optimal_threshold)

        else:
            raise ValueError(
                f"Method must be either 'cv' or 'bootstrap', got: {method}"
            )

        results = {
            "optimal_threshold": np.mean(thresholds_values),
            "threshold_std": np.std(thresholds_values),
            "threshold_values": thresholds_values,
        }

        # Add confidence intervals only for bootstrap method
        if method == "bootstrap":
            results.update(
                {
                    "ci_lower": np.percentile(thresholds_values, 2.5),
                    "ci_upper": np.percentile(thresholds_values, 97.5),
                }
            )

        return results

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

        Parameters
        ----------
        metrics : dict, optional
            Metrics to log.
        params : dict, optional
            Parameters to log.
        additional_artifacts : dict, optional
            Additional artifacts to log, e.g., {"parallel_coords_plot": plot_path}.
        sample_input : pd.DataFrame, optional
            Sample input for signature inference.
        sample_output : np.ndarray, optional
            Sample output for signature inference.

        Returns
        -------
        mlflow.models.ModelInfo
            Information about the logged model.
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


# Backward compatibility alias with deprecation warning
class ML_PIPELINE(MLPipeline):
    """
    Deprecated: Use MLPipeline instead to follow Python naming conventions (PEP 8).
    ML_PIPELINE will be removed in a future version.

    Please update your code:
        from mlarena import MLPipeline  # New (recommended)
        # instead of: from mlarena import ML_PIPELINE
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ML_PIPELINE is deprecated and will be removed in a future version. "
            "Please use MLPipeline instead. "
            "See upgrade guide: https://github.com/MenaWANG/mlarena/blob/master/docs/upgrading.md",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
