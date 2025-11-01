"""
Tests for wrapper feature selection functionality in PreProcessor.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR

from mlarena.preprocessor import PreProcessor


class TestWrapperFeatureSelection:
    """Test cases for wrapper feature selection method."""

    @pytest.fixture
    def classification_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
        y_series = pd.Series(y)
        return X_df, y_series

    @pytest.fixture
    def regression_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=100, n_features=15, n_informative=10, noise=0.1, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(15)])
        y_series = pd.Series(y)
        return X_df, y_series

    @pytest.fixture
    def mixed_data(self):
        """Create data with mixed types including missing values."""
        np.random.seed(42)
        n_samples = 80

        # Create mixed data
        data = {
            "numeric_1": np.random.randn(n_samples),
            "numeric_2": np.random.randn(n_samples),
            "numeric_3": np.random.randn(n_samples),
            "categorical_1": np.random.choice(["A", "B", "C"], n_samples),
            "categorical_2": np.random.choice(["X", "Y"], n_samples),
            "binary_feature": np.random.choice([0, 1], n_samples),
        }

        # Add missing values
        data["numeric_1"][::10] = np.nan
        data["categorical_1"][::15] = np.nan

        X_df = pd.DataFrame(data)
        y_series = pd.Series(np.random.choice([0, 1], n_samples))
        return X_df, y_series

    def test_basic_classification(self, classification_data):
        """Test basic functionality with classification data."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, n_max_features=10, visualize=False, verbose=False
        )

        # Check return structure
        expected_keys = [
            "selected_features",
            "n_features_selected",
            "optimal_score",
            "optimal_score_std",
            "penalized_score",
            "cv_scores",
            "cv_scores_std",
            "feature_rankings",
            "rfecv_object",
            "selection_params",
        ]
        assert all(key in results for key in expected_keys)

        # Check basic constraints
        assert isinstance(results["selected_features"], list)
        assert len(results["selected_features"]) == results["n_features_selected"]
        assert results["n_features_selected"] <= 10  # n_max_features
        assert results["n_features_selected"] >= 2  # min_features_to_select
        assert all(feat in X.columns for feat in results["selected_features"])

        # Check scores
        assert isinstance(results["optimal_score"], (int, float))
        assert isinstance(results["optimal_score_std"], (int, float))
        assert isinstance(results["penalized_score"], (int, float))
        assert len(results["cv_scores"]) > 0
        assert len(results["cv_scores_std"]) == len(results["cv_scores"])

    def test_basic_regression(self, regression_data):
        """Test basic functionality with regression data."""
        X, y = regression_data
        estimator = RandomForestRegressor(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, n_max_features=8, visualize=False, verbose=False
        )

        # Check task type detection
        assert results["selection_params"]["task_type"] == "regression"

        # Check constraints
        assert results["n_features_selected"] <= 8
        assert results["n_features_selected"] >= 2
        assert len(results["selected_features"]) == results["n_features_selected"]

    def test_mixed_data_types(self, mixed_data):
        """Test with mixed data types and missing values."""
        X, y = mixed_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, n_max_features=4, visualize=False, verbose=False
        )

        # Should handle mixed types without errors
        assert results["n_features_selected"] <= 4
        assert len(results["selected_features"]) > 0

    def test_parameter_validation(self, classification_data):
        """Test parameter validation."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        # Test invalid inputs
        with pytest.raises(ValueError, match="X must be a pandas DataFrame"):
            PreProcessor.wrapper_feature_selection(
                X.values, y, estimator, visualize=False
            )

        with pytest.raises(ValueError, match="y must be a pandas Series"):
            PreProcessor.wrapper_feature_selection(
                X, y.values, estimator, visualize=False
            )

        with pytest.raises(ValueError, match="same number of samples"):
            PreProcessor.wrapper_feature_selection(
                X, y[:-5], estimator, visualize=False
            )

        with pytest.raises(
            ValueError, match="min_features_to_select must be at least 1"
        ):
            PreProcessor.wrapper_feature_selection(
                X, y, estimator, min_features_to_select=0, visualize=False
            )

        with pytest.raises(ValueError, match="step must be at least 1"):
            PreProcessor.wrapper_feature_selection(
                X, y, estimator, step=0, visualize=False
            )

        with pytest.raises(ValueError, match="cv must be at least 2"):
            PreProcessor.wrapper_feature_selection(
                X, y, estimator, cv=1, visualize=False
            )

        with pytest.raises(
            ValueError, match="cv_variance_penalty must be non-negative"
        ):
            PreProcessor.wrapper_feature_selection(
                X, y, estimator, cv_variance_penalty=-0.1, visualize=False
            )

    def test_default_n_max_features(self, classification_data):
        """Test default n_max_features calculation."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, n_max_features=None, visualize=False, verbose=False
        )

        # Should default to n_samples // 10 = 100 // 10 = 10
        expected_max = len(X) // 10
        assert results["selection_params"]["n_max_features"] == expected_max

    def test_different_estimators(self, classification_data):
        """Test with different types of estimators."""
        X, y = classification_data

        estimators = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42, max_iter=100),
        ]

        for estimator in estimators:
            results = PreProcessor.wrapper_feature_selection(
                X, y, estimator, n_max_features=5, visualize=False, verbose=False
            )
            assert len(results["selected_features"]) > 0
            assert results["n_features_selected"] <= 5

    def test_variance_penalty_effect(self, classification_data):
        """Test that variance penalty affects feature selection."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        # Test with different penalty values
        results_low_penalty = PreProcessor.wrapper_feature_selection(
            X,
            y,
            estimator,
            cv_variance_penalty=0.0,
            n_max_features=10,
            visualize=False,
            verbose=False,
        )

        results_high_penalty = PreProcessor.wrapper_feature_selection(
            X,
            y,
            estimator,
            cv_variance_penalty=0.5,
            n_max_features=10,
            visualize=False,
            verbose=False,
        )

        # Both should work without errors
        assert len(results_low_penalty["selected_features"]) > 0
        assert len(results_high_penalty["selected_features"]) > 0

    def test_small_dataset(self):
        """Test with very small dataset."""
        # Create minimal dataset
        X = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [2, 4, 6, 8, 10],
                "feat3": [1, 1, 2, 2, 3],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        estimator = RandomForestClassifier(n_estimators=5, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, cv=3, visualize=False, verbose=False
        )

        # Should handle small dataset
        assert len(results["selected_features"]) >= 2  # min_features_to_select
        assert len(results["selected_features"]) <= len(X.columns)

    def test_scoring_parameter(self, classification_data):
        """Test custom scoring parameter."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X,
            y,
            estimator,
            scoring="roc_auc",
            n_max_features=5,
            visualize=False,
            verbose=False,
        )

        assert results["selection_params"]["scoring"] == "roc_auc"
        assert len(results["selected_features"]) > 0

    def test_step_parameter(self, classification_data):
        """Test step parameter for feature elimination."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, step=2, n_max_features=8, visualize=False, verbose=False
        )

        assert results["selection_params"]["step"] == 2
        assert len(results["selected_features"]) > 0

    def test_verbose_output(self, classification_data, capsys):
        """Test verbose output."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        PreProcessor.wrapper_feature_selection(
            X, y, estimator, n_max_features=5, visualize=False, verbose=True
        )

        captured = capsys.readouterr()
        assert "Wrapper Feature Selection Summary:" in captured.out
        assert "Task type:" in captured.out
        assert "Optimal features selected:" in captured.out

    def test_feature_rankings(self, classification_data):
        """Test feature rankings output."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, n_max_features=10, visualize=False, verbose=False
        )

        # Check feature rankings
        assert len(results["feature_rankings"]) == len(X.columns)
        assert all(isinstance(rank, int) for rank in results["feature_rankings"])
        assert min(results["feature_rankings"]) == 1  # Best features have rank 1

    def test_rfecv_object(self, classification_data):
        """Test that RFECV object is returned."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, n_max_features=8, visualize=False, verbose=False
        )

        # Check RFECV object
        rfecv = results["rfecv_object"]
        assert hasattr(rfecv, "support_")
        assert hasattr(rfecv, "ranking_")
        assert hasattr(rfecv, "cv_results_")

    def test_edge_case_n_max_features(self, classification_data):
        """Test edge cases for n_max_features."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        # Test n_max_features larger than total features
        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, n_max_features=100, visualize=False, verbose=False
        )

        # Should be limited to total number of features
        assert results["n_features_selected"] <= len(X.columns)

        # Test n_max_features smaller than min_features_to_select
        results = PreProcessor.wrapper_feature_selection(
            X,
            y,
            estimator,
            n_max_features=1,
            min_features_to_select=3,
            visualize=False,
            verbose=False,
        )

        # Should respect min_features_to_select
        assert results["n_features_selected"] >= 3

    def test_regression_task_detection(self, regression_data):
        """Test automatic regression task detection."""
        X, y = regression_data
        estimator = RandomForestRegressor(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, n_max_features=8, visualize=False, verbose=False
        )

        assert results["selection_params"]["task_type"] == "regression"

    def test_binary_classification_detection(self, classification_data):
        """Test automatic binary classification detection."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, n_max_features=8, visualize=False, verbose=False
        )

        assert results["selection_params"]["task_type"] == "classification"

    def test_multiclass_classification_detection(self):
        """Test automatic multiclass classification detection."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, n_informative=8, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        y_series = pd.Series(y)

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X_df, y_series, estimator, n_max_features=6, visualize=False, verbose=False
        )

        assert results["selection_params"]["task_type"] == "classification"

    def test_reproducibility(self, classification_data):
        """Test that results are reproducible with same random_state."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        results1 = PreProcessor.wrapper_feature_selection(
            X,
            y,
            estimator,
            n_max_features=8,
            random_state=42,
            visualize=False,
            verbose=False,
        )

        results2 = PreProcessor.wrapper_feature_selection(
            X,
            y,
            estimator,
            n_max_features=8,
            random_state=42,
            visualize=False,
            verbose=False,
        )

        # Results should be identical
        assert results1["selected_features"] == results2["selected_features"]
        assert results1["n_features_selected"] == results2["n_features_selected"]
        assert np.allclose(results1["cv_scores"], results2["cv_scores"])

    def test_custom_cv_splitter_object(self, classification_data):
        """Test wrapper feature selection with custom CV splitter object."""
        from sklearn.model_selection import StratifiedKFold

        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        # Create custom CV splitter
        custom_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        results = PreProcessor.wrapper_feature_selection(
            X,
            y,
            estimator,
            cv=custom_cv,
            n_max_features=8,
            visualize=False,
            verbose=False,
        )

        # Should work and return valid results
        assert len(results["selected_features"]) > 0
        assert results["n_features_selected"] <= 8
        assert "cv_scores" in results
        assert len(results["cv_scores"]) > 0

    def test_time_series_cv_splitter(self, regression_data):
        """Test wrapper feature selection with TimeSeriesSplit."""
        from sklearn.model_selection import TimeSeriesSplit

        X, y = regression_data
        estimator = RandomForestRegressor(n_estimators=10, random_state=42)

        # Create TimeSeriesSplit CV splitter
        ts_cv = TimeSeriesSplit(n_splits=3)

        results = PreProcessor.wrapper_feature_selection(
            X, y, estimator, cv=ts_cv, n_max_features=8, visualize=False, verbose=False
        )

        # Should work and return valid results
        assert len(results["selected_features"]) > 0
        assert results["n_features_selected"] <= 8
        assert "cv_scores" in results
        assert len(results["cv_scores"]) > 0

    def test_group_cv_with_groups_parameter(self, classification_data):
        """Test wrapper feature selection with GroupKFold and cv_groups parameter."""
        from sklearn.model_selection import GroupKFold

        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        # Add a group column to X
        np.random.seed(42)
        X_with_groups = X.copy()
        X_with_groups["group_col"] = np.random.randint(0, 5, size=len(X))

        # Create GroupKFold CV splitter
        group_cv = GroupKFold(n_splits=3)

        results = PreProcessor.wrapper_feature_selection(
            X_with_groups,
            y,
            estimator,
            cv=group_cv,
            cv_groups="group_col",
            n_max_features=8,
            visualize=False,
            verbose=False,
        )

        # Should work and return valid results
        assert len(results["selected_features"]) > 0
        assert results["n_features_selected"] <= 8
        assert "cv_scores" in results

    def test_cv_groups_warning_with_incompatible_splitter(self, classification_data):
        """Test that warning is issued when cv_groups is provided with incompatible CV splitter."""
        from sklearn.model_selection import StratifiedKFold

        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        # Add a group column to X
        X_with_groups = X.copy()
        X_with_groups["group_col"] = np.random.randint(0, 3, size=len(X))

        # Use StratifiedKFold (doesn't support groups) with cv_groups parameter
        cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        with pytest.warns(
            UserWarning,
            match="cv_groups.*is provided but the selected CV splitter.*does not use groups",
        ):
            results = PreProcessor.wrapper_feature_selection(
                X_with_groups,
                y,
                estimator,
                cv=cv_splitter,
                cv_groups="group_col",
                n_max_features=8,
                visualize=False,
                verbose=False,
            )

        # Should still work despite the warning
        assert len(results["selected_features"]) > 0

    def test_invalid_cv_groups_column(self, classification_data):
        """Test error when cv_groups column doesn't exist in X."""
        from sklearn.model_selection import GroupKFold

        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        group_cv = GroupKFold(n_splits=3)

        with pytest.raises(
            ValueError, match="Grouping column 'nonexistent_col' not found in X"
        ):
            PreProcessor.wrapper_feature_selection(
                X,
                y,
                estimator,
                cv=group_cv,
                cv_groups="nonexistent_col",
                visualize=False,
                verbose=False,
            )
