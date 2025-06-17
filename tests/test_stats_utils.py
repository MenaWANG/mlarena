import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mlarena.utils.stats_utils import (
    add_stratified_groups,
    compare_groups,
    optimize_stratification_strategy,
    calculate_threshold_stats,
    calculate_group_thresholds,
)


class TestCompareGroups:
    """Test the compare_groups function."""

    def test_basic_functionality(self):
        """Test basic functionality with mixed data types."""
        # Create test data
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "A", "B"] * 5,
                "numeric_var": np.random.normal(100, 15, 30),
                "categorical_var": ["X", "Y"] * 15,
                "another_numeric": np.random.normal(50, 10, 30),
            }
        )

        # Test function execution
        effect_size_sum, summary_df = compare_groups(
            df, "group", ["numeric_var", "categorical_var", "another_numeric"]
        )

        # Basic checks
        assert isinstance(effect_size_sum, (int, float))
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 3  # Three target variables

        # Check summary DataFrame structure
        expected_cols = [
            "grouping_col",
            "target_var",
            "stat_test",
            "p_value",
            "effect_size",
            "is_significant",
            "weight",
        ]
        assert all(col in summary_df.columns for col in expected_cols)

        # Check grouping column is preserved correctly
        assert all(summary_df["grouping_col"] == "group")

        # Check target variables are recorded correctly
        assert set(summary_df["target_var"]) == {
            "numeric_var",
            "categorical_var",
            "another_numeric",
        }

    def test_weights_functionality(self):
        """Test weighted effect size calculation."""
        np.random.seed(42)  # For reproducible test
        df = pd.DataFrame(
            {
                "group": ["A", "B"] * 20,
                "var1": np.concatenate(
                    [
                        np.random.normal(10, 2, 20),  # Group A: mean=10, std=2
                        np.random.normal(12, 2, 20),  # Group B: mean=12, std=2
                    ]
                ),
                "var2": np.concatenate(
                    [
                        np.random.normal(50, 5, 20),  # Group A: mean=50, std=5
                        np.random.normal(55, 5, 20),  # Group B: mean=55, std=5
                    ]
                ),
            }
        )

        weights = {"var1": 2.0, "var2": 0.5}

        effect_size_sum, summary_df = compare_groups(
            df, "group", ["var1", "var2"], weights=weights
        )

        # Check weights are applied correctly
        assert (
            summary_df.loc[summary_df["target_var"] == "var1", "weight"].iloc[0] == 2.0
        )
        assert (
            summary_df.loc[summary_df["target_var"] == "var2", "weight"].iloc[0] == 0.5
        )

    def test_missing_data_handling(self):
        """Test handling of missing data."""
        np.random.seed(123)
        df = pd.DataFrame(
            {
                "group": ["A"] * 10 + ["B"] * 10,
                "var_with_missing": np.concatenate(
                    [
                        [None, None],  # Some missing values
                        np.random.normal(10, 2, 8),  # Group A data
                        [None],  # Missing in group B
                        np.random.normal(15, 2, 9),  # Group B data
                    ]
                ),
            }
        )

        # Should not raise error with missing data
        effect_size_sum, summary_df = compare_groups(df, "group", ["var_with_missing"])

        assert isinstance(effect_size_sum, (int, float))
        assert len(summary_df) == 1

    @patch("mlarena.utils.plot_utils.plot_box_scatter")
    def test_visualization_numeric(self, mock_plot):
        """Test visualization for numeric variables."""
        # Mock the plot function to return expected structure
        mock_plot.return_value = (
            None,
            None,
            {"stat_test": {"method": "anova", "p_value": 0.05, "effect_size": 0.1}},
        )

        np.random.seed(456)
        df = pd.DataFrame(
            {
                "group": ["A", "B"] * 10,
                "numeric_var": np.concatenate(
                    [
                        np.random.normal(100, 10, 10),  # Group A
                        np.random.normal(105, 10, 10),  # Group B
                    ]
                ),
            }
        )

        effect_size_sum, summary_df = compare_groups(
            df, "group", ["numeric_var"], visualize=True
        )

        # Check that plot function was called
        mock_plot.assert_called_once()

    def test_alpha_threshold(self):
        """Test custom alpha threshold."""
        df = pd.DataFrame({"group": ["A", "B"] * 10, "var": np.random.normal(0, 1, 20)})

        # Test with different alpha values
        _, summary_strict = compare_groups(df, "group", ["var"], alpha=0.01)
        _, summary_lenient = compare_groups(df, "group", ["var"], alpha=0.10)

        # Both should have one row for the one variable
        assert len(summary_strict) == 1
        assert len(summary_lenient) == 1


class TestAddStratifiedGroups:
    """Test the add_stratified_groups function."""

    def test_basic_stratification(self):
        """Test basic stratification with single column."""
        df = pd.DataFrame(
            {"category": ["A", "B", "A", "B", "A", "B"] * 10, "value": range(60)}
        )

        result = add_stratified_groups(df, "category")

        # Check structure
        assert "stratified_group" in result.columns
        assert len(result) == len(df)

        # Check that both groups are present
        unique_groups = result["stratified_group"].unique()
        assert len(unique_groups) == 2
        assert set(unique_groups) == {0, 1}

        # Check stratification worked (roughly equal distribution)
        group_counts = result["stratified_group"].value_counts()
        assert abs(group_counts[0] - group_counts[1]) <= 2  # Allow small imbalance

    def test_multiple_column_stratification(self):
        """Test stratification with multiple columns."""
        df = pd.DataFrame(
            {
                "region": ["North", "South"] * 20,
                "segment": ["A", "B"] * 20,
                "value": range(40),
            }
        )

        result = add_stratified_groups(df, ["region", "segment"])

        # Check that groups are created
        assert "stratified_group" in result.columns
        assert len(result["stratified_group"].unique()) == 2

    def test_custom_group_labels(self):
        """Test custom group labels."""
        df = pd.DataFrame({"category": ["A", "B"] * 10, "value": range(20)})

        # Test with string labels
        result = add_stratified_groups(
            df, "category", group_labels=("control", "treatment")
        )
        unique_groups = result["stratified_group"].unique()
        assert set(unique_groups) == {"control", "treatment"}

        # Test with custom numeric labels
        result = add_stratified_groups(df, "category", group_labels=(100, 200))
        unique_groups = result["stratified_group"].unique()
        assert set(unique_groups) == {100, 200}

    def test_custom_group_column_name(self):
        """Test custom group column name."""
        df = pd.DataFrame({"category": ["A", "B"] * 10, "value": range(20)})

        result = add_stratified_groups(df, "category", group_col_name="treatment_group")

        assert "treatment_group" in result.columns
        assert "stratified_group" not in result.columns

    def test_missing_columns_error(self):
        """Test error handling for missing columns."""
        df = pd.DataFrame({"existing_col": [1, 2, 3]})

        with pytest.raises(ValueError, match="not found in DataFrame"):
            add_stratified_groups(df, "non_existent_col")

        with pytest.raises(ValueError, match="not found in DataFrame"):
            add_stratified_groups(df, ["existing_col", "non_existent_col"])

    def test_stratification_failure_warning(self):
        """Test warning when stratification fails."""
        # Create data where stratification will fail (groups with single member)
        df = pd.DataFrame(
            {
                "category": ["A", "B", "C"],  # Each category has only one member
                "value": [1, 2, 3],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = add_stratified_groups(df, "category")

            # Check that warning was issued
            assert len(w) == 1
            assert "Stratified split failed" in str(w[0].message)

            # Check that all rows are assigned to first group
            assert all(result["stratified_group"] == 0)

    def test_random_seed_reproducibility(self):
        """Test that random seed produces reproducible results."""
        df = pd.DataFrame({"category": ["A", "B"] * 20, "value": range(40)})

        result1 = add_stratified_groups(df, "category", random_seed=42)
        result2 = add_stratified_groups(df, "category", random_seed=42)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestOptimizeStratificationStrategy:
    """Test the optimize_stratification_strategy function."""

    def test_basic_functionality(self):
        """Test basic optimization functionality."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "region": ["North", "South"] * 50,
                "segment": ["A", "B"] * 50,
                "metric1": np.random.normal(100, 15, 100),
                "metric2": np.random.normal(50, 10, 100),
                "category": ["X", "Y"] * 50,
            }
        )

        results = optimize_stratification_strategy(
            df, ["region", "segment"], ["metric1", "metric2"]
        )

        # Check structure
        assert "best_stratifier" in results
        assert "results" in results
        assert "rankings" in results
        assert "summary" in results

        # Check that we got results for different stratifiers
        assert len(results["results"]) >= 2  # At least single columns

        # Check that rankings are in correct order
        rankings = results["rankings"]
        scores = [results["results"][k]["composite_score"] for k in rankings]
        assert scores == sorted(scores)  # Should be in ascending order

        # Check summary DataFrame structure
        summary_df = results["summary"]
        expected_cols = [
            "stratifier",
            "effect_size_sum",
            "n_significant",
            "composite_score",
            "rank",
        ]
        assert all(col in summary_df.columns for col in expected_cols)
        assert len(summary_df) == len(results["results"])
        assert summary_df["rank"].tolist() == list(range(1, len(summary_df) + 1))

    def test_max_combinations(self):
        """Test max_combinations parameter."""
        df = pd.DataFrame(
            {
                "col1": ["A", "B"] * 20,
                "col2": ["X", "Y"] * 20,
                "col3": ["P", "Q"] * 20,
                "metric": np.random.normal(0, 1, 40),
            }
        )

        # Test with max_combinations=1 (only single columns)
        results = optimize_stratification_strategy(
            df, ["col1", "col2", "col3"], ["metric"], max_combinations=1
        )

        # Should only have single column stratifiers
        expected_stratifiers = ["col1", "col2", "col3"]
        assert set(results["results"].keys()) == set(expected_stratifiers)

        # Test with max_combinations=2 (single + pairs)
        results = optimize_stratification_strategy(
            df, ["col1", "col2", "col3"], ["metric"], max_combinations=2
        )

        # Should have more stratifiers (singles + combinations)
        assert len(results["results"]) > 3

    def test_composite_scoring(self):
        """Test that composite scoring includes both effect size and significance count."""
        np.random.seed(999)
        df = pd.DataFrame(
            {
                "stratifier": ["A", "B"] * 25,
                "metric1": np.concatenate(
                    [
                        np.random.normal(10, 1, 25),  # Group A: small variance
                        np.random.normal(10.2, 1, 25),  # Group B: very small difference
                    ]
                ),
                "metric2": np.concatenate(
                    [
                        np.random.normal(10, 2, 25),  # Group A
                        np.random.normal(20, 2, 25),  # Group B: large difference
                    ]
                ),
            }
        )

        results = optimize_stratification_strategy(
            df, ["stratifier"], ["metric1", "metric2"]
        )

        # Check that composite score and components are calculated
        result_data = results["results"]["stratifier"]
        assert "effect_size_sum" in result_data
        assert "n_significant" in result_data
        assert "composite_score" in result_data

        # Composite score should be effect_size_sum + penalty (default 0.1)
        expected_composite = result_data["effect_size_sum"] + (
            result_data["n_significant"] * 0.1
        )
        assert abs(result_data["composite_score"] - expected_composite) < 1e-10

    def test_empty_candidate_list(self):
        """Test handling of empty candidate list."""
        df = pd.DataFrame({"metric": [1, 2, 3, 4], "group": ["A", "B", "A", "B"]})

        results = optimize_stratification_strategy(df, [], ["metric"])

        assert results["best_stratifier"] is None
        assert results["results"] == {}
        assert results["rankings"] == []
        assert len(results["summary"]) == 0  # Empty DataFrame

    def test_stratification_failure_handling(self):
        """Test graceful handling when stratification fails."""
        # Create problematic data
        df = pd.DataFrame(
            {"bad_stratifier": ["A", "B", "B"], "metric": [1, 2, 3]}  # All same value
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = optimize_stratification_strategy(
                df, ["bad_stratifier"], ["metric"]
            )

            # Should handle gracefully and continue
            assert isinstance(results, dict)

            # Should have issued warnings about stratification failure
            # Check for either the original stratification warning or our new one
            warning_messages = [str(warning.message) for warning in w]
            assert len(warning_messages) > 0  # At least one warning should be issued

            # Check for expected warning patterns
            stratification_warnings = any(
                "stratified split failed" in msg.lower()
                or "failed to create multiple groups" in msg.lower()
                for msg in warning_messages
            )
            assert (
                stratification_warnings
            ), f"Expected stratification warning not found. Warnings: {warning_messages}"

            # Should have no results since stratification failed
            assert len(results["results"]) == 0
            assert results["best_stratifier"] is None
            assert results["rankings"] == []
            assert len(results["summary"]) == 0  # Empty DataFrame

    def test_weights_integration(self):
        """Test that weights are properly passed to compare_groups."""
        np.random.seed(111)
        df = pd.DataFrame(
            {
                "stratifier": ["A", "B"] * 15,
                "metric1": np.concatenate(
                    [
                        np.random.normal(100, 10, 15),  # Group A
                        np.random.normal(110, 10, 15),  # Group B
                    ]
                ),
                "metric2": np.concatenate(
                    [
                        np.random.normal(50, 5, 15),  # Group A
                        np.random.normal(55, 5, 15),  # Group B
                    ]
                ),
            }
        )

        weights = {"metric1": 2.0, "metric2": 0.5}

        results = optimize_stratification_strategy(
            df, ["stratifier"], ["metric1", "metric2"], weights=weights
        )

        # Check that weights were used in the summary
        summary = results["results"]["stratifier"]["summary"]
        weight_values = summary["weight"].values
        assert 2.0 in weight_values
        assert 0.5 in weight_values

    def test_custom_significance_penalty(self):
        """Test custom significance penalty parameter."""
        np.random.seed(789)
        df = pd.DataFrame(
            {
                "stratifier": ["A", "B"] * 30,
                "metric1": np.concatenate(
                    [
                        np.random.normal(10, 2, 30),  # Group A
                        np.random.normal(12, 2, 30),  # Group B
                    ]
                ),
                "metric2": np.concatenate(
                    [
                        np.random.normal(50, 5, 30),  # Group A
                        np.random.normal(55, 5, 30),  # Group B
                    ]
                ),
            }
        )

        # Test with different penalty values
        results_low_penalty = optimize_stratification_strategy(
            df, ["stratifier"], ["metric1", "metric2"], significance_penalty=0.01
        )

        results_high_penalty = optimize_stratification_strategy(
            df, ["stratifier"], ["metric1", "metric2"], significance_penalty=1.0
        )

        # Both should have same effect_size_sum and n_significant
        low_data = results_low_penalty["results"]["stratifier"]
        high_data = results_high_penalty["results"]["stratifier"]

        assert low_data["effect_size_sum"] == high_data["effect_size_sum"]
        assert low_data["n_significant"] == high_data["n_significant"]

        # But different composite scores due to different penalties
        expected_low = low_data["effect_size_sum"] + (low_data["n_significant"] * 0.01)
        expected_high = high_data["effect_size_sum"] + (
            high_data["n_significant"] * 1.0
        )

        assert abs(low_data["composite_score"] - expected_low) < 1e-10
        assert abs(high_data["composite_score"] - expected_high) < 1e-10

        # Test with zero penalty (effect size only)
        results_no_penalty = optimize_stratification_strategy(
            df, ["stratifier"], ["metric1", "metric2"], significance_penalty=0.0
        )

        no_penalty_data = results_no_penalty["results"]["stratifier"]
        assert no_penalty_data["composite_score"] == no_penalty_data["effect_size_sum"]


class TestIntegrationTests:
    """Integration tests for the stats_utils functions working together."""

    def test_stratification_validation_workflow(self):
        """Test the complete workflow: stratify -> validate."""
        # Create test data with known patterns
        df = pd.DataFrame(
            {
                "region": ["North", "South"] * 50,
                "age_group": ["Young", "Old"] * 50,
                "income": np.random.normal(50000, 15000, 100),
                "satisfaction": np.random.normal(7, 2, 100),
            }
        )

        # Step 1: Create stratified groups
        df_stratified = add_stratified_groups(df, "region")

        # Step 2: Validate stratification
        effect_size, summary = compare_groups(
            df_stratified, "stratified_group", ["income", "satisfaction"]
        )

        # Check that both steps completed successfully
        assert "stratified_group" in df_stratified.columns
        assert isinstance(effect_size, (int, float))
        assert len(summary) == 2  # Two target metrics

    def test_optimization_workflow(self):
        """Test the optimization workflow with validation."""
        np.random.seed(123)
        df = pd.DataFrame(
            {
                "region": ["North", "South"] * 30,
                "segment": ["A", "B"] * 30,
                "age": ["Young", "Old"] * 30,
                "metric1": np.random.normal(100, 20, 60),
                "metric2": np.random.normal(50, 15, 60),
            }
        )

        # Find best stratification strategy
        optimization_results = optimize_stratification_strategy(
            df, ["region", "segment"], ["metric1", "metric2"]
        )

        # Use the best strategy
        best_stratifier = optimization_results["best_stratifier"]
        df_final = add_stratified_groups(
            df, best_stratifier, group_labels=("control", "treatment")
        )

        # Validate final result
        effect_size, validation_summary = compare_groups(
            df_final, "stratified_group", ["metric1", "metric2"]
        )

        # Check complete workflow
        assert best_stratifier is not None
        assert set(df_final["stratified_group"].unique()) == {"control", "treatment"}
        assert len(validation_summary) == 2
        assert isinstance(effect_size, (int, float))


class TestCalculateThresholdStats:
    """Test the calculate_threshold_stats function."""

    def test_basic_functionality(self):
        """Test basic functionality with different input types."""
        # Test with list
        data_list = [1, 2, 2, 3, 3, 3, 4, 4, 10]
        stats_list = calculate_threshold_stats(data_list)

        # Test with numpy array
        data_np = np.array(data_list)
        stats_np = calculate_threshold_stats(data_np)

        # Test with pandas Series
        data_pd = pd.Series(data_list)
        stats_pd = calculate_threshold_stats(data_pd)

        # All should give same results
        for stats in [stats_list, stats_np, stats_pd]:
            assert isinstance(stats, dict)
            assert set(stats.keys()) == {
                "mean",
                "median",
                "std",
                "count",
                "method",
                "threshold",
            }
            assert stats["count"] == len(data_list)
            assert stats["method"] == "std"
            assert np.isclose(stats["mean"], np.mean(data_list))
            assert np.isclose(stats["median"], np.median(data_list))
            assert np.isclose(stats["std"], np.std(data_list))

    def test_threshold_methods(self):
        """Test different threshold calculation methods."""
        data = [1, 2, 2, 3, 3, 3, 4, 4, 10]  # Intentionally include outlier

        # Test std method
        stats_std = calculate_threshold_stats(data, n_std=2, threshold_method="std")
        expected_threshold_std = np.mean(data) + 2 * np.std(data)
        assert np.isclose(stats_std["threshold"], expected_threshold_std)

        # Test IQR method
        stats_iqr = calculate_threshold_stats(data, threshold_method="iqr")
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        expected_threshold_iqr = q3 + 1.5 * iqr
        assert np.isclose(stats_iqr["threshold"], expected_threshold_iqr)

        # Test percentile method
        stats_percentile = calculate_threshold_stats(
            data, threshold_method="percentile"
        )
        expected_threshold_percentile = np.percentile(data, 95)
        assert np.isclose(stats_percentile["threshold"], expected_threshold_percentile)

    def test_invalid_method(self):
        """Test error handling for invalid threshold method."""
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="Invalid threshold_method"):
            calculate_threshold_stats(data, threshold_method="invalid_method")

    def test_empty_input(self):
        """Test handling of empty input."""
        # Test that appropriate warning is raised
        with pytest.warns(UserWarning, match="Empty input provided"):
            stats = calculate_threshold_stats([])

        # Verify expected values for empty input
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["median"])
        assert np.isnan(stats["std"])
        assert stats["count"] == 0
        assert np.isnan(stats["threshold"])
        assert stats["method"] == "std"  # Default method should be preserved

    def test_single_value(self):
        """Test handling of single value input."""
        data = [42]
        stats = calculate_threshold_stats(data)
        assert stats["mean"] == 42
        assert stats["median"] == 42
        assert stats["count"] == 1
        assert stats["std"] == 0

    @patch("matplotlib.pyplot.show")
    def test_visualization(self, mock_show):
        """Test visualization option."""
        data = list(range(10))
        calculate_threshold_stats(data, visualize=True)
        mock_show.assert_called_once()

        # Test without visualization
        mock_show.reset_mock()
        calculate_threshold_stats(data, visualize=False)
        mock_show.assert_not_called()


class TestCalculateGroupThresholds:
    """Test the calculate_group_thresholds function."""

    def test_basic_functionality(self):
        """Test basic functionality with simple grouped data."""
        df = pd.DataFrame(
            {"group": ["A", "A", "A", "B", "B", "B"], "value": [1, 2, 3, 4, 5, 6]}
        )

        results = calculate_group_thresholds(df, "group", "value")

        # Check structure
        assert isinstance(results, pd.DataFrame)
        expected_cols = {
            "group",
            "method",
            "mean",
            "median",
            "std",
            "threshold",
            "count",
        }
        assert set(results.columns) == expected_cols

        # Check groups
        assert set(results["group"].unique()) == {"A", "B"}

        # Check methods
        assert set(results["method"].unique()) == {"std", "iqr", "percentile"}

        # Basic value checks
        group_a_stats = results[results["group"] == "A"]
        assert np.isclose(
            group_a_stats[group_a_stats["method"] == "std"]["mean"].iloc[0], 2.0
        )
        assert np.isclose(
            group_a_stats[group_a_stats["method"] == "std"]["median"].iloc[0], 2.0
        )

    def test_multiple_methods(self):
        """Test with specific methods selection."""
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        # Test with single method
        results_single = calculate_group_thresholds(
            df, "group", "value", methods=["std"]
        )
        assert len(results_single["method"].unique()) == 1
        assert results_single["method"].iloc[0] == "std"

        # Test with multiple methods
        methods = ["std", "iqr"]
        results_multiple = calculate_group_thresholds(
            df, "group", "value", methods=methods
        )
        assert set(results_multiple["method"].unique()) == set(methods)

    def test_visualization_control(self):
        """Test visualization control for first group."""
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        with patch("matplotlib.pyplot.show") as mock_show:
            # With visualization
            calculate_group_thresholds(df, "group", "value", visualize_first_group=True)
            assert mock_show.call_count == len(["std", "iqr", "percentile"])

            # Without visualization
            mock_show.reset_mock()
            calculate_group_thresholds(
                df, "group", "value", visualize_first_group=False
            )
            mock_show.assert_not_called()

    def test_empty_groups(self):
        """Test handling of empty groups and DataFrames."""
        # Test empty DataFrame
        empty_df = pd.DataFrame({"group": [], "value": []})
        with pytest.warns(UserWarning, match="Empty DataFrame provided"):
            results = calculate_group_thresholds(empty_df, "group", "value")
            assert len(results) == 0
            assert set(results.columns) == {
                "group",
                "method",
                "mean",
                "median",
                "std",
                "threshold",
                "count",
            }

        # Test small group warning
        df_small = pd.DataFrame({"group": ["A", "B", "B", "B"], "value": [1, 2, 3, 4]})
        with pytest.warns(UserWarning, match="Group 'A' has fewer than 2 values"):
            results = calculate_group_thresholds(
                df_small, "group", "value", min_group_size=2
            )
            assert len(results["group"].unique()) == 2
            assert set(results["group"].unique()) == {"A", "B"}

    def test_invalid_column_names(self):
        """Test error handling for invalid column names."""
        df = pd.DataFrame({"group": ["A", "B"], "value": [1, 2]})

        with pytest.raises(KeyError):
            calculate_group_thresholds(df, "nonexistent_group", "value")

        with pytest.raises(KeyError):
            calculate_group_thresholds(df, "group", "nonexistent_value")

    def test_non_numeric_values(self):
        """Test error handling for non-numeric values."""
        df = pd.DataFrame({"group": ["A", "B"], "value": ["not", "numeric"]})

        with pytest.raises(Exception):  # Should raise some kind of error
            calculate_group_thresholds(df, "group", "value")

    def test_n_std_parameter(self):
        """Test that n_std parameter affects threshold calculation."""
        df = pd.DataFrame(
            {
                "group": ["A"] * 10,
                "value": [1, 2, 2, 3, 3, 3, 4, 4, 5, 10],  # mean=3.7, std≈2.45
            }
        )

        # Test with different n_std values
        results_2std = calculate_group_thresholds(
            df, "group", "value", methods=["std"], n_std=2.0
        )
        results_3std = calculate_group_thresholds(
            df, "group", "value", methods=["std"], n_std=3.0
        )

        # Get thresholds
        threshold_2std = results_2std[results_2std["method"] == "std"][
            "threshold"
        ].iloc[0]
        threshold_3std = results_3std[results_3std["method"] == "std"][
            "threshold"
        ].iloc[0]

        # 3 std threshold should be higher than 2 std
        assert threshold_3std > threshold_2std

        # Verify exact calculations
        mean = results_2std["mean"].iloc[0]
        std = results_2std["std"].iloc[0]
        assert np.isclose(threshold_2std, mean + 2.0 * std)
        assert np.isclose(threshold_3std, mean + 3.0 * std)

    def test_different_distributions(self):
        """Test with different underlying distributions."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "group": ["normal"] * 100 + ["uniform"] * 100 + ["exponential"] * 100,
                "value": np.concatenate(
                    [
                        np.random.normal(10, 2, 100),  # Normal distribution
                        np.random.uniform(0, 20, 100),  # Uniform distribution
                        np.random.exponential(10, 100),  # Exponential distribution
                    ]
                ),
            }
        )

        results = calculate_group_thresholds(df, "group", "value")

        # Each distribution should have different characteristics
        for group in df["group"].unique():
            group_stats = results[results["group"] == group]

            # Basic checks that make sense for the distributions
            if group == "normal":
                # For normal, mean and median should be similar
                std_stats = group_stats[group_stats["method"] == "std"]
                assert (
                    np.abs(std_stats["mean"].iloc[0] - std_stats["median"].iloc[0]) < 1
                )

            elif group == "exponential":
                # For exponential, mean should be greater than median (right-skewed)
                std_stats = group_stats[group_stats["method"] == "std"]
                assert std_stats["mean"].iloc[0] > std_stats["median"].iloc[0]
