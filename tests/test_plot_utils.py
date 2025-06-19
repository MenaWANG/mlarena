import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlarena.utils.plot_utils import plot_box_scatter


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "C", "C"] * 5,
            "values": np.random.normal(0, 1, 30),
            "color_group": ["X", "Y", "X", "Y", "X", "Y"] * 5,
        }
    )
    return data


@pytest.fixture
def time_series_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "values": np.random.normal(0, 1, len(dates)),
            "group": ["A", "B"] * (len(dates) // 2) + ["A"] * (len(dates) % 2),
        }
    )
    return data


def test_plot_box_scatter_missing_point_hue(sample_data):
    """Test that plot_box_scatter handles missing point_hue column correctly."""
    # Given: Data without the specified point_hue column
    data = sample_data.drop(columns=["color_group"])

    # When: We try to plot with a non-existent point_hue
    with pytest.warns(
        UserWarning, match="point_hue column 'non_existent_column' not found"
    ) as warn_record:
        fig, ax = plot_box_scatter(
            data=data, x="category", y="values", point_hue="non_existent_column"
        )

    # Then: Should raise a warning and create plot without point_hue
    assert len(warn_record) == 1  # Should raise exactly one warning
    assert "Proceeding with plot without point_hue coloring" in str(
        warn_record[0].message
    )

    # Verify the plot was created with default styling (no point_hue coloring)
    scatter_points = [
        c
        for c in ax.collections
        if isinstance(c, plt.matplotlib.collections.PathCollection)
    ]
    assert len(scatter_points) > 0  # Should have scatter points
    # All points should have the same color (no point_hue coloring)
    unique_colors = set(tuple(color) for color in scatter_points[0].get_facecolors())
    assert len(unique_colors) == 1

    plt.close(fig)


def test_plot_box_scatter_with_valid_point_hue(sample_data):
    """Test that plot_box_scatter works correctly with valid point_hue column."""
    # When: We plot with a valid point_hue column
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Convert warnings to errors
        fig, ax = plot_box_scatter(
            data=sample_data, x="category", y="values", point_hue="color_group"
        )

    # Then: Should create plot with point_hue coloring
    scatter_points = [
        c
        for c in ax.collections
        if isinstance(c, plt.matplotlib.collections.PathCollection)
    ]
    assert len(scatter_points) > 0  # Should have scatter points

    # Get all unique colors across all scatter collections
    all_colors = set()
    for points in scatter_points:
        colors = points.get_facecolors()
        all_colors.update(tuple(color) for color in colors)

    # Should have two unique colors for 'X' and 'Y'
    assert len(all_colors) == 2

    plt.close(fig)


def test_plot_box_scatter_without_point_hue(sample_data):
    """Test that plot_box_scatter works correctly when point_hue is not specified."""
    # When: We plot without specifying point_hue
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Convert warnings to errors
        fig, ax = plot_box_scatter(data=sample_data, x="category", y="values")

    # Then: Should create plot without any warnings
    scatter_points = [
        c
        for c in ax.collections
        if isinstance(c, plt.matplotlib.collections.PathCollection)
    ]
    assert len(scatter_points) > 0  # Should have scatter points
    # All points should have the same color
    unique_colors = set(tuple(color) for color in scatter_points[0].get_facecolors())
    assert len(unique_colors) == 1

    plt.close(fig)


def test_plot_distribution_over_time_missing_point_hue(time_series_data):
    """Test that plot_distribution_over_time handles missing point_hue column correctly."""
    from mlarena.utils.plot_utils import plot_distribution_over_time

    # Given: Data without the specified point_hue column
    data = time_series_data.drop(columns=["group"])

    # When: We try to plot with a non-existent point_hue
    with pytest.warns(
        UserWarning, match="point_hue column 'non_existent_column' not found"
    ) as warn_record:
        fig, ax = plot_distribution_over_time(
            data=data, x="date", y="values", freq="MS", point_hue="non_existent_column"
        )

    # Then: Should raise a warning and create plot without point_hue
    assert len(warn_record) == 1  # Should raise exactly one warning
    assert "Proceeding with plot without point_hue coloring" in str(
        warn_record[0].message
    )

    # Verify the plot was created with default styling (no point_hue coloring)
    scatter_points = [
        c
        for c in ax.collections
        if isinstance(c, plt.matplotlib.collections.PathCollection)
    ]
    assert len(scatter_points) > 0  # Should have scatter points

    # All points should have the same color (no point_hue coloring)
    unique_colors = set(tuple(color) for color in scatter_points[0].get_facecolors())
    assert len(unique_colors) == 1

    plt.close(fig)


def test_plot_distribution_over_time_without_point_hue(time_series_data):
    """Test that plot_distribution_over_time works correctly when point_hue is not specified."""
    from mlarena.utils.plot_utils import plot_distribution_over_time

    # When: We plot without specifying point_hue
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Convert warnings to errors
        fig, ax = plot_distribution_over_time(
            data=time_series_data, x="date", y="values", freq="MS"
        )

    # Then: Should create plot without any warnings
    scatter_points = [
        c
        for c in ax.collections
        if isinstance(c, plt.matplotlib.collections.PathCollection)
    ]
    assert len(scatter_points) > 0  # Should have scatter points

    # All points should have the same color
    unique_colors = set(tuple(color) for color in scatter_points[0].get_facecolors())
    assert len(unique_colors) == 1

    plt.close(fig)


def test_plot_distribution_over_time_with_valid_point_hue(time_series_data):
    """Test that plot_distribution_over_time works correctly with valid point_hue column."""
    from mlarena.utils.plot_utils import plot_distribution_over_time

    # When: We plot with a valid point_hue column
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Convert warnings to errors
        fig, ax = plot_distribution_over_time(
            data=time_series_data, x="date", y="values", freq="MS", point_hue="group"
        )

    # Then: Should create plot with point_hue coloring
    scatter_points = [
        c
        for c in ax.collections
        if isinstance(c, plt.matplotlib.collections.PathCollection)
    ]
    assert len(scatter_points) > 0  # Should have scatter points

    # Get all unique colors across all scatter collections
    all_colors = set()
    for points in scatter_points:
        colors = points.get_facecolors()
        all_colors.update(tuple(color) for color in colors)

    # Should have two unique colors for 'A' and 'B'
    assert len(all_colors) == 2

    plt.close(fig)


@pytest.fixture
def unequal_variance_data():
    """Create sample data with unequal variances for Welch ANOVA testing."""
    np.random.seed(42)

    # Group A: small variance
    group_a = np.random.normal(loc=10, scale=1, size=20)

    # Group B: medium variance
    group_b = np.random.normal(loc=12, scale=3, size=15)

    # Group C: large variance
    group_c = np.random.normal(loc=8, scale=5, size=25)

    data = pd.DataFrame(
        {
            "category": ["A"] * 20 + ["B"] * 15 + ["C"] * 25,
            "values": np.concatenate([group_a, group_b, group_c]),
        }
    )
    return data


def test_plot_box_scatter_welch_anova_basic(unequal_variance_data):
    """Test basic Welch ANOVA functionality in plot_box_scatter."""
    # When: We perform Welch ANOVA
    fig, ax = plot_box_scatter(
        data=unequal_variance_data, x="category", y="values", stat_test="welch"
    )

    # Then: Plot should be created successfully
    assert fig is not None
    assert ax is not None

    # Should have annotation on the plot (since show_stat_test defaults to True when stat_test is specified)
    annotations = [child for child in ax.get_children() if hasattr(child, "get_text")]
    annotation_texts = [
        ann.get_text() for ann in annotations if hasattr(ann, "get_text")
    ]
    welch_annotations = [text for text in annotation_texts if "Welch's ANOVA" in text]
    assert len(welch_annotations) > 0

    plt.close(fig)


def test_plot_box_scatter_welch_anova_return_stats(unequal_variance_data):
    """Test Welch ANOVA with return_stats=True."""
    # When: We perform Welch ANOVA and request stats
    fig, ax, results = plot_box_scatter(
        data=unequal_variance_data,
        x="category",
        y="values",
        stat_test="welch",
        return_stats=True,
    )

    # Then: Should return results with proper structure
    assert "summary_table" in results
    assert "stat_test" in results

    # Check summary table structure
    summary_df = results["summary_table"]
    assert isinstance(summary_df, pd.DataFrame)
    assert set(summary_df.columns) == {"category", "n", "mean", "median", "sd"}
    assert len(summary_df) == 3  # Three groups

    # Check statistical test results
    stat_test_results = results["stat_test"]
    assert stat_test_results["method"] == "welch"
    assert "statistic" in stat_test_results
    assert "p_value" in stat_test_results
    assert "effect_size" in stat_test_results

    # Statistical values should be numeric
    assert isinstance(stat_test_results["statistic"], (int, float))
    assert isinstance(stat_test_results["p_value"], (int, float))
    assert isinstance(stat_test_results["effect_size"], (int, float))

    # P-value should be between 0 and 1
    assert 0 <= stat_test_results["p_value"] <= 1

    plt.close(fig)


def test_plot_box_scatter_welch_anova_stats_only(unequal_variance_data):
    """Test Welch ANOVA with stats_only=True."""
    # When: We request only statistics without plotting
    results = plot_box_scatter(
        data=unequal_variance_data,
        x="category",
        y="values",
        stat_test="welch",
        stats_only=True,
    )

    # Then: Should return only results dictionary
    assert isinstance(results, dict)
    assert "summary_table" in results
    assert "stat_test" in results

    # Check statistical test results
    stat_test_results = results["stat_test"]
    assert stat_test_results["method"] == "welch"
    assert "statistic" in stat_test_results
    assert "p_value" in stat_test_results
    assert "effect_size" in stat_test_results


def test_plot_box_scatter_welch_anova_no_show_stat_test(unequal_variance_data):
    """Test Welch ANOVA with show_stat_test=False."""
    # When: We perform Welch ANOVA but don't want to show the test on plot
    fig, ax = plot_box_scatter(
        data=unequal_variance_data,
        x="category",
        y="values",
        stat_test="welch",
        show_stat_test=False,
    )

    # Then: Plot should be created without statistical annotation
    annotations = [child for child in ax.get_children() if hasattr(child, "get_text")]
    annotation_texts = [
        ann.get_text() for ann in annotations if hasattr(ann, "get_text")
    ]
    welch_annotations = [text for text in annotation_texts if "Welch's ANOVA" in text]
    assert len(welch_annotations) == 0

    plt.close(fig)


def test_plot_box_scatter_welch_anova_zero_variance_warning():
    """Test Welch ANOVA behavior with zero variance groups."""
    # Given: Data with one group having zero variance
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "category": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "values": [5.0] * 10
            + list(np.random.normal(10, 2, 10))
            + list(np.random.normal(15, 3, 10)),
        }
    )

    # When: We perform Welch ANOVA with zero variance group
    # Note: Zero variance groups will cause NaN results, which is mathematically correct
    # Suppress expected RuntimeWarnings from statsmodels division by zero
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in divide",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )

        fig, ax, results = plot_box_scatter(
            data=data, x="category", y="values", stat_test="welch", return_stats=True
        )

    # Then: Should still compute results (though they may be NaN due to zero variance)
    assert "stat_test" in results
    assert results["stat_test"]["method"] == "welch"

    # With zero variance groups, NaN results are mathematically correct
    # This is expected behavior, not an error
    stat_result = results["stat_test"]
    # We just verify that the statistical test was attempted and returned some result
    assert "statistic" in stat_result
    assert "p_value" in stat_result
    assert "effect_size" in stat_result

    plt.close(fig)


def test_plot_box_scatter_welch_vs_anova_different_results(unequal_variance_data):
    """Test that Welch ANOVA gives different results than regular ANOVA for unequal variances."""
    # When: We perform both regular ANOVA and Welch ANOVA
    anova_results = plot_box_scatter(
        data=unequal_variance_data,
        x="category",
        y="values",
        stat_test="anova",
        stats_only=True,
    )

    welch_results = plot_box_scatter(
        data=unequal_variance_data,
        x="category",
        y="values",
        stat_test="welch",
        stats_only=True,
    )

    # Then: Results should be different (due to unequal variances)
    anova_stat = anova_results["stat_test"]["statistic"]
    welch_stat = welch_results["stat_test"]["statistic"]

    anova_p = anova_results["stat_test"]["p_value"]
    welch_p = welch_results["stat_test"]["p_value"]

    # Statistics should be different (allowing for small numerical differences)
    assert abs(anova_stat - welch_stat) > 0.001
    assert abs(anova_p - welch_p) > 0.001


def test_plot_box_scatter_welch_anova_effect_size_bounds(unequal_variance_data):
    """Test that Welch ANOVA effect size is within reasonable bounds."""
    # When: We perform Welch ANOVA
    results = plot_box_scatter(
        data=unequal_variance_data,
        x="category",
        y="values",
        stat_test="welch",
        stats_only=True,
    )

    # Then: Effect size should be between 0 and 1 for omega squared approximation
    effect_size = results["stat_test"]["effect_size"]
    assert 0 <= effect_size <= 1


def test_plot_box_scatter_welch_anova_annotation_content(unequal_variance_data):
    """Test that Welch ANOVA annotation contains correct content."""
    # When: We perform Welch ANOVA with annotation
    fig, ax = plot_box_scatter(
        data=unequal_variance_data,
        x="category",
        y="values",
        stat_test="welch",
        show_stat_test=True,
    )

    # Then: Annotation should contain Welch-specific text
    annotations = [child for child in ax.get_children() if hasattr(child, "get_text")]
    annotation_texts = [
        ann.get_text() for ann in annotations if hasattr(ann, "get_text")
    ]
    welch_annotations = [text for text in annotation_texts if "Welch's ANOVA" in text]

    assert len(welch_annotations) > 0

    # Check that the annotation contains expected elements
    annotation_text = welch_annotations[0]
    assert "Welch's ANOVA" in annotation_text
    assert "ω²" in annotation_text  # Omega squared symbol
    assert "p" in annotation_text.lower()

    plt.close(fig)
