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
