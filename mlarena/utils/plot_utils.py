from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "boxplot_scatter_overlay",
    "plot_medical_timeseries",
    "plot_stacked_bar_over_time",
]


def boxplot_scatter_overlay(
    data,
    x,
    y,
    title: str = "Box Plot with Scatter Overlay",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    box_alpha=0.3,
    dot_size=50,
    dot_alpha=0.8,
    jitter=0.08,
    figsize=(10, 6),
    palette=None,
    return_summary=False,
):
    """
    Draws a box plot with semi-transparent boxes and overlays colored dots matching the box colors.

    Parameters:
    - data: pandas DataFrame containing the data.
    - x: str, the column name for categorical items.
    - y: str, the column name for numerical values.
    - title: str, the title of the plot. Default is "Box Plot with Scatter Overlay".
    - xlabel: str, optional, label for x-axis. If None, uses the x column name.
    - ylabel: str, optional, label for y-axis. If None, uses the y column name.
    - box_alpha: float, transparency level for box fill (default 0.3).
    - dot_size: int, size of the overlaid dots (default 50).
    - jitter: float, amount of horizontal jitter for dots (default 0.08).
    - figsize: tuple, size of the figure (default (10, 6)).
    - palette: list of colors or None. If None, uses Matplotlib's default color cycle.
    - return_summary: bool, whether to return a DataFrame of summary stats (default False).

    Returns:
    - fig, ax: The figure and axis objects for further customization.
    - (Optional) summary_df: DataFrame with count, mean, median, std per category if return_summary=True.
    """
    # Prepare data
    categories = sorted(data[x].unique())
    num_categories = len(categories)
    data_per_category = [data[data[x] == cat][y].values for cat in categories]

    # Define color palette
    if palette is None:
        # Use Matplotlib's default color cycle with 10 distinct colors at most
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = color_cycle[:num_categories]
    else:
        colors = palette[:num_categories]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot boxplots
    boxprops = dict(linewidth=1)
    medianprops = dict(color="black", linewidth=1)
    bp = ax.boxplot(
        data_per_category,
        patch_artist=True,
        showfliers=False,
        boxprops=boxprops,
        medianprops=medianprops,
    )

    # Set box colors and transparency
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(mcolors.to_rgba(color, alpha=box_alpha))

    # Overlay scatter dots
    for idx, (cat, y_values) in enumerate(zip(categories, data_per_category)):
        x_jittered = np.random.normal(loc=idx + 1, scale=jitter, size=len(y_values))
        ax.scatter(
            x_jittered,
            y_values,
            color=colors[idx],
            s=dot_size,
            alpha=dot_alpha,
            edgecolor="none",
        )

    # Customize axes
    ax.set_xticks(range(1, num_categories + 1))
    ax.set_xticklabels(categories, rotation=45)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()

    if return_summary:
        summary_df = (
            data.groupby(x)[y]
            .agg(n="count", mean="mean", median="median", sd="std")
            .reset_index()
        )
        return fig, ax, summary_df
    else:
        return fig, ax


def plot_medical_timeseries(
    data: pd.DataFrame,
    date_col: str,
    metrics: dict,
    treatment_dates: dict = None,
    title: str = "Medical Time Series with Treatments",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (12, 6),
    show_minmax: bool = True,
    alternate_years: bool = True,
):
    """
    Plot 1-2 medical metrics over time with treatments and annotations.

    Parameters:
        data: DataFrame containing the time series data
        date_col: Name of the date column
        metrics: Dictionary of metrics to plot, each with values and color (optional)
                e.g., {'Iron': {'values': 'iron', 'color': 'blue'},
                       'Ferritin': {'values': 'ferritin', 'color': 'red'}}
        treatment_dates: Dictionary of treatment dates
                       e.g., {'Iron Infusion': ['2022-09-01', '2024-03-28']}
        title: Plot title. Default is "Medical Time Series with Treatments"
        xlabel: str, optional, label for x-axis. If None, uses "Date".
        ylabel: str, optional, label for y-axis. If None, uses metric names.
        figsize: Figure size as (width, height)
        show_minmax: Whether to show min/max annotations
        alternate_years: Whether to show alternating year backgrounds
    """

    # Validate and set default colors for metrics (max 2 supported)
    if len(metrics) > 2:
        raise ValueError("This function supports plotting of up to 2 metrics only")
    default_colors = ["#000000", "#FF0000"]  # black, red
    for (metric_name, metric_info), default_color in zip(
        metrics.items(), default_colors
    ):
        if "color" not in metric_info:
            metric_info["color"] = default_color

    # Convert dates if needed
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    axes = [ax]

    # Create additional y-axes if needed
    for i in range(len(metrics) - 1):
        axes.append(ax.twinx())
        axes[-1].spines["right"].set_position(("outward", 60 * i))

    # Add alternating year backgrounds if requested
    if alternate_years:
        start_year = data[date_col].min().year
        end_year = data[date_col].max().year
        for year in range(start_year, end_year + 1):
            if year % 2 == 0:
                start = pd.Timestamp(f"{year}-01-01")
                end = pd.Timestamp(f"{year + 1}-01-01")
                ax.axvspan(start, end, color="gray", alpha=0.1)

    # Plot each metric
    for (metric_name, metric_info), ax in zip(metrics.items(), axes):
        values = data[metric_info["values"]]
        color = metric_info["color"]

        # Plot the metric (corrected line)
        ax.plot(data[date_col], values, "o-", color=color, label=metric_name)
        ax.set_ylabel(metric_name, color=color)
        ax.tick_params(axis="y", labelcolor=color)

        # Add min/max annotations if requested
        if show_minmax:
            min_idx = values.idxmin()
            max_idx = values.idxmax()

            # Calculate vertical offsets based on relative position
            # If points are close, stack annotations vertically
            for idx, label in [(min_idx, "Min"), (max_idx, "Max")]:
                # Check if this point is close to any previous annotations
                point_date = data[date_col][idx]
                point_value = values[idx]

                # Default offsets
                x_offset = 5
                y_offset = -5 if label == "Max" else 5

                # Check proximity to other metric's points
                for other_metric, other_info in metrics.items():
                    if other_metric != metric_name:
                        other_values = data[other_info["values"]]
                        date_diff = abs(
                            (point_date - data[date_col]).dt.total_seconds()
                        )
                        closest_idx = date_diff.idxmin()

                        # If points are close in time, adjust vertical position
                        if (
                            date_diff[closest_idx]
                            < pd.Timedelta(days=60).total_seconds()
                        ):
                            if point_value > other_values[closest_idx]:
                                y_offset += 10  # Move annotation higher
                            else:
                                y_offset += -10  # Move annotation lower

                ax.annotate(
                    f"{label} {metric_name}: {values[idx]}",
                    xy=(data[date_col][idx], values[idx]),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    color=color,
                    fontsize=8,
                )

    # Add treatment markers if provided
    if treatment_dates:
        for treatment, dates in treatment_dates.items():
            dates = pd.to_datetime(dates)
            for i, date in enumerate(dates):
                ax.axvline(x=date, color="green", linestyle="--", alpha=0.7)
                ax.annotate(
                    f"{treatment} {i + 1}",
                    xy=(date, 0),
                    xytext=(date, ax.get_ylim()[1] * 0.1),
                    rotation=90,
                    color="green",
                )

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Set x-axis range with padding
    date_min = data[date_col].min() - pd.Timedelta(days=30)
    date_max = data[date_col].max() + pd.Timedelta(days=30)
    ax.set_xlim([date_min, date_max])
    for axis in axes:
        axis.grid(True, axis="x")

    # Add title and labels
    if title:
        plt.title(title)
    ax.set_xlabel(xlabel or "Date")

    # Handle ylabels for multiple metrics
    if len(metrics) == 1:
        ax.set_ylabel(ylabel or list(metrics.keys())[0])
    else:
        # For multiple metrics, use their names as labels
        for axis, (metric_name, _) in zip(axes, metrics.items()):
            axis.set_ylabel(metric_name)

    # Adjust layout
    fig.autofmt_xdate(rotation=45, ha="right")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    ax.grid(True, axis="x", zorder=10)

    return fig, axes


def plot_stacked_bar_over_time(
    data: pd.DataFrame,
    x: str,
    y: str,
    freq: str = "ME",  # 'm'=minute, 'h'=hour, 'D'=day, 'ME'=month end, 'YE'=year end
    label_dict: Optional[Dict[str, str]] = None,
    is_pct: bool = True,
    title: str = "Time Series Stacked Bar Chart",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (12, 6),
    color_palette: Optional[List[str]] = None,
) -> None:
    """
    Plot a stacked bar chart showing the distribution of a categorical variable over time,
    either in percentage or actual counts.

    Parameters:
        data (pd.DataFrame): Input DataFrame.
        x (str): Name of the datetime column.
        y (str): Name of the categorical column.
        freq (str): Frequency for time grouping ('m'=minute, 'h'=hour, 'D'=day, 'ME'=month end, 'YE'=year end).
        label_dict (Dict[str, str], optional): Mapping of original category values to display labels.
        is_pct (bool): Whether to display percentage (True) or actual count (False).
        title (str): Title of the plot.
        xlabel (str, optional): Label for the x-axis. If None, will be set based on frequency.
        ylabel (str, optional): Label for the y-axis (default is auto-set based on is_pct).
        figsize (tuple): Figure size as (width, height) in inches. Default is (12, 6).
        color_palette (List[str], optional): List of colors for the bars.
    """

    # Use provided color palette or fallback to matplotlib's default color cycle
    num_categories = data[y].nunique()
    if label_dict:
        num_categories = len(label_dict)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = (
        color_palette if color_palette is not None else color_cycle[:num_categories]
    )

    # Convert x column to datetime and set as index for resampling
    df = data.copy()
    df[x] = pd.to_datetime(df[x])
    df = df.set_index(x)

    # Aggregate data with specified frequency
    class_agg = df.groupby([pd.Grouper(freq=freq), y]).size().unstack(fill_value=0)

    # Sort index for time order
    class_agg = class_agg.sort_index()

    # Compute percentage if requested
    if is_pct:
        data_to_plot = class_agg.div(class_agg.sum(axis=1), axis=0) * 100
        y_label = ylabel or "Percentage"
    else:
        data_to_plot = class_agg
        y_label = ylabel or "Count"

    # Set default xlabel based on frequency
    if xlabel is None:
        if freq == "h":
            x_label = "Hour"
        elif freq == "D":
            x_label = "Date"
        elif freq in ["ME", "MS"]:
            x_label = "Month"
        elif freq in ["YE", "YS"]:
            x_label = "Year"
        else:
            x_label = "Time"
    else:
        x_label = xlabel

    # Apply label mapping if provided
    if label_dict:
        data_to_plot.rename(columns=label_dict, inplace=True)

    # Format x-axis labels based on frequency
    if freq == "h":
        date_format = "%Y-%m-%d %H:00"
    elif freq == "D":
        date_format = "%Y-%m-%d"
    elif freq in ["ME", "MS"]:
        date_format = "%Y-%m"
    elif freq in ["YE", "YS"]:
        date_format = "%Y"
    else:  # other frequencies
        date_format = "%Y-%m-%d %H:%M"
    date_labels = data_to_plot.index.strftime(date_format)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    data_to_plot.plot(
        kind="bar", stacked=True, color=colors[: len(data_to_plot.columns)], ax=ax
    )

    ax.set_xticks(range(len(date_labels)))
    ax.set_xticklabels(date_labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title=y if not label_dict else "")
    ax.grid(True, axis="y")
    plt.tight_layout()

    return fig, ax
