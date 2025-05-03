from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "plot_box_scatter",
    "plot_medical_timeseries",
    "plot_stacked_bar_over_time",
    "plot_distribution_over_time",
]


def plot_box_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str = "Box Plot with Scatter Overlay",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    point_hue: Optional[str] = None,
    point_size: int = 50,
    point_alpha: float = 0.8,
    jitter: float = 0.08,
    box_alpha: float = 0.3,
    single_color_box: bool = False,
    figsize: tuple = (10, 6),
    palette: Optional[List[str]] = None,
    return_summary: bool = False,
):
    """
    Draws a box plot with optional scatter overlay and customizable coloring behavior.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    x : str
        Column name for categorical items.
    y : str
        Column name for numerical values.
    title : str, default="Box Plot with Scatter Overlay"
        Title of the plot.
    xlabel : str, optional
        Label for x-axis. If None, uses the x column name.
    ylabel : str, optional
        Label for y-axis. If None, uses the y column name.
    point_hue : str, optional
        Column name to color points by. If set, overrides color-by-x behavior.
    point_size : int, default=50
        Size of the overlaid scatter points.
    point_alpha : float, default=0.8
        Transparency level for points.
    jitter : float, default=0.08
        Amount of horizontal jitter for points.
    box_alpha : float, default=0.3
        Transparency level for box fill.
    single_color_box : bool, default=False
        Whether to use a single color for all boxes and points (if point_hue is None).
    figsize : tuple, default=(10, 6)
        Size of the figure as (width, height) in inches.
    palette : List[str], optional
        List of colors. If None, uses Matplotlib's default color cycle.
    return_summary : bool, default=False
        Whether to return a DataFrame of summary stats.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for further customization.
    ax : matplotlib.axes.Axes
        The axes object for further customization.
    summary_df : pd.DataFrame, optional
        DataFrame with count, mean, median, std per category if return_summary=True.
    """

    fig, ax = plt.subplots(figsize=figsize)

    categories = sorted(data[x].unique())
    num_categories = len(categories)
    data_per_category = [data[data[x] == cat][y].values for cat in categories]

    # Determine color palette
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if point_hue:
        # Boxes are transparent with black outlines
        box_colors = ["white"] * num_categories
        edge_colors = ["black"] * num_categories

        hue_levels = sorted(data[point_hue].dropna().unique())
        hue_colors = (
            palette if palette is not None else default_colors[: len(hue_levels)]
        )
        hue_color_map = dict(zip(hue_levels, hue_colors))
    elif single_color_box:
        color = (
            palette[0]
            if (palette and isinstance(palette, list))
            else (palette or default_colors[0])
        )
        box_colors = [mcolors.to_rgba(color, alpha=box_alpha)] * num_categories
        edge_colors = [color] * num_categories
    else:
        # Default: color by x, ensure boxes are semi-transparent
        box_colors = [
            mcolors.to_rgba(c, alpha=box_alpha)
            for c in (palette or default_colors[:num_categories])
        ]
        edge_colors = [c for c in (palette or default_colors[:num_categories])]

    # Boxplot
    bp = ax.boxplot(
        data_per_category,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1),
        medianprops=dict(color="black", linewidth=1),
    )

    for patch, face_color, edge_color in zip(bp["boxes"], box_colors, edge_colors):
        patch.set_facecolor(face_color)
        patch.set_edgecolor(edge_color)

    # Scatter overlay
    for idx, cat in enumerate(categories):
        y_values = data[data[x] == cat][y].values
        x_jittered = np.random.normal(loc=idx + 1, scale=jitter, size=len(y_values))

        if point_hue:
            hue_vals = data[data[x] == cat][point_hue].values
            for xv, yv, hv in zip(x_jittered, y_values, hue_vals):
                if pd.isna(hv):
                    continue
                ax.scatter(
                    xv,
                    yv,
                    color=hue_color_map.get(hv, "grey"),
                    s=point_size,
                    alpha=point_alpha,
                    edgecolor="none",
                    label=hv if hv not in ax.get_legend_handles_labels()[1] else None,
                    zorder=3,
                )
        else:
            ax.scatter(
                x_jittered,
                y_values,
                color=edge_colors[idx],
                s=point_size,
                alpha=point_alpha,
                edgecolor="none",
            )

    # Legend only for point_hue
    if point_hue:
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicates while preserving order
        seen = set()
        filtered = [
            (h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))
        ]
        if filtered:
            ax.legend(*zip(*filtered), title=point_hue, loc="best")
        ax.legend(
            bbox_to_anchor=(1.02, 1),  # (x, y) position outside the axes
            loc="upper left",  # anchor point of the legend box
            borderaxespad=0.0,  # padding between axes and legend box
        )

    # Axis labels and title
    ax.set_xticks(range(1, num_categories + 1))
    ax.set_xticklabels(categories)
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
    x: str,
    metrics: Dict[str, Dict[str, str]],
    treatment_dates: Optional[Dict[str, List[str]]] = None,
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
        data (pd.DataFrame): DataFrame containing the time series data
        x (str): Name of the datetime column
        metrics (Dict[str, Dict[str, str]]): Dictionary of metrics to plot, each with values and color (optional)
                e.g., {'Iron': {'values': 'iron', 'color': 'blue'},
                       'Ferritin': {'values': 'ferritin', 'color': 'red'}}
        treatment_dates (Dict[str, List[str]], optional): Dictionary of treatment dates
                       e.g., {'Iron Infusion': ['2022-09-01', '2024-03-28']}
        title (str): Plot title. Default is "Medical Time Series with Treatments"
        xlabel (str, optional): Label for x-axis. If None, uses "Date".
        ylabel (str, optional): Label for y-axis. If None, uses metric names.
        figsize (tuple): Figure size as (width, height) in inches. Default is (12, 6).
        show_minmax (bool): Whether to show min/max annotations. Default is True.
        alternate_years (bool): Whether to show alternating year backgrounds. Default is True.

    Returns:
        Tuple[plt.Figure, List[plt.Axes]]:
            - Figure object for further customization
            - List of Axes objects (1-2 axes depending on number of metrics)
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
    data[x] = pd.to_datetime(data[x])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    axes = [ax]

    # Create additional y-axes if needed
    for i in range(len(metrics) - 1):
        axes.append(ax.twinx())
        axes[-1].spines["right"].set_position(("outward", 60 * i))

    # Add alternating year backgrounds if requested
    if alternate_years:
        start_year = data[x].min().year
        end_year = data[x].max().year
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
        ax.plot(data[x], values, "o-", color=color, label=metric_name)
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
                point_date = data[x][idx]
                point_value = values[idx]

                # Default offsets
                x_offset = 5
                y_offset = -5 if label == "Max" else 5

                # Check proximity to other metric's points
                for other_metric, other_info in metrics.items():
                    if other_metric != metric_name:
                        other_values = data[other_info["values"]]
                        date_diff = abs((point_date - data[x]).dt.total_seconds())
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
                    xy=(data[x][idx], values[idx]),
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
    date_min = data[x].min() - pd.Timedelta(days=30)
    date_max = data[x].max() + pd.Timedelta(days=30)
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


def _get_date_format_for_freq(freq: str) -> str:
    """Helper function to get date format string based on frequency.

    Parameters:
        freq (str): Frequency identifier ('h', 'D', 'MS', 'ME', 'YS', 'YE', etc.)

    Returns:
        str: Date format string suitable for the specified frequency
    """
    if freq == "h":
        return "%Y-%m-%d %H:00"
    elif freq == "D":
        return "%Y-%m-%d"
    elif freq in ["ME", "MS"]:
        return "%Y-%m"
    elif freq in ["YE", "YS"]:
        return "%Y"
    else:  # other frequencies
        return "%Y-%m-%d %H:%M"


def _get_label_for_freq(freq: str) -> str:
    """Helper function to get default axis label based on frequency.

    Parameters:
        freq (str): Frequency identifier ('h', 'D', 'MS', 'ME', 'YS', 'YE', etc.)

    Returns:
        str: Appropriate axis label for the specified frequency
    """
    if freq == "h":
        return "Hour"
    elif freq == "D":
        return "Date"
    elif freq in ["ME", "MS"]:
        return "Month"
    elif freq in ["YE", "YS"]:
        return "Year"
    else:
        return "Time"


def plot_stacked_bar_over_time(
    data: pd.DataFrame,
    x: str,
    y: str,
    freq: str = "MS",  # 'm'=minute, 'h'=hour, 'D'=day, 'MS'=month start, 'ME' = month end, 'YS'=year start
    label_dict: Optional[Dict[str, str]] = None,
    is_pct: bool = True,
    title: str = "Time Series Stacked Bar Chart",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (12, 6),
    color_palette: Optional[List[str]] = None,
):
    """
    Plot a stacked bar chart showing the distribution of a categorical variable over time,
    either in percentage or actual counts.

    Parameters:
        data (pd.DataFrame): Input DataFrame.
        x (str): Name of the datetime column.
        y (str): Name of the categorical column.
        freq (str): Frequency for time grouping ('m'=minute, 'h'=hour, 'D'=day, 'MS'=month start, 'ME' = month end, 'YS'=year start).
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
    x_label = xlabel or _get_label_for_freq(freq)

    # Apply label mapping if provided
    if label_dict:
        data_to_plot.rename(columns=label_dict, inplace=True)

    # Format x-axis labels based on frequency
    date_format = _get_date_format_for_freq(freq)
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


def plot_distribution_over_time(
    data: pd.DataFrame,
    x: str,
    y: str,
    freq: str = "MS",  # 'm'=minute, 'h'=hour, 'D'=day, 'MS'=month start, 'ME' = month end, 'YS'=year start
    point_hue: Optional[str] = None,
    title: str = "Distribution Over Time",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (12, 6),
    box_alpha: float = 0.3,
    point_size: int = 50,
    point_alpha: float = 0.8,
    jitter: float = 0.08,
    return_summary: bool = False,
):
    """
    Plot the distribution of a continuous variable over time, showing box plots with scatter overlay.
    Users can optionally color the points with point_hue, otherwise all the boxes and points will be in one color.

    Parameters:
        data (pd.DataFrame): Input DataFrame.
        x (str): Name of the datetime column.
        y (str): Name of the continuous column.
        freq (str): Frequency for time grouping ('m'=minute, 'h'=hour, 'D'=day, 'MS'=month start, 'ME' = month end, 'YS'=year start).
        point_hue (str, optional): Column name to use for coloring the scatter points. If provided, points will be colored
                                  according to this variable.
        title (str): Title of the plot.
        xlabel (str, optional): Label for the x-axis. If None, will be set based on frequency.
        ylabel (str, optional): Label for the y-axis. If None, uses the y column name.
        figsize (tuple): Figure size as (width, height) in inches. Default is (12, 6).
        box_alpha (float): Transparency level for box fill (default 0.3).
        point_size (int): Size of the overlaid scatter points (default 50).
        point_alpha (float): Transparency level for points (default 0.8).
        jitter (float): Amount of horizontal jitter for points (default 0.08).
        return_summary (bool): Whether to return a DataFrame of summary statistics (default False).

    Returns:
        Tuple[plt.Figure, plt.Axes] or Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
            - Figure and axis objects for further customization
            - (Optional) DataFrame with count, mean, median, std per time period if return_summary=True
    """
    # Convert x column to datetime
    df = data.copy()
    df[x] = pd.to_datetime(df[x])

    # Create a DataFrame with time periods as index
    if point_hue is not None and point_hue in df.columns:
        period_df = pd.DataFrame(
            {y: df[y].values, point_hue: df[point_hue].values}, index=df[x]
        )
    else:
        period_df = pd.DataFrame({y: df[y].values}, index=df[x])

    # Get date format for the specified frequency
    date_format = _get_date_format_for_freq(freq)

    # Group by time period
    grouped = period_df.groupby(pd.Grouper(freq=freq))

    # Create a new DataFrame for plotting
    plot_data = []
    time_periods = []

    # Process each time group and collect time periods
    for name, group in grouped:
        if not group.empty:
            time_periods.append(name)
            formatted_name = name.strftime(date_format)
            if point_hue is not None and point_hue in group.columns:
                for val, hue_val in zip(group[y], group[point_hue]):
                    plot_data.append(
                        {"time_period": formatted_name, y: val, point_hue: hue_val}
                    )
            else:
                for val in group[y]:
                    plot_data.append({"time_period": formatted_name, y: val})

    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)

    # Sort time periods chronologically
    sorted_time_periods = sorted(time_periods)
    formatted_sorted_periods = [
        period.strftime(date_format) for period in sorted_time_periods
    ]

    # Create a categorical type with the correct order
    plot_df["time_period"] = pd.Categorical(
        plot_df["time_period"], categories=formatted_sorted_periods, ordered=True
    )

    # Sort the dataframe
    plot_df = plot_df.sort_values("time_period")

    # Set default xlabel based on frequency
    x_label = xlabel or _get_label_for_freq(freq)

    # Use boxplot_scatter_overlay for visualization
    if return_summary:
        fig, ax, summary_df = plot_box_scatter(
            data=plot_df,
            x="time_period",
            y=y,
            point_hue=point_hue if point_hue in plot_df.columns else None,
            title=title,
            xlabel=x_label,
            ylabel=ylabel,
            box_alpha=box_alpha,
            point_size=point_size,
            point_alpha=point_alpha,
            jitter=jitter,
            figsize=figsize,
            return_summary=True,
            single_color_box=True,
        )
        ax.tick_params(axis="x", labelrotation=90)
        return fig, ax, summary_df
    else:
        fig, ax = plot_box_scatter(
            data=plot_df,
            x="time_period",
            y=y,
            point_hue=point_hue if point_hue in plot_df.columns else None,
            title=title,
            xlabel=x_label,
            ylabel=ylabel,
            box_alpha=box_alpha,
            point_size=point_size,
            point_alpha=point_alpha,
            jitter=jitter,
            figsize=figsize,
            single_color_box=True,
        )
        ax.tick_params(axis="x", labelrotation=90)
        return fig, ax
