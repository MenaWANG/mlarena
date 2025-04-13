import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["boxplot_scatter_overlay"]


def boxplot_scatter_overlay(
    df,
    cat_col,
    value_col,
    title,
    box_alpha=0.3,
    dot_size=60,
    dot_alpha=0.8,
    jitter=0.08,
    figsize=(10, 6),
    palette=None,
):
    """
    Draws a box plot with semi-transparent boxes and overlays colored dots matching the box colors.

    Parameters:
    - df: pandas DataFrame containing the data.
    - item_col: str, the column name for categorical items.
    - value_col: str, the column name for numerical values.
    - title: str, the title of the plot.
    - box_alpha: float, transparency level for box fill (default 0.3).
    - dot_size: int, size of the overlaid dots (default 60).
    - jitter: float, amount of horizontal jitter for dots (default 0.08).
    - figsize: tuple, size of the figure (default (10, 6)).
    - palette: list of colors or None. If None, uses Matplotlib's default color cycle.

    Returns:
    - None (displays the plot).
    """
    # Prepare data
    categories = sorted(df[cat_col].unique())
    num_categories = len(categories)
    data_per_category = [df[df[cat_col] == cat][value_col].values for cat in categories]

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
    ax.set_xticklabels(categories, rotation=45, fontsize=12)
    ax.set_xlabel(cat_col, fontsize=14)
    ax.set_ylabel(value_col, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
