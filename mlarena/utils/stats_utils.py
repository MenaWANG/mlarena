import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import mlarena.utils.plot_utils as put

__all__ = [
    "compare_groups",
    "add_stratified_groups",
    "optimize_stratification_strategy",
    "calculate_threshold_stats",
    "calculate_group_thresholds",
]


def compare_groups(
    data: pd.DataFrame,
    grouping_col: str,
    target_cols: List[str],
    weights: Optional[Dict[str, float]] = None,
    num_test: str = "anova",
    cat_test: str = "chi2",
    alpha: float = 0.05,
    visualize: bool = False,
) -> Tuple[float, pd.DataFrame]:
    """
    Compare groups across specified target variables using statistical tests.

    Evaluates whether groups defined by grouping_col have equivalent distributions
    across target variables, useful for A/B testing and stratification validation.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    grouping_col : str
        Column used to divide groups. For A/B testing, should have two unique values.
    target_cols : List[str]
        List of column names to compare across the groups.
    weights : Optional[Dict[str, float]], optional
        Optional dictionary of weights for each target column.
    num_test : str, default="anova"
        Statistical test for numeric variables. Supported: "anova", "welch", "kruskal".
    cat_test : str, default="chi2"
        Statistical test for categorical variables.
    alpha : float, default=0.05
        Significance threshold for flagging imbalance.
    visualize : bool, default=False
        If True, generate plots for numeric and categorical variables.

    Returns
    -------
    effect_size_sum : float
        Weighted sum of effect sizes across all target variables.
    summary_df : pd.DataFrame
        Summary statistics and test results for each target variable.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'group': ['A', 'A', 'B', 'B', 'A', 'B'],
    ...     'metric1': [10, 12, 15, 13, 11, 14],
    ...     'metric2': [1.2, 1.5, 2.1, 1.8, 1.3, 2.0],
    ...     'category': ['X', 'Y', 'X', 'Y', 'X', 'Y']
    ... })
    >>> effect_size_sum, summary = compare_groups(
    ...     df, 'group', ['metric1', 'metric2', 'category']
    ... )
    """
    summary = []
    for col in target_cols:
        col_data = data[[grouping_col, col]].dropna()
        weight = weights[col] if weights and col in weights else 1.0
        if pd.api.types.is_numeric_dtype(col_data[col]):
            if visualize:
                fig, ax, results = put.plot_box_scatter(
                    data,
                    grouping_col,
                    col,
                    title=f"{col} across group",
                    stat_test=num_test,
                    show_stat_test=True,
                    return_stats=True,
                )
            else:
                results = put.plot_box_scatter(
                    data, grouping_col, col, stat_test=num_test, stats_only=True
                )
        else:
            if visualize:
                fig, ax, results = put.plot_stacked_bar(
                    data,
                    grouping_col,
                    col,
                    is_pct=False,
                    title=f"{col} across group",
                    stat_test=cat_test,
                    show_stat_test=True,
                    return_stats=True,
                )
            else:
                results = put.plot_stacked_bar(
                    data, grouping_col, col, stat_test=cat_test, stats_only=True
                )
        stat_result = results.get("stat_test", {})
        summary.append(
            {
                "grouping_col": grouping_col,
                "target_var": col,
                "stat_test": stat_result.get("method"),
                "p_value": stat_result.get("p_value"),
                "effect_size": stat_result.get("effect_size"),
                "is_significant": (
                    stat_result.get("p_value") < alpha
                    if stat_result.get("p_value") is not None
                    else None
                ),
                "weight": weight,
            }
        )

    summary_df = pd.DataFrame(summary)
    effect_size_sum = (summary_df["effect_size"] * summary_df["weight"]).sum()

    return effect_size_sum, summary_df


def add_stratified_groups(
    data: pd.DataFrame,
    stratifier_col: Union[str, List[str]],
    random_seed: int = 42,
    group_col_name: str = None,
    group_labels: Tuple[Union[str, int], Union[str, int]] = (0, 1),
) -> pd.DataFrame:
    """
    Add a column to stratify a DataFrame into two equal groups based on specified column(s).

    This function maintains the distribution of the stratifier column(s) across both groups,
    making it useful for creating balanced train/test splits or A/B testing groups.
    Use with compare_groups() to validate stratification effectiveness.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to be stratified.
    stratifier_col : Union[str, List[str]]
        The column name or list of column names to use as stratification factors.
        If a list is provided, the columns are combined for stratification.
    random_seed : int, default=42
        Random seed for reproducibility.
    group_col_name : str, optional
        Name for the new group column. If None, defaults to 'stratified_group'.
    group_labels : Tuple[Union[str, int], Union[str, int]], default=(0, 1)
        Labels for the two groups. First label for group 0, second for group 1.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with an additional column indicating group membership
        using the specified group_labels.

    Raises
    ------
    ValueError
        If stratifier_col contains column names that don't exist in the DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'category': ['A', 'B', 'A', 'B', 'A', 'C', 'C', 'C'],
    ...     'value_1': [10, 20, 30, 40, 50, 60, 70, 80],
    ...     'value_2': [15, 70, 37, 80, 90, 40, 70, 20],
    ... })
    >>> # Create stratified groups
    >>> result = add_stratified_groups(df, 'category')
    >>> # Validate stratification worked
    >>> from mlarena.utils.stats_utils import compare_groups
    >>> effect_size, summary = compare_groups(
    ...     result, 'stratified_group', ['value_1', 'value_2']
    ... )
    """
    # Validate columns exist
    cols_to_check = (
        [stratifier_col] if isinstance(stratifier_col, str) else stratifier_col
    )
    missing_cols = [col for col in cols_to_check if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Column(s) {missing_cols} not found in DataFrame")

    df = data.copy()

    # Handle single column or multiple columns
    if isinstance(stratifier_col, list):
        combined_col_name = "_".join(stratifier_col).lower()
        df[combined_col_name] = df[stratifier_col].astype(str).agg("_".join, axis=1)
        stratify_col = combined_col_name
        cleanup_temp_col = True
    else:
        stratify_col = stratifier_col
        cleanup_temp_col = False

    # Use provided name or default
    group_name = group_col_name or "stratified_group"

    try:
        # Perform stratified split
        train_df, test_df = train_test_split(
            df, test_size=0.5, stratify=df[stratify_col], random_state=random_seed
        )

        # Add group membership column with semantic labels
        df[group_name] = df.index.map(
            lambda x: group_labels[0] if x in train_df.index else group_labels[1]
        )

    except ValueError as e:
        # Handle cases where stratification fails (e.g., groups with only one member)
        stratifier_name = (
            str(stratifier_col)
            if isinstance(stratifier_col, str)
            else "_".join(stratifier_col)
        )
        warnings.warn(
            f"Stratifier '{stratifier_name}' failed: {e} Assigning all rows to {group_labels[0]}.",
            UserWarning,
        )
        df[group_name] = group_labels[0]

    # Clean up temporary combined column if created
    if cleanup_temp_col and combined_col_name in df.columns:
        df = df.drop(columns=[combined_col_name])

    return df


def optimize_stratification_strategy(
    data: pd.DataFrame,
    candidate_stratifiers: List[str],
    target_metrics: List[str],
    weights: Optional[Dict[str, float]] = None,
    max_combinations: int = 3,
    alpha: float = 0.05,
    significance_penalty: float = 0.2,
    num_test: str = "anova",
    cat_test: str = "chi2",
    visualize_best_strategy: bool = False,
    include_random_baseline: bool = True,
    random_seed: int = 42,
) -> Dict:
    """
    Find the best stratification strategy by testing different combinations of stratifier columns.

    Evaluates each candidate stratifier by creating stratified groups and measuring
    how well balanced the groups are across target metrics using compare_groups().
    Automatically generates combinations of candidate columns up to max_combinations.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    candidate_stratifiers : List[str]
        List of column names to test as stratifiers. Function will automatically
        generate combinations up to max_combinations length.
    target_metrics : List[str]
        List of target variables to evaluate balance across.
    weights : Optional[Dict[str, float]], optional
        Optional weights for target metrics in the comparison.
    max_combinations : int, default=3
        Maximum number of columns to combine when testing multi-column stratifiers.
    alpha : float, default=0.05
        Significance threshold for counting significant differences.
    significance_penalty : float, default=0.2
        Penalty weight applied per significant difference in composite scoring.
        Higher values more heavily penalize strategies with significant imbalances.
        Set to 0 to ignore significance count and use only effect sizes.
    num_test : str, default="anova"
        Statistical test for numeric variables. Supported: "anova", "welch", "kruskal".
    cat_test : str, default="chi2"
        Statistical test for categorical variables. Supported: "chi2", "g_test".
    visualize_best_strategy : bool, default=False
        If True, generates visualizations for the best stratification strategy only.
    include_random_baseline : bool, default=True
        If True, includes a random baseline strategy in the comparison.
        This creates a random 50/50 group assignment to serve as a baseline
        for evaluating whether stratification strategies perform better than chance.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary with results:
        - 'best_stratifier': The stratifier with best balance (lowest composite score)
        - 'results': Dict mapping each stratifier to its detailed metrics and summary DataFrame
        - 'rankings': List of stratifiers ranked by effectiveness (best to worst)
        - 'summary': DataFrame with overview of all tested strategies, ranked by performance

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'region': ['North', 'South', 'North', 'South'] * 50,
    ...     'segment': ['A', 'B', 'A', 'B'] * 50,
    ...     'metric1': np.random.normal(100, 15, 200),
    ...     'metric2': np.random.normal(50, 10, 200)
    ... })
    >>> results = optimize_stratification_strategy(
    ...     df, ['region', 'segment'], ['metric1', 'metric2']
    ... )
    >>> print(f"Best stratifier: {results['best_stratifier']}")
    >>> # View performance overview of all strategies
    >>> print(results['summary'])
    >>>
    >>> # Advanced analysis with custom penalty and tests
    >>> results_strict = optimize_stratification_strategy(
    ...     df, ['region', 'segment'], ['metric1', 'metric2'],
    ...     significance_penalty=0.5,  # Heavily penalize significant differences
    ...     num_test='kruskal',       # Use non-parametric test
    ...     visualize_best_strategy=True  # Generate plots for best strategy
    ... )
    >>> # Compare top 3 strategies
    >>> top_3 = results_strict['summary'].head(3)
    >>> print(top_3[['stratifier', 'composite_score', 'n_significant']])
    >>>
    >>> # Check if any stratifier beats random baseline
    >>> summary = results['summary']
    >>> random_score = summary[summary['stratifier'] == 'random_baseline']['composite_score'].iloc[0]
    >>> best_stratified_score = summary[summary['stratifier'] != 'random_baseline']['composite_score'].min()
    >>> improvement = (random_score - best_stratified_score) / random_score * 100
    >>> print(f"Best stratification improves over random by {improvement:.1f}%")
    """
    from itertools import combinations

    # Generate all possible combinations up to max_combinations
    all_stratifiers = []
    for r in range(1, min(max_combinations + 1, len(candidate_stratifiers) + 1)):
        for combo in combinations(candidate_stratifiers, r):
            all_stratifiers.append(list(combo) if len(combo) > 1 else combo[0])

    # Add random baseline if requested
    if include_random_baseline:
        all_stratifiers.append("random_baseline")

    results = {}

    for stratifier in all_stratifiers:
        try:
            # Handle random baseline differently
            if stratifier == "random_baseline":
                # Create random assignment baseline
                df_stratified = data.copy()
                np.random.seed(random_seed)
                group_col = "temp_group_random"
                df_stratified[group_col] = np.random.choice([0, 1], size=len(data))
            else:
                # Create stratified groups
                df_stratified = add_stratified_groups(
                    data,
                    stratifier,
                    random_seed=random_seed,
                    group_col_name=f"temp_group_{hash(str(stratifier)) % 10000}",
                )
                # Get the group column name
                group_col = f"temp_group_{hash(str(stratifier)) % 10000}"

            # Check if assignment actually worked (more than one unique group)
            unique_groups = df_stratified[group_col].nunique()
            if unique_groups < 2:
                # Skip evaluation silently since add_stratified_groups already warned
                continue

            # Evaluate balance
            effect_size_sum, summary_df = compare_groups(
                df_stratified,
                group_col,
                target_metrics,
                weights=weights,
                alpha=alpha,
                num_test=num_test,
                cat_test=cat_test,
                visualize=False,  # only the best strategy if requested
            )

            # Count significant differences
            n_significant = (
                summary_df["is_significant"].sum()
                if "is_significant" in summary_df.columns
                else 0
            )

            # Calculate composite score (effect size + penalty for significant differences)
            composite_score = effect_size_sum + (n_significant * significance_penalty)

            # Store results
            stratifier_key = (
                str(stratifier) if isinstance(stratifier, str) else "_".join(stratifier)
            )
            results[stratifier_key] = {
                "effect_size_sum": effect_size_sum,
                "n_significant": n_significant,
                "composite_score": composite_score,
                "summary": summary_df,
                "stratifier": stratifier,
            }

        except Exception as e:
            warnings.warn(
                f"Failed to evaluate stratifier {stratifier}: {e}", UserWarning
            )
            continue

    # Find best stratifier (lowest composite score)
    if results:
        best_key = min(results.keys(), key=lambda k: results[k]["composite_score"])
        best_stratifier = results[best_key]["stratifier"]

        # If requested, visualize the best strategy
        if visualize_best_strategy:
            # Re-run compare_groups with visualization for the best strategy
            df_best = add_stratified_groups(
                data,
                best_stratifier,
                random_seed=random_seed,
                group_col_name="best_strategy_group",
            )
            _, _ = compare_groups(
                df_best,
                "best_strategy_group",
                target_metrics,
                weights=weights,
                alpha=alpha,
                num_test=num_test,
                cat_test=cat_test,
                visualize=True,
            )

        # Create rankings by composite score
        rankings = sorted(results.keys(), key=lambda k: results[k]["composite_score"])

        # Create detailed summary DataFrame for analysis
        summary_data = []
        for i, key in enumerate(rankings):
            data = results[key]
            summary_data.append(
                {
                    "stratifier": key,
                    "effect_size_sum": data["effect_size_sum"],
                    "n_significant": data["n_significant"],
                    "composite_score": data["composite_score"],
                    "rank": i + 1,
                }
            )

        summary_df = pd.DataFrame(summary_data)

        return {
            "best_stratifier": best_stratifier,
            "results": results,
            "rankings": rankings,
            "summary": summary_df,
        }
    else:
        return {
            "best_stratifier": None,
            "results": {},
            "rankings": [],
            "summary": pd.DataFrame(),
        }


def calculate_threshold_stats(
    data: Union[pd.Series, np.ndarray, List[Union[int, float]]],
    n_std: float = 2.0,
    threshold_method: str = "std",
    visualize: bool = False,
) -> Dict[str, Union[float, int]]:
    """
    Calculate frequency statistics and threshold based on statistical criteria.

    This function computes basic statistics (mean, median, std, count) and
    determines a threshold based on the specified method. Useful for outlier
    detection and frequency analysis.

    Parameters
    ----------
    data : Union[pd.Series, np.ndarray, List[Union[int, float]]]
        Input data containing frequency or numeric values.
    n_std : float, default=2.0
        Number of standard deviations to use for threshold calculation
        when threshold_method is "std".
    threshold_method : str, default="std"
        Method to calculate threshold:
        - "std": mean + n_std * std
        - "iqr": Q3 + 1.5 * IQR (Interquartile Range)
        - "percentile": 95th percentile
    visualize : bool, default=False
        If True, creates a histogram with marked statistics.

    Returns
    -------
    Dict[str, Union[float, int]]
        Dictionary containing:
        - 'mean': mean of the data
        - 'median': median of the data
        - 'std': standard deviation
        - 'count': number of observations
        - 'threshold': calculated threshold value
        - 'method': threshold calculation method used

    Examples
    --------
    >>> data = [1, 2, 2, 3, 3, 3, 4, 4, 10]
    >>> stats = calculate_frequency_stats(data, n_std=2, visualize=True)
    >>> print(f"Mean: {stats['mean']:.2f}")
    >>> print(f"Threshold: {stats['threshold']:.2f}")

    >>> # Using different threshold method
    >>> stats_iqr = calculate_frequency_stats(
    ...     data, threshold_method='iqr', visualize=True
    ... )
    """
    # Convert input to numpy array
    if isinstance(data, pd.Series):
        values = data.values
    elif isinstance(data, list):
        values = np.array(data)
    else:
        values = data

    # Handle empty input explicitly
    if len(values) == 0:
        warnings.warn(
            "Empty input provided to calculate_threshold_stats. "
            "Returning NaN for all statistics.",
            UserWarning,
        )
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "threshold": np.nan,
            "count": 0,
            "method": threshold_method,
        }

    # Calculate basic statistics
    stats = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "count": int(len(values)),
        "method": threshold_method,
    }

    # Calculate threshold based on method
    if threshold_method == "std":
        stats["threshold"] = stats["mean"] + n_std * stats["std"]
    elif threshold_method == "iqr":
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        stats["threshold"] = q3 + 1.5 * iqr
    elif threshold_method == "percentile":
        stats["threshold"] = float(np.percentile(values, 95))
    else:
        raise ValueError(
            f"Invalid threshold_method: {threshold_method}. "
            "Must be one of: 'std', 'iqr', 'percentile'"
        )

    if visualize:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins="auto", alpha=0.7)
        plt.axvline(
            stats["mean"], color="r", linestyle="--", label=f"Mean: {stats['mean']:.2f}"
        )
        plt.axvline(
            stats["median"],
            color="g",
            linestyle="--",
            label=f"Median: {stats['median']:.2f}",
        )
        plt.axvline(
            stats["threshold"],
            color="b",
            linestyle="--",
            label=f"Threshold ({threshold_method}): {stats['threshold']:.2f}",
        )
        plt.title("Frequency Distribution with Statistics")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return stats


def calculate_group_thresholds(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    methods: List[str] = ["std", "iqr", "percentile"],
    n_std: float = 2.0,
    visualize_first_group: bool = True,
    min_group_size: int = 1,
) -> pd.DataFrame:
    """
    Calculate thresholds for values grouped by any categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data
    group_col : str
        Name of the column to group by
    value_col : str
        Name of the column containing the values to analyze
    methods : List[str], default=['std', 'iqr', 'percentile']
        List of threshold methods to use
    n_std : float, default=2.0
        Number of standard deviations to use for threshold calculation
        when threshold_method is "std".
    visualize_first_group : bool, default=True
        Whether to show visualizations for the first group

    Returns
    -------
    pd.DataFrame
        DataFrame containing threshold statistics for each group and method

    Examples
    --------
    >>> # Example with products and prices
    >>> df = pd.DataFrame({
    ...     'product': ['A', 'B', 'A', 'B'],
    ...     'price': [10, 20, 15, 25]
    ... })
    >>> results = calculate_group_thresholds(df, 'product', 'price')

    >>> # Example with locations and temperatures
    >>> weather_df = pd.DataFrame({
    ...     'location': ['NY', 'LA', 'NY', 'LA'],
    ...     'temperature': [75, 85, 72, 88]
    ... })
    >>> results = calculate_group_thresholds(weather_df, 'location', 'temperature')
    """
    if len(df) == 0:
        warnings.warn(
            "Empty DataFrame provided to calculate_group_thresholds. "
            "Returning empty DataFrame.",
            UserWarning,
        )
        return pd.DataFrame(
            columns=["group", "method", "mean", "median", "std", "threshold", "count"]
        )

    results = []

    for group in df[group_col].unique():
        group_values = df[df[group_col] == group][value_col]

        if len(group_values) < min_group_size:
            warnings.warn(
                f"Group '{group}' has fewer than {min_group_size} values "
                f"(found {len(group_values)}). Statistics may be unreliable.",
                UserWarning,
            )

        for method in methods:
            # Calculate stats with visualization for first group only
            stats = calculate_threshold_stats(
                group_values,
                n_std=n_std,
                threshold_method=method,
                visualize=(visualize_first_group and group == df[group_col].iloc[0]),
            )

            results.append(
                {
                    "group": group,
                    "count": stats["count"],
                    "method": method,
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "std": stats["std"],
                    "threshold": stats["threshold"],
                }
            )

    return pd.DataFrame(results)
