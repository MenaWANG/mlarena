import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.stats.power import tt_solve_power, zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

import mlarena.utils.plot_utils as put

__all__ = [
    "compare_groups",
    "add_stratified_groups",
    "optimize_stratification_strategy",
    "calculate_threshold_stats",
    "calculate_group_thresholds",
    "power_analysis_numeric",
    "power_analysis_proportion",
    "sample_size_numeric",
    "sample_size_proportion",
    "numeric_effectsize",
    "calculate_cooks_like_influence",
    "get_normal_data",
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


def power_analysis_numeric(
    effect_size: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
    test_type: str = "two_sample",
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Calculate statistical power for numeric outcomes (t-tests).

    Parameters
    ----------
    effect_size : float
        Cohen's d effect size (standardized difference between groups)
    sample_size_per_group : int
        Sample size per group
    alpha : float, default=0.05
        Type I error rate (significance level)
    test_type : str, default="two_sample"
        Type of test: "two_sample", "one_sample", "paired"
    alternative : str, default="two-sided"
        Alternative hypothesis: "two-sided", "greater", "less"

    Returns
    -------
    Dict[str, float]
        Dictionary containing power, effect_size, alpha, sample_size_per_group
    """
    # Input validation
    if not -3 <= effect_size <= 3:
        warnings.warn(
            f"Effect size {effect_size} is outside typical range (-3 to 3) for Cohen's d",
            UserWarning,
        )

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

    if sample_size_per_group < 2:
        raise ValueError("Sample size per group must be at least 2")

    if test_type not in ["two_sample", "one_sample", "paired"]:
        raise ValueError(f"Invalid test_type: {test_type}")

    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError(f"Invalid alternative: {alternative}")

    # Calculate power using tt_solve_power
    power = tt_solve_power(
        effect_size=effect_size,
        nobs=sample_size_per_group,
        alpha=alpha,
        power=None,  # Solve for power
        alternative=alternative,
    )

    return {
        "power": float(power),
        "effect_size": effect_size,
        "alpha": alpha,
        "sample_size_per_group": sample_size_per_group,
        "test_type": test_type,
        "alternative": alternative,
    }


def power_analysis_proportion(
    baseline_rate: float,
    treatment_rate: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Calculate statistical power for proportion/conversion rate tests.

    Parameters
    ----------
    baseline_rate : float
        Baseline conversion rate (between 0 and 1)
    treatment_rate : float
        Treatment conversion rate (between 0 and 1)
    sample_size_per_group : int
        Sample size per group
    alpha : float, default=0.05
        Type I error rate
    alternative : str, default="two-sided"
        Alternative hypothesis: "two-sided", "larger", "smaller"

    Returns
    -------
    Dict[str, float]
        Dictionary containing power, effect_size, rates, and sample size
    """
    # Input validation
    if not 0 <= baseline_rate <= 1:
        raise ValueError("Baseline rate must be between 0 and 1")

    if not 0 <= treatment_rate <= 1:
        raise ValueError("Treatment rate must be between 0 and 1")

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

    if sample_size_per_group < 2:
        raise ValueError("Sample size per group must be at least 2")

    if alternative not in ["two-sided", "larger", "smaller"]:
        raise ValueError(f"Invalid alternative: {alternative}")

    # Calculate effect size (Cohen's h)
    effect_size = proportion_effectsize(treatment_rate, baseline_rate)

    # Calculate power
    power = zt_ind_solve_power(
        effect_size=effect_size,
        nobs1=sample_size_per_group,
        alpha=alpha,
        alternative=alternative,
    )

    return {
        "power": float(power),
        "effect_size": float(effect_size),
        "baseline_rate": baseline_rate,
        "treatment_rate": treatment_rate,
        "relative_lift": (treatment_rate - baseline_rate) / baseline_rate,
        "absolute_lift": treatment_rate - baseline_rate,
        "sample_size_per_group": sample_size_per_group,
        "alpha": alpha,
        "alternative": alternative,
    }


def sample_size_numeric(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test_type: str = "two_sample",
    alternative: str = "two-sided",
) -> Dict[str, Union[int, float]]:
    """
    Calculate required sample size for numeric outcomes to achieve desired power.

    Parameters
    ----------
    effect_size : float
        Cohen's d effect size to detect
    power : float, default=0.8
        Desired statistical power (1 - Type II error rate).
        Common values: 0.7 (exploration), 0.8 (standard), 0.9 (high-stakes)
    alpha : float, default=0.05
        Type I error rate
    test_type : str, default="two_sample"
        Type of test: "two_sample", "one_sample", "paired"
    alternative : str, default="two-sided"
        Alternative hypothesis: "two-sided", "greater", "less"

    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary containing required sample_size_per_group and input parameters

    Examples
    --------
    >>> # Sample size needed to detect medium effect (d=0.5) with 80% power
    >>> sample_size_numeric(effect_size=0.5)
    {'sample_size_per_group': 64, 'total_sample_size': 128, 'power': 0.8, ...}
    """
    # Input validation
    if not -3 <= effect_size <= 3:
        warnings.warn(
            f"Effect size {effect_size} is outside typical range (-3 to 3) for Cohen's d",
            UserWarning,
        )

    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1")

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

    if test_type not in ["two_sample", "one_sample", "paired"]:
        raise ValueError(f"Invalid test_type: {test_type}")

    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError(f"Invalid alternative: {alternative}")

    # Calculate required sample size
    sample_size = tt_solve_power(
        effect_size=effect_size, power=power, alpha=alpha, alternative=alternative
    )

    sample_size_per_group = int(np.ceil(sample_size))

    # For two-sample tests, total sample size is 2x per group
    if test_type == "two_sample":
        total_sample_size = sample_size_per_group * 2
    else:
        total_sample_size = sample_size_per_group

    return {
        "sample_size_per_group": sample_size_per_group,
        "total_sample_size": total_sample_size,
        "power": power,
        "effect_size": effect_size,
        "alpha": alpha,
        "test_type": test_type,
        "alternative": alternative,
    }


def sample_size_proportion(
    baseline_rate: float,
    treatment_rate: float,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Dict[str, Union[int, float]]:
    """
    Calculate required sample size for proportion tests to achieve desired power.

    Parameters
    ----------
    baseline_rate : float
        Baseline conversion rate (between 0 and 1)
    treatment_rate : float
        Treatment conversion rate to detect (between 0 and 1)
    power : float, default=0.8
        Desired statistical power (1 - Type II error rate).
        Common values: 0.7 (exploration), 0.8 (standard), 0.9 (high-stakes)
    alpha : float, default=0.05
        Type I error rate
    alternative : str, default="two-sided"
        Alternative hypothesis: "two-sided", "larger", "smaller"

    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary containing required sample sizes and effect size metrics

    Examples
    --------
    >>> # Sample size to detect 5% -> 6% improvement with 80% power
    >>> sample_size_proportion(0.05, 0.06)
    {'sample_size_per_group': 23506, 'total_sample_size': 47012, ...}
    """
    # Input validation
    if not 0 <= baseline_rate <= 1:
        raise ValueError("Baseline rate must be between 0 and 1")

    if not 0 <= treatment_rate <= 1:
        raise ValueError("Treatment rate must be between 0 and 1")

    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1")

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

    if alternative not in ["two-sided", "larger", "smaller"]:
        raise ValueError(f"Invalid alternative: {alternative}")

    # Calculate effect size
    effect_size = proportion_effectsize(treatment_rate, baseline_rate)

    # Calculate required sample size
    sample_size = zt_ind_solve_power(
        effect_size=effect_size, power=power, alpha=alpha, alternative=alternative
    )

    sample_size_per_group = int(np.ceil(sample_size))
    total_sample_size = sample_size_per_group * 2

    return {
        "sample_size_per_group": sample_size_per_group,
        "total_sample_size": total_sample_size,
        "power": power,
        "effect_size": float(effect_size),
        "baseline_rate": baseline_rate,
        "treatment_rate": treatment_rate,
        "relative_lift": (treatment_rate - baseline_rate) / baseline_rate,
        "absolute_lift": treatment_rate - baseline_rate,
        "alpha": alpha,
        "alternative": alternative,
    }


def numeric_effectsize(
    mean_diff: float = None,
    mean1: float = None,
    mean2: float = None,
    std: float = None,
    std1: float = None,
    std2: float = None,
    n1: int = None,
    n2: int = None,
) -> float:
    """
    Compute Cohen's d for independent samples in a power analysis context.

    Parameters
    ----------
    mean_diff : float, optional
        Mean difference (mean1 - mean2). Optional if mean1 and mean2 are provided.
    mean1 : float, optional
        Mean of group 1 (e.g., treatment group).
    mean2 : float, optional
        Mean of group 2 (e.g., control group).
    std : float, optional
        Common standard deviation (assumed equal for both groups).
    std1 : float, optional
        Standard deviation of group 1 (optional if std is provided).
    std2 : float, optional
        Standard deviation of group 2 (optional if std is provided).
    n1 : int, optional
        Sample size of group 1 (used only if std1 and std2 are provided).
    n2 : int, optional
        Sample size of group 2 (used only if std1 and std2 are provided).

    Returns
    -------
    float
        Cohen's d (standardized effect size)

    Raises
    ------
    ValueError
        If required parameters are missing or invalid

    Assumptions:
    --------
    - Groups are independent
    - Standard deviations are assumed equal unless both std1/std2 and n1/n2 are provided
    - Appropriate for planning two-sample t-tests (e.g., A/B testing)
    - Not suitable for paired-sample designs or ANOVA with 3+ groups without modification

    Examples
    --------
    >>> # Using mean difference and common std
    >>> d = numeric_effectsize(mean_diff=0.5, std=2.0)
    >>> print(f"Cohen's d: {d:.3f}")  # 0.250

    >>> # Using separate means and standard deviations
    >>> d = numeric_effectsize(
    ...     mean1=100, mean2=95,  # 5-point difference
    ...     std1=10, std2=12,     # Different spreads
    ...     n1=50, n2=50          # Equal sample sizes
    ... )
    >>> print(f"Cohen's d: {d:.3f}")  # ~0.455

    >>> # Using means with common standard deviation
    >>> d = numeric_effectsize(mean1=10, mean2=8, std=3)
    >>> print(f"Cohen's d: {d:.3f}")  # ~0.667
    """
    # Validate and compute mean difference
    if mean_diff is None:
        if mean1 is not None and mean2 is not None:
            mean_diff = mean1 - mean2
        else:
            raise ValueError(
                "You must provide either mean_diff or both mean1 and mean2."
            )

    # Validate and compute pooled standard deviation
    if std is not None:
        pooled_std = std
    elif std1 is not None and std2 is not None and n1 is not None and n2 is not None:
        # Validate sample sizes
        if n1 <= 0 or n2 <= 0:
            raise ValueError("Sample sizes must be positive integers.")
        # Validate standard deviations
        if std1 <= 0 or std2 <= 0:
            raise ValueError("Standard deviations must be positive.")
        # Compute pooled std
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    else:
        raise ValueError(
            "You must provide either a common std, or std1, std2, n1, and n2."
        )

    # Validate standard deviation
    if pooled_std <= 0:
        raise ValueError("Computed pooled standard deviation must be positive.")

    return mean_diff / pooled_std


def calculate_cooks_like_influence(
    model_class: type,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    visualize: bool = True,
    save_path: Optional[str] = None,
    max_loo_points: Optional[int] = None,
    residual_outlier_method: str = "percentile",
    residual_threshold: float = 99,
    random_state: Optional[int] = None,
    **model_params: Any,
) -> np.ndarray:
    """
    Calculates a Cook's-like influence score for each data point for any scikit-learn compatible model.
    This is an extension of Cook's Distance that works with any ML model, not just linear regression.

    The influence score is calculated by:
    1. Training a model on the full dataset
    2. For each point (or selected points):
       - Remove the point
       - Train a new model
       - Calculate how much the predictions change across all points
    3. The influence score is proportional to how much the model's predictions change when the point is removed

    Parameters
    ----------
    model_class : type
        The class of the ML model to use (e.g., LinearRegression, LGBMRegressor).
        Must be scikit-learn compatible with fit() and predict() methods.
    X : Union[pd.DataFrame, np.ndarray]
        The feature matrix.
    y : Union[pd.Series, np.ndarray]
        The target vector.
    visualize : bool, default=True
        If True, plots the influence scores.
    save_path : Optional[str], default=None
        If provided, saves the plot to this file path. Requires visualize=True.
    max_loo_points : Optional[int], default=None
        If specified, only perform LOO calculations for this many points.
        Points are selected based on having the highest residuals.
    residual_outlier_method : str, default='percentile'
        Method to select high-residual points if max_loo_points is set.
        Options: 'percentile' or 'isolation_forest'.
    residual_threshold : float, default=99
        Percentile threshold for filtering high-residual points
        if residual_outlier_method is 'percentile'. E.g., 99 for top 1%.
    random_state : Optional[int], default=None
        Random state for reproducibility, passed to model if supported.
    **model_params : Any
        Additional parameters passed to the model constructor.

    Returns
    -------
    np.ndarray
        Array of influence scores, one for each data point.
        Higher scores indicate more influential points.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = pd.DataFrame({'feature1': [1, 2, 3, 4, 100], 'feature2': [1, 2, 3, 4, 0]})
    >>> y = pd.Series([1, 2, 3, 4, 100])
    >>> scores = calculate_cooks_like_influence(LinearRegression, X, y)
    >>> print("Most influential point index:", scores.argmax())
    Most influential point index: 4

    >>> # Using with a more complex model
    >>> from lightgbm import LGBMRegressor
    >>> scores = calculate_cooks_like_influence(
    ...     LGBMRegressor,
    ...     X,
    ...     y,
    ...     max_loo_points=2,  # Only analyze top 2 most residual points
    ...     residual_outlier_method='isolation_forest'
    ... )

    Notes
    -----
    - For large datasets, consider using max_loo_points to reduce computation time
    - The influence score is proportional to the mean squared difference in predictions
    - The score is scaled by n_samples to make it comparable across different dataset sizes
    """
    n_samples = X.shape[0]
    influence_scores = np.zeros(n_samples)

    # 1. Train the full model
    print(f"Training full model using {model_class.__name__}...")
    full_model = model_class(
        random_state=(
            random_state
            if "random_state" in model_class.__init__.__code__.co_varnames
            else None
        ),
        **model_params,
    )
    full_model.fit(X, y)
    y_pred_full = full_model.predict(X)

    # Calculate initial residuals
    residuals = np.abs(y - y_pred_full)

    # Determine which points to perform LOO on
    loo_indices_to_process = []
    if max_loo_points is None or max_loo_points >= n_samples:
        loo_indices_to_process = np.arange(n_samples)
        print("Performing Cook's-like influence calculation for ALL points.")
    else:
        print(
            f"Identifying up to {max_loo_points} points with highest residuals for LOO calculation..."
        )
        if residual_outlier_method == "percentile":
            threshold_value = np.percentile(residuals, residual_threshold)
            high_residual_indices = np.where(residuals >= threshold_value)[0]
            if len(high_residual_indices) > max_loo_points:
                top_indices = np.argsort(residuals)[::-1][:max_loo_points]
                loo_indices_to_process = top_indices
            else:
                loo_indices_to_process = high_residual_indices
            print(
                f"  Selected {len(loo_indices_to_process)} points based on {residual_threshold}th percentile of residuals."
            )
        elif residual_outlier_method == "isolation_forest":
            iso_forest = IsolationForest(
                random_state=random_state, contamination="auto"
            )
            iso_forest.fit(residuals.reshape(-1, 1))
            outlier_scores = iso_forest.decision_function(residuals.reshape(-1, 1))
            top_indices_by_iso_forest = np.argsort(outlier_scores)[:max_loo_points]
            loo_indices_to_process = top_indices_by_iso_forest
            print(
                f"  Selected {len(loo_indices_to_process)} points based on Isolation Forest on residuals."
            )
        else:
            raise ValueError(
                "Invalid residual_outlier_method. Choose 'percentile' or 'isolation_forest'."
            )

    loo_indices_to_process = np.array(loo_indices_to_process)

    print(
        f"Calculating Cook's-like influence for {len(loo_indices_to_process)} selected points using {model_class.__name__}..."
    )
    # Using boolean mask for efficient pandas subsetting
    loo_mask = np.ones(n_samples, dtype=bool)

    for i_idx, original_data_index in enumerate(loo_indices_to_process):
        loo_mask[original_data_index] = False  # Set current point to False

        if isinstance(X, pd.DataFrame):
            X_train_loo = X.iloc[loo_mask]
            y_train_loo = y.iloc[loo_mask] if isinstance(y, pd.Series) else y[loo_mask]
        else:
            X_train_loo = X[loo_mask]
            y_train_loo = y[loo_mask]

        loo_model = model_class(
            random_state=(
                random_state
                if "random_state" in model_class.__init__.__code__.co_varnames
                else None
            ),
            **model_params,
        )
        loo_model.fit(X_train_loo, y_train_loo)

        y_pred_loo_on_full_data = loo_model.predict(X)
        influence_scores[original_data_index] = (
            mean_squared_error(y_pred_full, y_pred_loo_on_full_data) * n_samples
        )

        # Reset mask for next iteration
        loo_mask[original_data_index] = True

        if (i_idx + 1) % (len(loo_indices_to_process) // 10 + 1) == 0 or i_idx == len(
            loo_indices_to_process
        ) - 1:
            print(
                f"  Processed {i_idx + 1}/{len(loo_indices_to_process)} selected samples."
            )

    if visualize:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(influence_scores)
        ax.set_title(f"Cook's-like Influence Scores for {model_class.__name__}")
        ax.set_xlabel("Data Point Index")
        ax.set_ylabel("Influence Score (Sum of Squared Differences)")

        calculated_scores = influence_scores[loo_indices_to_process]
        if len(calculated_scores) > 0:
            percentile_95 = np.percentile(calculated_scores, 95)
            ax.axhline(
                y=percentile_95,
                color="r",
                linestyle="--",
                label=f"95th Percentile ({percentile_95:.2f}) of calculated scores",
            )
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No influence scores were calculated or available for plotting.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("Influence Scores (No Data to Plot)")

        ax.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()
        plt.close()

    return influence_scores


def get_normal_data(
    model_class: type,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    influence_threshold_percentile: float = 99,
    visualize_influence: bool = True,
    save_path_influence_plot: Optional[str] = None,
    max_loo_points: Optional[int] = None,
    residual_outlier_method: str = "percentile",
    residual_threshold: float = 99,
    random_state: Optional[int] = None,
    **model_params: Any,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], np.ndarray]:
    """
    Identifies influential data points using the Cook's-like approach and returns
    the 'normal' (non-influential) subset of data for model training.

    This function helps in creating more robust models by identifying and optionally
    removing observations that have an unusually large influence on the model's behavior.

    Parameters
    ----------
    model_class : type
        The class of the ML model to use (e.g., LinearRegression, LGBMRegressor).
        Must be scikit-learn compatible with fit() and predict() methods.
    X : Union[pd.DataFrame, np.ndarray]
        The feature matrix.
    y : Union[pd.Series, np.ndarray]
        The target vector.
    influence_threshold_percentile : float, default=99
        The percentile threshold for influence scores. Points with influence scores
        above this percentile will be considered influential and excluded from
        the 'normal' dataset. Default is 99 (top 1% influential).
    visualize_influence : bool, default=True
        If True, plots the influence scores.
    save_path_influence_plot : Optional[str], default=None
        If provided, saves the influence plot to this path.
    max_loo_points : Optional[int], default=None
        Passed to calculate_cooks_like_influence.
    residual_outlier_method : str, default='percentile'
        Passed to calculate_cooks_like_influence.
    residual_threshold : float, default=99
        Passed to calculate_cooks_like_influence.
    random_state : Optional[int], default=None
        Random state for reproducibility, passed to model if supported.
    **model_params : Any
        Additional parameters passed to the model constructor.

    Returns
    -------
    Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], np.ndarray]
        A tuple containing:
        - X_normal: Features of non-influential points
        - y_normal: Target values of non-influential points
        - influence_scores: The full array of calculated influence scores

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = pd.DataFrame({'feature1': [1, 2, 3, 4, 100], 'feature2': [1, 2, 3, 4, 0]})
    >>> y = pd.Series([1, 2, 3, 4, 100])
    >>> X_normal, y_normal, scores = get_normal_data(
    ...     LinearRegression,
    ...     X,
    ...     y,
    ...     influence_threshold_percentile=95  # Remove top 5% influential points
    ... )
    >>> print(f"Original data size: {len(y)}, Normal data size: {len(y_normal)}")

    Notes
    -----
    - The function preserves the input data type (DataFrame/Series or ndarray)
    - Consider the trade-off between removing influential points and maintaining
      sufficient data for model training
    - Investigate removed points for potential data quality issues or important
      edge cases before discarding them
    """
    # Calculate influence scores
    influence_scores = calculate_cooks_like_influence(
        model_class=model_class,
        X=X,
        y=y,
        visualize=visualize_influence,
        save_path=save_path_influence_plot,
        max_loo_points=max_loo_points,
        residual_outlier_method=residual_outlier_method,
        residual_threshold=residual_threshold,
        random_state=random_state,
        **model_params,
    )

    # Calculate threshold for normal data
    if max_loo_points is not None:
        # Use only calculated scores for percentile if we didn't calculate all
        calculated_mask = influence_scores > 0
        threshold = np.percentile(
            influence_scores[calculated_mask], influence_threshold_percentile
        )
    else:
        threshold = np.percentile(influence_scores, influence_threshold_percentile)

    # Create mask for normal (non-influential) points
    normal_mask = influence_scores <= threshold

    # Return appropriate type based on input
    if isinstance(X, pd.DataFrame):
        X_normal = X.iloc[normal_mask]
    else:
        X_normal = X[normal_mask]

    if isinstance(y, pd.Series):
        y_normal = y.iloc[normal_mask]
    else:
        y_normal = y[normal_mask]

    print(
        f"Identified {(~normal_mask).sum()} influential points above the {influence_threshold_percentile}th percentile threshold."
    )
    print(f"Returning {normal_mask.sum()} normal points for model training.")

    return X_normal, y_normal, influence_scores
