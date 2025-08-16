import inspect
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    "calculate_cooks_d_like_influence",
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


def calculate_cooks_d_like_influence(
    model_class: type,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    visualize: bool = True,
    save_path: Optional[str] = None,
    max_loo_points: Optional[int] = None,
    influence_outlier_method: str = "percentile",
    influence_outlier_threshold: float = 99,
    random_state: Optional[int] = None,
    **model_params: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates a Cook's-like influence score for each data point for any scikit-learn compatible model.
    This is an extension of Cook's Distance that works with any ML model, not just linear regression.

    The influence score is calculated by:
    1. Training a model on the full dataset
    2. For each point (or selected points):
       - Remove the point
       - Train a new model
       - Calculate how much the predictions change across all points
    3. The influence score is the mean squared difference in predictions between the full model and the leave-one-out model.

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
        Points are selected based on having the highest absolute residuals.
        This is purely for computational efficiency and does not affect
        which points are considered influential.
    influence_outlier_method : str, default='percentile'
        Method to identify influential points (outliers) based on influence scores.
        Options:
        - 'percentile': Select points above a percentile threshold
        - 'zscore': Select points beyond N standard deviations from the mean
        - 'top_k': Select the K points with highest influence scores
        - 'iqr': Select points above Q3 + k * IQR threshold
        Note: When max_loo_points is set, only 'percentile' and 'top_k' methods are available
        since other methods require all influence scores to be calculated.
    influence_outlier_threshold : float or int, default=99
        Threshold for identifying influential points:
        - For 'percentile': Points above this percentile are marked influential
          (e.g., 95 means top 5% most influential points)
        - For 'zscore': Points with absolute z-scores above this value are marked influential
          (e.g., 3 means points more than 3 standard deviations from the mean)
        - For 'top_k': Number of most influential points to select (integer)
        - For 'iqr': Multiplier k for Q3 + k*IQR threshold (typically 1.5 or 3.0)
        Note: When max_loo_points is set:
        - The number of influential points will be capped at max_loo_points
        - Uncalculated points are assumed to have zero influence
    random_state : Optional[int], default=None
        Random state for reproducibility, passed to model if supported.
    **model_params : Any
        Additional parameters passed to the model constructor.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - influence_scores: Array of influence scores, one for each data point.
          Higher scores indicate more influential points.
        - influential_indices: Array of indices for points identified as influential
          based on the specified method and threshold, and are capped at max_loo_points.
        - normal_indices: Array of indices for non-influential points (complement of influential_indices).
          When max_loo_points is set, this properly accounts for uncalculated points.

    Examples
    --------
    >>> from lightgbm import LGBMRegressor
    >>> # Example 1: Using percentile method with max_loo_points
    >>> scores1, infl_idx1, normal_idx1 = calculate_cooks_d_like_influence(
    ...     LGBMRegressor,
    ...     X,
    ...     y,
    ...     max_loo_points=20,  # Only analyze top 20 points with highest residuals
    ...     influence_outlier_threshold=95  # Mark top 5% as influential (but capped at max_loo_points)
    ... )

    >>> # Example 2: Using z-score method on all points
    >>> scores2, infl_idx2, normal_idx2 = calculate_cooks_d_like_influence(
    ...     LGBMRegressor,
    ...     X,
    ...     y,
    ...     influence_outlier_method='zscore',
    ...     influence_outlier_threshold=3  # Mark points > 3 std devs from mean as influential
    ... )
    >>> print(f"Points with influence scores > 3 std devs: {infl_idx2}")

    Notes
    -----
    - For large datasets, consider using max_loo_points to reduce computation time
    - User can set influence_outlier_threshold, but it is capped at max_loo_points
    """
    n_samples = X.shape[0]
    influence_scores = np.zeros(n_samples)

    # Train the full model
    if random_state is not None:
        sig = inspect.signature(model_class.__init__)
        if "random_state" in sig.parameters:
            model_params["random_state"] = random_state

    full_model = model_class(**model_params)
    full_model.fit(X, y)
    y_pred_full = full_model.predict(X)

    # Calculate initial residuals
    residuals = np.abs(y - y_pred_full)

    # Determine which points to perform LOO on
    if max_loo_points is None or max_loo_points >= n_samples:
        loo_indices_to_process = np.arange(n_samples)
        print("Performing Cook's-like influence calculation for ALL points.")
    else:
        print(
            f"Selecting {max_loo_points} points with highest absolute residuals for LOO calculation..."
        )
        # Sort by absolute residuals and take top max_loo_points
        loo_indices_to_process = np.argsort(residuals)[::-1][:max_loo_points]

    loo_indices_to_process = np.array(loo_indices_to_process)

    # Using boolean mask for efficient pandas subsetting
    loo_mask = np.ones(n_samples, dtype=bool)

    for i_idx, original_data_index in enumerate(loo_indices_to_process):
        loo_mask[original_data_index] = False  # Set current point to False

        if isinstance(X, pd.DataFrame):
            X_train_loo = X.iloc[loo_mask]
        else:
            X_train_loo = X[loo_mask]

        if isinstance(y, pd.Series):
            y_train_loo = y.iloc[loo_mask]
        else:
            y_train_loo = y[loo_mask]

        loo_model = model_class(**model_params)
        loo_model.fit(X_train_loo, y_train_loo)

        y_pred_loo_on_full_data = loo_model.predict(X)
        influence_scores[original_data_index] = mean_squared_error(
            y_pred_full, y_pred_loo_on_full_data
        )

        # Reset mask for next iteration
        loo_mask[original_data_index] = True

        # Print progress every 10% of the way through the loop if there are at least 20 points
        if len(loo_indices_to_process) >= 20:
            if (i_idx + 1) % (
                len(loo_indices_to_process) // 10 + 1
            ) == 0 or i_idx == len(loo_indices_to_process) - 1:
                print(
                    f"  Processed {i_idx + 1}/{len(loo_indices_to_process)} selected samples."
                )

    # Identify influential points based on influence scores
    if max_loo_points is not None:
        # When max_loo_points is set:
        # 1. We assume uncalculated points have influence=0
        # 2. We can only use percentile or top_k methods
        # 3. We cap the number of influential points at max_loo_points
        if influence_outlier_method not in ["percentile", "top_k"]:
            warnings.warn(
                "When max_loo_points is set, only 'percentile' and 'top_k' methods are supported "
                "Falling back to 'percentile'."
            )
            influence_outlier_method = "percentile"

        if influence_outlier_method == "percentile":
            # Calculate how many points the percentile would select from all points
            n_points_by_percentile = max(
                1, int(n_samples * (1 - influence_outlier_threshold / 100))
            )
            # Take the minimum between max_loo_points and percentile-based count
            n_influential = min(max_loo_points, n_points_by_percentile)
            # Sort scores and get indices of top influential points
            influential_indices = np.argsort(influence_scores)[::-1][:n_influential]

            print(f"Identified {len(influential_indices)} influential points")
            if max_loo_points < n_points_by_percentile:
                warnings.warn(
                    f"Top {influence_outlier_threshold}% threshold would have chosen {n_points_by_percentile} observations, "
                    f"but this is capped at max_loo_points of {max_loo_points}. Consider increasing max_loo_points or decreasing the threshold."
                )
        else:  # top_k method
            if not isinstance(influence_outlier_threshold, (int, np.integer)):
                raise ValueError(
                    "For top_k method, threshold must be an integer representing K."
                )
            k = int(influence_outlier_threshold)
            if k < 1:
                raise ValueError("K must be at least 1 for top_k method.")

            # Cap K at max_loo_points
            n_influential = min(k, max_loo_points)
            # Sort scores and get indices of top K (capped at max_loo_points)
            influential_indices = np.argsort(influence_scores)[::-1][:n_influential]
            print(f"Selected top {len(influential_indices)} most influential points")
            if max_loo_points < k:
                warnings.warn(
                    f"Requested top {k} points but this is capped at max_loo_points of {max_loo_points}. "
                    f"Consider increasing max_loo_points if you need more points."
                )

    else:
        # When all points are calculated, we can use any method
        if influence_outlier_method == "percentile":
            threshold = np.percentile(influence_scores, influence_outlier_threshold)
            influential_indices = np.where(influence_scores >= threshold)[0]
            print(
                f"Identified {len(influential_indices)} points above the "
                f"{influence_outlier_threshold}th percentile threshold."
            )
        elif influence_outlier_method == "zscore":
            z_scores = (influence_scores - np.mean(influence_scores)) / np.std(
                influence_scores
            )
            influential_indices = np.where(
                np.abs(z_scores) >= influence_outlier_threshold
            )[0]
            print(
                f"Identified {len(influential_indices)} points with absolute z-scores >= "
                f"{influence_outlier_threshold} standard deviations."
            )
        elif influence_outlier_method == "top_k":
            if not isinstance(influence_outlier_threshold, (int, np.integer)):
                raise ValueError(
                    "For top_k method, threshold must be an integer representing K."
                )
            k = int(influence_outlier_threshold)
            if k < 1:
                raise ValueError("K must be at least 1 for top_k method.")
            # Sort scores in descending order and get indices of top K
            influential_indices = np.argsort(influence_scores)[::-1][:k]
            print(f"Selected top {len(influential_indices)} most influential points.")
        elif influence_outlier_method == "iqr":
            if not isinstance(influence_outlier_threshold, (int, float)):
                raise ValueError(
                    "For IQR method, threshold must be a number representing the IQR multiplier."
                )
            q1, q3 = np.percentile(influence_scores, [25, 75])
            iqr = q3 - q1
            threshold = q3 + influence_outlier_threshold * iqr
            influential_indices = np.where(influence_scores >= threshold)[0]
            print(
                f"Identified {len(influential_indices)} points above "
                f"Q3 + {influence_outlier_threshold}*IQR threshold."
            )
        else:
            raise ValueError(
                "Invalid influence_outlier_method. Choose from: 'percentile', 'zscore', 'top_k', or 'iqr'."
            )

    if visualize:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        MPL_BLUE = colors[0]  # Main distribution/points color
        MPL_RED = colors[3]  # Highlight influential points
        MPL_YELLOW = colors[1]  # For influence score distribution

        # Get all features and their types if DataFrame
        if isinstance(X, pd.DataFrame):
            n_features = X.shape[1]
            feature_names = X.columns
            # Identify categorical columns (including bool)
            cat_cols = X.select_dtypes(include=["category", "object", "bool"]).columns
        else:
            n_features = X.shape[1]
            feature_names = [f"Feature {i+1}" for i in range(n_features)]
            cat_cols = []

        # Calculate grid dimensions
        n_scatter_rows = (n_features + 1) // 2
        n_rows = 2 + n_scatter_rows  # 2 rows for distributions + scatter plots
        n_cols = 2

        # Create figure with constrained_layout for better automatic spacing
        fig = plt.figure(figsize=(12, 6 * n_rows), constrained_layout=True)

        # Add height ratios to give more space to distribution plots
        gs = fig.add_gridspec(
            n_rows,
            n_cols,
            height_ratios=[1.2, 1.2]
            + [1] * n_scatter_rows,  # Make distribution plots slightly larger
            hspace=0.1,
        )  # Increase vertical spacing between subplots

        # Influence Score Distribution (spans both columns)
        ax_infl = fig.add_subplot(gs[0, :])

        # Plot influence score distribution
        sns.histplot(
            influence_scores,
            bins=50,
            kde=True,
            ax=ax_infl,
            color=MPL_YELLOW,
            alpha=0.5,
            stat="density",
            edgecolor="grey",
        )

        # Add vertical line for threshold
        if influence_outlier_method == "percentile":
            if max_loo_points is not None:
                # When max_loo_points is set, use the same threshold logic as when identifying influential points
                n_points_by_percentile = max(
                    1, int(n_samples * (1 - influence_outlier_threshold / 100))
                )
                n_influential = min(max_loo_points, n_points_by_percentile)
                # Get the threshold as the score at the n_influential position
                sorted_scores = np.sort(influence_scores)[
                    ::-1
                ]  # Sort in descending order
                threshold = sorted_scores[n_influential - 1]
            else:
                # When all scores are calculated, use regular percentile
                threshold = np.percentile(influence_scores, influence_outlier_threshold)

            ax_infl.axvline(
                x=threshold,
                color=MPL_RED,
                linestyle="--",
                label=f"{influence_outlier_threshold}th Percentile",
            )
        elif influence_outlier_method == "zscore" and max_loo_points is None:
            z_scores = (influence_scores - np.mean(influence_scores)) / np.std(
                influence_scores
            )
            threshold = influence_outlier_threshold * np.std(
                influence_scores
            ) + np.mean(influence_scores)
            ax_infl.axvline(
                x=threshold,
                color=MPL_RED,
                linestyle="--",
                label=f"{influence_outlier_threshold} Standard Deviations",
            )
        elif influence_outlier_method == "top_k":
            # For top_k, show the threshold at the Kth point
            k = min(
                int(influence_outlier_threshold),
                max_loo_points if max_loo_points is not None else len(influence_scores),
            )
            sorted_scores = np.sort(influence_scores)[::-1]  # Sort in descending order
            if k <= len(sorted_scores):
                threshold = sorted_scores[k - 1]
                ax_infl.axvline(
                    x=threshold,
                    color=MPL_RED,
                    linestyle="--",
                    label=f"Top {k} Threshold",
                )
        elif influence_outlier_method == "iqr" and max_loo_points is None:
            # For IQR method, show Q3 + k*IQR threshold
            q1, q3 = np.percentile(influence_scores, [25, 75])
            iqr = q3 - q1
            threshold = q3 + influence_outlier_threshold * iqr
            ax_infl.axvline(
                x=threshold,
                color=MPL_RED,
                linestyle="--",
                label=f"Q3 + {influence_outlier_threshold}*IQR",
            )

        ax_infl.set_title("Distribution of Influence Scores", fontsize=13, pad=15)
        ax_infl.set_xlabel("Influence Score", fontsize=12)
        ax_infl.set_ylabel("Density", fontsize=12)
        ax_infl.legend()

        # Add note about uncalculated points if max_loo_points is set
        if max_loo_points is not None:
            n_uncalculated = n_samples - len(loo_indices_to_process)
            note = (
                f"Note: The influence scores \n of {n_uncalculated:,} points ({n_uncalculated/n_samples*100:.1f}%) \n"
                "were not calculated\n"
                f"due to max_loo_points ({max_loo_points}) \n and are shown as 0."
            )
            ax_infl.text(
                0.98,
                0.75,
                note,
                transform=ax_infl.transAxes,
                horizontalalignment="right",
                verticalalignment="bottom",
                fontsize=10,
                style="italic",
                color="dimgray",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
        sns.despine(ax=ax_infl)

        # Target Distribution (spans both columns)
        ax_dist = fig.add_subplot(gs[1, :])

        # Plot target distribution
        sns.histplot(
            y,
            bins=50,
            kde=True,
            ax=ax_dist,
            color=MPL_BLUE,
            alpha=0.5,
            label="All Points",
            stat="density",
            edgecolor="grey",
        )

        # Add markers for influential points
        if isinstance(y, pd.Series):
            y_influential = y.iloc[influential_indices]
        else:
            y_influential = y[influential_indices]

        # Get max density for marker placement
        max_density = max(p.get_height() for p in ax_dist.patches)
        y_offset = max_density * 0.02

        ax_dist.scatter(
            y_influential,
            [y_offset] * len(influential_indices),
            color=MPL_RED,
            s=60,
            alpha=0.85,
            label=f"High Influence Points (n={len(influential_indices)})",
            zorder=5,
            marker="v",
        )

        ax_dist.set_title(
            "Target Distribution with High-Influence Points", fontsize=13, pad=15
        )
        ax_dist.set_xlabel("Target Value", fontsize=12)
        ax_dist.set_ylabel("Density", fontsize=12)
        ax_dist.legend()
        sns.despine(ax=ax_dist)

        # Feature scatter plots
        row_idx = 2  # Start from third row (after both distributions)
        col_idx = 0

        if isinstance(X, pd.DataFrame):
            for col in feature_names:
                ax = fig.add_subplot(gs[row_idx, col_idx])

                is_categorical = col in cat_cols
                x_data = X[col]

                if is_categorical:
                    sns.stripplot(
                        x=x_data,
                        y=y,
                        ax=ax,
                        color=MPL_BLUE,
                        alpha=0.5,
                        size=5,
                        jitter=0.2,
                    )

                    # Plot influential points on top
                    sns.stripplot(
                        x=x_data.iloc[influential_indices],
                        y=y_influential,
                        ax=ax,
                        color=MPL_RED,
                        alpha=0.85,
                        size=8,
                        jitter=0.2,
                    )
                else:
                    # Regular scatter for numeric features
                    ax.scatter(x_data, y, color=MPL_BLUE, alpha=0.5, s=30)
                    ax.scatter(
                        x_data.iloc[influential_indices],
                        y_influential,
                        color=MPL_RED,
                        s=60,
                        alpha=0.85,
                    )

                ax.set_xlabel(col)
                ax.set_ylabel("Target" if col_idx == 0 else "")
                sns.despine(ax=ax)

                # Update grid position
                col_idx = (col_idx + 1) % 2
                if col_idx == 0:
                    row_idx += 1
        else:
            # Handle numpy array case (assume all numeric)
            for i in range(n_features):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.scatter(X[:, i], y, color=MPL_BLUE, alpha=0.3, s=30)
                ax.scatter(
                    X[influential_indices, i],
                    y_influential,
                    color=MPL_RED,
                    s=40,
                    alpha=0.85,
                )
                ax.set_xlabel(f"Feature {i+1}")
                ax.set_ylabel("Target" if col_idx == 0 else "")
                sns.despine(ax=ax)

                col_idx = (col_idx + 1) % 2
                if col_idx == 0:
                    row_idx += 1

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()

    # Calculate normal indices as the complement of influential indices
    normal_mask = ~np.isin(np.arange(n_samples), influential_indices)
    normal_indices = np.where(normal_mask)[0]

    return influence_scores, influential_indices, normal_indices


def get_normal_data(
    model_class: type,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    influence_outlier_method: str = "percentile",
    influence_outlier_threshold: float = 99,
    max_loo_points: Optional[int] = None,
    random_state: Optional[int] = None,
    **model_params: Any,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Identifies and removes influential points from the dataset using Cook's D-like influence scores.

    .. warning::
        This function is deprecated and will be removed in a future version.
        Use `calculate_cooks_d_like_influence()` instead, which returns both influence scores
        and indices, allowing more flexibility in how you handle influential points.

    Parameters
    ----------
    model_class : type
        The class of the ML model to use (e.g., LinearRegression, LGBMRegressor).
        Must be scikit-learn compatible with fit() and predict() methods.
    X : Union[pd.DataFrame, np.ndarray]
        The feature matrix.
    y : Union[pd.Series, np.ndarray]
        The target vector.
    influence_outlier_method : str, default='percentile'
        Method to identify influential points. See `calculate_cooks_d_like_influence()`.
    influence_outlier_threshold : float, default=99
        Threshold for identifying influential points. See `calculate_cooks_d_like_influence()`.
    max_loo_points : Optional[int], default=None
        If specified, only calculate influence scores for this many points.
        See `calculate_cooks_d_like_influence()`.
    random_state : Optional[int], default=None
        Random state for reproducibility, passed to model if supported.
    **model_params : Any
        Additional parameters passed to the model constructor.

    Returns
    -------
    Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]
        A tuple containing:
        - X_normal: Features of non-influential points
        - y_normal: Target values of non-influential points

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = pd.DataFrame({'feature1': [1, 2, 3, 4, 100], 'feature2': [1, 2, 3, 4, 0]})
    >>> y = pd.Series([1, 2, 3, 4, 100])
    >>> # This will show a deprecation warning
    >>> X_normal, y_normal = get_normal_data(
    ...     LinearRegression,
    ...     X,
    ...     y,
    ...     influence_outlier_threshold=99  # Remove top 1% influential points
    ... )
    >>> print(f"Original data size: {len(y)}, Normal data size: {len(y_normal)}")

    # Recommended way using calculate_cooks_d_like_influence:
    >>> scores, infl_idx, normal_idx = calculate_cooks_d_like_influence(
    ...     LinearRegression, X, y, influence_outlier_threshold=99
    ... )
    >>> X_normal, y_normal = X.iloc[normal_idx], y.iloc[normal_idx]

    Notes
    -----
    This function is deprecated. Use `calculate_cooks_d_like_influence()` instead,
    which provides more flexibility in how you handle influential points.
    """
    warnings.warn(
        "get_normal_data() is deprecated and will be removed in a future version. "
        "Use calculate_cooks_d_like_influence() instead, which returns both influence scores "
        "and indices, allowing more flexibility in how you handle influential points.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Get influence scores and indices
    _, _, normal_indices = calculate_cooks_d_like_influence(
        model_class=model_class,
        X=X,
        y=y,
        visualize=False,  # Don't show visualization by default
        influence_outlier_method=influence_outlier_method,
        influence_outlier_threshold=influence_outlier_threshold,
        max_loo_points=max_loo_points,
        random_state=random_state,
        **model_params,
    )

    # Return normal subset based on input type
    if isinstance(X, pd.DataFrame):
        X_normal = X.iloc[normal_indices]
    else:
        X_normal = X[normal_indices]

    if isinstance(y, pd.Series):
        y_normal = y.iloc[normal_indices]
    else:
        y_normal = y[normal_indices]

    return X_normal, y_normal
