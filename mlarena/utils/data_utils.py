from typing import List, Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "clean_dollar_cols",
    "value_counts_with_pct",
    "transform_date_cols",
    "drop_fully_null_cols",
    "print_schema_alphabetically",
    "is_primary_key",
    "select_existing_cols",
    "filter_rows_by_substring",
    "filter_columns_by_substring",
    "find_duplicates",
    "deduplicate_by_rank",
    "pivot_by_group",
    "clean_null_representations",
]


def clean_dollar_cols(data: pd.DataFrame, cols_to_clean: List[str]) -> pd.DataFrame:
    """
    Clean specified columns of a Pandas DataFrame by removing '$' symbols, commas,
    and converting to floating-point numbers.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to clean.
    cols_to_clean : List[str]
        List of column names to clean.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns cleaned of '$' symbols and commas,
        and converted to floating-point numbers.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'price': ['$1,234.56', '$789.00', '$2,000'],
    ...     'revenue': ['$50,000', '$75,000.50', '$100,000'],
    ...     'name': ['A', 'B', 'C']
    ... })
    >>> clean_dollar_cols(df, ['price', 'revenue'])
       price  revenue name
    0  1234.56  50000.00    A
    1   789.00  75000.50    B
    2  2000.00 100000.00    C
    """
    df_ = data.copy()

    for col_name in cols_to_clean:
        df_[col_name] = (
            df_[col_name]
            .astype(str)
            .str.replace(r"^\$", "", regex=True)  # Remove $ at start
            .str.replace(",", "", regex=False)  # Remove commas
        )

        df_[col_name] = pd.to_numeric(df_[col_name], errors="coerce").astype("float64")

    return df_


def value_counts_with_pct(
    data: pd.DataFrame,
    cols: Union[str, List[str]],
    dropna: bool = False,
    decimals: int = 2,
) -> pd.DataFrame:
    """
    Calculate the count and percentage of occurrences for unique values or value combinations.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    cols : str or List[str]
        Column name or list of column names to analyze. If multiple columns are provided,
        counts unique combinations of values across these columns.
    dropna : bool, default=False
        Whether to exclude NA/null values.
    decimals : int, default=2
        Number of decimal places to round the percentage.

    Returns
    -------
    pd.DataFrame
        A DataFrame with:
        - For single column: unique values, their counts, and percentages
        - For multiple columns: unique value combinations, their counts, and percentages

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'category': ['A', 'A', 'B', 'B', 'B', None],
    ...     'status': ['Active', 'Active', 'Inactive', None, None, None]
    ... })
    >>> # Single column
    >>> value_counts_with_pct(df, 'category')
      category  count   pct
    0       B      3  50.0
    1       A      2  33.3
    2    None      1  16.7
    >>> # Multiple columns - counts combinations
    >>> value_counts_with_pct(df, ['category', 'status'])
      category   status  count   pct
    0       B     None      2  33.3
    1       A   Active      2  33.3
    2       B Inactive      1  16.7
    3    None     None      1  16.7
    """
    # Convert single column to list for consistent processing
    cols_list = [cols] if isinstance(cols, str) else cols

    # Validate all columns exist
    missing_cols = [col for col in cols_list if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Column(s) {missing_cols} not found in DataFrame")

    # Handle single column case differently to avoid MultiIndex
    if isinstance(cols, str):
        counts = data[cols].value_counts(dropna=dropna)
        percentages = (counts / counts.sum() * 100).round(decimals)
        result = pd.DataFrame(
            {cols: counts.index, "count": counts.values, "pct": percentages.values}
        )
    else:
        # Multiple columns case - use value_counts on the DataFrame
        counts = data[cols_list].value_counts(dropna=dropna)
        percentages = (counts / counts.sum() * 100).round(decimals)
        result = counts.reset_index().rename(columns={0: "count"})
        result["pct"] = percentages.values

    return result.sort_values(by="count", ascending=False).reset_index(drop=True)


def transform_date_cols(
    data: pd.DataFrame,
    date_cols: Union[str, List[str]],
    str_date_format: str = "%Y%m%d",
) -> pd.DataFrame:
    """
    Transforms specified columns in a Pandas DataFrame to datetime format.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    date_cols : Union[str, List[str]]
        A column name or list of column names to be transformed to dates.
    str_date_format : str, default="%Y%m%d"
        The string format of the dates, using Python's `strftime`/`strptime` directives.
        Common directives include:
            %d: Day of the month as a zero-padded decimal (e.g., 25)
            %m: Month as a zero-padded decimal number (e.g., 08)
            %b: Abbreviated month name (e.g., Aug)
            %B: Full month name (e.g., August)
            %Y: Four-digit year (e.g., 2024)
            %y: Two-digit year (e.g., 24)

        Example formats:
            "%Y%m%d"   → '20240825'
            "%d-%m-%Y" → '25-08-2024'
            "%d%b%Y"   → '25Aug2024'
            "%d%B%Y"   → '25August2024'
            "%d%b%y"   → '25Aug24'

        Note:
            If the format uses %b or %B (month names),
            strings like '25AUG2024' or '25august2024'
            will be automatically converted to title case before parsing.

    Returns
    -------
    pd.DataFrame
        The DataFrame with specified columns transformed to datetime format (datetime64[ns]).
        When only date information is provided (no time component), the time will be set to midnight (00:00:00).

        To extract just the date component later, you can use:
            - df['date_col'].dt.date  # Returns datetime.date objects
            - df['date_col'].dt.normalize()  # Returns datetime64[ns] at midnight
            - df['date_col'].dt.floor('D')  # Returns datetime64[ns] at midnight

    Raises
    ------
    ValueError
        If date_cols is empty.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': ['25Aug2024', '26AUG2024', '27aug2024']
    ... })
    >>> # Convert to datetime
    >>> df = transform_date_cols(df, 'date', str_date_format='%d%b%Y')
    >>> print(df['date'].dtype)
    datetime64[ns]
    >>>
    >>> # Extract date-only if needed
    >>> df['date_only'] = df['date'].dt.date
    >>> print(df['date_only'].iloc[0])
    2024-08-25
    """
    if isinstance(date_cols, str):
        date_cols = [date_cols]

    if not date_cols:
        raise ValueError("date_cols list cannot be empty")

    df_ = data.copy()
    for date_col in date_cols:
        if not pd.api.types.is_datetime64_any_dtype(df_[date_col]):
            if "%b" in str_date_format or "%B" in str_date_format:
                df_[date_col] = pd.to_datetime(
                    df_[date_col].astype(str).str.title(),
                    format=str_date_format,
                    errors="coerce",
                )
            else:
                df_[date_col] = pd.to_datetime(
                    df_[date_col], format=str_date_format, errors="coerce"
                )

    return df_


def drop_fully_null_cols(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Drops columns where all values are missing/null in a pandas DataFrame.

    This function is particularly useful when working with Databricks' display() function,
    which can break when encountering columns that are entirely null as it cannot
    infer the schema. Running this function before display() helps prevent such issues.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to check for missing columns.
    verbose : bool, default=False
        If True, prints information about which columns were dropped.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with fully-null columns removed.

    Examples
    --------
    >>> # In Databricks notebook:
    >>> drop_fully_null_cols(df).display()  # this won't affect the original df, just ensure .display() work
    >>> # To see which columns were dropped:
    >>> drop_fully_null_cols(df, verbose=True)
    """
    null_counts = data.isnull().sum()
    all_missing_cols = null_counts[null_counts == len(data)].index.tolist()

    if all_missing_cols and verbose:
        print(f"🗑️ Dropped fully-null columns: {all_missing_cols}")

    data_ = data.drop(columns=all_missing_cols)
    return data_


def print_schema_alphabetically(data: pd.DataFrame) -> None:
    """
    Prints the schema (column names and dtypes) of the DataFrame with columns sorted alphabetically.

    This is particularly useful when comparing schemas between different DataFrames
    or versions of the same DataFrame, as the alphabetical ordering makes it easier
    to spot differences.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame whose schema is to be printed.

    Returns
    -------
    None
        Prints the schema to stdout.

    Examples
    --------
    >>> df = pd.DataFrame({'c': [1], 'a': [2], 'b': ['text']})
    >>> print_schema_alphabetically(df)
    a    int64
    b    object
    c    int64
    """
    sorted_dtypes = data[sorted(data.columns)].dtypes
    print(sorted_dtypes)


def is_primary_key(
    data: pd.DataFrame, cols: Union[str, List[str]], verbose: bool = True
) -> bool:
    """
    Check if the combination of specified columns forms a primary key in the DataFrame.

    A primary key traditionally requires:
    1. Uniqueness: Each combination of values must be unique across all rows
    2. No null values: Primary key columns cannot contain null/missing values

    This implementation will:
    1. Alert if there are any missing values in the potential key columns
    2. Check if the columns form a unique identifier after removing rows with missing values

    This approach is practical for real-world data analysis where some missing values
    might exist but we want to understand the column(s)' potential to serve as a key.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to check.
    cols : str or List[str]
        Column name or list of column names to check for forming a primary key.
    verbose : bool, default=True
        If True, print detailed information.

    Returns
    -------
    bool
        True if the combination of columns forms a primary key (after removing nulls),
        False otherwise.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, None, 4],
    ...     'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
    ...     'value': [10, 20, 30, 40]
    ... })
    >>> is_primary_key(df, 'id')  # Single column as string
    >>> is_primary_key(df, ['id', 'date'])  # Multiple columns as list
    """
    # Convert single string to list
    cols_list = [cols] if isinstance(cols, str) else cols

    # Check if DataFrame is empty
    if data.empty:
        if verbose:
            print("❌ DataFrame is empty.")
        return False

    # Check if all columns exist in the DataFrame
    missing_cols = [col for col in cols_list if col not in data.columns]
    if missing_cols:
        if verbose:
            quoted_missing = [f"'{col}'" for col in missing_cols]
            print(
                f"❌ Column(s) {', '.join(quoted_missing)} do not exist in the DataFrame."
            )
        return False

    # Check and report missing values in each specified column
    cols_with_missing = []
    cols_without_missing = []
    for col in cols_list:
        missing_count = data[col].isna().sum()
        if missing_count > 0:
            cols_with_missing.append(col)
            if verbose:
                print(
                    f"⚠️ There are {missing_count:,} row(s) with missing values in column '{col}'."
                )
        else:
            cols_without_missing.append(col)

    if verbose:
        if cols_without_missing:
            quoted_cols = [f"'{col}'" for col in cols_without_missing]
            if len(quoted_cols) == 1:
                print(f"✅ There are no missing values in column {quoted_cols[0]}.")
            else:
                print(
                    f"✅ There are no missing values in columns {', '.join(quoted_cols)}."
                )

    # Filter out rows with missing values
    filtered_df = data.dropna(subset=cols_list)

    # Get counts for comparison
    total_row_count = len(filtered_df)
    unique_row_count = filtered_df.groupby(cols_list).size().reset_index().shape[0]

    if verbose:
        print(f"ℹ️ Total row count after filtering out missings: {total_row_count:,}")
        print(f"ℹ️ Unique row count after filtering out missings: {unique_row_count:,}")

    is_primary = unique_row_count == total_row_count

    if verbose:
        quoted_cols = [f"'{col}'" for col in cols_list]
        if is_primary:
            message = "form a primary key"
            if cols_with_missing:
                message += " after removing rows with missing values"
            print(f"🔑 The column(s) {', '.join(quoted_cols)} {message}.")
        else:
            print(
                f"❌ The column(s) {', '.join(quoted_cols)} do not form a primary key."
            )

    return is_primary


def select_existing_cols(
    data: pd.DataFrame,
    cols: Union[str, List[str]],
    verbose: bool = False,
    case_sensitive: bool = True,
) -> pd.DataFrame:
    """
    Select columns from a DataFrame if they exist.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    cols : Union[str, List[str]]
        Column name or list of column names to select.
    verbose : bool, default=False
        If True, print which columns exist vs. are missing.
    case_sensitive : bool, default=True
        If True, match column names exactly (case-sensitive).
        If False, match case-insensitively by lowering both data columns and input list.
        Returned DataFrame will still use original column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the matched columns (with original column casing).

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
    >>> select_existing_cols(df, ['A', 'D', 'b'], case_sensitive=True)  # Only returns 'A'
    >>> select_existing_cols(df, ['A', 'D', 'b'], case_sensitive=False)  # Returns 'A' and 'B'
    >>> select_existing_cols(df, ['A', 'D'], verbose=True)  # Shows found/missing columns
    """
    if not hasattr(data, "columns"):
        raise TypeError(
            "Input `data` must be a DataFrame-like object with a `.columns` attribute."
        )

    if isinstance(cols, str):
        cols = [cols]

    df_columns = list(data.columns)

    if case_sensitive:
        existing = [col for col in cols if col in df_columns]
    else:
        # Case-insensitive match
        lower_map = {col.lower(): col for col in df_columns}
        existing = [lower_map[col.lower()] for col in cols if col.lower() in lower_map]

    missing = [
        col
        for col in cols
        if col not in existing
        and (
            col
            if case_sensitive
            else col.lower() not in [c.lower() for c in df_columns]
        )
    ]

    if verbose:
        print(f"✅ Columns found: {existing}")
        if missing:
            print(f"⚠️ Columns not found: {missing}")

    return data[existing]


def filter_rows_by_substring(
    data: pd.DataFrame,
    column: str,
    substring: str,
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    Filter rows in a DataFrame where a specified column contains a given substring.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to filter.
    column : str
        The name of the column to search within.
    substring : str
        The substring to search for in the column values.
    case_sensitive : bool, default=False
        Whether the matching should be case-sensitive.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only the rows where the column values
        contain the specified substring.

    Raises
    ------
    KeyError
        If the specified column does not exist in the DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'alice']})
    >>> filter_rows_by_substring(df, 'name', 'alice')
         name
    0    Alice
    3    alice

    >>> filter_rows_by_substring(df, 'name', 'alice', case_sensitive=True)
         name
    3    alice
    """
    if column not in data.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    mask = (
        data[column].astype(str).str.contains(substring, case=case_sensitive, na=False)
    )
    return data[mask]


def filter_columns_by_substring(
    data: pd.DataFrame,
    substring: str,
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    Filter columns in a DataFrame by keeping only those whose names contain a given substring.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame whose columns are to be filtered.
    substring : str
        The substring to search for in column names.
    case_sensitive : bool, default=False
        Whether the matching should be case-sensitive.

    Returns
    -------
    pd.DataFrame
        A DataFrame with only the columns whose names contain the specified substring.

    Raises
    ------
    ValueError
        If no columns match the substring.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'price_usd': [100, 200],
    ...     'price_eur': [90, 180],
    ...     'name': ['A', 'B']
    ... })
    >>> filter_columns_by_substring(df, 'price')
       price_usd  price_eur
    0        100         90
    1        200        180

    >>> filter_columns_by_substring(df, 'USD', case_sensitive=True)
    Empty DataFrame
    Columns: []
    Index: [0, 1]

    >>> filter_columns_by_substring(df, 'usd', case_sensitive=False)
       price_usd
    0        100
    1        200
    """
    if case_sensitive:
        matching_cols = [col for col in data.columns if substring in str(col)]
    else:
        matching_cols = [
            col for col in data.columns if substring.lower() in str(col).lower()
        ]

    if not matching_cols:
        # Return empty DataFrame with same index but no columns
        return pd.DataFrame(index=data.index)

    return data[matching_cols]


def find_duplicates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Function to find duplicate rows based on specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    cols : List[str]
        List of column names to check for duplicates.

    Returns
    -------
    pd.DataFrame
        DataFrame containing duplicate rows based on the specified columns,
        with the specified columns and the 'count' column as the first columns,
        along with the rest of the columns from the original DataFrame,
        ordered by the specified columns.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3, 4, 5],
    ...     'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    ...     'age': [25, 30, 25, 35, 35]
    ... })
    >>> find_duplicates(df, ['name'])
       count   name  id  age
    0      2  Alice   1   25
    1      2  Alice   3   25
    2      2    Bob   2   30
    3      2    Bob   5   35

    >>> find_duplicates(df, ['name', 'age'])
       count   name  age  id
    0      2  Alice   25   1
    1      2  Alice   25   3
    """
    # Remove rows with NULLs in any of the specified columns
    filtered_df = df.dropna(subset=cols)

    # Group by the specified columns and count occurrences
    dup_counts = filtered_df.groupby(cols).size().reset_index(name="count")

    # Keep only groups with count > 1
    duplicates = dup_counts[dup_counts["count"] > 1]

    if duplicates.empty:
        # No duplicates found
        return pd.DataFrame(
            columns=["count"] + cols + [c for c in df.columns if c not in cols]
        )

    # Merge to get full rows
    result = (
        pd.merge(duplicates, filtered_df, on=cols, how="inner")
        .sort_values(by=cols)
        .reset_index(drop=True)
    )

    # Reorder columns: count + specified cols + other cols
    other_cols = [c for c in df.columns if c not in cols]
    result = result[["count"] + cols + other_cols]

    return result


def deduplicate_by_rank(
    df: pd.DataFrame,
    id_cols: Union[str, List[str]],
    ranking_col: str,
    ascending: bool = False,
    tiebreaker_col: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Deduplicate rows by keeping the best-ranked row per group of id_cols,
    optionally breaking ties by preferring non-missing tiebreaker_col.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to deduplicate.
    id_cols : Union[str, List[str]]
        Column(s) defining the unique entity (e.g., customer_id, product_id).
    ranking_col : str
        The column to rank within each group (e.g., 'date', 'score', 'priority').
    ascending : bool, default=False
        Sort order for ranking_col:
        - True: smallest value kept (e.g., earliest date)
        - False: largest value kept (e.g., most recent date, highest score)
    tiebreaker_col : Optional[str], default=None
        Column where non-missing values are preferred in case of ties.
        Useful when ranking_col has identical values.
    verbose : bool, default=False
        If True, print information about the deduplication process.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with one row per unique combination of id_cols.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'customer_id': ['C001', 'C001', 'C002', 'C002', 'C003'],
    ...     'transaction_date': ['2024-01-01', '2024-01-15', '2024-01-05', '2024-01-10', '2024-01-20'],
    ...     'amount': [100, 200, 150, 150, 300],
    ...     'email': ['old@email.com', 'new@email.com', None, 'email@test.com', 'test@email.com']
    ... })

    >>> # Keep most recent transaction per customer
    >>> deduplicate_by_rank(df, 'customer_id', 'transaction_date', ascending=False)
       customer_id transaction_date  amount           email
    0        C001       2024-01-15     200   new@email.com
    1        C002       2024-01-10     150  email@test.com
    2        C003       2024-01-20     300  test@email.com

    >>> # Keep highest amount, break ties by preferring non-null email
    >>> deduplicate_by_rank(df, 'customer_id', 'amount', ascending=False, tiebreaker_col='email')
       customer_id transaction_date  amount           email
    0        C001       2024-01-15     200   new@email.com
    1        C002       2024-01-10     150  email@test.com
    2        C003       2024-01-20     300  test@email.com
    """
    # Handle empty DataFrame
    if df.empty:
        if verbose:
            print("⚠️ Input DataFrame is empty. Returning empty DataFrame.")
        return df.copy()

    # Normalize id_cols to list
    if isinstance(id_cols, str):
        id_cols = [id_cols]

    # Validate that all columns exist
    missing_cols = [col for col in id_cols + [ranking_col] if col not in df.columns]
    if tiebreaker_col and tiebreaker_col not in df.columns:
        missing_cols.append(tiebreaker_col)

    if missing_cols:
        raise ValueError(f"Column(s) {missing_cols} not found in DataFrame")

    if verbose:
        initial_count = len(df)
        unique_groups = df[id_cols].drop_duplicates().shape[0]
        print(f"🔄 Deduplicating {initial_count} rows by {id_cols}")
        print(f"ℹ️ Found {unique_groups} unique groups")

    # Prepare sorting columns and orders
    sort_cols = id_cols + [ranking_col]
    sort_orders = [True] * len(id_cols) + [ascending]

    df_sorted = df.copy()

    # Add tiebreaker logic if specified
    if tiebreaker_col:
        df_sorted["_tiebreaker_isna"] = df_sorted[tiebreaker_col].isna().astype(int)
        sort_cols.append("_tiebreaker_isna")
        sort_orders.append(True)  # Prefer non-missing (0) over missing (1)

    # Sort by all criteria
    df_sorted = df_sorted.sort_values(by=sort_cols, ascending=sort_orders)

    # Keep first row per group (best-ranked after sorting)
    dedup_df = df_sorted.drop_duplicates(subset=id_cols, keep="first").reset_index(
        drop=True
    )

    # Remove temporary tiebreaker column
    if tiebreaker_col:
        dedup_df = dedup_df.drop(columns="_tiebreaker_isna")

    if verbose:
        final_count = len(dedup_df)
        removed_count = initial_count - final_count
        print(f"✅ Removed {removed_count} duplicate rows")
        print(f"📊 Final dataset: {final_count} rows")

    return dedup_df


def pivot_by_group(
    data: pd.DataFrame,
    id_column: str = "id",
    group_column: str = "group",
    separator: str = "_",
    agg_func: str = "first",
    sort_columns: bool = True,
    handle_duplicates: str = "raise",
) -> pd.DataFrame:
    """
    Pivot a DataFrame by group values, transforming data from long to wide format.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with separate ID and group columns
    id_column : str, default='id'
        Name of the column that uniquely identifies each entity
    group_column : str, default='group'
        Name of the column containing the values to be used as column suffixes
    separator : str, default='_'
        Separator to use between column name and group value
    agg_func : str, default='first'
        Aggregation function to use when there are duplicate id-group combinations.
        Options: 'first', 'last', 'mean', 'sum', 'min', 'max'
    sort_columns : bool, default=True
        Whether to sort the output columns alphabetically
    handle_duplicates : str, default='raise'
        How to handle duplicate id-group combinations:
        - 'raise': Raise an error with details about duplicates
        - 'warn': Print warning with duplicate details and apply agg_func


    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame with ID as primary key and grouped columns

    Raises
    ------
    ValueError
        - If required columns are not found in DataFrame
        - If group_column contains null values
        - If handle_duplicates='raise' and duplicates are found
        - If agg_func is not one of the supported functions
        - If handle_duplicates is not one of 'raise' or 'warn'

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'id': ['aaa', 'aaa', 'bbb'],
    ...     'group': [1, 2, 1],
    ...     'surname': ['smith', 'cook', 'jones'],
    ...     'age': [25, 30, 35]
    ... })
    >>> pivot_by_group(df)
       id  surname_1  surname_2  age_1  age_2
    0  aaa     smith      cook     25     30
    1  bbb     jones      NaN     35    NaN

    >>> # Handle duplicates with aggregation
    >>> df_dups = pd.DataFrame({
    ...     'id': ['aaa', 'aaa', 'bbb'],
    ...     'group': [1, 1, 1],
    ...     'value': [10, 20, 30]
    ... })
    >>> pivot_by_group(df_dups, handle_duplicates='warn', agg_func='mean')
       id  value_1
    0  aaa     15.0
    1  bbb     30.0
    """
    # Handle empty DataFrame first
    if data.empty:
        return pd.DataFrame()

    # Input validation
    if id_column not in data.columns:
        raise ValueError(f"ID column '{id_column}' not found in DataFrame")
    if group_column not in data.columns:
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")

    # Validate aggregation function
    valid_agg_funcs = ["first", "last", "mean", "sum", "min", "max"]
    if agg_func not in valid_agg_funcs:
        raise ValueError(f"agg_func must be one of {valid_agg_funcs}")

    # Validate handle_duplicates
    valid_duplicate_handlers = ["raise", "warn"]
    if handle_duplicates not in valid_duplicate_handlers:
        raise ValueError(f"handle_duplicates must be one of {valid_duplicate_handlers}")

    # Handle nulls in group column
    if data[group_column].isna().any():
        raise ValueError(f"Found null values in group column '{group_column}'")

    # Check for duplicates using is_primary_key
    is_unique = is_primary_key(data=data, cols=[id_column, group_column], verbose=False)

    if not is_unique:
        duplicate_counts = (
            data.groupby([id_column, group_column])
            .size()
            .reset_index(name="count")
            .query("count > 1")
        )

        if handle_duplicates == "raise":
            raise ValueError(
                "ℹ️ Found duplicate id-group combinations. "
                f"Clean up dups before pivoting, or set handle_duplicates='warn' and aggregate duplicates using '{agg_func}'."
            )
        elif handle_duplicates == "warn":
            import warnings

            warnings.warn(
                f"ℹ️ Found {len(duplicate_counts)} duplicate id-group combinations. "
                f"Aggregating using {agg_func}."
            )

    # Get columns to pivot (exclude ID and group columns)
    value_cols = [col for col in data.columns if col not in [id_column, group_column]]

    # Melt the DataFrame to long format
    df_melted = data.melt(
        id_vars=[id_column, group_column],
        value_vars=value_cols,
        var_name="field",
        value_name="_temp_value_",
    )

    # Create new column names with group values
    df_melted["new_field"] = (
        df_melted["field"] + separator + df_melted[group_column].astype(str)
    )

    # Pivot to get the final structure
    result = df_melted.pivot_table(
        index=id_column, columns="new_field", values="_temp_value_", aggfunc=agg_func
    ).reset_index()

    # Clean up column names
    result.columns.name = None

    # Sort columns if requested
    if sort_columns:
        # Keep id_column as first column
        other_cols = sorted(col for col in result.columns if col != id_column)
        result = result[[id_column] + other_cols]

    # Remove columns with all NaN values
    result = result.dropna(axis=1, how="all")

    return result


def clean_null_representations(
    data: pd.DataFrame,
    numeric_sentinel_values: Optional[dict] = None,
    numeric_range_limits: Optional[dict] = None,
    replace_inf: bool = True,
    zero_as_null_cols: Optional[List[str]] = None,
    custom_string_nulls: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Clean null representations in both numeric and categorical columns.

    This function standardizes various null representations across different data types:
    - Categorical columns: Converts string nulls ('nan', 'NULL', etc.) to None
    - Numeric columns: Converts inf, sentinel values, and out-of-range values to NaN

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to clean
    numeric_sentinel_values : dict, optional
        Dictionary mapping column names to lists of values that should be treated as null.
        Example: {'age': [-1, 999], 'score': [-999, 9999]}
    numeric_range_limits : dict, optional
        Dictionary mapping column names to min/max valid ranges. Values outside
        these ranges will be converted to NaN.
        Example: {'age': {'min': 0, 'max': 150}, 'percentage': {'min': 0, 'max': 100}}
    replace_inf : bool, default True
        Whether to replace infinite values (inf, -inf) with NaN in numeric columns
    zero_as_null_cols : List[str], optional
        List of numeric column names where zero values should be treated as null.
        Example: ['optional_id', 'bonus_amount']
    custom_string_nulls : List[str], optional
        Additional string representations to treat as null beyond the default list.
        Default nulls: ['nan', 'NaN', 'None', 'null', 'NULL', '', '<NA>', 'N/A', 'n/a']
        Example: ['missing', 'unknown', '--', 'TBD']
    verbose : bool, default False
        If True, print information about the cleaning process.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized null representations

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'name': ['John', 'nan', None, 'Alice'],
    ...     'age': [25, -1, 30, np.inf],
    ...     'score': [85.5, -999, np.nan, 88.0],
    ...     'status': ['active', 'NULL', '', 'inactive']
    ... })

    >>> # Define custom cleaning rules
    >>> sentinel_values = {
    ...     'age': [-1, 999],
    ...     'score': [-999, 9999]
    ... }
    >>> range_limits = {
    ...     'age': {'min': 0, 'max': 120},
    ...     'score': {'min': 0, 'max': 100}
    ... }
    >>> custom_nulls = ['missing', 'unknown', '--']
    >>>
    >>> cleaned_df = clean_null_representations(
    ...     df,
    ...     numeric_sentinel_values=sentinel_values,
    ...     numeric_range_limits=range_limits,
    ...     custom_string_nulls=custom_nulls,
    ...     verbose=True
    ... )

    Notes
    -----
    - The function preserves original data types where possible
    - Numeric columns use np.nan for null values
    - Categorical columns use None for null values
    - The function creates a copy of the input DataFrame
    - Processing order: inf replacement → sentinel values → range limits → zero replacement

    See Also
    --------
    drop_fully_null_cols : Drop columns where all values are missing/null
    value_counts_with_pct : Calculate value counts including null values
    """
    df_clean = data.copy()

    # Default string null representations
    default_string_nulls = [
        "nan",
        "NaN",
        "None",
        "null",
        "NULL",
        "",
        "<NA>",
        "N/A",
        "n/a",
    ]
    string_nulls_to_replace = default_string_nulls + (custom_string_nulls or [])

    # Get column types
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    categorical_cols = df_clean.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    if verbose:
        print(
            f"🔍 Found {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns"
        )

    # === CLEAN CATEGORICAL COLUMNS ===
    if len(categorical_cols) > 0:
        if verbose:
            print(f"🧹 Cleaning categorical columns: {list(categorical_cols)}")
            print(f"   Replacing values: {string_nulls_to_replace}")

        df_clean[categorical_cols] = df_clean[categorical_cols].apply(
            lambda col: col.where(
                ~col.astype(str).str.strip().isin(string_nulls_to_replace), None
            )
        )

    # === CLEAN NUMERIC COLUMNS ===
    for col in numeric_cols:
        if verbose:
            print(f"\n🔢 Cleaning numeric column: {col}")

        # 1. Replace infinity values
        if replace_inf:
            inf_mask = np.isinf(df_clean[col])
            inf_count = inf_mask.sum()
            if inf_count > 0 and verbose:
                print(f"   Replaced {inf_count} infinite values")
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

        # 2. Replace sentinel values
        if numeric_sentinel_values and col in numeric_sentinel_values:
            sentinel_mask = df_clean[col].isin(numeric_sentinel_values[col])
            sentinel_count = sentinel_mask.sum()
            if sentinel_count > 0 and verbose:
                print(
                    f"   Replaced {sentinel_count} sentinel values: {numeric_sentinel_values[col]}"
                )
            df_clean[col] = df_clean[col].replace(numeric_sentinel_values[col], np.nan)

        # 3. Apply range limits
        if numeric_range_limits and col in numeric_range_limits:
            limits = numeric_range_limits[col]
            original_valid = df_clean[col].notna().sum()

            if "min" in limits:
                df_clean[col] = df_clean[col].where(
                    df_clean[col] >= limits["min"], np.nan
                )
            if "max" in limits:
                df_clean[col] = df_clean[col].where(
                    df_clean[col] <= limits["max"], np.nan
                )

            new_valid = df_clean[col].notna().sum()
            if verbose and original_valid != new_valid:
                print(f"   Replaced {original_valid - new_valid} out-of-range values")
                if "min" in limits:
                    print(f"   - Below minimum ({limits['min']})")
                if "max" in limits:
                    print(f"   - Above maximum ({limits['max']})")

        # 4. Replace zeros if specified
        if zero_as_null_cols and col in zero_as_null_cols:
            zero_mask = df_clean[col] == 0
            zero_count = zero_mask.sum()
            if zero_count > 0 and verbose:
                print(f"   Replaced {zero_count} zero values")
            df_clean[col] = df_clean[col].replace(0, np.nan)

    if verbose:
        total_nulls = df_clean.isnull().sum().sum()
        print(f"\n✨ Cleaning complete! Total null values: {total_nulls:,}")

    return df_clean
