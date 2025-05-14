from typing import List, Union

import pandas as pd

__all__ = [
    "clean_dollar_cols",
    "value_counts_with_pct",
    "transform_date_cols",
    "drop_fully_null_columns",
    "print_schema_alphabetically",
    "is_primary_key",
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
    data: pd.DataFrame, column_name: str, dropna: bool = False, decimals: int = 2
) -> pd.DataFrame:
    """
    Calculate the count and percentage of occurrences for each unique value in the specified column.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the column for which to calculate value counts.
    dropna : bool, default=False
        Whether to exclude NA/null values.
    decimals : int, default=2
        Number of decimal places to round the percentage.

    Returns
    -------
    pd.DataFrame
        A DataFrame with unique values, their counts, and percentages.
    """
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    counts = data[column_name].value_counts(dropna=dropna, normalize=False)
    percentages = (counts / counts.sum() * 100).round(decimals)

    result = (
        pd.DataFrame(
            {
                column_name: counts.index,
                "count": counts.values,
                "pct": percentages.values,
            }
        )
        .sort_values(by="count", ascending=False)
        .reset_index(drop=True)
    )

    return result


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
            %Y: Four-digit year (e.g., 2024)

        Example formats:
            "%Y%m%d"   → '20240825'
            "%d-%m-%Y" → '25-08-2024'
            "%d%b%Y"   → '25Aug2024'

        Note:
            If the format uses %b (abbreviated month), strings like '25AUG2024'
            will be handled automatically by converting to title case before parsing.

    Returns
    -------
    pd.DataFrame
        The DataFrame with specified columns transformed to datetime format.

    Raises
    ------
    ValueError
        If date_cols is empty.
    """
    if isinstance(date_cols, str):
        date_cols = [date_cols]

    if not date_cols:
        raise ValueError("date_cols list cannot be empty")

    df_ = data.copy()
    for date_col in date_cols:
        if not pd.api.types.is_datetime64_any_dtype(df_[date_col]):
            if "%b" in str_date_format:
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


def drop_fully_null_columns(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Drops columns where all values are missing/null in a pandas DataFrame.

    This function is particularly useful when working with Databricks' display() function,
    which can break when encountering columns that are entirely null as it cannot
    infer the schema. Running this function before display() helps prevent such issues.

    Parameters
    ----------
    df : pd.DataFrame
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
    >>> drop_fully_null_columns(df).display()  # this won't affect the original df, just ensure .display() work
    >>> # To see which columns were dropped:
    >>> drop_fully_null_columns(df, verbose=True)
    """
    null_counts = df.isnull().sum()
    all_missing_cols = null_counts[null_counts == len(df)].index.tolist()

    if all_missing_cols and verbose:
        print(f"Dropped fully-null columns: {all_missing_cols}")

    df_ = df.drop(columns=all_missing_cols)
    return df_


def print_schema_alphabetically(df: pd.DataFrame) -> None:
    """
    Prints the schema (column names and dtypes) of the DataFrame with columns sorted alphabetically.

    This is particularly useful when comparing schemas between different DataFrames
    or versions of the same DataFrame, as the alphabetical ordering makes it easier
    to spot differences.

    Parameters
    ----------
    df : pd.DataFrame
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
    sorted_dtypes = df[sorted(df.columns)].dtypes
    print(sorted_dtypes)


def is_primary_key(
    df: pd.DataFrame, cols: Union[str, List[str]], verbose: bool = True
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
    df : pd.DataFrame
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
    if df.empty:
        if verbose:
            print("DataFrame is empty.")
        return False

    # Check if all columns exist in the DataFrame
    missing_cols = [col for col in cols_list if col not in df.columns]
    if missing_cols:
        if verbose:
            print(f"Column(s) {', '.join(missing_cols)} do not exist in the DataFrame.")
        return False

    # Check and report missing values in each specified column
    has_missing = False
    for col in cols_list:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            has_missing = True
            if verbose:
                print(
                    f"There are {missing_count:,} row(s) with missing values in column '{col}'."
                )

    # Filter out rows with missing values
    filtered_df = df.dropna(subset=cols_list)

    # Get counts for comparison
    total_row_count = len(filtered_df)
    unique_row_count = filtered_df.groupby(cols_list).size().reset_index().shape[0]

    if verbose:
        print(f"Total row count after filtering out missings: {total_row_count:,}")
        print(f"Unique row count after filtering out missings: {unique_row_count:,}")

    is_primary = unique_row_count == total_row_count

    if verbose:
        if is_primary:
            message = "form a primary key"
            if has_missing:
                message += " after removing rows with missing values"
            print(f"The column(s) {', '.join(cols_list)} {message}.")
        else:
            print(f"The column(s) {', '.join(cols_list)} do not form a primary key.")

    return is_primary
