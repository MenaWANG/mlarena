from typing import List, Union

import pandas as pd

__all__ = ["clean_dollar_cols", "value_counts_with_pct", "transform_date_cols"]


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
