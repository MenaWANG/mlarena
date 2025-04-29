from typing import List

import pandas as pd

__all__ = ["clean_dollar_cols", "value_counts_with_pct", "transform_date_cols"]


def clean_dollar_cols(df: pd.DataFrame, cols_to_clean: List[str]) -> pd.DataFrame:
    """
    Clean specified columns of a Pandas DataFrame by removing '$' symbols, commas,
    and converting to floating-point numbers.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        cols_to_clean (List[str]): List of column names to clean.

    Returns:
        pd.DataFrame: DataFrame with specified columns cleaned of '$' symbols and commas,
                      and converted to floating-point numbers.
    """
    df_ = df.copy()

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
    df: pd.DataFrame, column_name: str, dropna: bool = False, decimals: int = 2
) -> pd.DataFrame:
    """
    Calculate the count and percentage of occurrences for each unique value in the specified column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column for which to calculate value counts.
    - dropna (bool): Whether to exclude NA/null values. Default is False.
    - decimals (int): Number of decimal places to round the percentage. Default is 2.

    Returns:
    - pd.DataFrame: A DataFrame with unique values, their counts, and percentages.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    counts = df[column_name].value_counts(dropna=dropna, normalize=False)
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
    df: pd.DataFrame, date_cols: List[str], str_date_format: str = "%Y%m%d"
) -> pd.DataFrame:
    """
    Transforms specified columns in a Pandas DataFrame to datetime format.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_cols (List[str]): A list of column names to be transformed to dates.
        str_date_format (str, optional): The string format of the dates. Defaults to "%Y%m%d".

    Returns:
        pd.DataFrame: The DataFrame with specified columns transformed to datetime format.
    """
    if not date_cols:
        raise ValueError("date_cols list cannot be empty")

    df_ = df.copy()
    for date_col in date_cols:
        if not pd.api.types.is_datetime64_any_dtype(df_[date_col]):
            df_[date_col] = pd.to_datetime(
                df_[date_col], format=str_date_format, errors="coerce"
            )

    return df_
