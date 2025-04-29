import pandas as pd
import pytest

from mlarena.utils.data_utils import (
    clean_dollar_cols,
    transform_date_cols,
    value_counts_with_pct,
)


def test_clean_dollar_cols():
    # Test data
    df = pd.DataFrame(
        {
            "price": ["$1,000", "$2,000", "$3,000"],
            "salary": ["$50,000", "$60,000", "$70,000"],
            "other": ["a", "b", "c"],
        }
    )

    # Test cleaning single column
    result = clean_dollar_cols(df, ["price"])
    assert result["price"].dtype == "float64"
    assert result["price"].tolist() == [1000.0, 2000.0, 3000.0]
    assert result["salary"].tolist() == ["$50,000", "$60,000", "$70,000"]  # unchanged

    # Test cleaning multiple columns
    result = clean_dollar_cols(df, ["price", "salary"])
    assert result["price"].dtype == "float64"
    assert result["salary"].dtype == "float64"
    assert result["price"].tolist() == [1000.0, 2000.0, 3000.0]
    assert result["salary"].tolist() == [50000.0, 60000.0, 70000.0]

    # Test with invalid values
    df_invalid = pd.DataFrame({"price": ["$1,000", "invalid", "$3,000"]})
    result = clean_dollar_cols(df_invalid, ["price"])
    assert pd.isna(result["price"][1])  # invalid value should be NaN


def test_value_counts_with_pct():
    # Test data
    df = pd.DataFrame({"category": ["A", "A", "B", "B", "B", "C", None]})

    # Test basic functionality
    result = value_counts_with_pct(df, "category")
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"category", "count", "pct"}
    assert result["count"].sum() == 7  # including None
    assert abs(result["pct"].sum() - 100.0) < 0.1  # accomodate for rounding

    # Test with dropna=True
    result = value_counts_with_pct(df, "category", dropna=True)
    assert result["count"].sum() == 6  # excluding None
    assert abs(result["pct"].sum() - 100.0) < 0.1  # accomodate for rounding

    # Test with different decimal places
    result = value_counts_with_pct(df, "category", decimals=1)
    assert all(result["pct"].apply(lambda x: len(str(x).split(".")[1]) <= 1))

    # Test with non-existent column
    with pytest.raises(ValueError):
        value_counts_with_pct(df, "non_existent_column")


def test_transform_date_cols():
    # Test data
    df = pd.DataFrame(
        {
            "date1": ["20230101", "20230102", "20230103"],
            "date2": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "other": ["a", "b", "c"],
        }
    )

    # Test with default format
    result = transform_date_cols(df, ["date1"])
    assert pd.api.types.is_datetime64_any_dtype(result["date1"])
    assert not pd.api.types.is_datetime64_any_dtype(result["date2"])  # unchanged

    # Test with custom format
    result = transform_date_cols(df, ["date2"], str_date_format="%Y-%m-%d")
    assert pd.api.types.is_datetime64_any_dtype(result["date2"])

    # Test with multiple columns
    result = transform_date_cols(df, ["date1", "date2"])
    assert pd.api.types.is_datetime64_any_dtype(result["date1"])
    assert pd.api.types.is_datetime64_any_dtype(result["date2"])

    # Test with empty date_cols list
    with pytest.raises(ValueError):
        transform_date_cols(df, [])

    # Test with invalid dates
    df_invalid = pd.DataFrame({"date": ["20230101", "invalid", "20230103"]})
    result = transform_date_cols(df_invalid, ["date"])
    assert pd.isna(result["date"][1])  # invalid date should be NaT
