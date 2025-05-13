import pandas as pd
import pytest

from mlarena.utils.data_utils import (
    clean_dollar_cols,
    drop_fully_null_columns,
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


def test_drop_fully_null_columns(capsys):
    # Test data with various null patterns
    df = pd.DataFrame(
        {
            "all_null": [None, None, None],
            "partial_null": [1, None, 3],
            "no_null": [1, 2, 3],
            "all_null_2": [pd.NA, pd.NA, pd.NA],
        }
    )

    # Keep original for comparison
    df_original = df.copy()

    # Test dropping fully null columns
    result = drop_fully_null_columns(df)

    # Check that fully null columns are dropped
    assert "all_null" not in result.columns
    assert "all_null_2" not in result.columns

    # Check that partially null and non-null columns are kept
    assert "partial_null" in result.columns
    assert "no_null" in result.columns

    # Check that original DataFrame is not modified
    pd.testing.assert_frame_equal(df, df_original)

    # Check that the function prints the dropped columns
    captured = capsys.readouterr()
    assert "Dropped fully-null columns" in captured.out
    assert "all_null" in captured.out
    assert "all_null_2" in captured.out

    # Test with no null columns
    df_no_nulls = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    result_no_nulls = drop_fully_null_columns(df_no_nulls)
    pd.testing.assert_frame_equal(result_no_nulls, df_no_nulls)

    # Test with all null columns
    df_all_nulls = pd.DataFrame({"col1": [None, None], "col2": [pd.NA, pd.NA]})
    result_all_nulls = drop_fully_null_columns(df_all_nulls)
    assert len(result_all_nulls.columns) == 0  # All columns should be dropped
