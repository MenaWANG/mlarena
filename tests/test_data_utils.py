import pandas as pd
import pytest

from mlarena.utils.data_utils import (
    clean_dollar_cols,
    drop_fully_null_columns,
    is_primary_key,
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

    # Test dropping fully null columns (default verbose=False)
    result = drop_fully_null_columns(df)
    captured = capsys.readouterr()
    assert captured.out == ""  # No output by default

    # Check that fully null columns are dropped
    assert "all_null" not in result.columns
    assert "all_null_2" not in result.columns

    # Check that partially null and non-null columns are kept
    assert "partial_null" in result.columns
    assert "no_null" in result.columns

    # Check that original DataFrame is not modified
    pd.testing.assert_frame_equal(df, df_original)

    # Test with verbose=True
    result = drop_fully_null_columns(df, verbose=True)
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


def test_is_primary_key(capsys):
    # Test data with various scenarios
    df = pd.DataFrame(
        {
            "id": [1, 2, None, 4, 5],
            "code": ["A1", "A2", "A3", "A1", "A5"],  # Has duplicates
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
            ],
            "value": [100, 200, 300, 400, 500],
        }
    )

    # Test 1: Single column that is a primary key (after removing nulls)
    assert is_primary_key(df, "id")
    captured = capsys.readouterr()
    assert "missing values in column 'id'" in captured.out
    assert "form a primary key after removing rows with missing values" in captured.out

    # Test 2: Single column that is not a primary key (has duplicates)
    assert not is_primary_key(df, "code")
    captured = capsys.readouterr()
    assert "do not form a primary key" in captured.out

    # Test 3: Multiple columns that form a primary key
    assert is_primary_key(df, ["code", "date"])
    captured = capsys.readouterr()
    assert "form a primary key" in captured.out

    # Test 4: Empty DataFrame
    empty_df = pd.DataFrame(columns=["id", "value"])
    assert not is_primary_key(empty_df, "id")
    captured = capsys.readouterr()
    assert "DataFrame is empty" in captured.out

    # Test 5: Non-existent column
    assert not is_primary_key(df, "non_existent")
    captured = capsys.readouterr()
    assert "do not exist in the DataFrame" in captured.out

    # Test 6: Verbose mode off
    result = is_primary_key(df, "id", verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""  # No output when verbose=False
    assert result  # Should still return True

    # Test 7: Multiple columns with one having null
    df_composite = pd.DataFrame(
        {
            "id": [1, 1, None, 2],
            "sub_id": ["A", "B", "C", "A"],
        }
    )
    assert is_primary_key(df_composite, ["id", "sub_id"])
    captured = capsys.readouterr()
    assert "missing values in column 'id'" in captured.out
    assert "form a primary key after removing rows with missing values" in captured.out

    # Test 8: String input vs List input equivalence
    single_col = is_primary_key(df, "value")
    list_col = is_primary_key(df, ["value"])
    assert single_col == list_col  # Both forms should give same result
