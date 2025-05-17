import pandas as pd
import pytest

from mlarena.utils.data_utils import (
    clean_dollar_cols,
    drop_fully_null_cols,
    is_primary_key,
    select_existing_cols,
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
    """Test value_counts_with_pct function with various input combinations.

    Tests:
    - Basic functionality with single column
    - Multiple columns handling (value combinations)
    - NA handling (dropna parameter)
    - Decimal places control
    - Error handling for non-existent columns
    """
    # Test data
    df = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "B", None],
            "status": ["Active", "Active", "Inactive", None, None, None],
        }
    )

    # Test 1: Basic functionality with single column
    result = value_counts_with_pct(df, "category")
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"category", "count", "pct"}
    assert result["count"].sum() == 6  # including None
    assert abs(result["pct"].sum() - 100.0) < 0.1  # accommodate for rounding

    # Test 2: Multiple columns - should count unique combinations
    result_multi = value_counts_with_pct(df, ["category", "status"])
    assert isinstance(result_multi, pd.DataFrame)
    assert set(result_multi.columns) == {"category", "status", "count", "pct"}
    assert result_multi["count"].sum() == 6  # total rows
    assert abs(result_multi["pct"].sum() - 100.0) < 0.1
    # Check specific combinations
    assert len(result_multi) == 4  # should have 4 unique combinations
    # Most frequent combinations
    assert result_multi.iloc[0]["count"] == 2  # B-None or A-Active should have count 2
    assert result_multi.iloc[1]["count"] == 2  # B-None or A-Active should have count 2

    # Test 3: dropna parameter
    result_no_na = value_counts_with_pct(df, "category", dropna=True)
    assert result_no_na["count"].sum() == 5  # excluding None
    assert abs(result_no_na["pct"].sum() - 100.0) < 0.1

    # Test 4: decimals parameter
    result_decimals = value_counts_with_pct(df, "category", decimals=1)
    assert all(result_decimals["pct"].apply(lambda x: len(str(x).split(".")[1]) <= 1))

    # Test 5: Error handling - non-existent column
    with pytest.raises(ValueError):
        value_counts_with_pct(df, "non_existent_column")

    # Test 6: Error handling - non-existent columns in list
    with pytest.raises(ValueError):
        value_counts_with_pct(df, ["category", "non_existent"])


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


def test_drop_fully_null_cols(capsys):
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
    result = drop_fully_null_cols(df)
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
    result = drop_fully_null_cols(df, verbose=True)
    captured = capsys.readouterr()
    assert "Dropped fully-null columns" in captured.out
    assert "all_null" in captured.out
    assert "all_null_2" in captured.out

    # Test with no null columns
    df_no_nulls = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    result_no_nulls = drop_fully_null_cols(df_no_nulls)
    pd.testing.assert_frame_equal(result_no_nulls, df_no_nulls)

    # Test with all null columns
    df_all_nulls = pd.DataFrame({"col1": [None, None], "col2": [pd.NA, pd.NA]})
    result_all_nulls = drop_fully_null_cols(df_all_nulls)
    assert len(result_all_nulls.columns) == 0  # All columns should be dropped


def test_is_primary_key(capsys):
    """Test is_primary_key function with various scenarios.

    Tests:
    - Single column with missing values
    - Single column without missing values
    - Multiple columns with missing values
    - Multiple columns without missing values
    - Non-primary key columns
    - Empty DataFrame
    - Non-existent columns
    - Verbose mode on/off
    """
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
            "complete": [1, 2, 3, 4, 5],  # No missing values, unique
        }
    )

    # Test 1: Single column that is a primary key (after removing nulls)
    assert is_primary_key(data=df, cols="id")
    captured = capsys.readouterr()
    assert "missing values in column 'id'" in captured.out
    assert "form a primary key after removing rows with missing values" in captured.out

    # Test 2: Single column with no missing values that is a primary key
    assert is_primary_key(data=df, cols="complete")
    captured = capsys.readouterr()
    assert "There are no missing values in column 'complete'" in captured.out
    assert "form a primary key" in captured.out

    # Test 3: Single column that is not a primary key (has duplicates)
    assert not is_primary_key(data=df, cols="code")
    captured = capsys.readouterr()
    assert "There are no missing values in column 'code'" in captured.out
    assert "do not form a primary key" in captured.out

    # Test 4: Multiple columns that form a primary key
    assert is_primary_key(data=df, cols=["code", "date"])
    captured = capsys.readouterr()
    assert "There are no missing values in columns 'code', 'date'" in captured.out
    assert "form a primary key" in captured.out

    # Test 5: Empty DataFrame
    empty_df = pd.DataFrame(columns=["id", "value"])
    assert not is_primary_key(data=empty_df, cols="id")
    captured = capsys.readouterr()
    assert "DataFrame is empty" in captured.out

    # Test 6: Non-existent column
    assert not is_primary_key(data=df, cols="non_existent")
    captured = capsys.readouterr()
    assert "do not exist in the DataFrame" in captured.out

    # Test 7: Verbose mode off
    result = is_primary_key(data=df, cols="id", verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""  # No output when verbose=False
    assert result  # Should still return True

    # Test 8: Multiple columns with one having null
    df_composite = pd.DataFrame(
        {
            "id": [1, 1, None, 2],
            "sub_id": ["A", "B", "C", "A"],
        }
    )
    assert is_primary_key(data=df_composite, cols=["id", "sub_id"])
    captured = capsys.readouterr()
    assert "missing values in column 'id'" in captured.out
    assert "form a primary key after removing rows with missing values" in captured.out

    # Test 9: String input vs List input equivalence
    single_col = is_primary_key(data=df, cols="value")
    list_col = is_primary_key(data=df, cols=["value"])
    assert single_col == list_col  # Both forms should give same result


def test_select_existing_cols():
    """Test select_existing_cols function with various input combinations.

    Tests:
    - Basic column selection
    - Case-sensitive matching
    - Case-insensitive matching
    - String input handling
    - Non-DataFrame input handling
    - Verbose output
    """
    # Test data
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "Mixed_Case": [7, 8, 9]})

    # Test basic selection
    result = select_existing_cols(df, ["A", "C"])
    assert list(result.columns) == ["A"]

    # Test case-sensitive matching
    result = select_existing_cols(df, ["a", "B"], strict=True)
    assert list(result.columns) == ["B"]

    # Test case-insensitive matching
    result = select_existing_cols(df, ["a", "mixed_case"], strict=False)
    assert set(result.columns) == {"A", "Mixed_Case"}

    # Test with string input
    result = select_existing_cols(df, "A")
    assert list(result.columns) == ["A"]

    # Test with non-DataFrame input
    with pytest.raises(TypeError):
        select_existing_cols({"not": "a dataframe"}, ["A"])

    # Test verbose output
    result = select_existing_cols(df, ["A", "NonExistent"], verbose=True)
    assert list(result.columns) == ["A"]
