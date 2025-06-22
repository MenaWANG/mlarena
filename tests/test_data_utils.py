import pandas as pd
import pytest

from mlarena.utils.data_utils import (
    clean_dollar_cols,
    deduplicate_by_rank,
    drop_fully_null_cols,
    filter_columns_by_substring,
    filter_rows_by_substring,
    find_duplicates,
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

    # Test with abbreviated month names and two-digit year
    df_abbr = pd.DataFrame({"date": ["25AUG24", "26aug24", "27Aug24"]})
    result = transform_date_cols(df_abbr, "date", str_date_format="%d%b%y")
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result["date"].dt.year.tolist() == [2024, 2024, 2024]
    assert result["date"].dt.month.tolist() == [8, 8, 8]
    assert result["date"].dt.day.tolist() == [25, 26, 27]

    # Test with full month names
    df_full = pd.DataFrame({"date": ["25AUGUST2024", "26august2024", "27August2024"]})
    result = transform_date_cols(df_full, "date", str_date_format="%d%B%Y")
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result["date"].dt.year.tolist() == [2024, 2024, 2024]
    assert result["date"].dt.month.tolist() == [8, 8, 8]
    assert result["date"].dt.day.tolist() == [25, 26, 27]


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
    result = select_existing_cols(df, ["a", "B"], case_sensitive=True)
    assert list(result.columns) == ["B"]

    # Test case-insensitive matching
    result = select_existing_cols(df, ["a", "mixed_case"], case_sensitive=False)
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


def test_filter_rows_by_substring():
    """Test filter_rows_by_substring function with various scenarios.

    Tests:
    - Basic substring filtering
    - Case-sensitive filtering
    - Case-insensitive filtering
    - Handling of NaN values
    - Non-existent column error
    - Empty results
    """
    # Test data
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "alice", None],
            "city": ["New York", "San Francisco", "Chicago", "Boston", "Seattle"],
        }
    )

    # Test basic case-insensitive filtering (default)
    result = filter_rows_by_substring(df, "name", "alice")
    assert len(result) == 2
    assert set(result["name"].dropna()) == {"Alice", "alice"}

    # Test case-sensitive filtering
    result = filter_rows_by_substring(df, "name", "alice", case_sensitive=True)
    assert len(result) == 1
    assert result["name"].iloc[0] == "alice"

    # Test case-insensitive filtering explicitly
    result = filter_rows_by_substring(df, "name", "ALICE", case_sensitive=False)
    assert len(result) == 2
    assert set(result["name"].dropna()) == {"Alice", "alice"}

    # Test with different column and substring
    result = filter_rows_by_substring(df, "city", "San")
    assert len(result) == 1
    assert result["city"].iloc[0] == "San Francisco"

    # Test with substring not found
    result = filter_rows_by_substring(df, "name", "xyz")
    assert len(result) == 0

    # Test with non-existent column
    with pytest.raises(KeyError):
        filter_rows_by_substring(df, "non_existent", "test")

    # Test handling of NaN values
    result = filter_rows_by_substring(
        df, "name", "e"
    )  # Should match "Alice", "Charlie", "alice", and None (as "None")
    assert len(result) == 4
    # Note: None gets converted to string "None" which contains "e"
    assert set(result["name"].astype(str)) == {"Alice", "Charlie", "alice", "None"}

    # Test with realistic business scenario - filtering transaction descriptions
    transactions_df = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6],
            "description": [
                "AMAZON.COM*ORDER12345 SEATTLE WA",
                "STARBUCKS STORE #1234 NEW YORK NY",
                "amazon prime subscription renewal",
                "WALMART SUPERCENTER #567 AUSTIN TX",
                "Starbucks Coffee - Downtown Location",
                "walmart.com GROCERY PICKUP",
            ],
            "amount": [25.99, 4.95, 12.99, 45.67, 3.75, 23.45],
        }
    )

    # Find all Amazon transactions (case-insensitive)
    amazon_transactions = filter_rows_by_substring(
        transactions_df, "description", "amazon"
    )
    assert len(amazon_transactions) == 2
    assert amazon_transactions["transaction_id"].tolist() == [1, 3]

    # Find Starbucks transactions (case-insensitive)
    starbucks_transactions = filter_rows_by_substring(
        transactions_df, "description", "starbucks"
    )
    assert len(starbucks_transactions) == 2
    assert starbucks_transactions["transaction_id"].tolist() == [2, 5]

    # Find Walmart transactions with case-sensitive search (should miss some variants)
    walmart_exact = filter_rows_by_substring(
        transactions_df, "description", "WALMART", case_sensitive=True
    )
    assert len(walmart_exact) == 1
    assert walmart_exact["transaction_id"].iloc[0] == 4

    # Find all Walmart variants with case-insensitive search
    walmart_all = filter_rows_by_substring(
        transactions_df, "description", "walmart", case_sensitive=False
    )
    assert len(walmart_all) == 2
    assert set(walmart_all["transaction_id"]) == {4, 6}


def test_filter_columns_by_substring():
    """Test filter_columns_by_substring function with various scenarios.

    Tests:
    - Basic substring filtering
    - Case-sensitive filtering
    - Case-insensitive filtering
    - No matching columns
    - All columns match
    - Non-string column names
    """
    # Test data
    df = pd.DataFrame(
        {
            "price_usd": [100, 200],
            "price_eur": [90, 180],
            "Price_GBP": [80, 160],
            "name": ["A", "B"],
            "date": ["2024-01-01", "2024-01-02"],
        }
    )

    # Test basic case-insensitive filtering (default)
    result = filter_columns_by_substring(df, "price")
    assert len(result.columns) == 3
    assert set(result.columns) == {"price_usd", "price_eur", "Price_GBP"}

    # Test case-sensitive filtering
    result = filter_columns_by_substring(df, "price", case_sensitive=True)
    assert len(result.columns) == 2
    assert set(result.columns) == {"price_usd", "price_eur"}

    # Test case-insensitive filtering explicitly
    result = filter_columns_by_substring(df, "PRICE", case_sensitive=False)
    assert len(result.columns) == 3
    assert set(result.columns) == {"price_usd", "price_eur", "Price_GBP"}

    # Test with different substring
    result = filter_columns_by_substring(df, "usd")
    assert len(result.columns) == 1
    assert list(result.columns) == ["price_usd"]

    # Test with no matching columns
    result = filter_columns_by_substring(df, "xyz")
    assert len(result.columns) == 0
    assert len(result.index) == len(df.index)  # Should preserve index

    # Test with all columns matching
    result = filter_columns_by_substring(df, "")  # Empty string matches all
    assert len(result.columns) == len(df.columns)

    # Test with DataFrame having non-string column names
    df_with_numeric_cols = pd.DataFrame(
        {123: [1, 2], "price_456": [3, 4], "name": ["A", "B"]}
    )
    result = filter_columns_by_substring(df_with_numeric_cols, "price")
    assert len(result.columns) == 1
    assert list(result.columns) == ["price_456"]


def test_find_duplicates():
    """Test find_duplicates function with various scenarios.

    Tests:
    - Basic duplicate finding with single column
    - Multiple column combinations
    - No duplicates scenario
    - Handling of NaN values
    - Column ordering in output
    - Empty DataFrame input
    - Various data types
    """
    # Test 1: Basic duplicate finding with single column
    df_basic = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Alice", "Charlie", "Bob"],
            "age": [25, 30, 25, 35, 30],
        }
    )

    result = find_duplicates(df_basic, ["name"])
    assert len(result) == 4  # 2 Alice rows + 2 Bob rows
    assert list(result.columns) == ["count", "name", "id", "age"]
    assert set(result["name"]) == {"Alice", "Bob"}
    assert all(result["count"] == 2)

    # Test 2: Multiple column combinations
    result_multi = find_duplicates(df_basic, ["name", "age"])
    assert len(result_multi) == 4  # Same as single column in this case
    assert list(result_multi.columns) == ["count", "name", "age", "id"]
    assert all(result_multi["count"] == 2)

    # Test 3: No duplicates scenario
    df_unique = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    )

    result_unique = find_duplicates(df_unique, ["name"])
    assert len(result_unique) == 0
    assert list(result_unique.columns) == ["count", "name", "id", "age"]

    # Test 4: Handling of NaN values - should be excluded from analysis
    df_nan = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["Alice", "Bob", None, "Alice", None, "Charlie"],
            "score": [85, 90, 88, 85, 92, 78],
        }
    )

    result_nan = find_duplicates(df_nan, ["name"])
    assert len(result_nan) == 2  # Only 2 Alice rows (None values excluded)
    assert set(result_nan["name"]) == {"Alice"}
    assert all(result_nan["count"] == 2)

    # Test 5: Complex business scenario - customer transactions
    transactions_df = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "customer_id": [
                "C001",
                "C002",
                "C001",
                "C003",
                "C002",
                "C001",
                "C004",
                "C002",
            ],
            "product": ["A", "B", "A", "C", "B", "C", "A", "A"],
            "amount": [100, 200, 100, 150, 200, 300, 100, 100],
            "date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
                "2024-01-07",
                "2024-01-08",
            ],
        }
    )

    # Find customers with duplicate product purchases
    customer_product_dups = find_duplicates(transactions_df, ["customer_id", "product"])
    assert len(customer_product_dups) == 4  # C001-A (2), C002-B (2)
    assert list(customer_product_dups.columns) == [
        "count",
        "customer_id",
        "product",
        "transaction_id",
        "amount",
        "date",
    ]

    # Find duplicate amounts
    amount_dups = find_duplicates(transactions_df, ["amount"])
    # Should find: 100 (4 times), 200 (2 times)
    assert len(amount_dups) == 6  # 4 + 2 = 6 rows
    amounts_with_dups = set(amount_dups["amount"])
    assert amounts_with_dups == {100, 200}

    # Test 6: Empty DataFrame
    df_empty = pd.DataFrame(columns=["name", "age"])
    result_empty = find_duplicates(df_empty, ["name"])
    assert len(result_empty) == 0
    assert list(result_empty.columns) == ["count", "name", "age"]

    # Test 7: Single row DataFrame (no duplicates possible)
    df_single = pd.DataFrame({"name": ["Alice"], "age": [25]})
    result_single = find_duplicates(df_single, ["name"])
    assert len(result_single) == 0
    assert list(result_single.columns) == ["count", "name", "age"]

    # Test 8: All rows identical
    df_identical = pd.DataFrame(
        {"name": ["Alice", "Alice", "Alice"], "age": [25, 25, 25]}
    )

    result_identical = find_duplicates(df_identical, ["name"])
    assert len(result_identical) == 3
    assert all(result_identical["count"] == 3)
    assert all(result_identical["name"] == "Alice")

    # Test 9: Mixed data types
    df_mixed = pd.DataFrame(
        {
            "str_col": ["A", "B", "A", "C"],
            "int_col": [1, 2, 1, 3],
            "float_col": [1.1, 2.2, 1.1, 3.3],
            "bool_col": [True, False, True, False],
        }
    )

    result_mixed = find_duplicates(df_mixed, ["str_col", "int_col"])
    assert len(result_mixed) == 2  # Two 'A', 1 combinations
    assert all(result_mixed["count"] == 2)

    # Test 10: Column ordering - make sure specified columns come first after count
    df_ordering = pd.DataFrame(
        {"z_col": ["A", "B", "A"], "a_col": [1, 2, 1], "b_col": [10, 20, 10]}
    )

    result_ordering = find_duplicates(df_ordering, ["a_col", "z_col"])
    expected_columns = ["count", "a_col", "z_col", "b_col"]
    assert list(result_ordering.columns) == expected_columns


def test_deduplicate_by_rank():
    """Test deduplicate_by_rank function with various scenarios.

    Tests:
    - Basic deduplication with single id column
    - Multiple id columns
    - Ascending vs descending ranking
    - Tiebreaker column functionality
    - Edge cases (empty DataFrame, single row)
    - Error handling for missing columns
    - Verbose mode output
    - Business scenarios
    """
    # Test 1: Basic deduplication - keep most recent record per customer
    df_basic = pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C002", "C002", "C003"],
            "transaction_date": [
                "2024-01-01",
                "2024-01-15",
                "2024-01-05",
                "2024-01-10",
                "2024-01-20",
            ],
            "amount": [100, 200, 150, 175, 300],
            "email": [
                "old@email.com",
                "new@email.com",
                "test@email.com",
                "updated@email.com",
                "customer@email.com",
            ],
        }
    )

    # Keep most recent transaction (descending date)
    result_recent = deduplicate_by_rank(
        df_basic, "customer_id", "transaction_date", ascending=False
    )
    assert len(result_recent) == 3  # One per customer
    assert result_recent["customer_id"].tolist() == ["C001", "C002", "C003"]
    assert result_recent["transaction_date"].tolist() == [
        "2024-01-15",
        "2024-01-10",
        "2024-01-20",
    ]
    assert result_recent["amount"].tolist() == [200, 175, 300]

    # Keep earliest transaction (ascending date)
    result_earliest = deduplicate_by_rank(
        df_basic, "customer_id", "transaction_date", ascending=True
    )
    assert len(result_earliest) == 3
    assert result_earliest["transaction_date"].tolist() == [
        "2024-01-01",
        "2024-01-05",
        "2024-01-20",
    ]
    assert result_earliest["amount"].tolist() == [100, 150, 300]

    # Test 2: Multiple id columns
    df_multi_id = pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C001", "C002", "C002"],
            "product_id": ["P001", "P001", "P002", "P001", "P001"],
            "date": [
                "2024-01-01",
                "2024-01-15",
                "2024-01-10",
                "2024-01-05",
                "2024-01-20",
            ],
            "quantity": [1, 2, 3, 1, 5],
        }
    )

    result_multi = deduplicate_by_rank(
        df_multi_id, ["customer_id", "product_id"], "date", ascending=False
    )
    assert len(result_multi) == 3  # C001-P001, C001-P002, C002-P001
    expected_combos = [("C001", "P001"), ("C001", "P002"), ("C002", "P001")]
    actual_combos = list(zip(result_multi["customer_id"], result_multi["product_id"]))
    assert set(actual_combos) == set(expected_combos)

    # Test 3: Tiebreaker functionality
    df_ties = pd.DataFrame(
        {
            "id": ["A", "A", "B", "B", "C", "C"],
            "score": [100, 100, 90, 90, 80, 80],  # Tied scores
            "email": [
                "old@test.com",
                None,
                None,
                "new@test.com",
                "valid@test.com",
                "invalid@test.com",
            ],
            "date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
            ],
        }
    )

    # Without tiebreaker - should be somewhat random based on original order
    result_no_tie = deduplicate_by_rank(df_ties, "id", "score", ascending=False)
    assert len(result_no_tie) == 3

    # With tiebreaker - should prefer non-null email
    result_with_tie = deduplicate_by_rank(
        df_ties, "id", "score", ascending=False, tiebreaker_col="email"
    )
    assert len(result_with_tie) == 3
    # Check that non-null emails are preferred
    assert (
        result_with_tie[result_with_tie["id"] == "A"]["email"].iloc[0] == "old@test.com"
    )  # Non-null preferred
    assert (
        result_with_tie[result_with_tie["id"] == "B"]["email"].iloc[0] == "new@test.com"
    )  # Non-null preferred

    # Test 4: Business scenario - customer data cleanup
    customer_data = pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C001", "C002", "C002", "C003", "C003"],
            "last_login": [
                "2024-01-01",
                "2024-01-15",
                "2024-01-10",
                "2024-01-05",
                "2024-01-20",
                "2024-01-25",
                "2024-01-30",
            ],
            "profile_completeness": [0.3, 0.8, 0.6, 0.4, 0.9, 0.7, 0.5],
            "email_verified": [False, True, False, False, True, True, False],
            "phone": [None, "555-0001", "555-0002", None, "555-0003", "555-0004", None],
        }
    )

    # Keep most recent login with highest profile completeness as tiebreaker
    result_customer = deduplicate_by_rank(
        customer_data,
        "customer_id",
        "last_login",
        ascending=False,
        tiebreaker_col="phone",
    )
    assert len(result_customer) == 3
    assert result_customer["customer_id"].tolist() == ["C001", "C002", "C003"]

    # Test 5: Edge cases
    # Empty DataFrame
    df_empty = pd.DataFrame(columns=["id", "score", "value"])
    result_empty = deduplicate_by_rank(df_empty, "id", "score")
    assert len(result_empty) == 0
    assert list(result_empty.columns) == ["id", "score", "value"]

    # Single row
    df_single = pd.DataFrame({"id": ["A"], "score": [100], "value": ["test"]})
    result_single = deduplicate_by_rank(df_single, "id", "score")
    assert len(result_single) == 1
    pd.testing.assert_frame_equal(result_single, df_single)

    # No duplicates
    df_unique = pd.DataFrame(
        {"id": ["A", "B", "C"], "score": [100, 90, 80], "value": ["x", "y", "z"]}
    )
    result_unique = deduplicate_by_rank(df_unique, "id", "score")
    assert len(result_unique) == 3
    pd.testing.assert_frame_equal(
        result_unique.sort_values("id").reset_index(drop=True),
        df_unique.sort_values("id").reset_index(drop=True),
    )

    # Test 6: Error handling
    df_test = pd.DataFrame({"id": ["A", "B"], "score": [1, 2]})

    # Missing id column
    with pytest.raises(ValueError, match="not found in DataFrame"):
        deduplicate_by_rank(df_test, "missing_id", "score")

    # Missing ranking column
    with pytest.raises(ValueError, match="not found in DataFrame"):
        deduplicate_by_rank(df_test, "id", "missing_score")

    # Missing tiebreaker column
    with pytest.raises(ValueError, match="not found in DataFrame"):
        deduplicate_by_rank(df_test, "id", "score", tiebreaker_col="missing_col")

    # Test 7: String input handling
    result_string_input = deduplicate_by_rank(
        df_basic, "customer_id", "transaction_date"
    )
    assert len(result_string_input) == 3

    # Test 8: Data types preservation
    df_types = pd.DataFrame(
        {
            "id": ["A", "A", "B"],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "bool_col": [True, False, True],
            "date_col": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )

    result_types = deduplicate_by_rank(df_types, "id", "int_col", ascending=False)
    assert len(result_types) == 2
    assert result_types["int_col"].dtype == df_types["int_col"].dtype
    assert result_types["float_col"].dtype == df_types["float_col"].dtype
    assert result_types["bool_col"].dtype == df_types["bool_col"].dtype
    assert result_types["date_col"].dtype == df_types["date_col"].dtype


def test_deduplicate_by_rank_verbose(capsys):
    """Test verbose output of deduplicate_by_rank function."""
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C002"],
            "date": ["2024-01-01", "2024-01-15", "2024-01-10"],
            "amount": [100, 200, 150],
        }
    )

    # Test verbose mode
    result = deduplicate_by_rank(df, "customer_id", "date", verbose=True)
    captured = capsys.readouterr()

    assert "Deduplicating 3 rows" in captured.out
    assert "Found 2 unique groups" in captured.out
    assert "Removed 1 duplicate rows" in captured.out
    assert "Final dataset: 2 rows" in captured.out

    # Test verbose with empty DataFrame
    df_empty = pd.DataFrame(columns=["id", "score"])
    result_empty = deduplicate_by_rank(df_empty, "id", "score", verbose=True)
    captured_empty = capsys.readouterr()

    assert "Input DataFrame is empty" in captured_empty.out
