import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mlarena.utils.io_utils import load_object, save_object


# Setup and teardown
@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    yield tmp_path
    # Cleanup after tests
    try:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
    except PermissionError:
        # On Windows, sometimes files need a moment to be released
        import time

        time.sleep(0.1)
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        "array": np.array([1, 2, 3]),
        "dict": {"a": 1, "b": 2},
        "list": [1, 2, 3],
        "string": "test",
    }


@pytest.fixture
def large_array():
    """Create a moderately sized array for compression testing."""
    # Using 100x100 instead of 1000x1000 - still good for testing compression
    # but uses 100x less memory (80KB vs 8MB)
    return np.random.rand(100, 100)


def test_save_load_pickle(temp_dir, sample_data):
    """Test basic save and load functionality with pickle."""
    try:
        # Save without date
        path = save_object(
            sample_data, temp_dir, name="test", use_date=False, backend="pickle"
        )

        # Check file exists
        assert path.exists()
        assert path.suffix == ".pkl"

        # Load and verify
        loaded_data = load_object(path, backend="pickle")
        assert loaded_data["dict"] == sample_data["dict"]
        assert loaded_data["string"] == sample_data["string"]
        assert loaded_data["list"] == sample_data["list"]
        np.testing.assert_array_equal(loaded_data["array"], sample_data["array"])
    finally:
        # Ensure file handles are closed
        import gc

        gc.collect()


def test_save_load_joblib(temp_dir, sample_data):
    """Test basic save and load functionality with joblib."""
    pytest.importorskip("joblib")  # Skip if joblib not installed

    try:
        # Save without date
        path = save_object(
            sample_data, temp_dir, name="test", use_date=False, backend="joblib"
        )

        # Check file exists
        assert path.exists()
        assert path.suffix == ".joblib"

        # Load and verify
        loaded_data = load_object(path, backend="joblib")
        assert loaded_data["dict"] == sample_data["dict"]
        assert loaded_data["string"] == sample_data["string"]
        assert loaded_data["list"] == sample_data["list"]
        np.testing.assert_array_equal(loaded_data["array"], sample_data["array"])
    finally:
        # Ensure file handles are closed
        import gc

        gc.collect()


def test_date_suffix(temp_dir, sample_data):
    """Test date suffix functionality."""
    # Get today's date
    today = datetime.today().strftime("%Y-%m-%d")

    # Save with date
    path = save_object(sample_data, temp_dir, name="test", use_date=True)

    # Check filename contains date
    assert today in path.name
    assert path.exists()


def test_compression_joblib(temp_dir, sample_data):
    """Test joblib compression options."""
    pytest.importorskip("joblib")  # Skip if joblib not installed

    try:
        # Save with different compression settings
        path1 = save_object(
            sample_data,
            temp_dir,
            name="test1",
            use_date=False,
            backend="joblib",
            compress=False,
        )

        path2 = save_object(
            sample_data,
            temp_dir,
            name="test2",
            use_date=False,
            backend="joblib",
            compress=True,
            compression_level=9,
        )

        # Verify both files exist
        assert path1.exists()
        assert path2.exists()

        # Check compression worked (compressed file should be smaller)
        assert path1.stat().st_size > path2.stat().st_size

        # Verify data integrity
        loaded1 = load_object(path1, backend="joblib")
        loaded2 = load_object(path2, backend="joblib")

        assert loaded1["dict"] == loaded2["dict"]
        np.testing.assert_array_equal(loaded1["array"], loaded2["array"])
    finally:
        # Ensure file handles are closed
        import gc

        gc.collect()


def test_path_handling(temp_dir, sample_data):
    """Test different path input formats."""
    # Test with string path
    path1 = save_object(sample_data, str(temp_dir), name="test1", use_date=False)
    assert path1.exists()

    # Test with Path object
    path2 = save_object(sample_data, Path(temp_dir), name="test2", use_date=False)
    assert path2.exists()

    # Test nested directory creation
    nested_dir = temp_dir / "nested" / "path"
    path3 = save_object(sample_data, nested_dir, name="test3", use_date=False)
    assert path3.exists()


def test_error_handling(temp_dir, sample_data):
    """Test error cases."""
    # Test invalid backend
    with pytest.raises(ValueError, match="backend must be either 'pickle' or 'joblib'"):
        save_object(sample_data, temp_dir, backend="invalid")

    # Test loading non-existent file
    with pytest.raises(FileNotFoundError):
        load_object(temp_dir / "nonexistent.pkl")

    # Test loading with wrong backend
    path = save_object(
        sample_data, temp_dir, name="test", use_date=False, backend="pickle"
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "File extension .pkl does not match expected .joblib for joblib backend"
        ),
    ):
        load_object(path, backend="joblib")


def test_large_array_comparison(temp_dir, large_array):
    """Test saving/loading large array with different backends."""
    pytest.importorskip("joblib")  # Skip if joblib not installed

    try:
        # Save with both backends
        path_pickle = save_object(
            large_array, temp_dir, name="large_pickle", use_date=False, backend="pickle"
        )

        path_joblib = save_object(
            large_array,
            temp_dir,
            name="large_joblib",
            use_date=False,
            backend="joblib",
            compress=True,
        )

        # Verify files exist
        assert path_pickle.exists()
        assert path_joblib.exists()

        # Load and verify data integrity
        loaded_pickle = load_object(path_pickle, backend="pickle")
        loaded_joblib = load_object(path_joblib, backend="joblib")

        # Verify data integrity
        np.testing.assert_array_equal(large_array, loaded_pickle)
        np.testing.assert_array_equal(large_array, loaded_joblib)
        np.testing.assert_array_equal(loaded_pickle, loaded_joblib)

        # Print file sizes for information (but don't assert)
        pickle_size = path_pickle.stat().st_size
        joblib_size = path_joblib.stat().st_size
        print(f"\nFile sizes for information:")
        print(f"Pickle file size: {pickle_size:,} bytes")
        print(f"Joblib file size: {joblib_size:,} bytes")
    finally:
        # Ensure file handles are closed
        import gc

        gc.collect()
