import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

try:
    import joblib
except ImportError:
    joblib = None

__all__ = ["save_object", "load_object"]


def save_object(
    obj: Any,
    directory: Union[str, Path],  # Directory
    basename: str,  # Base filename (without extension)
    use_date: bool = True,
    backend: str = "pickle",  # Explicit backend, determines extension
    compress: bool = True,
    compression_level: Optional[int] = None,
    pickle_protocol: Optional[int] = None,
) -> Path:
    """
    Save a Python object to disk using either pickle or joblib.

    Parameters
    ----------
    obj : Any
        The Python object to save.
    directory : Union[str, Path]
        Directory to save the file.
    basename : str
        Base name for the file (without extension, e.g., 'my_model').
        The extension will be added based on the backend.
    use_date : bool, default=True
        Whether to append the current date to the file name stem.
    backend : str, default="pickle"
        Storage backend to use ('pickle' or 'joblib'). This determines the
        file extension.
    compress : bool, default=True
        Whether to use compression for joblib backend.
    compression_level : Optional[int], default=None
        Compression level (0-9, joblib only). None uses backend default.
    pickle_protocol : Optional[int], default=None
        The protocol version to use for pickling. The default (None) uses pickle's
        default protocol. Only specify if needed for backward compatibility.

    Returns
    -------
    Path
        The full filepath to the saved object. This path can be directly used
        as the `filepath` argument for the `load_object` function.

    Examples
    --------
    >>> # model = RandomForestClassifier() # Assuming model is trained
    >>> # model.fit(X_train, y_train)
    >>> # Save with date and pickle backend (default)
    >>> # save_object(model, directory="models", basename="my_model") # -> models/my_model_YYYY-MM-DD.pkl
    >>> # Save without date and with joblib backend
    >>> # save_object(model, directory="output/data", basename="raw_features", use_date=False, backend="joblib") # -> output/data/raw_features.joblib
    >>> # Save with joblib and specific compression level
    >>> # save_object(model, directory="models", basename="compressed_model", backend="joblib", compression_level=5)
    >>> # Save with specific pickle protocol
    >>> # save_object(model, directory="models", basename="old_model", pickle_protocol=4)
    """
    # Validate and create directory
    save_dir = Path(directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Validate and process backend
    actual_backend_str = backend.lower()
    if actual_backend_str not in ["pickle", "joblib"]:
        raise ValueError("Backend must be 'pickle' or 'joblib'.")

    if actual_backend_str == "joblib" and joblib is None:
        raise ImportError(
            "joblib is not installed. Install it with 'pip install joblib'"
        )

    # Create filename components
    date_suffix = f"_{datetime.today().strftime('%Y-%m-%d')}" if use_date else ""
    final_extension = ".joblib" if actual_backend_str == "joblib" else ".pkl"

    output_filename_stem = f"{basename}{date_suffix}"
    final_filename_with_ext = f"{output_filename_stem}{final_extension}"
    final_filepath = save_dir / final_filename_with_ext

    # Save the object
    if actual_backend_str == "joblib":
        if not compress:  # compress parameter of save_object is False
            joblib.dump(
                obj, final_filepath, compress=False
            )  # joblib.dump compress=False means no compression
        elif (
            compression_level is not None
        ):  # compress is True (default or explicit) AND compression_level is set
            joblib.dump(obj, final_filepath, compress=("zlib", compression_level))
        else:  # compress is True (default or explicit) AND compression_level is None
            joblib.dump(
                obj, final_filepath, compress=True
            )  # Use joblib's default compression behavior
    else:  # pickle backend
        with open(final_filepath, "wb") as f:
            if pickle_protocol is not None:
                pickle.dump(obj, f, protocol=pickle_protocol)
            else:
                pickle.dump(obj, f)  # Use pickle's default protocol

    print(f"Object saved to {final_filepath}")
    return final_filepath


def load_object(filepath: Union[str, Path]) -> Any:
    """
    Load a Python object from disk, inferring backend from file extension.

    Parameters
    ----------
    filepath : Union[str, Path]
        Full path to the file to load (e.g., "models/model.pkl" or "models/model.joblib").
        This should be the full path, including the directory, basename and extension.

    Returns
    -------
    Any
        The loaded object.

    Examples
    --------
    >>> # saved_filepath = save_object(my_data, directory="data_dir", basename="my_data_file")
    >>> # loaded_data = load_object(filepath=saved_filepath)
    >>> # Load a pickle file directly by path
    >>> # model = load_object(filepath="models/model_2024-02-27.pkl")
    >>> # Load a joblib file directly by path
    >>> # model = load_object(filepath="models/archive/old_model.joblib")
    """
    input_file_path = Path(filepath)

    # Validate file exists
    if not input_file_path.exists():
        raise FileNotFoundError(f"File not found: {input_file_path}")

    # Infer backend from file extension
    file_suffix = input_file_path.suffix.lower()
    actual_backend_str = ""
    if file_suffix == ".pkl":
        actual_backend_str = "pickle"
    elif file_suffix == ".joblib":
        actual_backend_str = "joblib"
    else:
        raise ValueError(
            f"Unsupported file extension: {file_suffix}. Expected '.pkl' or '.joblib'."
        )

    if actual_backend_str == "joblib" and joblib is None:
        raise ImportError(
            "joblib is not installed. Install it with 'pip install joblib'"
        )

    # Load the object
    obj: Any
    if actual_backend_str == "joblib":
        obj = joblib.load(input_file_path)
    else:  # pickle backend
        with open(input_file_path, "rb") as f:
            obj = pickle.load(f)

    print(f"Object loaded from {input_file_path}")
    return obj
