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
    path: Union[str, Path],
    name: str = "model",
    use_date: bool = True,
    backend: str = "pickle",
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
    path : Union[str, Path]
        Directory to save the file.
    name : str, default="model"
        Base name for the file.
    use_date : bool, default=True
        Whether to append the current date to the file name.
    backend : str, default="pickle"
        Storage backend to use ('pickle' or 'joblib').
    compress : bool, default=True
        Whether to use compression for joblib backend.
    compression_level : Optional[int], default=None
        Compression level (0-9, joblib only). None uses backend default.
    pickle_protocol : Optional[int], default=None
        The protocol version to use for pickling. The default (None) uses pickle's
        default protocol, which is optimized for the current Python version.
        Only specify this if you need backward compatibility with older Python versions.

    Returns
    -------
    Path
        Full path to the saved file.

    Examples
    --------
    >>> model = RandomForestClassifier()
    >>> model.fit(X_train, y_train)
    >>> # Save with date in filename (e.g., "model_2024-02-27.pkl")
    >>> save_object(model, "models")
    >>> # Save without date
    >>> save_object(model, "models", use_date=False)
    >>> # Save with joblib and default compression
    >>> save_object(model, "models", backend="joblib")
    >>> # Save with joblib and specific compression level
    >>> save_object(model, "models", backend="joblib", compression_level=3)
    >>> # Save with specific protocol only if needed for compatibility
    >>> save_object(model, "models", pickle_protocol=4)  # for Python 3.4+ compatibility
    """
    # Validate and create directory
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Validate backend
    backend = backend.lower()
    if backend not in ["pickle", "joblib"]:
        raise ValueError("backend must be either 'pickle' or 'joblib'")

    if backend == "joblib" and joblib is None:
        raise ImportError(
            "joblib is not installed. Install it with 'pip install joblib'"
        )

    # Create filename
    date_suffix = f"_{datetime.today().strftime('%Y-%m-%d')}" if use_date else ""
    extension = "joblib" if backend == "joblib" else "pkl"
    filename = f"{name}{date_suffix}.{extension}"
    full_path = save_dir / filename

    # Save the object
    if backend == "joblib":
        if not compress:
            joblib.dump(obj, full_path, compress=False)
        elif compression_level is not None:
            joblib.dump(obj, full_path, compress=("zlib", compression_level))
        else:
            joblib.dump(obj, full_path)  # Use joblib's default compression
    else:
        with open(full_path, "wb") as f:
            if pickle_protocol is not None:
                pickle.dump(obj, f, protocol=pickle_protocol)
            else:
                pickle.dump(obj, f)  # Use pickle's default protocol

    print(f"Object saved to {full_path}")
    return full_path


def load_object(path: Union[str, Path], backend: str = "pickle") -> Any:
    """
    Load a Python object from disk using either pickle or joblib.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the file to load.
    backend : str, default="pickle"
        Storage backend to use ('pickle' or 'joblib').

    Returns
    -------
    Any
        The loaded object.

    Examples
    --------
    >>> # Load pickle file
    >>> model = load_object("models/model_2024-02-27.pkl")
    >>> # Load joblib file
    >>> model = load_object("models/model_2024-02-27.joblib", backend="joblib")
    """
    file_path = Path(path)

    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Validate backend
    backend = backend.lower()
    if backend not in ["pickle", "joblib"]:
        raise ValueError("backend must be either 'pickle' or 'joblib'")

    if backend == "joblib" and joblib is None:
        raise ImportError(
            "joblib is not installed. Install it with 'pip install joblib'"
        )

    # Validate file extension matches backend
    expected_ext = ".joblib" if backend == "joblib" else ".pkl"
    if file_path.suffix.lower() != expected_ext:
        raise ValueError(
            f"File extension {file_path.suffix} does not match expected {expected_ext} for {backend} backend"
        )

    # Load the object
    if backend == "joblib":
        obj = joblib.load(file_path)
    else:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)

    print(f"Object loaded from {file_path}")
    return obj
