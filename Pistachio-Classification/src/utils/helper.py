import os
from pathlib import Path
from typing import Any, Dict, Union, Optional

import numpy as np
import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse YAML configuration file.

    Args:
                    path: Path to YAML config file (string or Path object)

    Returns:
                    Dict containing configuration parameters

    Raises:
                    FileNotFoundError: If config file doesn't exist
                    yaml.YAMLError: If the file contains invalid YAML
    """
    path = Path(path)

    try:
        with path.open("r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {path}")


def save_numpy_array(
    array_data: np.ndarray, file_path: str, show_details: bool = True
) -> bool:
    """
    Saves a NumPy array to a specified file path.

    Args:
            array_data (np.ndarray): The NumPy array to be saved.
            file_path (str): The full path to the file where the array will be saved.
            show_details (bool, optional): If True, prints details about the saved array,
                    including file path, size, shape, and data type. Defaults to True.

    Returns:
            bool: True if the array was saved successfully, False otherwise.

    Example:
    >>> import numpy as np
    >>> data = np.array([[1, 2], [3, 4]])
    >>> file = "my_array.npy"
    >>> success = save_numpy_array(data, file)
    Successfully saved to: my_array.npy
    File size: 0.00 MB
    Data shape: (2, 2), Type: int64
    >>> print(f"Save successful: {success}")
    Save successful: True
    """

    try:
        np.save(file_path, array_data)

        if show_details:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            array_shape = array_data.shape
            array_type = array_data.dtype

            print(f"Successfully saved to: {file_path}")
            print(f"File size: {file_size_mb:.2f} MB")
            print(f"Data shape: {array_shape}, Type: {array_type}")

        return True

    except PermissionError:
        print(
            f"Error: Permission denied when saving to {file_path}. Check file permissions."
        )
        return False

    except MemoryError:
        print(
            f"Error: Not enough memory to save array of size {array_data.nbytes / (1024 ** 2):.2f} MB."
        )
        return False

    except Exception as e:
        print(f"Error saving array to {file_path}: {str(e)}")

        import traceback

        traceback.print_exc()

        return False


def load_numpy_array(file_path: str, show_details: bool = True) -> Optional[np.ndarray]:
    """
    Loads a NumPy array from a specified file path.

    Args:
            file_path (str): The full path to the file from which the array will be loaded.
            show_details (bool, optional): If True, prints details about the loaded array,
                    including file path and size. Defaults to True.

    Returns:
            Optional[np.ndarray]: The loaded NumPy array if successful, None otherwise.

    Example:
            >>> import numpy as np
            >>> data = np.array([[1, 2], [3, 4]])
            >>> file = "my_array.npy"
            >>> np.save(file, data)  # Save the array first
            >>> loaded_array = load_numpy_array(file)
            Successfully loaded from: my_array.npy
            File size: 0.00 MB
            >>> print(loaded_array)
            [[1 2]
            [3 4]]
    """
    try:
        loaded_data = np.load(file_path)

        if show_details:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"Successfully loaded from: {file_path}")
            print(f"File size: {file_size_mb:.2f} MB")

        return loaded_data

    except PermissionError:
        print(
            f"Error: Permission denied when accessing {file_path}. Check file permissions."
        )
        return None

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        return None

    except Exception as e:
        print(f"Error loading array from {file_path}: {str(e)}")

        import traceback

        traceback.print_exc()

        return None
