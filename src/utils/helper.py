from typing import Any, Dict
from pathlib import Path
import numpy as np
import yaml


def read_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Read and parse YAML configuration file

    Args:
        config_path: Path to YAML config file

    Returns:
        Dict containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    return yaml.safe_load(config_path.read_text())


def save_dataset(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path,
) -> None:
    """
    Save train and test datasets to specified directory

    Args:
        X_train: Training images array
        X_test: Testing images array
        y_train: Training labels array
        y_test: Testing labels array
        save_dir: Directory to save arrays
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "X_train.npy", X_train)
    np.save(save_dir / "X_test.npy", X_test)
    np.save(save_dir / "y_train.npy", y_train)
    np.save(save_dir / "y_test.npy", y_test)
