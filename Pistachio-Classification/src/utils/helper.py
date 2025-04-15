from pathlib import Path
from typing import Any, Dict, Union

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
