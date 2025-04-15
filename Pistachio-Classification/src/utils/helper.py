from pathlib import Path

import yaml


def load_yaml(path):
	"""
	Read and parse YAML configuration file

	Args:
	    path: Path to YAML config file

	Returns:
	    Dict containing configuration parameters

	Raises:
	    FileNotFoundError: If config file doesn't exist
	"""
	path = Path(path)

	try:
		with path.open("r") as file:
			return yaml.safe_load(file)
	except FileNotFoundError:
		raise FileNotFoundError(f"Config file not found at: {path}")
