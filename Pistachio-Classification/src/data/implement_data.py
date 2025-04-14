from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
from functools import lru_cache
from pathlib import Path
import numpy as np
import cv2
import os


class ImageDataset:
	def __init__(self, dataset_root_dir: str | Path) -> None:
		"""
		Initialize the image dataset with a root directory.

		Args:
				dataset_root_dir: Root directory containing category subdirectories with images
		"""
		self.dataset_root_dir: Path = Path(dataset_root_dir)
		self.cached_image_paths: List[Tuple[str, int]] = []

	@lru_cache(maxsize=32)
	def get_labeled_images(
			self,
			supported_formats: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif"),
	) -> List[Tuple[str, int]]:
		"""
		Get all images with their corresponding labels.

		Args:
				supported_formats: File extensions to include in the dataset

		Returns:
				List of (image_path, label) tuples where label is 0 for "Kirmizi" and 1 for others
		"""
		try:
			return [
				(str(image_file_path), 0 if "Kirmizi" in str(image_file_path) else 1)
				for category_folder in os.scandir(self.dataset_root_dir)
				if category_folder.is_dir()
				for image_file_path in Path(category_folder.path).glob("*")
				if image_file_path.is_file()
				   and image_file_path.suffix.lower() in supported_formats
			]

		except (PermissionError, OSError) as e:
			print(f"Error accessing {self.dataset_root_dir}: {e}")
			return []

	def _load_single_image(self, path_label: Tuple[str, int]) -> Tuple[Optional[np.ndarray], int]:
		"""
		Load and process a single image.

		Args:
			path_label: Tuple containing (image_path, label)

		Returns:
			Tuple of (processed_image, label) or (None, label) if loading fails
		"""
		image_path, label = path_label
		try:
			image = cv2.imread(image_path)
			if image is None:
				print(f"Warning: Failed to load image from {image_path}")
				return None, label


			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			return image, label
		except Exception as e:
			print(f"Error processing image {image_path}: {e}")
			return None, label

	def load_images(self, paths: List[Tuple[str, int]], num_workers: int) -> List[Tuple[np.ndarray, int]]:
		"""
		Load multiple images in parallel.

		Args:
			paths: List of (image_path, label) tuples
			num_workers: Number of parallel workers for loading images

		Returns:
			List of (image, label) tuples with successfully loaded images
		"""
		with ThreadPoolExecutor(max_workers=num_workers) as executor:
			results = list(executor.map(self._load_single_image, paths))

		return [result for result in results if result[0] is not None]
