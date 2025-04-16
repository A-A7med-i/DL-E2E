from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np


class ImageProcessor:
	"""
	Class for processing a collection of images with their labels.

	This class provides functionality to resize and normalize images in parallel,
	while preserving their aspect ratios and associated labels.
	"""

	def __init__(self,
	             image_label_pairs: List[Tuple[np.ndarray, Any]],
	             target_width: int = 300,
	             max_workers: Optional[int] = None):
		"""
		Initialize the ImageProcessor with image-label pairs.

		Args:
			image_label_pairs: List of tuples containing (image, label) pairs
			target_width: Target width for resizing images (default: 300)
			max_workers: Maximum number of threads for parallel processing.
						 If None, uses the default from ThreadPoolExecutor.
		"""
		self.image_label_pairs = image_label_pairs
		self.processed_pairs: List[Tuple[np.ndarray, Any]] = []
		self.target_width = target_width
		self.max_workers = max_workers

	def process_all(self) -> List[Tuple[np.ndarray, Any]]:
		"""
		Process all image-label pairs in the collection using parallel processing.

		Returns:
			List of processed (image, label) pairs where each image is resized
			and normalized to float32 values between 0 and 1.
		"""

		with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
			self.processed_pairs = list(executor.map(self.process_single, self.image_label_pairs))

		return self.processed_pairs

	def process_single(self, image_label: Tuple[np.ndarray, Any]) -> Tuple[np.ndarray, Any]:
		"""
		Process a single image-label pair by resizing and normalizing the image.

		Args:
			image_label: Tuple containing (image, label) where image is a numpy array
						and label can be of any type

		Returns:
			Tuple containing (processed_image, label) where processed_image is resized
			and normalized to float32 values between 0 and 1

		Raises:
			ValueError: If the image data is invalid
		"""

		try:
			image, label = image_label

			target_height = int(self.target_width * image.shape[0] / image.shape[1])

			processed_image = cv2.resize(image, (self.target_width, target_height))

			normalized_image = processed_image.astype(np.float32) / 255.0

			return (normalized_image, label)

		except Exception as e:
			print(f"Error processing image: {e}")
			return image_label
