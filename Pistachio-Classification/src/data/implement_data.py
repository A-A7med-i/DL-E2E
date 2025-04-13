import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple


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
