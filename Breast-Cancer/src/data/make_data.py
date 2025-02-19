from sklearn.model_selection import train_test_split
from typing import List, Tuple
from sklearn.utils import shuffle
import albumentations as A
from pathlib import Path
import numpy as np
import cv2
import os


class ImageDataProcessor:
    """Class for processing, augmenting and splitting image datasets"""

    def __init__(self, config: dict) -> None:
        """
        Initialize processor with configuration

        Args:
            config: Dictionary containing processing parameters
        """

        self.config = config
        self.transforms = [
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(p=1),
            A.Rotate(limit=45, p=1),
            A.RGBShift(p=1),
            A.GaussianBlur(p=1),
            A.VerticalFlip(p=1),
            A.RandomGamma(p=1),
        ]

    def load_image(self, path: Path, label: int) -> List[Tuple[np.ndarray, int]]:
        """
        Load images from directory with labels

        Args:
            path: Directory path containing images
            label: Class label for the images

        Returns:
            List of tuples containing (image, label)
        """
        images = [
            (cv2.imread(os.path.join(path, image)), label) for image in os.listdir(path)
        ]

        return images

    def generate_augmentations(
        self, data: Tuple[np.ndarray, int]
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Generate augmented versions of an image

        Args:
            data: Tuple of (image, label)

        Returns:
            List of tuples containing (augmented_image, label)
        """
        image, label = data

        return [
            (transform(image=image)["image"], label) for transform in self.transforms
        ]

    def preprocess_images(
        self,
        data: List[Tuple[np.ndarray, int]],
        target_size: Tuple[int, int] = (224, 224),
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Resize and normalize images

        Args:
            data: List of (image, label) tuples
            target_size: Target image dimensions

        Returns:
            List of processed (image, label) tuples
        """
        return [
            (cv2.resize(image, target_size).astype(np.float32) / 255.0, label)
            for image, label in data
        ]

    def split_dataset(
        self, data: List[Tuple[np.ndarray, int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into train and test sets

        Args:
            data: List of (image, label) tuples

        Returns:
            X_train, X_test, y_train, y_test arrays
        """
        shuffled_data = shuffle(data)

        X, y = zip(*shuffled_data)

        return train_test_split(
            np.array(X),
            np.array(y),
            test_size=0.1,
            random_state=0,
            shuffle=True,
            stratify=y,
        )

    def create_complete_dataset(
        self, benign_path: Path, malignant_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create and split a complete dataset with original and augmented images

        Args:
            benign_path: Directory path containing benign images
            malignant_path: Directory path containing malignant images

        Returns:
            Tuple containing:
                X_train: Training images array
                X_test: Testing images array
                y_train: Training labels array
                y_test: Testing labels array
        """
        # Load original image sets
        benign_dataset = self.load_image(benign_path, label=0)
        malignant_dataset = self.load_image(malignant_path, label=1)

        # Generate augmented versions
        augmented_benign_dataset = [
            augmented_img
            for original_img in benign_dataset
            for augmented_img in self.generate_augmentations(original_img)
        ]

        # Combine and preprocess all images
        processed_dataset = self.preprocess_images(
            benign_dataset + malignant_dataset + augmented_benign_dataset
        )

        # Split into train/test sets
        return self.split_dataset(processed_dataset)
