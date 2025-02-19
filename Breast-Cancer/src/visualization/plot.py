import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy.typing as npt
import numpy as np


def visualize_medical_scans(
    dataset: List[Tuple[npt.NDArray, int]], sample_count: int
) -> None:
    """
    Visualize a random selection of medical scan images with their diagnosis labels.

    Parameters:
        dataset: List of tuples containing (image_array, diagnosis_label)
                where diagnosis_label is 0 for Benign, 1 for Malignant
        sample_count: Number of random images to display

    Returns:
        None - Displays a matplotlib figure with the selected images

    Example:
        visualize_medical_scans(medical_data, 5)  # Shows 5 random scans
    """
    fig, subplots = plt.subplots(nrows=1, ncols=sample_count, figsize=(10, 10))

    selected_indices = np.random.choice(len(dataset), sample_count, replace=False)

    diagnosis_types = {0: "Benign", 1: "Malignant"}

    for subplot, index in zip(subplots, selected_indices):
        scan_image = dataset[index][0]
        diagnosis = dataset[index][1]

        subplot.imshow(scan_image)
        subplot.axis("off")
        subplot.set_title(diagnosis_types[diagnosis])

    plt.tight_layout()
    plt.show()
