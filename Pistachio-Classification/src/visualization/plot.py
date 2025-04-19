import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_random_images(
    num_samples: int,
    dataset: List[Tuple[np.ndarray, bool]],
    title: str = "Random Image Samples",
    save_path: str = None,
) -> None:
    """
    Plot a grid of random images from the dataset with their classifications.

    Parameters
    ----------
    num_samples : int
            Number of random images to display. Will be capped at the dataset size.
    dataset : List[Tuple[np.ndarray, bool]]
            List of tuples where each tuple contains (image_array, is_sirt_type).
            Images should be numpy arrays, and the boolean indicates the classification.
    title : str, optional
            Main title for the figure, by default 'Random Image Samples'
    save_path : str, optional
    Path to save the figure (default: None, no saving)

    Returns
    -------
    None
            This function displays the plot but does not return any value.

    Notes
    -----
    - Images are displayed in a grid with a maximum of 5 columns
    - Each image is labeled as either "Sirt Type" or "Kirmizi Type"
    - Unused subplot spaces are turned off
    """
    num_samples = max(1, min(len(dataset), num_samples))

    num_columns = min(5, num_samples)
    num_rows = (num_samples + num_columns - 1) // num_columns

    fig, axes = plt.subplots(
        num_rows, num_columns, figsize=(3 * num_columns, 3 * num_rows)
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    if num_samples > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    random_indices = random.sample(range(len(dataset)), num_samples)
    selected_samples = [dataset[idx] for idx in random_indices]

    for i, (image, is_sirt) in enumerate(selected_samples):
        classification_label = "Sirt Type" if is_sirt else "Kirmizi Type"

        axes[i].imshow(image)
        axes[i].set_title(f"{classification_label}", fontsize=12, fontweight="bold")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for k in range(num_samples, len(axes)):
        axes[k].axis("off")

    if save_path:
        plt.savefig(save_path)

    plt.tight_layout()
    plt.show()
