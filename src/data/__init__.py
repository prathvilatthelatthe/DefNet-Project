"""
DeforestNet - Data Package
Dataset generation, loading, and augmentation utilities.
"""

from .synthetic_generator import SyntheticDataGenerator, generate_full_dataset
from .augmentation import (
    TrainAugmentation,
    ValAugmentation,
    random_horizontal_flip,
    random_vertical_flip,
    random_rotate90,
    random_brightness,
    random_contrast,
    random_band_noise
)
from .visualization import (
    visualize_sample,
    visualize_batch,
    plot_class_distribution,
    plot_band_statistics,
    visualize_all_bands,
    create_class_colormap
)
from .deforest_dataset import (
    DeforestationDataset,
    SyntheticDataset,
    create_dataloaders,
    get_dataset_info
)

__all__ = [
    # Generators
    "SyntheticDataGenerator",
    "generate_full_dataset",
    # Augmentation
    "TrainAugmentation",
    "ValAugmentation",
    "random_horizontal_flip",
    "random_vertical_flip",
    "random_rotate90",
    "random_brightness",
    "random_contrast",
    "random_band_noise",
    # Visualization
    "visualize_sample",
    "visualize_batch",
    "plot_class_distribution",
    "plot_band_statistics",
    "visualize_all_bands",
    "create_class_colormap",
    # Dataset
    "DeforestationDataset",
    "SyntheticDataset",
    "create_dataloaders",
    "get_dataset_info"
]
