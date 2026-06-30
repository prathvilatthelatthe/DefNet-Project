"""
DeforestNet - Dataset Visualization Utilities
Tools for visualizing synthetic and real satellite data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import CLASS_NAMES, CLASS_COLORS, BAND_NAMES, NUM_CLASSES


def create_class_colormap() -> ListedColormap:
    """Create matplotlib colormap for class visualization."""
    colors = np.array(CLASS_COLORS) / 255.0
    return ListedColormap(colors)


def visualize_sample(
    image: np.ndarray,
    mask: np.ndarray,
    title: str = "Sample",
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Visualize a single sample with multiple views.

    Args:
        image: Shape (11, H, W) or (H, W, 11)
        mask: Shape (H, W)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    # Ensure image is (C, H, W)
    if image.ndim == 3 and image.shape[-1] == 11:
        image = np.transpose(image, (2, 0, 1))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=14)

    # RGB composite (B4, B3, B2 = Red, Green, Blue)
    rgb = np.stack([image[4], image[3], image[2]], axis=-1)
    rgb = np.clip(rgb * 2.5, 0, 1)  # Enhance brightness
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB (B4, B3, B2)")
    axes[0, 0].axis('off')

    # False color (NIR, Red, Green)
    false_color = np.stack([image[5], image[4], image[3]], axis=-1)
    false_color = np.clip(false_color * 2, 0, 1)
    axes[0, 1].imshow(false_color)
    axes[0, 1].set_title("False Color (NIR, R, G)")
    axes[0, 1].axis('off')

    # NDVI
    ndvi_display = axes[0, 2].imshow(image[6], cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, 2].set_title("NDVI")
    axes[0, 2].axis('off')
    plt.colorbar(ndvi_display, ax=axes[0, 2], fraction=0.046)

    # SAR VV
    vv_display = axes[0, 3].imshow(image[0], cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title("SAR VV")
    axes[0, 3].axis('off')
    plt.colorbar(vv_display, ax=axes[0, 3], fraction=0.046)

    # SAR VH
    vh_display = axes[1, 0].imshow(image[1], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("SAR VH")
    axes[1, 0].axis('off')
    plt.colorbar(vh_display, ax=axes[1, 0], fraction=0.046)

    # VV/VH Ratio
    ratio_display = axes[1, 1].imshow(image[9], cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title("VV/VH Ratio")
    axes[1, 1].axis('off')
    plt.colorbar(ratio_display, ax=axes[1, 1], fraction=0.046)

    # Ground Truth Mask
    cmap = create_class_colormap()
    mask_display = axes[1, 2].imshow(mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
    axes[1, 2].set_title("Ground Truth")
    axes[1, 2].axis('off')

    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=np.array(CLASS_COLORS[i])/255)
                       for i in range(NUM_CLASSES)]
    axes[1, 3].legend(legend_elements, CLASS_NAMES, loc='center', fontsize=10)
    axes[1, 3].axis('off')
    axes[1, 3].set_title("Legend")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def visualize_batch(
    images: np.ndarray,
    masks: np.ndarray,
    num_samples: int = 4,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Visualize multiple samples in a grid.

    Args:
        images: Shape (N, 11, H, W)
        masks: Shape (N, H, W)
        num_samples: Number of samples to show
        save_path: Path to save figure
        show: Whether to display
    """
    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    cmap = create_class_colormap()

    for i in range(num_samples):
        image = images[i]
        mask = masks[i]

        # RGB
        rgb = np.stack([image[4], image[3], image[2]], axis=-1)
        rgb = np.clip(rgb * 2.5, 0, 1)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Sample {i+1} - RGB")
        axes[i, 0].axis('off')

        # NDVI
        axes[i, 1].imshow(image[6], cmap='RdYlGn', vmin=0, vmax=1)
        axes[i, 1].set_title("NDVI")
        axes[i, 1].axis('off')

        # Mask
        axes[i, 2].imshow(mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(
    masks: np.ndarray,
    title: str = "Class Distribution",
    save_path: Optional[Path] = None,
    show: bool = True
) -> Dict[str, float]:
    """
    Plot class distribution in dataset.

    Args:
        masks: Shape (N, H, W)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display

    Returns:
        Dictionary with class percentages
    """
    # Count pixels per class
    total_pixels = masks.size
    class_counts = {}
    class_percentages = {}

    for i in range(NUM_CLASSES):
        count = np.sum(masks == i)
        class_counts[CLASS_NAMES[i]] = count
        class_percentages[CLASS_NAMES[i]] = (count / total_pixels) * 100

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [np.array(CLASS_COLORS[i]) / 255 for i in range(NUM_CLASSES)]
    bars = ax.bar(CLASS_NAMES, class_percentages.values(), color=colors, edgecolor='black')

    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title)

    # Add percentage labels on bars
    for bar, pct in zip(bars, class_percentages.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return class_percentages


def plot_band_statistics(
    images: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Plot band-wise statistics.

    Args:
        images: Shape (N, 11, H, W)
        save_path: Path to save figure
        show: Whether to display
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute statistics
    means = np.mean(images, axis=(0, 2, 3))
    stds = np.std(images, axis=(0, 2, 3))
    mins = np.min(images, axis=(0, 2, 3))
    maxs = np.max(images, axis=(0, 2, 3))

    x = np.arange(len(BAND_NAMES))

    # Mean values
    axes[0, 0].bar(x, means, color='steelblue', edgecolor='black')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(BAND_NAMES, rotation=45, ha='right')
    axes[0, 0].set_title("Mean Values per Band")
    axes[0, 0].set_ylabel("Mean")

    # Standard deviation
    axes[0, 1].bar(x, stds, color='coral', edgecolor='black')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(BAND_NAMES, rotation=45, ha='right')
    axes[0, 1].set_title("Std Dev per Band")
    axes[0, 1].set_ylabel("Std")

    # Min-Max range
    axes[1, 0].bar(x, maxs - mins, bottom=mins, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(BAND_NAMES, rotation=45, ha='right')
    axes[1, 0].set_title("Value Range per Band")
    axes[1, 0].set_ylabel("Range")

    # Histogram of NDVI (band 6)
    ndvi_values = images[:, 6, :, :].flatten()
    axes[1, 1].hist(ndvi_values, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title("NDVI Distribution")
    axes[1, 1].set_xlabel("NDVI Value")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def visualize_all_bands(
    image: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Visualize all 11 bands of a single image.

    Args:
        image: Shape (11, H, W)
        save_path: Path to save figure
        show: Whether to display
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(11):
        ax = axes[i]
        im = ax.imshow(image[i], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(BAND_NAMES[i])
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide last empty subplot
    axes[11].axis('off')

    plt.suptitle("All 11 Bands", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test with random data
    print("Testing visualization utilities...")

    # Create dummy data
    dummy_images = np.random.rand(4, 11, 256, 256).astype(np.float32)
    dummy_masks = np.random.randint(0, 6, (4, 256, 256)).astype(np.int64)

    # Test class distribution
    dist = plot_class_distribution(dummy_masks, show=False)
    print(f"Class distribution: {dist}")

    print("Visualization utilities test passed!")
