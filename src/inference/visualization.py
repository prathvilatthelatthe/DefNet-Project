"""
DeforestNet - Prediction Visualization
Visualize model predictions with colored segmentation maps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import (
    NUM_CLASSES, CLASS_NAMES, CLASS_COLORS,
    VISUALIZATION_DIR, PREDICTIONS_DIR
)


def create_colormap() -> ListedColormap:
    """Create colormap for segmentation visualization."""
    colors = np.array(CLASS_COLORS) / 255.0
    return ListedColormap(colors)


def prediction_to_rgb(prediction: np.ndarray) -> np.ndarray:
    """
    Convert prediction mask to RGB image.

    Args:
        prediction: (H, W) class labels (0-5)

    Returns:
        (H, W, 3) RGB image
    """
    rgb = np.zeros((*prediction.shape, 3), dtype=np.uint8)

    for class_idx, color in enumerate(CLASS_COLORS):
        mask = prediction == class_idx
        rgb[mask] = color

    return rgb


def confidence_to_heatmap(
    confidence: np.ndarray,
    colormap: str = 'hot'
) -> np.ndarray:
    """
    Convert confidence map to colored heatmap.

    Args:
        confidence: (H, W) confidence scores (0-1)
        colormap: Matplotlib colormap name

    Returns:
        (H, W, 3) RGB heatmap
    """
    # Normalize to 0-255
    conf_normalized = (confidence * 255).astype(np.uint8)

    # Apply colormap (compatible with both old and new matplotlib)
    try:
        cmap = plt.colormaps.get_cmap(colormap)
    except AttributeError:
        cmap = plt.cm.get_cmap(colormap)

    heatmap = cmap(conf_normalized / 255.0)[:, :, :3]  # Remove alpha
    heatmap = (heatmap * 255).astype(np.uint8)

    return heatmap


def create_legend_image(
    figsize: Tuple[int, int] = (4, 3),
    dpi: int = 100
) -> np.ndarray:
    """
    Create legend image for class colors.

    Returns:
        (H, W, 3) RGB legend image
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    patches = []
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        color_normalized = np.array(color) / 255.0
        patch = mpatches.Patch(color=color_normalized, label=name)
        patches.append(patch)

    ax.legend(handles=patches, loc='center', fontsize=12, frameon=False)
    ax.axis('off')

    fig.canvas.draw()
    legend_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    legend_img = legend_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return legend_img


def visualize_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Prediction",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (16, 8)
) -> Optional[np.ndarray]:
    """
    Visualize prediction with optional ground truth comparison.

    Args:
        image: (C, H, W) or (H, W, C) input image (uses RGB bands for display)
        prediction: (H, W) predicted class labels
        confidence: (H, W) confidence scores (optional)
        ground_truth: (H, W) ground truth labels (optional)
        title: Figure title
        save_path: Path to save figure (optional)
        show: Whether to display figure
        figsize: Figure size

    Returns:
        Figure as numpy array if save_path is None and show is False
    """
    # Determine number of subplots
    n_cols = 2  # Input + Prediction
    if confidence is not None:
        n_cols += 1
    if ground_truth is not None:
        n_cols += 1

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    # Prepare input image for display
    if image.ndim == 3:
        if image.shape[0] <= image.shape[-1]:
            # (C, H, W) -> (H, W, C)
            image = image.transpose(1, 2, 0)
        # Use RGB-like bands (B4, B3, B2 = indices 4, 3, 2 in our stack)
        if image.shape[-1] >= 5:
            rgb_img = image[:, :, [4, 3, 2]]  # Red, Green, Blue
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
        else:
            rgb_img = image[:, :, :3]
    else:
        rgb_img = image

    # Plot input
    ax_idx = 0
    axes[ax_idx].imshow(rgb_img)
    axes[ax_idx].set_title("Input (RGB)", fontsize=12)
    axes[ax_idx].axis('off')

    # Plot prediction
    ax_idx += 1
    pred_rgb = prediction_to_rgb(prediction)
    axes[ax_idx].imshow(pred_rgb)
    axes[ax_idx].set_title("Prediction", fontsize=12)
    axes[ax_idx].axis('off')

    # Plot confidence if provided
    if confidence is not None:
        ax_idx += 1
        conf_heatmap = confidence_to_heatmap(confidence)
        axes[ax_idx].imshow(conf_heatmap)
        axes[ax_idx].set_title(f"Confidence (mean: {confidence.mean():.2f})", fontsize=12)
        axes[ax_idx].axis('off')

    # Plot ground truth if provided
    if ground_truth is not None:
        ax_idx += 1
        gt_rgb = prediction_to_rgb(ground_truth)
        axes[ax_idx].imshow(gt_rgb)
        axes[ax_idx].set_title("Ground Truth", fontsize=12)
        axes[ax_idx].axis('off')

    # Add legend
    patches = []
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        color_normalized = np.array(color) / 255.0
        patch = mpatches.Patch(color=color_normalized, label=name)
        patches.append(patch)

    fig.legend(handles=patches, loc='lower center', ncol=NUM_CLASSES,
               fontsize=10, frameon=True)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    if show:
        plt.show()
        plt.close(fig)
        return None

    # Return as numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def visualize_batch(
    images: np.ndarray,
    predictions: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    ground_truths: Optional[np.ndarray] = None,
    max_samples: int = 8,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Visualize a batch of predictions.

    Args:
        images: (N, C, H, W) input images
        predictions: (N, H, W) predictions
        confidences: (N, H, W) confidence scores (optional)
        ground_truths: (N, H, W) ground truth (optional)
        max_samples: Maximum samples to show
        save_path: Path to save figure
        show: Whether to display
    """
    n_samples = min(len(images), max_samples)
    n_cols = 4 if ground_truths is not None else 3

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 3 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Input RGB
        img = images[i].transpose(1, 2, 0)
        if img.shape[-1] >= 5:
            rgb = img[:, :, [4, 3, 2]]
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        else:
            rgb = img[:, :, :3]

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Input" if i == 0 else "")
        axes[i, 0].axis('off')

        # Prediction
        pred_rgb = prediction_to_rgb(predictions[i])
        axes[i, 1].imshow(pred_rgb)
        axes[i, 1].set_title("Prediction" if i == 0 else "")
        axes[i, 1].axis('off')

        # Confidence
        if confidences is not None:
            conf_heatmap = confidence_to_heatmap(confidences[i])
            axes[i, 2].imshow(conf_heatmap)
            axes[i, 2].set_title("Confidence" if i == 0 else "")
            axes[i, 2].axis('off')

        # Ground truth
        if ground_truths is not None:
            gt_rgb = prediction_to_rgb(ground_truths[i])
            col_idx = 3 if confidences is not None else 2
            axes[i, col_idx].imshow(gt_rgb)
            axes[i, col_idx].set_title("Ground Truth" if i == 0 else "")
            axes[i, col_idx].axis('off')

    # Legend
    patches = [mpatches.Patch(color=np.array(c)/255, label=n)
               for n, c in zip(CLASS_NAMES, CLASS_COLORS)]
    fig.legend(handles=patches, loc='lower center', ncol=NUM_CLASSES, fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()
        plt.close(fig)


def create_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create overlay of prediction on input image.

    Args:
        image: (C, H, W) or (H, W, C) input image
        prediction: (H, W) class labels
        alpha: Transparency (0=image only, 1=prediction only)

    Returns:
        (H, W, 3) blended RGB image
    """
    # Prepare input
    if image.ndim == 3 and image.shape[0] <= image.shape[-1]:
        image = image.transpose(1, 2, 0)

    if image.shape[-1] >= 5:
        rgb = image[:, :, [4, 3, 2]]
    else:
        rgb = image[:, :, :3]

    # Normalize to 0-255
    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)

    # Get prediction colors
    pred_rgb = prediction_to_rgb(prediction)

    # Blend
    overlay = cv2.addWeighted(rgb, 1 - alpha, pred_rgb, alpha, 0)

    return overlay


def save_prediction_outputs(
    image: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    output_dir: Union[str, Path],
    name: str = "prediction",
    summary: Optional[Dict] = None
):
    """
    Save all prediction outputs to directory.

    Args:
        image: Input image
        prediction: Prediction mask
        confidence: Confidence scores
        output_dir: Output directory
        name: Base filename
        summary: Optional summary dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save prediction mask
    np.save(output_dir / f"{name}_mask.npy", prediction)

    # Save confidence
    np.save(output_dir / f"{name}_confidence.npy", confidence)

    # Save colored prediction
    pred_rgb = prediction_to_rgb(prediction)
    cv2.imwrite(str(output_dir / f"{name}_colored.png"),
                cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

    # Save confidence heatmap
    conf_heatmap = confidence_to_heatmap(confidence)
    cv2.imwrite(str(output_dir / f"{name}_confidence.png"),
                cv2.cvtColor(conf_heatmap, cv2.COLOR_RGB2BGR))

    # Save overlay
    overlay = create_overlay(image, prediction, alpha=0.4)
    cv2.imwrite(str(output_dir / f"{name}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Save visualization
    visualize_prediction(
        image, prediction, confidence,
        title=name,
        save_path=output_dir / f"{name}_visualization.png",
        show=False
    )

    # Save summary
    if summary:
        import json
        with open(output_dir / f"{name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    print("Testing Visualization Module...")
    print("=" * 50)

    # Create dummy data
    image = np.random.rand(11, 256, 256).astype(np.float32)
    prediction = np.random.randint(0, NUM_CLASSES, (256, 256))
    confidence = np.random.rand(256, 256).astype(np.float32)

    # Test colormap
    cmap = create_colormap()
    print(f"[OK] Created colormap with {len(cmap.colors)} colors")

    # Test RGB conversion
    rgb = prediction_to_rgb(prediction)
    print(f"[OK] Prediction to RGB: {rgb.shape}")

    # Test confidence heatmap
    heatmap = confidence_to_heatmap(confidence)
    print(f"[OK] Confidence heatmap: {heatmap.shape}")

    # Test overlay
    overlay = create_overlay(image, prediction)
    print(f"[OK] Overlay: {overlay.shape}")

    # Test legend
    legend = create_legend_image()
    print(f"[OK] Legend image: {legend.shape}")

    # Test full visualization (save only, don't show)
    output_dir = PREDICTIONS_DIR / "test_viz"
    save_prediction_outputs(
        image, prediction, confidence,
        output_dir=output_dir,
        name="test"
    )

    print(f"\n[OK] All visualizations saved to {output_dir}")
