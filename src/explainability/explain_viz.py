"""
DeforestNet - Explainability Visualization
Visualize GradCAM heatmaps, band importance, and explanation reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import (
    NUM_CLASSES, CLASS_NAMES, CLASS_COLORS, BAND_NAMES,
    VISUALIZATION_DIR
)


def heatmap_to_rgb(
    heatmap: np.ndarray,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Convert heatmap to RGB image using colormap.

    Args:
        heatmap: (H, W) values in [0, 1]
        colormap: Matplotlib colormap name

    Returns:
        (H, W, 3) RGB image
    """
    try:
        cmap = plt.colormaps.get_cmap(colormap)
    except AttributeError:
        cmap = plt.cm.get_cmap(colormap)

    rgb = cmap(heatmap)[:, :, :3]
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay GradCAM heatmap on input image.

    Args:
        image: (C, H, W) or (H, W, C) input image
        heatmap: (H, W) GradCAM heatmap
        alpha: Heatmap transparency
        colormap: Colormap for heatmap

    Returns:
        (H, W, 3) overlaid RGB image
    """
    # Prepare base image
    if image.ndim == 3 and image.shape[0] <= image.shape[-1]:
        image = image.transpose(1, 2, 0)

    if image.shape[-1] >= 5:
        rgb = image[:, :, [4, 3, 2]]  # R, G, B from our band stack
    elif image.shape[-1] >= 3:
        rgb = image[:, :, :3]
    else:
        rgb = np.stack([image[:, :, 0]] * 3, axis=-1)

    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)

    # Create heatmap RGB
    heatmap_rgb = heatmap_to_rgb(heatmap, colormap)

    # Resize heatmap if needed
    if heatmap_rgb.shape[:2] != rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (rgb.shape[1], rgb.shape[0]))

    # Overlay
    overlay = cv2.addWeighted(rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def visualize_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    class_name: str = "Unknown",
    confidence: float = 0.0,
    band_importance: Optional[Dict[str, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (18, 6)
):
    """
    Full GradCAM visualization with all components.

    Args:
        image: (C, H, W) input image
        heatmap: (H, W) GradCAM heatmap
        prediction: (H, W) prediction mask (optional)
        class_name: Target class name
        confidence: Mean confidence score
        band_importance: Band importance scores (optional)
        save_path: Path to save figure
        show: Whether to display
        figsize: Figure size
    """
    n_cols = 3
    if prediction is not None:
        n_cols = 4
    if band_importance is not None:
        n_cols += 1

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # 1. Input image (RGB)
    if image.shape[0] <= image.shape[-1] if image.ndim == 3 else True:
        img_display = image.transpose(1, 2, 0) if image.ndim == 3 and image.shape[0] < image.shape[-1] else image
    else:
        img_display = image

    if img_display.ndim == 3 and img_display.shape[0] <= 11:
        img_display = image.transpose(1, 2, 0)

    if img_display.shape[-1] >= 5:
        rgb = img_display[:, :, [4, 3, 2]]
    elif img_display.shape[-1] >= 3:
        rgb = img_display[:, :, :3]
    else:
        rgb = img_display

    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    ax_idx = 0
    axes[ax_idx].imshow(rgb)
    axes[ax_idx].set_title("Input (RGB)", fontsize=11)
    axes[ax_idx].axis('off')

    # 2. GradCAM heatmap
    ax_idx += 1
    axes[ax_idx].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[ax_idx].set_title("GradCAM Attention", fontsize=11)
    axes[ax_idx].axis('off')

    # 3. Overlay
    ax_idx += 1
    overlay = overlay_heatmap(image, heatmap)
    axes[ax_idx].imshow(overlay)
    axes[ax_idx].set_title(f"Overlay ({class_name}, {confidence:.0%})", fontsize=11)
    axes[ax_idx].axis('off')

    # 4. Prediction mask
    if prediction is not None:
        ax_idx += 1
        pred_rgb = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for c_idx, color in enumerate(CLASS_COLORS):
            pred_rgb[prediction == c_idx] = color
        axes[ax_idx].imshow(pred_rgb)
        axes[ax_idx].set_title("Prediction", fontsize=11)
        axes[ax_idx].axis('off')

    # 5. Band importance bar chart
    if band_importance is not None:
        ax_idx += 1
        sorted_bands = sorted(band_importance.items(), key=lambda x: x[1], reverse=True)
        names = [b[0].replace('S1_', '').replace('S2_', '').replace('_', '\n') for b in sorted_bands]
        values = [b[1] * 100 for b in sorted_bands]

        colors = ['#e74c3c' if v == max(values) else '#3498db' for v in values]
        bars = axes[ax_idx].barh(names[::-1], values[::-1], color=colors[::-1])
        axes[ax_idx].set_xlabel("Importance (%)", fontsize=9)
        axes[ax_idx].set_title("Band Importance", fontsize=11)
        axes[ax_idx].tick_params(axis='y', labelsize=8)

    plt.suptitle(
        f"GradCAM Explanation: {class_name}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)


def visualize_all_class_heatmaps(
    image: np.ndarray,
    heatmaps: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Visualize GradCAM heatmaps for all classes.

    Args:
        image: (C, H, W) input image
        heatmaps: Dict mapping class names to heatmaps
        save_path: Path to save
        show: Whether to display
    """
    n_classes = len(heatmaps)
    fig, axes = plt.subplots(2, (n_classes + 1) // 2 + 1, figsize=(20, 8))
    axes = axes.flatten()

    # Show input image
    if image.ndim == 3 and image.shape[0] < image.shape[-1]:
        img = image.transpose(1, 2, 0)
    else:
        img = image
    if img.ndim == 3 and img.shape[0] <= 11:
        img = image.transpose(1, 2, 0)

    if img.shape[-1] >= 5:
        rgb = img[:, :, [4, 3, 2]]
    else:
        rgb = img[:, :, :3]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    axes[0].imshow(rgb)
    axes[0].set_title("Input (RGB)", fontsize=10)
    axes[0].axis('off')

    # Show each class heatmap
    for i, (class_name, heatmap) in enumerate(heatmaps.items()):
        ax = axes[i + 1]
        overlay = overlay_heatmap(image, heatmap)
        ax.imshow(overlay)
        ax.set_title(f"{class_name}", fontsize=10)
        ax.axis('off')

    # Hide remaining axes
    for j in range(len(heatmaps) + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("GradCAM per Class", fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)


def save_explanation_report(
    report: Dict,
    image: np.ndarray,
    output_dir: Union[str, Path],
    name: str = "explanation"
):
    """
    Save complete explanation report to directory.

    Args:
        report: Report dictionary from ExplainabilityReport
        image: Input image
        output_dir: Output directory
        name: Base filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save heatmap
    heatmap = report['gradcam_heatmap']
    np.save(output_dir / f"{name}_heatmap.npy", heatmap)

    # Save heatmap as image
    heatmap_rgb = heatmap_to_rgb(heatmap)
    cv2.imwrite(
        str(output_dir / f"{name}_heatmap.png"),
        cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    )

    # Save overlay
    overlay = overlay_heatmap(image, heatmap)
    cv2.imwrite(
        str(output_dir / f"{name}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    # Save visualization
    visualize_gradcam(
        image, heatmap,
        class_name=report['predicted_class'],
        confidence=report['mean_confidence'],
        band_importance=report.get('band_importance'),
        save_path=output_dir / f"{name}_visualization.png",
        show=False
    )

    # Save text report
    json_report = {
        'predicted_class': report['predicted_class'],
        'predicted_class_idx': report['predicted_class_idx'],
        'mean_confidence': report['mean_confidence'],
        'band_importance': report['band_importance'],
        'top_bands': report['top_bands'],
        'explanation_text': report['explanation_text'],
        'class_distribution': report.get('class_distribution', {})
    }

    with open(output_dir / f"{name}_report.json", 'w') as f:
        json.dump(json_report, f, indent=2)

    # Save readable text explanation
    with open(output_dir / f"{name}_explanation.txt", 'w') as f:
        f.write(report['explanation_text'])

    print(f"Explanation saved to {output_dir}")
