"""
DeforestNet - Explainability Package
GradCAM, band importance, and explanation visualization.
"""

from .gradcam import (
    GradCAM,
    BandImportanceAnalyzer,
    ExplainabilityReport
)

from .explain_viz import (
    heatmap_to_rgb,
    overlay_heatmap,
    visualize_gradcam,
    visualize_all_class_heatmaps,
    save_explanation_report
)

__all__ = [
    # Core
    "GradCAM",
    "BandImportanceAnalyzer",
    "ExplainabilityReport",
    # Visualization
    "heatmap_to_rgb",
    "overlay_heatmap",
    "visualize_gradcam",
    "visualize_all_class_heatmaps",
    "save_explanation_report"
]
