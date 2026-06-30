"""
DeforestNet - Inference Package
Prediction engine and visualization utilities.
"""

from .engine import (
    InferenceEngine,
    BatchPredictor,
    load_inference_engine
)

from .visualization import (
    prediction_to_rgb,
    confidence_to_heatmap,
    create_colormap,
    create_legend_image,
    create_overlay,
    visualize_prediction,
    visualize_batch,
    save_prediction_outputs
)

__all__ = [
    # Engine
    "InferenceEngine",
    "BatchPredictor",
    "load_inference_engine",
    # Visualization
    "prediction_to_rgb",
    "confidence_to_heatmap",
    "create_colormap",
    "create_legend_image",
    "create_overlay",
    "visualize_prediction",
    "visualize_batch",
    "save_prediction_outputs"
]
