"""
DeforestNet - Models Package
Neural network models for deforestation detection.
"""

from .unet import (
    UNet,
    UNetResNet34,
    build_model,
    count_parameters,
    model_summary,
    ResNet34Encoder,
    UNetDecoder,
    SegmentationHead,
    DecoderBlock,
    BasicBlock,
    ConvBNReLU
)

__all__ = [
    # Main model
    "UNet",
    "UNetResNet34",  # Alias
    "build_model",
    # Utilities
    "count_parameters",
    "model_summary",
    # Components
    "ResNet34Encoder",
    "UNetDecoder",
    "SegmentationHead",
    "DecoderBlock",
    "BasicBlock",
    "ConvBNReLU"
]
