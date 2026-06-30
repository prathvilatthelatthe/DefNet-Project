"""
DeforestNet - U-Net with ResNet-34 Encoder
6-class semantic segmentation model for deforestation detection.

Architecture:
  Encoder: ResNet-34 backbone adapted for 11-channel satellite input
  Decoder: U-Net style with skip connections and bilinear upsampling

Input:  [B, 11, 256, 256]  (11 feature bands)
Output: [B, 6, 256, 256]   (6-class logits)

Classes:
  0: Forest
  1: Logging
  2: Mining
  3: Agriculture
  4: Fire
  5: Infrastructure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import (
    TOTAL_CHANNELS, NUM_CLASSES, MODEL_CONFIG, CLASS_NAMES
)


# ============================================================
# BUILDING BLOCKS
# ============================================================

class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """
    ResNet basic block with two 3x3 convolutions and residual connection.

    Structure:
        x → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+x) → ReLU → out
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ============================================================
# ENCODER
# ============================================================

class ResNet34Encoder(nn.Module):
    """
    ResNet-34 encoder adapted for multi-channel satellite imagery.

    Produces feature maps at 5 resolution levels for skip connections:
      - x0: [B, 64, H/2, W/2]     (after initial conv)
      - x1: [B, 64, H/4, W/4]     (after layer1)
      - x2: [B, 128, H/8, W/8]    (after layer2)
      - x3: [B, 256, H/16, W/16]  (after layer3)
      - x4: [B, 512, H/32, W/32]  (bottleneck, after layer4)

    For 256x256 input:
      x0: 128x128, x1: 64x64, x2: 32x32, x3: 16x16, x4: 8x8
    """

    def __init__(self, in_channels: int = 11):
        super().__init__()

        # Initial convolution: adapt to N input channels
        # 7x7 conv with stride 2 reduces spatial dimensions by half
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers (ResNet-34 configuration: [3, 4, 6, 3] blocks)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

        # Channel dimensions at each level (for decoder reference)
        self.channels = [64, 64, 128, 256, 512]

    def _make_layer(self, in_channels: int, out_channels: int,
                    blocks: int, stride: int) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass returning features at each resolution.

        Returns:
            Tuple of (x0, x1, x2, x3, x4) feature maps
        """
        x0 = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2]
        x1 = self.layer1(self.maxpool(x0))        # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)                      # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)                      # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)                      # [B, 512, H/32, W/32]

        return x0, x1, x2, x3, x4


# ============================================================
# DECODER
# ============================================================

class DecoderBlock(nn.Module):
    """
    U-Net decoder block: upsample → concat skip → double conv.

    Uses bilinear upsampling followed by two 3x3 convolutions
    with BatchNorm and ReLU.

    Args:
        in_channels: Channels from lower level (being upsampled)
        skip_channels: Channels from skip connection
        out_channels: Output channels
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True)

        # After concatenation: in_channels + skip_channels
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map from lower level [B, in_channels, H, W]
            skip: Skip connection from encoder [B, skip_channels, 2H, 2W]

        Returns:
            Refined feature map [B, out_channels, 2H, 2W]
        """
        x = self.upsample(x)

        # Handle spatial size mismatches from odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetDecoder(nn.Module):
    """
    U-Net decoder with skip connections from encoder.

    Takes encoder features and progressively upsamples while
    incorporating skip connections for fine detail recovery.
    """

    def __init__(self, encoder_channels: List[int] = [64, 64, 128, 256, 512]):
        super().__init__()

        # Decoder blocks (reverse order)
        # decoder4: 512 → 256 (with 256 skip)
        # decoder3: 256 → 128 (with 128 skip)
        # decoder2: 128 → 64 (with 64 skip)
        # decoder1: 64 → 32 (with 64 skip)

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 32)

        # Final upsample to restore original resolution
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                           align_corners=True)

        self.out_channels = 32

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            features: Tuple of encoder features (x0, x1, x2, x3, x4)

        Returns:
            Decoded feature map [B, 32, H, W]
        """
        x0, x1, x2, x3, x4 = features

        d4 = self.decoder4(x4, x3)  # [B, 256, 16, 16]
        d3 = self.decoder3(d4, x2)  # [B, 128, 32, 32]
        d2 = self.decoder2(d3, x1)  # [B, 64, 64, 64]
        d1 = self.decoder1(d2, x0)  # [B, 32, 128, 128]

        d0 = self.final_upsample(d1)  # [B, 32, 256, 256]

        return d0


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

class SegmentationHead(nn.Module):
    """
    Segmentation head for pixel-wise classification.

    Applies dropout and 1x1 convolution to produce class logits.
    """

    def __init__(self, in_channels: int, num_classes: int, dropout_p: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, in_channels, H, W]

        Returns:
            Class logits [B, num_classes, H, W]
        """
        x = self.dropout(x)
        return self.conv(x)


# ============================================================
# FULL U-NET MODEL
# ============================================================

class UNet(nn.Module):
    """
    U-Net with ResNet-34 encoder for 6-class deforestation segmentation.

    Architecture Overview:

    Encoder (ResNet-34):
        Input [B, 11, 256, 256]
          → x0 [B, 64, 128, 128]   (initial conv)
          → x1 [B, 64, 64, 64]     (layer1)
          → x2 [B, 128, 32, 32]    (layer2)
          → x3 [B, 256, 16, 16]    (layer3)
          → x4 [B, 512, 8, 8]      (layer4 / bottleneck)

    Decoder (U-Net):
        d4 = Up(x4) + x3 → [B, 256, 16, 16]
        d3 = Up(d4) + x2 → [B, 128, 32, 32]
        d2 = Up(d3) + x1 → [B, 64, 64, 64]
        d1 = Up(d2) + x0 → [B, 32, 128, 128]
        d0 = Up(d1)      → [B, 32, 256, 256]

    Head:
        logits = Conv1x1(d0) → [B, 6, 256, 256]

    Args:
        in_channels: Number of input bands (default: 11)
        num_classes: Number of output classes (default: 6)
        dropout_p: Dropout probability in segmentation head
    """

    def __init__(self, in_channels: int = 11, num_classes: int = 6,
                 dropout_p: float = 0.2):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.encoder = ResNet34Encoder(in_channels=in_channels)

        # Decoder
        self.decoder = UNetDecoder(encoder_channels=self.encoder.channels)

        # Segmentation head
        self.head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            num_classes=num_classes,
            dropout_p=dropout_p
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Kaiming Normal initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            Class logits [B, num_classes, H, W] (not softmax)
        """
        # Encode
        features = self.encoder(x)

        # Decode
        decoded = self.decoder(features)

        # Classify
        logits = self.head(decoded)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions (argmax of softmax).

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            Class predictions [B, H, W] with values in [0, num_classes-1]
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (softmax).

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            Class probabilities [B, num_classes, H, W]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for visualization/GradCAM.

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            Dictionary of feature maps at each level
        """
        x0, x1, x2, x3, x4 = self.encoder(x)
        decoded = self.decoder((x0, x1, x2, x3, x4))

        return {
            "encoder_0": x0,
            "encoder_1": x1,
            "encoder_2": x2,
            "encoder_3": x3,
            "bottleneck": x4,
            "decoder_final": decoded
        }


# ============================================================
# MODEL FACTORY & UTILITIES
# ============================================================

def build_model(in_channels: int = None, num_classes: int = None,
                dropout_p: float = 0.2, pretrained: bool = False) -> UNet:
    """
    Factory function to create the U-Net model.

    Args:
        in_channels: Number of input bands (default: from config)
        num_classes: Number of output classes (default: from config)
        dropout_p: Dropout probability
        pretrained: Not used (no pretrained weights for 11-channel input)

    Returns:
        UNet model instance
    """
    if in_channels is None:
        in_channels = TOTAL_CHANNELS
    if num_classes is None:
        num_classes = NUM_CLASSES

    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_p=dropout_p
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Returns:
        Dictionary with total, trainable, and frozen parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_mb": total * 4 / (1024 * 1024)  # Assuming float32
    }


def model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 11, 256, 256),
                  device: str = "cpu") -> str:
    """
    Generate a summary of the model architecture.

    Args:
        model: The model to summarize
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on

    Returns:
        String summary
    """
    model = model.to(device)
    model.eval()

    # Count parameters
    params = count_parameters(model)

    # Test forward pass
    x = torch.randn(*input_size).to(device)
    with torch.no_grad():
        output = model(x)

    summary = []
    summary.append("=" * 60)
    summary.append("  U-Net Model Summary")
    summary.append("=" * 60)
    summary.append(f"  Input shape:  {list(input_size)}")
    summary.append(f"  Output shape: {list(output.shape)}")
    summary.append(f"  Parameters:   {params['total']:,}")
    summary.append(f"  Trainable:    {params['trainable']:,}")
    summary.append(f"  Model size:   {params['total_mb']:.1f} MB")
    summary.append("=" * 60)
    summary.append("")
    summary.append("  Architecture:")
    summary.append("  ├─ Encoder (ResNet-34)")
    summary.append("  │   ├─ Initial Conv: 11 → 64 channels")
    summary.append("  │   ├─ Layer1: 64 → 64 (3 blocks)")
    summary.append("  │   ├─ Layer2: 64 → 128 (4 blocks)")
    summary.append("  │   ├─ Layer3: 128 → 256 (6 blocks)")
    summary.append("  │   └─ Layer4: 256 → 512 (3 blocks)")
    summary.append("  │")
    summary.append("  ├─ Decoder (U-Net)")
    summary.append("  │   ├─ Block4: 512+256 → 256")
    summary.append("  │   ├─ Block3: 256+128 → 128")
    summary.append("  │   ├─ Block2: 128+64 → 64")
    summary.append("  │   └─ Block1: 64+64 → 32")
    summary.append("  │")
    summary.append("  └─ Head")
    summary.append("      ├─ Dropout(0.2)")
    summary.append(f"      └─ Conv 1×1: 32 → {model.num_classes}")
    summary.append("")
    summary.append(f"  Classes: {CLASS_NAMES}")
    summary.append("=" * 60)

    return "\n".join(summary)


# ============================================================
# BACKWARDS COMPATIBILITY
# ============================================================

# Alias for old code
UNetResNet34 = UNet


if __name__ == "__main__":
    print("Testing U-Net model...")
    print()

    # Build model
    model = build_model()

    # Print summary
    print(model_summary(model))

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 11, 256, 256)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        predictions = model.predict(x)
        probabilities = model.predict_proba(x)

    print(f"  Input:        {x.shape}")
    print(f"  Logits:       {logits.shape}")
    print(f"  Predictions:  {predictions.shape}")
    print(f"  Probabilities:{probabilities.shape}")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"  Unique preds: {torch.unique(predictions).tolist()}")

    # Test feature extraction
    print("\nTesting feature extraction...")
    features = model.get_feature_maps(x)
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    print("\n[OK] All model tests passed!")
