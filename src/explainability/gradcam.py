"""
DeforestNet - GradCAM Explainability
Gradient-weighted Class Activation Mapping for U-Net.

Shows "where the model is looking" for each prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import UNet, build_model
from src.utils.logger import get_logger
from configs.config import (
    DEVICE, NUM_CLASSES, CLASS_NAMES, BAND_NAMES, TOTAL_CHANNELS
)

logger = get_logger("gradcam")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for U-Net.

    GradCAM highlights which regions of the input image
    contributed most to a particular class prediction.

    How it works:
    1. Forward pass through the model
    2. Hook into a target layer to capture activations
    3. Backward pass from target class score
    4. Compute importance weights = global avg pool of gradients
    5. Weighted sum of feature maps = heatmap
    """

    def __init__(
        self,
        model: UNet,
        target_layer: str = 'bottleneck',
        device: str = DEVICE
    ):
        """
        Initialize GradCAM.

        Args:
            model: The U-Net model
            target_layer: Which layer to visualize
                - 'bottleneck': Deepest encoder layer (default)
                - 'encoder_3': Third encoder block
                - 'encoder_2': Second encoder block
                - 'decoder': Final decoder output
            device: Computation device
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_layer = target_layer

        # Storage for hooks
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

        logger.info(f"GradCAM initialized, target_layer={target_layer}")

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        target = self._get_target_module()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def _get_target_module(self) -> torch.nn.Module:
        """Get the target layer module."""
        if self.target_layer == 'bottleneck':
            return self.model.encoder.layer4
        elif self.target_layer == 'encoder_3':
            return self.model.encoder.layer3
        elif self.target_layer == 'encoder_2':
            return self.model.encoder.layer2
        elif self.target_layer == 'encoder_1':
            return self.model.encoder.layer1
        elif self.target_layer == 'decoder':
            return self.model.decoder
        else:
            raise ValueError(f"Unknown target layer: {self.target_layer}")

    def generate(
        self,
        image: Union[np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap for a single image.

        Args:
            image: Input image (C, H, W) or (1, C, H, W)
            target_class: Class to explain (None = predicted class)
            normalize: Whether to normalize heatmap to [0, 1]

        Returns:
            Heatmap (H, W) values in [0, 1]
        """
        # Prepare input
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.astype(np.float32))
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        image.requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        logits = self.model(image)  # (1, C, H, W)

        # Determine target class
        if target_class is None:
            # Use the most predicted class (by total pixel score)
            class_scores = logits.sum(dim=(2, 3))  # (1, C)
            target_class = class_scores.argmax(dim=1).item()

        # Get score for target class (sum over all pixels)
        target_score = logits[0, target_class].sum()

        # Backward pass
        target_score.backward(retain_graph=False)

        # Get activations and gradients
        activations = self.activations  # (1, channels, h, w)
        gradients = self.gradients      # (1, channels, h, w)

        if activations is None or gradients is None:
            logger.warning("No activations/gradients captured")
            h, w = image.shape[2], image.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        # Compute importance weights (global average pooling of gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, channels, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam, size=(image.shape[2], image.shape[3]),
            mode='bilinear', align_corners=False
        )

        # Convert to numpy
        heatmap = cam[0, 0].cpu().numpy()

        # Normalize
        if normalize and heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap.astype(np.float32)

    def generate_all_classes(
        self,
        image: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Generate GradCAM heatmaps for all classes.

        Args:
            image: Input image (C, H, W)

        Returns:
            Dictionary mapping class names to heatmaps
        """
        heatmaps = {}

        for class_idx, class_name in enumerate(CLASS_NAMES):
            heatmap = self.generate(image, target_class=class_idx)
            heatmaps[class_name] = heatmap

        return heatmaps


class BandImportanceAnalyzer:
    """
    Analyze which input bands contribute most to predictions.

    Uses gradient-based importance: the magnitude of gradients
    w.r.t. each input band indicates its importance.
    """

    def __init__(self, model: UNet, device: str = DEVICE):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def compute_band_importance(
        self,
        image: Union[np.ndarray, torch.Tensor],
        target_class: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute importance score for each input band.

        Args:
            image: Input image (C, H, W)
            target_class: Class to analyze (None = predicted class)

        Returns:
            Dictionary mapping band names to importance scores (0-1)
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.astype(np.float32))
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        image.requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        logits = self.model(image)

        # Determine target class
        if target_class is None:
            class_scores = logits.sum(dim=(2, 3))
            target_class = class_scores.argmax(dim=1).item()

        # Backward pass from target class
        target_score = logits[0, target_class].sum()
        target_score.backward()

        # Get gradient w.r.t. input
        input_gradients = image.grad[0].cpu().numpy()  # (C, H, W)

        # Compute importance per band (mean absolute gradient)
        band_importance = {}
        importance_scores = []

        for i, name in enumerate(BAND_NAMES):
            score = float(np.abs(input_gradients[i]).mean())
            importance_scores.append(score)

        # Normalize to sum to 1.0
        total = sum(importance_scores) + 1e-8
        for i, name in enumerate(BAND_NAMES):
            band_importance[name] = importance_scores[i] / total

        return band_importance

    def compute_batch_importance(
        self,
        images: np.ndarray,
        max_samples: int = 20
    ) -> Dict[str, float]:
        """
        Compute average band importance over multiple images.

        Args:
            images: (N, C, H, W) batch of images
            max_samples: Maximum samples to use

        Returns:
            Average band importance scores
        """
        n = min(len(images), max_samples)
        all_importance = {name: 0.0 for name in BAND_NAMES}

        for i in range(n):
            importance = self.compute_band_importance(images[i])
            for name, score in importance.items():
                all_importance[name] += score

        # Average
        for name in all_importance:
            all_importance[name] /= n

        return all_importance


class ExplainabilityReport:
    """
    Generate comprehensive explanation for a prediction.
    """

    def __init__(self, model: UNet, device: str = DEVICE):
        self.model = model
        self.device = device
        self.gradcam = GradCAM(model, target_layer='bottleneck', device=device)
        self.band_analyzer = BandImportanceAnalyzer(model, device=device)

    def generate_report(
        self,
        image: np.ndarray,
        prediction: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Generate full explanation report for an image.

        Args:
            image: Input image (C, H, W)
            prediction: Pre-computed prediction (optional)

        Returns:
            Dictionary with:
                - predicted_class: dominant class name
                - confidence: prediction confidence
                - gradcam_heatmap: GradCAM visualization
                - band_importance: per-band importance scores
                - top_bands: top 3 most important bands
                - explanation_text: human-readable explanation
        """
        # Get prediction if not provided
        if prediction is None:
            image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            with torch.no_grad():
                logits = self.model(image_tensor)
                probs = F.softmax(logits, dim=1)
                confidence_map, pred_map = probs.max(dim=1)
                prediction = pred_map[0].cpu().numpy()
                confidence = confidence_map[0].cpu().numpy()
        else:
            confidence = np.ones_like(prediction, dtype=np.float32) * 0.5

        # Find dominant deforestation class
        unique, counts = np.unique(prediction, return_counts=True)
        non_forest_mask = unique > 0
        if non_forest_mask.any():
            non_forest_classes = unique[non_forest_mask]
            non_forest_counts = counts[non_forest_mask]
            dominant_idx = non_forest_classes[np.argmax(non_forest_counts)]
        else:
            dominant_idx = unique[np.argmax(counts)]

        dominant_class = CLASS_NAMES[dominant_idx]

        # Generate GradCAM for dominant class
        heatmap = self.gradcam.generate(image, target_class=int(dominant_idx))

        # Compute band importance
        band_importance = self.band_analyzer.compute_band_importance(
            image, target_class=int(dominant_idx)
        )

        # Sort bands by importance
        sorted_bands = sorted(band_importance.items(), key=lambda x: x[1], reverse=True)
        top_bands = sorted_bands[:3]

        # Generate explanation text
        explanation = self._generate_explanation(
            dominant_class, top_bands, confidence.mean()
        )

        report = {
            'predicted_class': dominant_class,
            'predicted_class_idx': int(dominant_idx),
            'mean_confidence': float(confidence.mean()),
            'gradcam_heatmap': heatmap,
            'band_importance': band_importance,
            'top_bands': {name: float(score) for name, score in top_bands},
            'explanation_text': explanation,
            'class_distribution': {
                CLASS_NAMES[c]: int(count)
                for c, count in zip(unique, counts)
            }
        }

        return report

    def _generate_explanation(
        self,
        dominant_class: str,
        top_bands: List[Tuple[str, float]],
        confidence: float
    ) -> str:
        """Generate human-readable explanation."""
        band_explanations = {
            'S1_VV': 'radar vertical backscatter (structural changes)',
            'S1_VH': 'radar cross-polarized signal (volume scattering)',
            'S2_B2_Blue': 'blue light reflectance (water/mineral detection)',
            'S2_B3_Green': 'green light reflectance (healthy vegetation)',
            'S2_B4_Red': 'red light reflectance (soil exposure/brown areas)',
            'S2_B8_NIR': 'near-infrared reflectance (vegetation vitality)',
            'NDVI': 'vegetation health index (green = high, brown = low)',
            'EVI': 'enhanced vegetation index (forest density)',
            'SAVI': 'soil-adjusted vegetation index',
            'VV_VH_Ratio': 'forest structure ratio (high = dense forest)',
            'RVI': 'radar vegetation index (canopy condition)'
        }

        class_explanations = {
            'Forest': 'healthy forest cover with no signs of disturbance',
            'Logging': 'tree removal patterns with exposed soil and reduced canopy',
            'Mining': 'excavation patterns with water accumulation and bare earth',
            'Agriculture': 'regular geometric patterns of crop cultivation',
            'Fire': 'burned areas with charred vegetation and heat signatures',
            'Infrastructure': 'roads, buildings, or constructed surfaces'
        }

        explanation = f"AI Classification: {dominant_class} (confidence: {confidence:.0%})\n\n"
        explanation += f"The model detected {class_explanations.get(dominant_class, 'unknown pattern')}.\n\n"
        explanation += "Key factors in this decision:\n"

        for i, (band, score) in enumerate(top_bands, 1):
            pct = score * 100
            desc = band_explanations.get(band, 'spectral feature')
            explanation += f"  {i}. {band} ({pct:.0f}% importance): {desc}\n"

        total_top = sum(s for _, s in top_bands) * 100
        explanation += f"\nThese top 3 bands account for {total_top:.0f}% of the decision."

        return explanation


if __name__ == "__main__":
    print("Testing GradCAM...")
    print("=" * 50)

    import numpy as np

    model = build_model()
    model.eval()

    # Create dummy image
    image = np.random.rand(TOTAL_CHANNELS, 256, 256).astype(np.float32)

    # Test GradCAM
    gradcam = GradCAM(model)
    heatmap = gradcam.generate(image, target_class=1)
    print(f"[OK] GradCAM heatmap: {heatmap.shape}, range [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # Test all classes
    all_heatmaps = gradcam.generate_all_classes(image)
    print(f"[OK] All-class heatmaps: {list(all_heatmaps.keys())}")

    # Test band importance
    analyzer = BandImportanceAnalyzer(model)
    importance = analyzer.compute_band_importance(image)
    print(f"[OK] Band importance: {len(importance)} bands")
    for name, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"     Top band: {name} = {score:.3f}")

    print("\n[OK] All GradCAM tests passed!")
