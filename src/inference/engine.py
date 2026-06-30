"""
DeforestNet - Inference Engine
Run predictions on satellite imagery with visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import build_model, UNet
from src.utils.logger import get_logger
from configs.config import (
    DEVICE, NUM_CLASSES, CLASS_NAMES, CLASS_COLORS,
    CHECKPOINTS_DIR, PREDICTIONS_DIR, TOTAL_CHANNELS
)

logger = get_logger("inference")


class InferenceEngine:
    """
    Inference engine for deforestation detection.

    Handles:
    - Model loading from checkpoints
    - Single image prediction
    - Batch prediction
    - Probability maps
    - Confidence scores
    """

    def __init__(
        self,
        model: Optional[UNet] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = DEVICE
    ):
        """
        Initialize inference engine.

        Args:
            model: Pre-loaded model (optional)
            checkpoint_path: Path to checkpoint file (optional)
            device: Device to run inference on
        """
        self.device = device
        self.num_classes = NUM_CLASSES
        self.class_names = CLASS_NAMES

        if model is not None:
            self.model = model.to(device)
        elif checkpoint_path is not None:
            self.model = self._load_from_checkpoint(checkpoint_path)
        else:
            # Build fresh model (untrained)
            self.model = build_model().to(device)
            logger.warning("Using untrained model - load checkpoint for real predictions")

        self.model.eval()
        logger.info(f"InferenceEngine initialized on {device}")

    def _load_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> UNet:
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        model = build_model()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"  Metrics: {checkpoint.get('metrics', {})}")
        else:
            model.load_state_dict(checkpoint)

        return model.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_probs: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run prediction on a single image.

        Args:
            image: Input image (C, H, W) or (H, W, C) numpy array or tensor
            return_probs: Whether to return probability maps

        Returns:
            Dictionary with:
                - 'prediction': (H, W) class labels
                - 'confidence': (H, W) confidence scores
                - 'probabilities': (C, H, W) if return_probs=True
        """
        # Prepare input
        if isinstance(image, np.ndarray):
            # Handle different formats
            if image.ndim == 2:
                raise ValueError("Expected multi-band image, got single band")
            if image.shape[-1] == TOTAL_CHANNELS:
                # (H, W, C) -> (C, H, W)
                image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image.astype(np.float32))

        # Add batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Run model
        logits = self.model(image)  # (1, C, H, W)
        probs = F.softmax(logits, dim=1)  # (1, C, H, W)

        # Get predictions and confidence
        confidence, prediction = probs.max(dim=1)  # (1, H, W)

        result = {
            'prediction': prediction[0].cpu().numpy(),
            'confidence': confidence[0].cpu().numpy()
        }

        if return_probs:
            result['probabilities'] = probs[0].cpu().numpy()

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: Union[np.ndarray, torch.Tensor],
        return_probs: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run prediction on a batch of images.

        Args:
            images: Input batch (N, C, H, W)
            return_probs: Whether to return probability maps

        Returns:
            Dictionary with batched results
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images.astype(np.float32))

        images = images.to(self.device)

        # Run model
        logits = self.model(images)
        probs = F.softmax(logits, dim=1)

        confidence, prediction = probs.max(dim=1)

        result = {
            'prediction': prediction.cpu().numpy(),
            'confidence': confidence.cpu().numpy()
        }

        if return_probs:
            result['probabilities'] = probs.cpu().numpy()

        return result

    def get_class_areas(
        self,
        prediction: np.ndarray,
        pixel_area_hectares: float = 0.01
    ) -> Dict[str, float]:
        """
        Calculate area per class from prediction.

        Args:
            prediction: (H, W) class labels
            pixel_area_hectares: Area of each pixel in hectares (default: 10m resolution)

        Returns:
            Dictionary mapping class names to areas in hectares
        """
        areas = {}
        for i, name in enumerate(self.class_names):
            pixel_count = np.sum(prediction == i)
            areas[name] = pixel_count * pixel_area_hectares
        return areas

    def get_deforestation_summary(
        self,
        prediction: np.ndarray,
        confidence: np.ndarray,
        pixel_area_hectares: float = 0.01
    ) -> Dict:
        """
        Generate deforestation summary from prediction.

        Args:
            prediction: (H, W) class labels
            confidence: (H, W) confidence scores
            pixel_area_hectares: Area per pixel

        Returns:
            Summary dictionary
        """
        areas = self.get_class_areas(prediction, pixel_area_hectares)

        # Deforestation classes (non-forest)
        deforestation_classes = ['Logging', 'Mining', 'Agriculture', 'Fire', 'Infrastructure']
        total_deforestation = sum(areas[c] for c in deforestation_classes)

        # Find dominant deforestation cause
        deforestation_areas = {c: areas[c] for c in deforestation_classes}
        if total_deforestation > 0:
            dominant_cause = max(deforestation_areas, key=deforestation_areas.get)
        else:
            dominant_cause = None

        # Calculate average confidence for deforestation areas
        deforestation_mask = prediction > 0  # Non-forest
        if deforestation_mask.sum() > 0:
            avg_confidence = float(confidence[deforestation_mask].mean())
        else:
            avg_confidence = 0.0

        summary = {
            'total_area_hectares': float(prediction.size * pixel_area_hectares),
            'forest_area_hectares': areas['Forest'],
            'deforestation_area_hectares': total_deforestation,
            'deforestation_percentage': (total_deforestation / (prediction.size * pixel_area_hectares)) * 100,
            'dominant_cause': dominant_cause,
            'average_confidence': avg_confidence,
            'areas_by_class': areas
        }

        return summary

    def save_prediction(
        self,
        prediction: np.ndarray,
        output_path: Union[str, Path],
        metadata: Optional[Dict] = None
    ):
        """
        Save prediction to file.

        Args:
            prediction: (H, W) class labels
            output_path: Path to save (supports .npy, .npz)
            metadata: Optional metadata to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.npy':
            np.save(output_path, prediction)
        elif output_path.suffix == '.npz':
            if metadata:
                np.savez(output_path, prediction=prediction, **metadata)
            else:
                np.savez(output_path, prediction=prediction)
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")

        logger.info(f"Saved prediction to {output_path}")


class BatchPredictor:
    """
    Process multiple images efficiently.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        batch_size: int = 8
    ):
        self.engine = engine
        self.batch_size = batch_size

    def predict_dataset(
        self,
        images: np.ndarray,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on entire dataset.

        Args:
            images: (N, C, H, W) input images
            show_progress: Whether to show progress

        Returns:
            Tuple of (predictions, confidences)
        """
        n_samples = len(images)
        predictions = []
        confidences = []

        for i in range(0, n_samples, self.batch_size):
            batch = images[i:i + self.batch_size]
            result = self.engine.predict_batch(batch)
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])

            if show_progress:
                progress = min(i + self.batch_size, n_samples)
                print(f"  Processed {progress}/{n_samples} images", end='\r')

        if show_progress:
            print()

        return np.concatenate(predictions), np.concatenate(confidences)


def load_inference_engine(
    checkpoint_name: str = 'best.pt',
    experiment_name: str = 'deforestnet'
) -> InferenceEngine:
    """
    Convenience function to load inference engine.

    Args:
        checkpoint_name: Name of checkpoint file
        experiment_name: Name of experiment folder

    Returns:
        Configured InferenceEngine
    """
    checkpoint_path = CHECKPOINTS_DIR / experiment_name / checkpoint_name

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Using untrained model")
        return InferenceEngine()

    return InferenceEngine(checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    print("Testing Inference Engine...")
    print("=" * 50)

    # Test with untrained model
    engine = InferenceEngine()

    # Create dummy input
    dummy_image = np.random.rand(TOTAL_CHANNELS, 256, 256).astype(np.float32)

    # Run prediction
    result = engine.predict(dummy_image, return_probs=True)

    print(f"Prediction shape: {result['prediction'].shape}")
    print(f"Confidence shape: {result['confidence'].shape}")
    print(f"Probabilities shape: {result['probabilities'].shape}")
    print(f"Unique classes: {np.unique(result['prediction'])}")
    print(f"Confidence range: [{result['confidence'].min():.3f}, {result['confidence'].max():.3f}]")

    # Test summary
    summary = engine.get_deforestation_summary(
        result['prediction'],
        result['confidence']
    )
    print(f"\nDeforestation Summary:")
    print(f"  Total area: {summary['total_area_hectares']:.2f} hectares")
    print(f"  Forest: {summary['forest_area_hectares']:.2f} hectares")
    print(f"  Deforestation: {summary['deforestation_area_hectares']:.2f} hectares")
    print(f"  Dominant cause: {summary['dominant_cause']}")

    print("\n[OK] Inference engine working!")
