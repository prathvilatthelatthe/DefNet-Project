#!/usr/bin/env python
"""
DeforestNet - Inference Script
Run predictions on satellite imagery.

Usage:
    python predict.py                           # Run on test set
    python predict.py --input image.npy         # Run on single image
    python predict.py --checkpoint best.pt      # Use specific checkpoint
    python predict.py --visualize               # Generate visualizations
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.inference import (
    InferenceEngine, BatchPredictor,
    visualize_prediction, visualize_batch,
    save_prediction_outputs, prediction_to_rgb
)
from src.data import SyntheticDataset
from src.training import MetricTracker
from src.utils.logger import get_logger
from configs.config import (
    CHECKPOINTS_DIR, PREDICTIONS_DIR, DEVICE,
    CLASS_NAMES, NUM_CLASSES
)

logger = get_logger("predict")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run DeforestNet predictions")

    parser.add_argument(
        '--input', type=str, default=None,
        help='Path to input .npy file (single image or batch)'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='best.pt',
        help='Checkpoint filename'
    )
    parser.add_argument(
        '--experiment', type=str, default='test_quick',
        help='Experiment name (checkpoint folder)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--eval-test', action='store_true',
        help='Evaluate on test set'
    )
    parser.add_argument(
        '--num-samples', type=int, default=10,
        help='Number of samples to visualize'
    )

    return parser.parse_args()


def evaluate_on_test_set(engine: InferenceEngine, batch_size: int = 8):
    """Evaluate model on test set."""
    logger.info("Loading test dataset...")
    test_dataset = SyntheticDataset("test")

    logger.info(f"  Test samples: {len(test_dataset)}")

    # Initialize metrics
    metric_tracker = MetricTracker(NUM_CLASSES, CLASS_NAMES)

    # Run inference
    logger.info("Running inference on test set...")
    predictor = BatchPredictor(engine, batch_size=batch_size)

    images = test_dataset.images
    masks = test_dataset.masks

    predictions, confidences = predictor.predict_dataset(images, show_progress=True)

    # Compute metrics
    metric_tracker.update(
        torch.from_numpy(predictions),
        torch.from_numpy(masks)
    )

    metrics = metric_tracker.compute()

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Mean IoU:         {metrics['mean_iou']:.4f}")
    logger.info(f"  Mean Dice:        {metrics['mean_dice']:.4f}")
    logger.info(f"  Mean F1:          {metrics['mean_f1']:.4f}")

    logger.info("\nPer-class IoU:")
    for name in CLASS_NAMES:
        iou = metrics.get(f'{name}_iou', 0.0)
        dice = metrics.get(f'{name}_dice', 0.0)
        logger.info(f"  {name:15s}: IoU={iou:.4f}, Dice={dice:.4f}")

    logger.info("=" * 60)

    return metrics, predictions, confidences, images, masks


def predict_single_image(engine: InferenceEngine, image_path: Path, output_dir: Path):
    """Run prediction on a single image."""
    logger.info(f"Loading image: {image_path}")
    image = np.load(image_path)

    logger.info(f"  Shape: {image.shape}")

    # Handle batch input
    if image.ndim == 4:
        logger.info(f"  Detected batch of {len(image)} images")
        for i, img in enumerate(image):
            result = engine.predict(img, return_probs=True)
            summary = engine.get_deforestation_summary(
                result['prediction'],
                result['confidence']
            )

            sample_dir = output_dir / f"sample_{i:03d}"
            save_prediction_outputs(
                img, result['prediction'], result['confidence'],
                sample_dir, name="prediction", summary=summary
            )
    else:
        result = engine.predict(image, return_probs=True)
        summary = engine.get_deforestation_summary(
            result['prediction'],
            result['confidence']
        )

        save_prediction_outputs(
            image, result['prediction'], result['confidence'],
            output_dir, name="prediction", summary=summary
        )

        # Print summary
        logger.info("\nDeforestation Summary:")
        logger.info(f"  Forest area: {summary['forest_area_hectares']:.2f} hectares")
        logger.info(f"  Deforestation: {summary['deforestation_area_hectares']:.2f} hectares")
        logger.info(f"  Percentage: {summary['deforestation_percentage']:.1f}%")
        logger.info(f"  Dominant cause: {summary['dominant_cause']}")
        logger.info(f"  Average confidence: {summary['average_confidence']:.2f}")


def main():
    """Main inference function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("DeforestNet Inference")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PREDICTIONS_DIR / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint_path = CHECKPOINTS_DIR / args.experiment / args.checkpoint

    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        engine = InferenceEngine(checkpoint_path=checkpoint_path)
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Using untrained model")
        engine = InferenceEngine()

    # Run inference
    if args.input:
        # Single image/batch inference
        image_path = Path(args.input)
        predict_single_image(engine, image_path, output_dir)

    elif args.eval_test:
        # Evaluate on test set
        metrics, preds, confs, images, masks = evaluate_on_test_set(
            engine, batch_size=args.batch_size
        )

        # Generate visualizations
        if args.visualize:
            logger.info(f"\nGenerating visualizations for {args.num_samples} samples...")
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # Save batch visualization
            n = min(args.num_samples, len(images))
            visualize_batch(
                images[:n], preds[:n], confs[:n], masks[:n],
                max_samples=n,
                save_path=viz_dir / "batch_visualization.png"
            )
            logger.info(f"  Saved batch visualization")

            # Save individual visualizations
            for i in range(n):
                save_prediction_outputs(
                    images[i], preds[i], confs[i],
                    viz_dir / f"sample_{i:03d}",
                    name="prediction"
                )

            logger.info(f"  Saved {n} individual visualizations to {viz_dir}")

    else:
        # Default: run on a few test samples
        logger.info("Running quick inference on test samples...")
        test_dataset = SyntheticDataset("test")

        n_samples = min(5, len(test_dataset))
        images = test_dataset.images[:n_samples]
        masks = test_dataset.masks[:n_samples]

        predictor = BatchPredictor(engine, batch_size=n_samples)
        preds, confs = predictor.predict_dataset(images, show_progress=False)

        # Visualize
        if args.visualize:
            visualize_batch(
                images, preds, confs, masks,
                save_path=output_dir / "quick_inference.png"
            )
            logger.info(f"Saved visualization to {output_dir / 'quick_inference.png'}")

        # Print sample deforestation summary
        for i in range(n_samples):
            summary = engine.get_deforestation_summary(preds[i], confs[i])
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Deforestation: {summary['deforestation_percentage']:.1f}%")
            logger.info(f"  Dominant cause: {summary['dominant_cause']}")

    logger.info("\n" + "=" * 60)
    logger.info("Inference complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
