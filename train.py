#!/usr/bin/env python
"""
DeforestNet - Training Script
Run model training with configurable parameters.

Usage:
    python train.py                     # Train with defaults
    python train.py --epochs 50         # Train for 50 epochs
    python train.py --batch-size 8      # Use batch size 8
    python train.py --quick             # Quick test (5 epochs)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models import build_model
from src.data import create_dataloaders, SyntheticDataset
from src.training import Trainer, CombinedLoss, create_optimizer, create_scheduler
from src.utils.logger import get_logger
from configs.config import (
    TRAINING_CONFIG, DEVICE, CHECKPOINTS_DIR,
    NUM_CLASSES, CLASS_NAMES
)

logger = get_logger("train")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DeforestNet model")

    parser.add_argument(
        '--epochs', type=int, default=TRAINING_CONFIG["num_epochs"],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=TRAINING_CONFIG["batch_size"],
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr', type=float, default=TRAINING_CONFIG["learning_rate"],
        help='Learning rate'
    )
    parser.add_argument(
        '--experiment', type=str, default='deforestnet',
        help='Experiment name for checkpoints'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick training run (5 epochs, smaller batches)'
    )
    parser.add_argument(
        '--no-augment', action='store_true',
        help='Disable data augmentation'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Quick mode adjustments
    if args.quick:
        args.epochs = 5
        args.batch_size = 4
        logger.info("Quick mode: 5 epochs, batch size 4")

    logger.info("=" * 60)
    logger.info("DeforestNet Training")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Experiment: {args.experiment}")

    # Create dataloaders
    logger.info("\nLoading datasets...")
    loaders = create_dataloaders(batch_size=args.batch_size)
    logger.info(f"  Train: {len(loaders['train'])} batches")
    logger.info(f"  Val: {len(loaders['val'])} batches")
    logger.info(f"  Test: {len(loaders['test'])} batches")

    # Compute class weights from training data
    logger.info("\nComputing class weights...")
    train_dataset = SyntheticDataset("train")
    class_weights = train_dataset.get_class_weights().tolist()
    logger.info(f"  Class weights: {[f'{w:.3f}' for w in class_weights]}")

    # Build model
    logger.info("\nBuilding model...")
    model = build_model()
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {num_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Create loss function
    criterion = CombinedLoss(
        class_weights=class_weights,
        ce_weight=0.5,
        dice_weight=0.3,
        focal_weight=0.2,
        focal_gamma=2.0
    )

    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type='adamw',
        lr=args.lr,
        weight_decay=1e-4
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type='plateau',
        num_epochs=args.epochs
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        checkpoint_dir=CHECKPOINTS_DIR / args.experiment,
        experiment_name=args.experiment,
        class_weights=class_weights
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\nResuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train!
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    history = trainer.train(
        num_epochs=args.epochs,
        save_every=10,
        metric_name='mean_iou'
    )

    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final evaluation on test set...")
    logger.info("=" * 60)

    # Load best model
    trainer.load_checkpoint('best.pt')

    # Evaluate on test set
    test_trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['test'],  # Use test set for final eval
        criterion=criterion,
        device=DEVICE,
        checkpoint_dir=CHECKPOINTS_DIR / args.experiment,
        experiment_name=args.experiment
    )

    test_metrics = test_trainer.validate()

    logger.info("\nTest Set Results:")
    logger.info("-" * 40)
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Mean IoU: {test_metrics['mean_iou']:.4f}")
    logger.info(f"  Mean Dice: {test_metrics['mean_dice']:.4f}")
    logger.info(f"  Mean F1: {test_metrics['mean_f1']:.4f}")

    logger.info("\nPer-class IoU:")
    for i, name in enumerate(CLASS_NAMES):
        iou = test_metrics.get(f'{name}_iou', 0.0)
        logger.info(f"  {name}: {iou:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best model saved to: {CHECKPOINTS_DIR / args.experiment / 'best.pt'}")
    logger.info("=" * 60)

    return history


if __name__ == "__main__":
    main()
