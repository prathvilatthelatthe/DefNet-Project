"""
DeforestNet - Training Pipeline
Complete training orchestration with validation, checkpointing, and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, StepLR
)
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple, Any
import json
import time
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.losses import CombinedLoss, build_loss
from src.training.metrics import MetricTracker, EarlyStopping, compute_class_weights
from src.utils.logger import get_logger
from configs.config import (
    TRAINING_CONFIG, NUM_CLASSES, CLASS_NAMES,
    CHECKPOINTS_DIR, DEVICE
)

logger = get_logger("trainer")


class Trainer:
    """
    Complete training pipeline for DeforestNet.

    Handles:
    - Training and validation loops
    - Loss computation and optimization
    - Metric tracking
    - Learning rate scheduling
    - Checkpointing (best and periodic)
    - Early stopping
    - Training history logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = DEVICE,
        checkpoint_dir: Optional[Path] = None,
        experiment_name: str = "deforestnet",
        class_weights: Optional[List[float]] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            criterion: Loss function (default: CombinedLoss)
            optimizer: Optimizer (default: Adam)
            scheduler: LR scheduler (default: ReduceLROnPlateau)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name for this experiment
            class_weights: Class weights for loss function
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.experiment_name = experiment_name

        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = CHECKPOINTS_DIR
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup loss function
        if criterion is None:
            self.criterion = CombinedLoss(
                class_weights=class_weights,
                ce_weight=0.5,
                dice_weight=0.3,
                focal_weight=0.2
            )
        else:
            self.criterion = criterion

        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=TRAINING_CONFIG["learning_rate"],
                weight_decay=TRAINING_CONFIG.get("weight_decay", 1e-4)
            )
        else:
            self.optimizer = optimizer

        # Setup scheduler
        if scheduler is None:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = scheduler

        # Metric trackers
        self.train_metrics = MetricTracker(NUM_CLASSES, CLASS_NAMES)
        self.val_metrics = MetricTracker(NUM_CLASSES, CLASS_NAMES)

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_iou': [],
            'val_iou': [],
            'lr': []
        }

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=TRAINING_CONFIG.get("early_stopping_patience", 15),
            mode='max'
        )

        logger.info(f"Trainer initialized: {experiment_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)

            # Compute loss
            if isinstance(self.criterion, CombinedLoss):
                loss, loss_dict = self.criterion(logits, masks)
            else:
                loss = self.criterion(logits, masks)
                loss_dict = {'total': loss.item()}

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            preds = logits.argmax(dim=1)
            self.train_metrics.update(preds, masks, loss_dict['total'])
            total_loss += loss_dict['total']

            # Progress logging
            if (batch_idx + 1) % 20 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"  Batch {batch_idx + 1}/{num_batches} - Loss: {avg_loss:.4f}")

        return self.train_metrics.compute()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            logits = self.model(images)

            # Compute loss
            if isinstance(self.criterion, CombinedLoss):
                loss, loss_dict = self.criterion(logits, masks)
            else:
                loss = self.criterion(logits, masks)
                loss_dict = {'total': loss.item()}

            # Update metrics
            preds = logits.argmax(dim=1)
            self.val_metrics.update(preds, masks, loss_dict['total'])

        return self.val_metrics.compute()

    def train(
        self,
        num_epochs: int = None,
        save_every: int = 10,
        metric_name: str = 'mean_iou'
    ) -> Dict[str, List]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs (default from config)
            save_every: Save checkpoint every N epochs
            metric_name: Metric to use for best model selection

        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG["num_epochs"]

        logger.info(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 40)

            # Training
            train_metrics = self.train_epoch()
            logger.info(f"Train - Loss: {train_metrics.get('loss', 0):.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"IoU: {train_metrics['mean_iou']:.4f}")

            # Validation
            val_metrics = self.validate()
            logger.info(f"Val   - Loss: {val_metrics.get('loss', 0):.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"IoU: {val_metrics['mean_iou']:.4f}")

            # Update history
            self.history['train_loss'].append(train_metrics.get('loss', 0))
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_iou'].append(train_metrics['mean_iou'])
            self.history['val_iou'].append(val_metrics['mean_iou'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Learning rate scheduling
            current_metric = val_metrics[metric_name]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(current_metric)
            elif self.scheduler is not None:
                self.scheduler.step()

            # Save best model
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.save_checkpoint('best.pt', val_metrics)
                logger.info(f"  New best model! {metric_name}: {current_metric:.4f}")

            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt', val_metrics)

            # Early stopping
            if self.early_stopping(current_metric):
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break

            epoch_time = time.time() - epoch_start
            logger.info(f"  Epoch time: {epoch_time:.1f}s")

        # Training complete
        total_time = time.time() - start_time
        logger.info(f"\nTraining complete in {total_time/60:.1f} minutes")
        logger.info(f"Best {metric_name}: {self.best_metric:.4f} at epoch {self.best_epoch}")

        # Save final checkpoint and history
        self.save_checkpoint('final.pt', val_metrics)
        self.save_history()

        return self.history

    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': TRAINING_CONFIG,
            'timestamp': datetime.now().isoformat()
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def save_history(self):
        """Save training history to JSON."""
        path = self.checkpoint_dir / 'history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {path}")

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    lr: float = None,
    weight_decay: float = 1e-4
) -> optim.Optimizer:
    """
    Create optimizer.

    Args:
        model: Model to optimize
        optimizer_type: 'adam', 'adamw', 'sgd'
        lr: Learning rate
        weight_decay: Weight decay

    Returns:
        Optimizer instance
    """
    if lr is None:
        lr = TRAINING_CONFIG["learning_rate"]

    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'plateau',
    num_epochs: int = None,
    steps_per_epoch: int = None
) -> Any:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer
        scheduler_type: 'plateau', 'cosine', 'onecycle', 'step'
        num_epochs: Number of training epochs
        steps_per_epoch: Number of batches per epoch

    Returns:
        Scheduler instance
    """
    if num_epochs is None:
        num_epochs = TRAINING_CONFIG["num_epochs"]

    if scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'onecycle':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        return OneCycleLR(
            optimizer,
            max_lr=TRAINING_CONFIG["learning_rate"] * 10,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs
        )
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


if __name__ == "__main__":
    # Quick test of training components
    print("Testing training components...")

    from src.models import build_model
    from src.data import create_dataloaders

    # Build model
    model = build_model()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders (small batch for testing)
    loaders = create_dataloaders(batch_size=4)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        experiment_name='test_run'
    )

    # Test one epoch
    print("\nRunning one training epoch...")
    train_metrics = trainer.train_epoch()
    print(f"  Train metrics: {train_metrics}")

    print("\nRunning validation...")
    val_metrics = trainer.validate()
    print(f"  Val metrics: {val_metrics}")

    print("\n[OK] Training components working!")
