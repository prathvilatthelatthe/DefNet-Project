"""
DeforestNet - Training Package
Training pipeline, loss functions, and metrics.
"""

from .losses import (
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    IoULoss,
    build_loss
)

from .metrics import (
    MetricTracker,
    EarlyStopping,
    compute_confusion_matrix,
    iou_from_cm,
    dice_from_cm,
    f1_from_cm,
    precision_from_cm,
    recall_from_cm,
    overall_accuracy,
    compute_class_weights
)

from .trainer import (
    Trainer,
    create_optimizer,
    create_scheduler
)

__all__ = [
    # Losses
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "IoULoss",
    "build_loss",
    # Metrics
    "MetricTracker",
    "EarlyStopping",
    "compute_confusion_matrix",
    "iou_from_cm",
    "dice_from_cm",
    "f1_from_cm",
    "precision_from_cm",
    "recall_from_cm",
    "overall_accuracy",
    "compute_class_weights",
    # Trainer
    "Trainer",
    "create_optimizer",
    "create_scheduler"
]
