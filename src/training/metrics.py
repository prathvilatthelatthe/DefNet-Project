"""
DeforestNet - Evaluation Metrics for Multi-Class Segmentation
Segmentation metrics for monitoring training and evaluating model performance.

All metrics operate on:
  - predictions: [B, H, W] integer class predictions (argmax of logits)
  - targets:     [B, H, W] integer ground truth labels (0 to C-1)

Provides both per-batch computation and an accumulator class for
computing metrics across an entire epoch/dataset.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import NUM_CLASSES, CLASS_NAMES


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES
) -> torch.Tensor:
    """
    Compute confusion matrix from predictions and targets.

    Args:
        preds:   [B, H, W] predicted class labels
        targets: [B, H, W] ground truth labels
        num_classes: Number of classes

    Returns:
        [num_classes, num_classes] confusion matrix where
        cm[i, j] = count of pixels with true=i, pred=j
    """
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    # Filter out invalid indices
    mask = (targets_flat >= 0) & (targets_flat < num_classes)
    preds_flat = preds_flat[mask]
    targets_flat = targets_flat[mask]

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=preds.device)

    for t in range(num_classes):
        for p in range(num_classes):
            cm[t, p] = ((targets_flat == t) & (preds_flat == p)).sum()

    return cm


def iou_from_cm(cm: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class IoU (Intersection over Union) from confusion matrix.

    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    """
    num_classes = cm.shape[0]
    iou = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        tp = cm[c, c].float()
        fp = cm[:, c].sum().float() - tp
        fn = cm[c, :].sum().float() - tp
        denom = tp + fp + fn
        iou[c] = tp / denom if denom > 0 else 0.0

    return iou


def dice_from_cm(cm: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class Dice coefficient from confusion matrix.

    Dice_c = 2*TP_c / (2*TP_c + FP_c + FN_c)
    """
    num_classes = cm.shape[0]
    dice = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        tp = cm[c, c].float()
        fp = cm[:, c].sum().float() - tp
        fn = cm[c, :].sum().float() - tp
        denom = 2.0 * tp + fp + fn
        dice[c] = (2.0 * tp) / denom if denom > 0 else 0.0

    return dice


def precision_from_cm(cm: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class precision from confusion matrix.

    Precision_c = TP_c / (TP_c + FP_c)
    """
    num_classes = cm.shape[0]
    precision = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        tp = cm[c, c].float()
        fp = cm[:, c].sum().float() - tp
        denom = tp + fp
        precision[c] = tp / denom if denom > 0 else 0.0

    return precision


def recall_from_cm(cm: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class recall from confusion matrix.

    Recall_c = TP_c / (TP_c + FN_c)
    """
    num_classes = cm.shape[0]
    recall = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        tp = cm[c, c].float()
        fn = cm[c, :].sum().float() - tp
        denom = tp + fn
        recall[c] = tp / denom if denom > 0 else 0.0

    return recall


def f1_from_cm(cm: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class F1 score from confusion matrix.

    F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
    """
    prec = precision_from_cm(cm)
    rec = recall_from_cm(cm)
    num_classes = cm.shape[0]
    f1 = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        denom = prec[c] + rec[c]
        f1[c] = (2.0 * prec[c] * rec[c]) / denom if denom > 0 else 0.0

    return f1


def overall_accuracy(cm: torch.Tensor) -> torch.Tensor:
    """
    Compute overall pixel accuracy.

    Accuracy = sum(TP_c) / total_pixels
    """
    correct = cm.diagonal().sum().float()
    total = cm.sum().float()
    return correct / total if total > 0 else torch.tensor(0.0)


def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class accuracy.

    Accuracy_c = TP_c / (TP_c + FN_c)
    """
    return recall_from_cm(cm)  # Same as recall


class MetricTracker:
    """
    Accumulates confusion matrix across batches and computes
    epoch-level metrics.

    Usage:
        tracker = MetricTracker()
        for batch in loader:
            preds = model(images).argmax(dim=1)
            tracker.update(preds, masks)
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.total_loss = 0.0
        self.num_batches = 0

    def reset(self):
        """Reset accumulated state for new epoch."""
        self.cm.zero_()
        self.total_loss = 0.0
        self.num_batches = 0

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None
    ):
        """
        Update with a batch of predictions and targets.

        Args:
            preds:   [B, H, W] predicted class labels (integer)
            targets: [B, H, W] ground truth labels (integer)
            loss:    Optional loss value for this batch
        """
        batch_cm = compute_confusion_matrix(
            preds.detach().cpu(),
            targets.detach().cpu(),
            self.num_classes
        )
        self.cm += batch_cm

        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dictionary with all computed metrics
        """
        iou = iou_from_cm(self.cm)
        dice = dice_from_cm(self.cm)
        prec = precision_from_cm(self.cm)
        rec = recall_from_cm(self.cm)
        f1 = f1_from_cm(self.cm)
        acc = overall_accuracy(self.cm)

        metrics = {
            'accuracy': acc.item(),
            'mean_iou': iou.mean().item(),
            'mean_dice': dice.mean().item(),
            'mean_f1': f1.mean().item(),
            'mean_precision': prec.mean().item(),
            'mean_recall': rec.mean().item(),
        }

        # Average loss
        if self.num_batches > 0:
            metrics['loss'] = self.total_loss / self.num_batches

        # Per-class metrics
        for c in range(self.num_classes):
            name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            metrics[f'{name}_iou'] = iou[c].item()
            metrics[f'{name}_dice'] = dice[c].item()
            metrics[f'{name}_precision'] = prec[c].item()
            metrics[f'{name}_recall'] = rec[c].item()
            metrics[f'{name}_f1'] = f1[c].item()

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Return confusion matrix as numpy array."""
        return self.cm.numpy()

    def summary(self, include_per_class: bool = True) -> str:
        """Return a formatted string summary of current metrics."""
        m = self.compute()
        lines = [
            f"Accuracy:   {m['accuracy']:.4f}",
            f"Mean IoU:   {m['mean_iou']:.4f}",
            f"Mean Dice:  {m['mean_dice']:.4f}",
            f"Mean F1:    {m['mean_f1']:.4f}",
        ]

        if 'loss' in m:
            lines.insert(0, f"Loss:       {m['loss']:.4f}")

        if include_per_class:
            lines.append("\nPer-class metrics:")
            for c in range(self.num_classes):
                name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
                lines.append(
                    f"  {name:15s}: IoU={m[f'{name}_iou']:.3f}  "
                    f"Dice={m[f'{name}_dice']:.3f}  "
                    f"P={m[f'{name}_precision']:.3f}  "
                    f"R={m[f'{name}_recall']:.3f}  "
                    f"F1={m[f'{name}_f1']:.3f}"
                )

        return "\n".join(lines)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics like accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def compute_class_weights(masks: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Compute class weights based on inverse frequency.

    Args:
        masks: [N, H, W] tensor of ground truth masks
        num_classes: Number of classes

    Returns:
        Tensor of class weights
    """
    class_counts = torch.zeros(num_classes)

    for c in range(num_classes):
        class_counts[c] = (masks == c).sum().float()

    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1)

    # Inverse frequency
    total = masks.numel()
    weights = total / (num_classes * class_counts)

    # Normalize
    weights = weights / weights.sum() * num_classes

    return weights


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Create dummy data
    batch_size = 4
    preds = torch.randint(0, NUM_CLASSES, (batch_size, 256, 256))
    targets = torch.randint(0, NUM_CLASSES, (batch_size, 256, 256))

    print(f"\nInput shapes: preds={preds.shape}, targets={targets.shape}")

    # Test confusion matrix
    cm = compute_confusion_matrix(preds, targets)
    print(f"  Confusion matrix shape: {cm.shape}")

    # Test individual metrics
    iou = iou_from_cm(cm)
    dice = dice_from_cm(cm)
    f1 = f1_from_cm(cm)
    acc = overall_accuracy(cm)

    print(f"  Overall Accuracy: {acc:.4f}")
    print(f"  Mean IoU: {iou.mean():.4f}")
    print(f"  Mean Dice: {dice.mean():.4f}")
    print(f"  Mean F1: {f1.mean():.4f}")

    # Test MetricTracker
    print("\nTesting MetricTracker...")
    tracker = MetricTracker()

    for _ in range(5):
        batch_preds = torch.randint(0, NUM_CLASSES, (batch_size, 256, 256))
        batch_targets = torch.randint(0, NUM_CLASSES, (batch_size, 256, 256))
        tracker.update(batch_preds, batch_targets, loss=0.5)

    print(tracker.summary())

    # Test EarlyStopping
    print("\nTesting EarlyStopping...")
    es = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.7, 0.65, 0.64, 0.63, 0.62]
    for i, score in enumerate(scores):
        stop = es(score)
        print(f"  Epoch {i+1}: score={score:.2f}, stop={stop}")

    # Test class weights
    print("\nTesting class weights...")
    weights = compute_class_weights(targets)
    print(f"  Class weights: {weights.tolist()}")

    print("\n[OK] All metrics working!")
