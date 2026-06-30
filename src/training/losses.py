"""
DeforestNet - Loss Functions for Multi-Class Segmentation
Loss functions for 6-class deforestation cause classification.

All losses expect:
  - logits:  [B, C, H, W] raw model output (before softmax), C=6 classes
  - targets: [B, H, W] integer class labels (0-5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import NUM_CLASSES, CLASS_NAMES


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.

    Measures overlap between predicted and ground truth masks.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice

    Naturally handles class imbalance since it measures overlap
    rather than per-pixel accuracy.
    """

    def __init__(self, smooth: float = 1.0, class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero.
            class_weights: Optional weights per class for weighted average.
        """
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.num_classes = NUM_CLASSES

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, H, W] raw model output
            targets: [B, H, W] ground truth labels (0 to C-1)
        Returns:
            Scalar dice loss
        """
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: [B, H, W] -> [B, C, H, W]
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        # Compute per-class dice
        dice_per_class = torch.zeros(self.num_classes, device=logits.device)

        for c in range(self.num_classes):
            pred_c = probs[:, c].contiguous().view(-1)
            true_c = targets_onehot[:, c].contiguous().view(-1)

            intersection = (pred_c * true_c).sum()
            union = pred_c.sum() + true_c.sum()

            dice_per_class[c] = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Weighted average
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            dice_score = (dice_per_class * weights).sum() / weights.sum()
        else:
            dice_score = dice_per_class.mean()

        return 1.0 - dice_score


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for multi-class segmentation.

    Down-weights easy examples and focuses on hard misclassified pixels.
    FL = -alpha * (1 - p_t)^gamma * log(p_t)

    Especially useful for deforestation detection where some classes
    (like mining, infrastructure) are rarer and harder to detect.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Per-class weights as a tensor [w_class0, ..., w_class5].
                   If None, no class weighting is applied.
            gamma: Focusing parameter. Higher = more focus on hard examples.
                   gamma=0 reduces to standard CE. Typical: 1.0 - 3.0.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        self.num_classes = NUM_CLASSES

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, H, W] raw model output
            targets: [B, H, W] ground truth labels (0 to C-1)
        Returns:
            Scalar focal loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B, H, W]

        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        # Gather the probability of the true class for each pixel
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, H, W]

        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]  # [B, H, W]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted sum of CrossEntropy + Dice + optional Focal.

    This is the recommended loss for DeforestNet. Each component contributes:
      - CE: Stable pixel-wise gradients, good for overall learning
      - Dice: Overlap-based, counteracts class imbalance
      - Focal: Hard example mining for boundary pixels and rare classes
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.3,
        focal_weight: float = 0.2,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            class_weights: Per-class weights [w_class0, ..., w_class5] for CE/Focal.
            ce_weight: Weight for cross-entropy component.
            dice_weight: Weight for dice loss component.
            focal_weight: Weight for focal loss component (0 = disabled).
            focal_gamma: Gamma for focal loss.
            dice_smooth: Smoothing for dice loss.
            label_smoothing: Label smoothing for CE (0-1).
        """
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # Convert class weights to tensor
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weight_tensor = None

        # Cross-Entropy with optional class weights
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=label_smoothing
        )

        # Dice Loss
        self.dice_loss = DiceLoss(smooth=dice_smooth, class_weights=weight_tensor)

        # Focal Loss
        if focal_weight > 0:
            self.focal_loss = FocalLoss(alpha=weight_tensor, gamma=focal_gamma)
        else:
            self.focal_loss = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits:  [B, C, H, W] raw model output
            targets: [B, H, W] ground truth labels (0 to C-1)
        Returns:
            total_loss: Scalar combined loss
            loss_dict: Dictionary with individual loss components
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total = self.ce_weight * ce + self.dice_weight * dice

        loss_dict = {
            'ce': ce.item(),
            'dice': dice.item()
        }

        if self.focal_loss is not None:
            focal = self.focal_loss(logits, targets)
            total = total + self.focal_weight * focal
            loss_dict['focal'] = focal.item()

        loss_dict['total'] = total.item()

        return total, loss_dict


class IoULoss(nn.Module):
    """
    IoU (Jaccard) Loss for segmentation.

    IoU = Intersection / Union
    Loss = 1 - IoU
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.num_classes = NUM_CLASSES

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        iou_per_class = torch.zeros(self.num_classes, device=logits.device)

        for c in range(self.num_classes):
            pred_c = probs[:, c].contiguous().view(-1)
            true_c = targets_onehot[:, c].contiguous().view(-1)

            intersection = (pred_c * true_c).sum()
            union = pred_c.sum() + true_c.sum() - intersection

            iou_per_class[c] = (intersection + self.smooth) / (union + self.smooth)

        return 1.0 - iou_per_class.mean()


def build_loss(
    loss_type: str = 'combined',
    class_weights: Optional[List[float]] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: 'ce', 'dice', 'focal', 'iou', or 'combined'
        class_weights: Per-class weights for imbalance handling
        **kwargs: Additional arguments passed to the loss constructor

    Returns:
        Loss module instance
    """
    if loss_type == 'ce':
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        return nn.CrossEntropyLoss(weight=weight)

    elif loss_type == 'dice':
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        return DiceLoss(class_weights=weight, **kwargs)

    elif loss_type == 'focal':
        alpha = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        return FocalLoss(alpha=alpha, **kwargs)

    elif loss_type == 'iou':
        return IoULoss(**kwargs)

    elif loss_type == 'combined':
        return CombinedLoss(class_weights=class_weights, **kwargs)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")

    # Create dummy data
    batch_size = 4
    logits = torch.randn(batch_size, NUM_CLASSES, 256, 256)
    targets = torch.randint(0, NUM_CLASSES, (batch_size, 256, 256))

    # Test each loss
    print(f"\nInput shapes: logits={logits.shape}, targets={targets.shape}")

    # Dice Loss
    dice_loss = DiceLoss()
    loss = dice_loss(logits, targets)
    print(f"  Dice Loss: {loss.item():.4f}")

    # Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"  Focal Loss: {loss.item():.4f}")

    # IoU Loss
    iou_loss = IoULoss()
    loss = iou_loss(logits, targets)
    print(f"  IoU Loss: {loss.item():.4f}")

    # Combined Loss
    combined_loss = CombinedLoss()
    loss, loss_dict = combined_loss(logits, targets)
    print(f"  Combined Loss: {loss.item():.4f}")
    print(f"    Components: {loss_dict}")

    # With class weights
    weights = [0.1, 1.0, 1.2, 1.5, 1.0, 2.0]
    weighted_loss = CombinedLoss(class_weights=weights)
    loss, loss_dict = weighted_loss(logits, targets)
    print(f"  Weighted Combined: {loss.item():.4f}")

    print("\n[OK] All loss functions working!")
