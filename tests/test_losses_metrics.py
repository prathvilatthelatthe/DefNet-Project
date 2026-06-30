"""
Test script for Step 6: Loss Functions & Evaluation Metrics.
Tests all loss functions, all metrics, and runs repeated training loops
to ensure everything works correctly end-to-end.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np

from src.training.losses import DiceLoss, FocalLoss, CombinedLoss, build_loss
from src.training.metrics import (
    compute_confusion_matrix, iou_from_cm, dice_from_cm,
    precision_from_cm, recall_from_cm, f1_from_cm,
    overall_accuracy, MetricTracker
)
from src.models.unet import build_model
from src.data.dataset import get_dataloaders
from configs.config import (
    IN_CHANNELS, NUM_CLASSES, PATCH_SIZE, PREPROCESSED_DIR,
    CLASS_WEIGHTS, DICE_LOSS_WEIGHT, CE_LOSS_WEIGHT,
    FOCAL_LOSS_WEIGHT, FOCAL_GAMMA, DICE_SMOOTH,
    CLASS_NAMES, LOSS_TYPE
)


# ============================================================
# LOSS FUNCTION TESTS
# ============================================================

def test_dice_loss():
    """Test DiceLoss with known inputs."""
    print("TEST 1: DiceLoss...")
    
    loss_fn = DiceLoss(smooth=1.0)
    
    # Perfect prediction → loss should be near 0
    logits_perfect = torch.zeros(2, 2, 4, 4)
    targets = torch.zeros(2, 4, 4, dtype=torch.long)
    targets[0, :2, :] = 1
    targets[1, 2:, :] = 1
    logits_perfect[:, 0] = -10.0  # Low prob for class 0 where target is 1
    logits_perfect[:, 1] = 10.0   # High prob for class 1 where target is 1
    logits_perfect[0, :, 2:, :] = logits_perfect[0, :, 2:, :].flip(0)
    logits_perfect[1, :, :2, :] = logits_perfect[1, :, :2, :].flip(0)
    
    loss_perfect = loss_fn(logits_perfect, targets)
    print(f"  Near-perfect prediction loss: {loss_perfect.item():.6f}")
    assert loss_perfect.item() < 0.05, f"Perfect prediction loss too high: {loss_perfect.item()}"
    
    # Random prediction → loss should be moderate
    logits_random = torch.randn(2, 2, 4, 4)
    loss_random = loss_fn(logits_random, targets)
    print(f"  Random prediction loss: {loss_random.item():.4f}")
    assert 0.0 < loss_random.item() < 1.0, f"Random loss out of range: {loss_random.item()}"
    
    # Loss should be differentiable
    logits_random.requires_grad = True
    loss = loss_fn(logits_random, targets)
    loss.backward()
    assert logits_random.grad is not None, "DiceLoss not differentiable"
    
    print("  PASSED")


def test_focal_loss():
    """Test FocalLoss with known inputs."""
    print("\nTEST 2: FocalLoss...")
    
    # Without class weights
    fl = FocalLoss(alpha=None, gamma=2.0)
    logits = torch.randn(2, 2, 4, 4)
    targets = torch.randint(0, 2, (2, 4, 4))
    
    loss = fl(logits, targets)
    print(f"  Focal (no alpha, gamma=2): {loss.item():.4f}")
    assert loss.item() > 0, "Focal loss should be positive"
    assert torch.isfinite(loss), "Focal loss is NaN/Inf"
    
    # With class weights
    fl_weighted = FocalLoss(alpha=CLASS_WEIGHTS, gamma=2.0)
    loss_w = fl_weighted(logits, targets)
    print(f"  Focal (weighted, gamma=2): {loss_w.item():.4f}")
    assert loss_w.item() > 0
    
    # Focal with gamma=0 should approximate CE
    fl_g0 = FocalLoss(alpha=None, gamma=0.0)
    ce = nn.CrossEntropyLoss()
    loss_g0 = fl_g0(logits, targets)
    loss_ce = ce(logits, targets)
    diff = abs(loss_g0.item() - loss_ce.item())
    print(f"  Focal(gamma=0)={loss_g0.item():.4f} vs CE={loss_ce.item():.4f} (diff={diff:.6f})")
    assert diff < 0.01, f"Focal(gamma=0) should match CE, diff={diff}"
    
    # Differentiable
    logits.requires_grad = True
    fl(logits, targets).backward()
    assert logits.grad is not None
    
    print("  PASSED")


def test_combined_loss():
    """Test CombinedLoss returns correct structure."""
    print("\nTEST 3: CombinedLoss...")
    
    loss_fn = CombinedLoss(
        class_weights=CLASS_WEIGHTS,
        dice_weight=DICE_LOSS_WEIGHT,
        ce_weight=CE_LOSS_WEIGHT,
        focal_weight=0.0,
        dice_smooth=DICE_SMOOTH,
    )
    
    logits = torch.randn(4, 2, 32, 32)
    targets = torch.randint(0, 2, (4, 32, 32))
    
    total_loss, loss_dict = loss_fn(logits, targets)
    
    print(f"  Total: {total_loss.item():.4f}")
    print(f"  Components: {loss_dict}")
    
    assert torch.isfinite(total_loss), "Combined loss is NaN/Inf"
    assert total_loss.item() > 0, "Combined loss should be positive"
    assert 'ce' in loss_dict, "Missing 'ce' in loss_dict"
    assert 'dice' in loss_dict, "Missing 'dice' in loss_dict"
    assert 'total' in loss_dict, "Missing 'total' in loss_dict"
    
    # Verify weighted sum
    expected_total = CE_LOSS_WEIGHT * loss_dict['ce'] + DICE_LOSS_WEIGHT * loss_dict['dice']
    assert abs(loss_dict['total'] - expected_total) < 1e-4, \
        f"Total {loss_dict['total']:.6f} != weighted sum {expected_total:.6f}"
    
    # With focal loss enabled
    loss_fn_focal = CombinedLoss(
        class_weights=CLASS_WEIGHTS,
        dice_weight=0.4, ce_weight=0.4, focal_weight=0.2, focal_gamma=2.0
    )
    total_f, dict_f = loss_fn_focal(logits, targets)
    assert 'focal' in dict_f, "Missing 'focal' when focal_weight > 0"
    print(f"  With focal: {dict_f}")
    
    # Differentiable
    logits.requires_grad = True
    total, _ = loss_fn(logits, targets)
    total.backward()
    assert logits.grad is not None
    
    print("  PASSED")


def test_build_loss_factory():
    """Test the build_loss factory function."""
    print("\nTEST 4: build_loss factory...")
    
    logits = torch.randn(2, 2, 16, 16)
    targets = torch.randint(0, 2, (2, 16, 16))
    
    for name in ['ce', 'dice', 'focal', 'combined']:
        loss_fn = build_loss(name, class_weights=CLASS_WEIGHTS)
        if name == 'combined':
            result, _ = loss_fn(logits, targets)
        else:
            result = loss_fn(logits, targets)
        assert torch.isfinite(result), f"{name} loss is NaN/Inf"
        assert result.item() > 0, f"{name} loss should be positive"
        print(f"  {name}: {result.item():.4f}")
    
    print("  PASSED")


# ============================================================
# METRICS TESTS
# ============================================================

def test_confusion_matrix():
    """Test confusion matrix computation with known values."""
    print("\nTEST 5: Confusion matrix...")
    
    # Known predictions
    preds   = torch.tensor([[[0, 0, 1, 1],
                             [0, 1, 1, 1],
                             [0, 0, 0, 1],
                             [0, 0, 1, 1]]])
    targets = torch.tensor([[[0, 0, 0, 1],
                             [0, 0, 1, 1],
                             [1, 0, 0, 1],
                             [0, 0, 1, 1]]])
    
    cm = compute_confusion_matrix(preds, targets, num_classes=2)
    print(f"  CM:\n    {cm[0].tolist()}\n    {cm[1].tolist()}")
    
    # Manually: true=0/pred=0=6, true=0/pred=1=2, true=1/pred=0=1, true=1/pred=1=7 -> wait let me recount
    # targets:  0 0 0 1 | 0 0 1 1 | 1 0 0 1 | 0 0 1 1  -> 8 zeros, 8 ones... wait
    # targets row0: 0,0,0,1 -> 3 zeros, 1 one
    # targets row1: 0,0,1,1 -> 2 zeros, 2 ones
    # targets row2: 1,0,0,1 -> 2 zeros, 2 ones  (wait: target has 1,0,0,1)
    # targets row3: 0,0,1,1 -> 2 zeros, 2 ones
    # Total: 9 zeros, 7 ones
    
    # preds row0: 0,0,1,1 | row1: 0,1,1,1 | row2: 0,0,0,1 | row3: 0,0,1,1
    
    # TP(class 0) = true=0 & pred=0: count positions
    # (0,0):t=0,p=0 ✓ (0,1):t=0,p=0 ✓ (0,2):t=0,p=1 ✗ (0,3):t=1,p=1
    # (1,0):t=0,p=0 ✓ (1,1):t=0,p=1 ✗ (1,2):t=1,p=1 (1,3):t=1,p=1
    # (2,0):t=1,p=0 (2,1):t=0,p=0 ✓ (2,2):t=0,p=0 ✓ (2,3):t=1,p=1
    # (3,0):t=0,p=0 ✓ (3,1):t=0,p=0 ✓ (3,2):t=1,p=1 (3,3):t=1,p=1
    # cm[0,0] = 7, cm[0,1] = 2, cm[1,0] = 1, cm[1,1] = 6
    
    assert cm[0, 0].item() == 7, f"TN: {cm[0,0].item()} != 7"
    assert cm[0, 1].item() == 2, f"FP: {cm[0,1].item()} != 2"
    assert cm[1, 0].item() == 1, f"FN: {cm[1,0].item()} != 1"
    assert cm[1, 1].item() == 6, f"TP: {cm[1,1].item()} != 6"
    
    total = cm.sum().item()
    assert total == 16, f"Total pixels: {total} != 16"
    
    print("  PASSED")


def test_metrics_from_cm():
    """Test IoU, Dice, Precision, Recall, F1 from a known confusion matrix."""
    print("\nTEST 6: Metrics from known CM...")
    
    # cm[true, pred]: TN=70, FP=10, FN=5, TP=15
    cm = torch.tensor([[70, 10],
                        [5,  15]], dtype=torch.long)
    
    iou = iou_from_cm(cm)
    dice = dice_from_cm(cm)
    prec = precision_from_cm(cm)
    rec = recall_from_cm(cm)
    f1 = f1_from_cm(cm)
    acc = overall_accuracy(cm)
    
    # Class 1 (deforestation): TP=15, FP=10, FN=5
    # IoU = 15/(15+10+5) = 0.5
    # Dice = 30/(30+10+5) = 30/45 = 0.6667
    # Precision = 15/(15+10) = 0.6
    # Recall = 15/(15+5) = 0.75
    # F1 = 2*0.6*0.75/(0.6+0.75) = 0.9/1.35 = 0.6667
    
    assert abs(iou[1].item() - 0.5) < 1e-4, f"IoU[1]: {iou[1].item()}"
    assert abs(dice[1].item() - 0.6667) < 1e-3, f"Dice[1]: {dice[1].item()}"
    assert abs(prec[1].item() - 0.6) < 1e-4, f"Prec[1]: {prec[1].item()}"
    assert abs(rec[1].item() - 0.75) < 1e-4, f"Recall[1]: {rec[1].item()}"
    assert abs(f1[1].item() - 0.6667) < 1e-3, f"F1[1]: {f1[1].item()}"
    assert abs(acc.item() - 0.85) < 1e-4, f"Accuracy: {acc.item()}"
    
    print(f"  Class 1: IoU={iou[1]:.4f} Dice={dice[1]:.4f} P={prec[1]:.4f} R={rec[1]:.4f} F1={f1[1]:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print("  PASSED")


def test_metric_tracker():
    """Test MetricTracker accumulation across multiple batches."""
    print("\nTEST 7: MetricTracker accumulation...")
    
    tracker = MetricTracker(num_classes=2, class_names=CLASS_NAMES)
    
    # Simulate multiple batches
    torch.manual_seed(42)
    for i in range(5):
        preds = torch.randint(0, 2, (4, 32, 32))
        targets = torch.randint(0, 2, (4, 32, 32))
        tracker.update(preds, targets)
    
    metrics = tracker.compute()
    
    assert 'accuracy' in metrics
    assert 'mean_iou' in metrics
    assert 'mean_dice' in metrics
    assert 'Deforest_iou' in metrics
    assert 'Non-Deforest_precision' in metrics
    assert 'Deforest_recall' in metrics
    
    # Random predictions on balanced data → accuracy ~ 0.5
    assert 0.3 < metrics['accuracy'] < 0.7, f"Random acc: {metrics['accuracy']}"
    
    print(f"  Accumulated over 5 batches (20 samples, 32x32):")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Mean Dice: {metrics['mean_dice']:.4f}")
    
    # Test reset
    tracker.reset()
    assert tracker.cm.sum().item() == 0, "Reset didn't clear CM"
    
    # Test summary string
    tracker.update(torch.randint(0, 2, (2, 16, 16)), torch.randint(0, 2, (2, 16, 16)))
    summary = tracker.summary()
    assert "Accuracy" in summary
    assert "Non-Deforest" in summary
    assert "Deforest" in summary
    print(f"  Summary:\n    " + summary.replace("\n", "\n    "))
    
    print("  PASSED")


def test_perfect_metrics():
    """Test that perfect predictions produce perfect metric scores."""
    print("\nTEST 8: Perfect prediction metrics...")
    
    tracker = MetricTracker(num_classes=2, class_names=CLASS_NAMES)
    
    targets = torch.randint(0, 2, (8, 32, 32))
    preds = targets.clone()  # Perfect match
    
    tracker.update(preds, targets)
    m = tracker.compute()
    
    assert abs(m['accuracy'] - 1.0) < 1e-6, f"Perfect acc: {m['accuracy']}"
    assert abs(m['mean_iou'] - 1.0) < 1e-6, f"Perfect mIoU: {m['mean_iou']}"
    assert abs(m['mean_dice'] - 1.0) < 1e-6, f"Perfect mDice: {m['mean_dice']}"
    assert abs(m['Deforest_precision'] - 1.0) < 1e-6
    assert abs(m['Deforest_recall'] - 1.0) < 1e-6
    
    print(f"  Accuracy={m['accuracy']:.4f}, mIoU={m['mean_iou']:.4f}, mDice={m['mean_dice']:.4f}")
    print("  PASSED")


# ============================================================
# END-TO-END MODEL TRAINING TESTS
# ============================================================

def test_training_with_combined_loss():
    """Test complete training loop with CombinedLoss and MetricTracker."""
    print("\nTEST 9: Training loop with CombinedLoss + MetricTracker...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.train()
    
    loss_fn = build_loss(LOSS_TYPE, class_weights=CLASS_WEIGHTS,
                         dice_weight=DICE_LOSS_WEIGHT, ce_weight=CE_LOSS_WEIGHT,
                         focal_weight=FOCAL_LOSS_WEIGHT)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tracker = MetricTracker(num_classes=NUM_CLASSES, class_names=CLASS_NAMES)
    
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=4)
    
    # Run 3 training steps
    losses = []
    for i, (images, masks) in enumerate(loaders['train']):
        if i >= 3:
            break
        
        logits = model(images)
        total_loss, loss_dict = loss_fn(logits, masks)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        preds = logits.argmax(dim=1)
        tracker.update(preds, masks)
        
        losses.append(loss_dict['total'])
        print(f"  Step {i+1}: loss={loss_dict['total']:.4f}  ce={loss_dict['ce']:.4f}  dice={loss_dict['dice']:.4f}")
    
    metrics = tracker.compute()
    print(f"  After 3 steps: Acc={metrics['accuracy']:.4f}, mIoU={metrics['mean_iou']:.4f}")
    
    assert all(np.isfinite(l) for l in losses), "NaN/Inf in loss"
    assert metrics['accuracy'] > 0, "Accuracy should be > 0"
    print("  PASSED")


def test_repeated_training_runs():
    """Run multiple independent training runs to verify stability."""
    print("\nTEST 10: Repeated training stability (3 runs x 5 steps)...")
    
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=4)
    # Grab a fixed batch for overfitting test
    fixed_images, fixed_masks = next(iter(loaders['train']))
    
    all_final_losses = []
    all_final_acc = []
    
    for run in range(3):
        torch.manual_seed(run * 100)
        
        model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
        model.train()
        
        loss_fn = CombinedLoss(
            class_weights=CLASS_WEIGHTS,
            dice_weight=DICE_LOSS_WEIGHT,
            ce_weight=CE_LOSS_WEIGHT,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        tracker = MetricTracker(num_classes=NUM_CLASSES, class_names=CLASS_NAMES)
        
        step_losses = []
        for step in range(5):
            logits = model(fixed_images)
            total_loss, loss_dict = loss_fn(logits, fixed_masks)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            step_losses.append(loss_dict['total'])
            
            if step == 4:
                preds = logits.detach().argmax(dim=1)
                tracker.update(preds, fixed_masks)
        
        metrics = tracker.compute()
        all_final_losses.append(step_losses[-1])
        all_final_acc.append(metrics['accuracy'])
        
        print(f"  Run {run+1}: loss {step_losses[0]:.4f}->{step_losses[-1]:.4f}  "
              f"acc={metrics['accuracy']:.4f}  "
              f"deforest_IoU={metrics['Deforest_iou']:.4f}")
        
        # Loss should decrease in each run
        assert step_losses[-1] < step_losses[0], \
            f"Run {run+1}: loss didn't decrease {step_losses[0]:.4f}->{step_losses[-1]:.4f}"
    
    # All runs should produce finite results
    assert all(np.isfinite(l) for l in all_final_losses), "NaN in final losses"
    assert all(a > 0 for a in all_final_acc), "Zero accuracy in some runs"
    
    print("  PASSED")


def test_val_evaluation():
    """Test full validation evaluation loop."""
    print("\nTEST 11: Full validation evaluation loop...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.eval()
    
    loss_fn = CombinedLoss(class_weights=CLASS_WEIGHTS,
                           dice_weight=DICE_LOSS_WEIGHT,
                           ce_weight=CE_LOSS_WEIGHT)
    
    tracker = MetricTracker(num_classes=NUM_CLASSES, class_names=CLASS_NAMES)
    
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=8)
    
    total_loss = 0.0
    n_batches = 0
    
    t0 = time.time()
    with torch.no_grad():
        for images, masks in loaders['val']:
            logits = model(images)
            loss, _ = loss_fn(logits, masks)
            total_loss += loss.item()
            n_batches += 1
            
            preds = logits.argmax(dim=1)
            tracker.update(preds, masks)
    
    elapsed = time.time() - t0
    avg_loss = total_loss / n_batches
    metrics = tracker.compute()
    
    print(f"  Val loss (avg): {avg_loss:.4f}")
    print(f"  Val time: {elapsed:.2f}s ({n_batches} batches)")
    print(f"  " + tracker.summary().replace("\n", "\n  "))
    
    assert np.isfinite(avg_loss), "Val loss is NaN/Inf"
    assert metrics['accuracy'] > 0
    assert metrics['mean_iou'] >= 0
    print("  PASSED")


def test_all_loss_types_training():
    """Test that every loss type works in a training step."""
    print("\nTEST 12: All loss types in training step...")
    
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=4)
    images, masks = next(iter(loaders['train']))
    
    for loss_type in ['ce', 'dice', 'focal', 'combined']:
        model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        loss_fn = build_loss(loss_type, class_weights=CLASS_WEIGHTS)
        
        logits = model(images)
        if loss_type == 'combined':
            loss, _ = loss_fn(logits, masks)
        else:
            loss = loss_fn(logits, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert torch.isfinite(loss), f"{loss_type}: loss is NaN/Inf"
        print(f"  {loss_type:10s}: loss={loss.item():.4f} — train step OK")
    
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 62)
    print("  DeforestNet — Step 6: Loss Functions & Metrics Tests")
    print("=" * 62)
    
    # Loss function tests
    test_dice_loss()
    test_focal_loss()
    test_combined_loss()
    test_build_loss_factory()
    
    # Metric tests
    test_confusion_matrix()
    test_metrics_from_cm()
    test_metric_tracker()
    test_perfect_metrics()
    
    # End-to-end training tests
    test_training_with_combined_loss()
    test_repeated_training_runs()
    test_val_evaluation()
    test_all_loss_types_training()
    
    print("\n" + "=" * 62)
    print("  ALL 12 TESTS PASSED!")
    print("=" * 62)
