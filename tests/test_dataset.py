"""
Test script for Step 4: PyTorch Dataset & DataLoader.
Validates data loading, shapes, types, augmentation integration, and iteration.
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from src.data.dataset import DeforestNetDataset, get_dataloaders
from src.data.augmentation import TrainAugmentation, ValAugmentation
from configs.config import PREPROCESSED_DIR, BATCH_SIZE, IN_CHANNELS, PATCH_SIZE


def test_dataset_loading():
    """Test that datasets load all chunks correctly."""
    print("TEST 1: Dataset loading...")
    
    train_ds = DeforestNetDataset(
        os.path.join(PREPROCESSED_DIR, "train"),
        augmentation=TrainAugmentation()
    )
    val_ds = DeforestNetDataset(
        os.path.join(PREPROCESSED_DIR, "val"),
        augmentation=ValAugmentation()
    )
    test_ds = DeforestNetDataset(
        os.path.join(PREPROCESSED_DIR, "test"),
        augmentation=ValAugmentation()
    )
    
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    
    assert len(train_ds) > 0, "Train dataset is empty"
    assert len(val_ds) > 0, "Val dataset is empty"
    assert len(test_ds) > 0, "Test dataset is empty"
    assert len(train_ds) > len(val_ds), "Train should be larger than val"
    
    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"  Total: {total} samples")
    print("  PASSED")
    return train_ds, val_ds, test_ds


def test_single_sample(train_ds, val_ds):
    """Test that individual samples have correct shapes and types."""
    print("\nTEST 2: Single sample shapes & types...")
    
    image, mask = train_ds[0]
    
    assert isinstance(image, torch.Tensor), f"Image is {type(image)}, expected torch.Tensor"
    assert isinstance(mask, torch.Tensor), f"Mask is {type(mask)}, expected torch.Tensor"
    assert image.dtype == torch.float32, f"Image dtype is {image.dtype}, expected float32"
    assert mask.dtype == torch.int64, f"Mask dtype is {mask.dtype}, expected int64 (long)"
    assert image.shape == (IN_CHANNELS, PATCH_SIZE, PATCH_SIZE), \
        f"Image shape {image.shape}, expected ({IN_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE})"
    assert mask.shape == (PATCH_SIZE, PATCH_SIZE), \
        f"Mask shape {mask.shape}, expected ({PATCH_SIZE}, {PATCH_SIZE})"
    
    print(f"  Image: shape={image.shape}, dtype={image.dtype}")
    print(f"  Mask:  shape={mask.shape}, dtype={mask.dtype}")
    
    # Check mask is binary
    unique_vals = torch.unique(mask)
    assert all(v in [0, 1] for v in unique_vals), f"Mask has unexpected values: {unique_vals}"
    print(f"  Mask unique values: {unique_vals.tolist()}")
    
    # Check no NaN/Inf
    assert torch.isfinite(image).all(), "Image contains NaN or Inf!"
    print("  No NaN/Inf in image")
    
    # Val sample should also work
    v_img, v_mask = val_ds[0]
    assert v_img.shape == (IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    assert v_mask.shape == (PATCH_SIZE, PATCH_SIZE)
    
    print("  PASSED")


def test_augmentation_effect(train_ds):
    """Test that train augmentation produces different outputs across calls."""
    print("\nTEST 3: Augmentation produces variation...")
    
    results = []
    for _ in range(5):
        img, _ = train_ds[0]
        results.append(img)
    
    # At least one pair should differ (augmentation is stochastic)
    any_different = False
    for i in range(1, len(results)):
        if not torch.equal(results[0], results[i]):
            any_different = True
            break
    
    assert any_different, "TrainAugmentation never produced a different result in 5 tries"
    print("  PASSED: Multiple calls to same index produce different augmented outputs")


def test_val_no_augmentation(val_ds):
    """Test that val augmentation is deterministic (no changes)."""
    print("\nTEST 4: Val augmentation is deterministic...")
    
    img1, mask1 = val_ds[0]
    img2, mask2 = val_ds[0]
    
    assert torch.equal(img1, img2), "Val images differ between calls"
    assert torch.equal(mask1, mask2), "Val masks differ between calls"
    print("  PASSED: Val dataset returns identical results each time")


def test_dataloader_batching():
    """Test DataLoader creation and batch iteration."""
    print("\nTEST 5: DataLoader batching...")
    
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=BATCH_SIZE)
    
    assert "train" in loaders, "Missing 'train' loader"
    assert "val" in loaders, "Missing 'val' loader"
    assert "test" in loaders, "Missing 'test' loader"
    
    # Iterate one batch from each loader
    for split_name, loader in loaders.items():
        batch_images, batch_masks = next(iter(loader))
        
        if split_name == "train":
            assert batch_images.shape[0] == BATCH_SIZE, \
                f"Train batch size {batch_images.shape[0]} != {BATCH_SIZE}"
        
        assert batch_images.shape[1:] == (IN_CHANNELS, PATCH_SIZE, PATCH_SIZE), \
            f"{split_name}: batch image shape {batch_images.shape}"
        assert batch_masks.shape[1:] == (PATCH_SIZE, PATCH_SIZE), \
            f"{split_name}: batch mask shape {batch_masks.shape}"
        assert batch_images.dtype == torch.float32
        assert batch_masks.dtype == torch.int64
        
        print(f"  {split_name}: images={batch_images.shape}, masks={batch_masks.shape}")
    
    print("  PASSED")


def test_full_epoch_iteration():
    """Test that we can iterate through an entire epoch without errors."""
    print("\nTEST 6: Full epoch iteration...")
    
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=BATCH_SIZE)
    
    for split_name, loader in loaders.items():
        n_batches = 0
        n_samples = 0
        t0 = time.time()
        
        for images, masks in loader:
            n_batches += 1
            n_samples += images.shape[0]
            
            # Verify every batch
            assert torch.isfinite(images).all(), f"NaN/Inf in {split_name} batch {n_batches}"
            assert images.dtype == torch.float32
            assert masks.dtype == torch.int64
        
        elapsed = time.time() - t0
        print(f"  {split_name}: {n_batches} batches, {n_samples} samples, {elapsed:.2f}s")
    
    print("  PASSED: All splits iterated without errors")


def test_class_distribution():
    """Check the class distribution in the training set."""
    print("\nTEST 7: Class distribution analysis...")
    
    train_ds = DeforestNetDataset(
        os.path.join(PREPROCESSED_DIR, "train"),
        augmentation=ValAugmentation()  # No augmentation for stats
    )
    
    deforest_pixels = 0
    total_pixels = 0
    
    for i in range(len(train_ds)):
        _, mask = train_ds[i]
        deforest_pixels += mask.sum().item()
        total_pixels += mask.numel()
    
    deforest_ratio = deforest_pixels / total_pixels
    non_deforest_ratio = 1.0 - deforest_ratio
    
    print(f"  Deforestation:     {deforest_ratio:.2%}")
    print(f"  Non-deforestation: {non_deforest_ratio:.2%}")
    
    # Calculate class weights (inverse frequency) for loss function
    weight_deforest = 1.0 / (2.0 * deforest_ratio) if deforest_ratio > 0 else 1.0
    weight_non_deforest = 1.0 / (2.0 * non_deforest_ratio) if non_deforest_ratio > 0 else 1.0
    print(f"  Suggested class weights: [non_deforest={weight_non_deforest:.4f}, deforest={weight_deforest:.4f}]")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 55)
    print("  DeforestNet — Step 4: Dataset & DataLoader Tests")
    print("=" * 55)
    
    train_ds, val_ds, test_ds = test_dataset_loading()
    test_single_sample(train_ds, val_ds)
    test_augmentation_effect(train_ds)
    test_val_no_augmentation(val_ds)
    test_dataloader_batching()
    test_full_epoch_iteration()
    test_class_distribution()
    
    print("\n" + "=" * 55)
    print("  ALL 7 TESTS PASSED!")
    print("=" * 55)
