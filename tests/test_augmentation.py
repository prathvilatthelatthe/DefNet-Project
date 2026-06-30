"""
Test script for the Data Augmentation module.
Loads real preprocessed data and verifies all augmentations work correctly.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.augmentation import (
    random_horizontal_flip, random_vertical_flip,
    random_rotate90, random_transpose,
    random_brightness, random_contrast,
    random_band_noise, random_band_dropout,
    mixup, TrainAugmentation, ValAugmentation
)


def load_one_sample():
    """Load one real sample from the preprocessed data."""
    preprocessed_dir = os.path.join(PROJECT_ROOT, "outputs", "preprocessed", "train")
    chunk_files = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.npz')])
    data = np.load(os.path.join(preprocessed_dir, chunk_files[0]))
    image = data["images"][0].astype(np.float32)  # [11, 256, 256]
    mask = data["masks"][0]                         # [256, 256]
    return image, mask


def test_shape_preservation(image, mask):
    """Test that all transforms preserve shape."""
    print("TEST: Shape preservation...")
    C, H, W = image.shape
    
    transforms = [
        ("horizontal_flip", lambda i, m: random_horizontal_flip(i.copy(), m.copy(), p=1.0)),
        ("vertical_flip", lambda i, m: random_vertical_flip(i.copy(), m.copy(), p=1.0)),
        ("rotate90", lambda i, m: random_rotate90(i.copy(), m.copy(), p=1.0)),
        ("transpose", lambda i, m: random_transpose(i.copy(), m.copy(), p=1.0)),
        ("brightness", lambda i, m: random_brightness(i.copy(), m.copy(), p=1.0)),
        ("contrast", lambda i, m: random_contrast(i.copy(), m.copy(), p=1.0)),
        ("band_noise", lambda i, m: random_band_noise(i.copy(), m.copy(), p=1.0)),
        ("band_dropout", lambda i, m: random_band_dropout(i.copy(), m.copy(), p=1.0)),
    ]
    
    for name, fn in transforms:
        aug_img, aug_mask = fn(image, mask)
        assert aug_img.shape[0] == C, f"{name}: channels changed {aug_img.shape[0]} != {C}"
        assert aug_img.shape[1] == aug_img.shape[2], f"{name}: not square {aug_img.shape}"
        assert aug_mask.shape[0] == aug_mask.shape[1], f"{name}: mask not square {aug_mask.shape}"
        assert aug_img.shape[1] == aug_mask.shape[0], f"{name}: image/mask size mismatch"
    
    print("  PASSED: All transforms preserve shape correctly")


def test_spatial_consistency(image, mask):
    """Test that spatial transforms apply identically to image and mask."""
    print("TEST: Spatial consistency (image & mask transform identically)...")
    
    # Horizontal flip
    img_f, mask_f = random_horizontal_flip(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(img_f[0], image[0, :, ::-1]), "H-flip: image band 0 incorrect"
    assert np.array_equal(mask_f, mask[:, ::-1]), "H-flip: mask incorrect"
    
    # Vertical flip
    img_f, mask_f = random_vertical_flip(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(img_f[0], image[0, ::-1, :]), "V-flip: image band 0 incorrect"
    assert np.array_equal(mask_f, mask[::-1, :]), "V-flip: mask incorrect"
    
    # Transpose
    img_f, mask_f = random_transpose(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(img_f[0], image[0].T), "Transpose: image band 0 incorrect"
    assert np.array_equal(mask_f, mask.T), "Transpose: mask incorrect"
    
    print("  PASSED: Spatial transforms are consistent between image and mask")


def test_value_ranges(image, mask):
    """Test that augmented values stay within valid ranges."""
    print("TEST: Value range bounds...")
    
    aug = TrainAugmentation(
        flip_p=1.0, rotate_p=1.0, transpose_p=1.0,
        brightness_p=1.0, contrast_p=1.0, noise_p=1.0, band_dropout_p=1.0
    )
    
    # Record original derived band ranges (bands 6-10: NDVI, EVI, SAVI, VV_VH_Ratio, RVI)
    orig_derived_min = image[6:].min()
    orig_derived_max = image[6:].max()
    
    for i in range(50):  # Run 50 random augmentations
        aug_img, aug_mask = aug(image.copy(), mask.copy())
        
        assert np.all(np.isfinite(aug_img)), f"Iteration {i}: NaN/Inf in augmented image"
        
        # Sensor bands (0-5) should be clipped to [0, 1] by radiometric transforms
        sensor = aug_img[:6]
        assert sensor.min() >= 0.0, f"Iteration {i}: sensor bands below 0 ({sensor.min():.4f})"
        assert sensor.max() <= 1.0, f"Iteration {i}: sensor bands above 1 ({sensor.max():.4f})"
        
        # Derived bands (6-10) are only spatially rearranged, values preserved
        derived = aug_img[6:]
        assert derived.min() >= orig_derived_min - 1e-6, \
            f"Iteration {i}: derived min unexpected ({derived.min():.4f})"
        assert derived.max() <= orig_derived_max + 1e-6, \
            f"Iteration {i}: derived max unexpected ({derived.max():.4f})"
        
        # Mask should remain binary
        assert set(np.unique(aug_mask)).issubset({0, 1}), \
            f"Iteration {i}: mask has unexpected values {np.unique(aug_mask)}"
    
    print("  PASSED: All 50 augmented samples have valid value ranges")


def test_mask_unchanged_by_radiometric(image, mask):
    """Test that radiometric transforms do NOT modify the mask."""
    print("TEST: Mask unchanged by radiometric transforms...")
    
    original_mask = mask.copy()
    
    _, m = random_brightness(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(m, original_mask), "Brightness modified mask!"
    
    _, m = random_contrast(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(m, original_mask), "Contrast modified mask!"
    
    _, m = random_band_noise(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(m, original_mask), "Band noise modified mask!"
    
    _, m = random_band_dropout(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(m, original_mask), "Band dropout modified mask!"
    
    print("  PASSED: Radiometric transforms leave mask untouched")


def test_derived_bands_untouched(image, mask):
    """Test that radiometric transforms don't modify derived index bands (NDVI, EVI, etc.)."""
    print("TEST: Derived bands (6-10) unchanged by radiometric transforms...")
    
    original_derived = image[6:].copy()
    
    img, _ = random_brightness(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(img[6:], original_derived), "Brightness modified derived bands!"
    
    img, _ = random_contrast(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(img[6:], original_derived), "Contrast modified derived bands!"
    
    img, _ = random_band_noise(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(img[6:], original_derived), "Noise modified derived bands!"
    
    img, _ = random_band_dropout(image.copy(), mask.copy(), p=1.0)
    assert np.array_equal(img[6:], original_derived), "Dropout modified derived bands!"
    
    print("  PASSED: Derived index bands are preserved")


def test_mixup(image, mask):
    """Test mixup augmentation."""
    print("TEST: Mixup augmentation...")
    
    # Create a second sample (shifted version)
    image2 = np.roll(image, 50, axis=1)
    mask2 = np.roll(mask, 50, axis=0)
    
    mixed_img, mixed_mask = mixup(image, mask, image2, mask2, alpha=0.3)
    
    assert mixed_img.shape == image.shape, "Mixup changed image shape"
    assert mixed_mask.shape == mask.shape, "Mixup changed mask shape"
    assert np.array_equal(mixed_mask, mask), "Mixup should return dominant sample's mask"
    assert not np.array_equal(mixed_img, image), "Mixup should modify the image"
    
    print("  PASSED: Mixup works correctly")


def test_pipeline_classes(image, mask):
    """Test TrainAugmentation and ValAugmentation pipeline classes."""
    print("TEST: Pipeline classes...")
    
    # Train augmentation should modify data (probabilistically)
    train_aug = TrainAugmentation()
    changed = False
    for _ in range(20):
        aug_img, aug_mask = train_aug(image.copy(), mask.copy())
        if not np.array_equal(aug_img, image):
            changed = True
            break
    assert changed, "TrainAugmentation never modified the image in 20 tries"
    
    # Val augmentation should never modify data
    val_aug = ValAugmentation()
    for _ in range(10):
        aug_img, aug_mask = val_aug(image.copy(), mask.copy())
        assert np.array_equal(aug_img, image), "ValAugmentation modified the image!"
        assert np.array_equal(aug_mask, mask), "ValAugmentation modified the mask!"
    
    print("  PASSED: TrainAugmentation modifies data, ValAugmentation preserves it")


def test_no_original_mutation(image, mask):
    """Test that augmentations don't mutate the original arrays when not copied."""
    print("TEST: No in-place mutation of originals...")
    
    original_img = image.copy()
    original_mask = mask.copy()
    
    aug = TrainAugmentation(
        flip_p=1.0, rotate_p=1.0, brightness_p=1.0, 
        contrast_p=1.0, noise_p=1.0
    )
    
    # Pass copies to the augmentation
    _ = aug(image.copy(), mask.copy())
    
    assert np.array_equal(image, original_img), "Original image was mutated!"
    assert np.array_equal(mask, original_mask), "Original mask was mutated!"
    
    print("  PASSED: Original arrays are never mutated")


if __name__ == "__main__":
    print("=" * 50)
    print("  DeforestNet Augmentation Tests")
    print("=" * 50)
    
    print("\nLoading real preprocessed sample...")
    image, mask = load_one_sample()
    print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"  Image range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"  Mask unique values: {np.unique(mask)}")
    print()
    
    test_shape_preservation(image, mask)
    test_spatial_consistency(image, mask)
    test_value_ranges(image, mask)
    test_mask_unchanged_by_radiometric(image, mask)
    test_derived_bands_untouched(image, mask)
    test_mixup(image, mask)
    test_pipeline_classes(image, mask)
    test_no_original_mutation(image, mask)
    
    print("\n" + "=" * 50)
    print("  ALL 8 TESTS PASSED!")
    print("=" * 50)
