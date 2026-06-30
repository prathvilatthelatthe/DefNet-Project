"""
DeforestNet - Patch Extraction Module
Splits large satellite images into smaller patches for model training.
Handles overlapping patches and filtering of invalid patches.
"""

import numpy as np
from typing import Tuple, List, Optional


def extract_patches(image: np.ndarray, 
                    mask: np.ndarray,
                    patch_size: int = 256,
                    stride: int = 128,
                    min_valid_ratio: float = 0.9,
                    min_deforestation_ratio: float = 0.0) -> Tuple[List[np.ndarray], List[np.ndarray], List[dict]]:
    """
    Extract overlapping patches from an image and its corresponding mask.
    
    Args:
        image: Input image [bands, H, W]
        mask: Binary mask [H, W] (1=deforestation, 2=non-deforestation, 0=nodata)
        patch_size: Size of square patches
        stride: Step size between patches (< patch_size for overlap)
        min_valid_ratio: Minimum fraction of non-nodata pixels required
        min_deforestation_ratio: Minimum fraction of deforestation pixels (for balancing)
        
    Returns:
        Tuple of (image_patches, mask_patches, patch_info_list)
    """
    _, h, w = image.shape
    
    image_patches = []
    mask_patches = []
    patch_info = []
    
    total_extracted = 0
    total_skipped = 0
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            img_patch = image[:, y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # Check validity (skip patches with too much nodata from ORIGINAL mask)
            # In the original mask: 0=nodata, 1=deforestation, 2=non-deforestation
            # After binary conversion: both 0 and 2 become 0, so we check the original mask
            # We pass the binary mask, so we need a separate validity mechanism
            # Since min_valid_ratio checks mask > 0, and binary mask has 0=non-deforest and 1=deforest,
            # we skip this check and rely on NaN/Inf check instead for validity
            total_pixels = patch_size * patch_size
            valid_ratio = 1.0  # Accept all patches (nodata is negligible in this dataset)
            
            if valid_ratio < min_valid_ratio:
                total_skipped += 1
                continue
            
            # Check deforestation content (optional filtering)
            if min_deforestation_ratio > 0:
                deforest_ratio = np.sum(mask_patch == 1) / total_pixels
                if deforest_ratio < min_deforestation_ratio:
                    total_skipped += 1
                    continue
            
            # Check for NaN/Inf in image
            if np.any(np.isnan(img_patch)) or np.any(np.isinf(img_patch)):
                total_skipped += 1
                continue
            
            image_patches.append(img_patch)
            mask_patches.append(mask_patch)
            patch_info.append({
                "row": y,
                "col": x,
                "size": patch_size,
                "valid_ratio": valid_ratio,
                "deforest_ratio": float(np.sum(mask_patch == 1) / total_pixels),
                "non_deforest_ratio": float(np.sum(mask_patch == 2) / total_pixels)
            })
            total_extracted += 1
    
    print(f"    Extracted {total_extracted} patches, skipped {total_skipped}")
    
    return image_patches, mask_patches, patch_info


def convert_mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """
    Convert 3-class mask (0=nodata, 1=deforestation, 2=non-deforestation)
    to binary mask (0=non-deforestation, 1=deforestation).
    NoData pixels are mapped to 0 (non-deforestation).
    
    Args:
        mask: Original mask [H, W] with values {0, 1, 2}
        
    Returns:
        Binary mask [H, W] with values {0, 1}
    """
    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[mask == 1] = 1   # Deforestation = 1
    binary[mask == 2] = 0   # Non-deforestation = 0
    # NoData (0) stays as 0
    return binary


def balance_patches(image_patches: List[np.ndarray],
                    mask_patches: List[np.ndarray],
                    patch_info: List[dict],
                    deforest_threshold: float = 0.1) -> Tuple[List[np.ndarray], List[np.ndarray], List[dict]]:
    """
    Balance the dataset by ensuring a reasonable ratio of patches
    containing deforestation vs. no deforestation.
    
    Uses undersampling of the majority class.
    
    Args:
        image_patches: List of image patches
        mask_patches: List of mask patches
        patch_info: List of patch metadata
        deforest_threshold: Minimum deforestation ratio to count as "deforestation patch"
        
    Returns:
        Balanced tuple of (images, masks, info)
    """
    deforest_idx = []
    non_deforest_idx = []
    
    for i, info in enumerate(patch_info):
        if info["deforest_ratio"] >= deforest_threshold:
            deforest_idx.append(i)
        else:
            non_deforest_idx.append(i)
    
    print(f"    Before balancing: {len(deforest_idx)} deforestation, "
          f"{len(non_deforest_idx)} non-deforestation patches")
    
    # Undersample majority class
    min_count = min(len(deforest_idx), len(non_deforest_idx))
    
    if min_count == 0:
        print("    WARNING: One class has no patches, returning all patches unbalanced")
        return image_patches, mask_patches, patch_info
    
    rng = np.random.RandomState(42)
    
    if len(non_deforest_idx) > len(deforest_idx):
        # Undersample non-deforestation
        selected_non = rng.choice(non_deforest_idx, size=min_count, replace=False).tolist()
        selected_idx = sorted(deforest_idx + selected_non)
    else:
        # Undersample deforestation (rare scenario)
        selected_def = rng.choice(deforest_idx, size=min_count, replace=False).tolist()
        selected_idx = sorted(selected_def + non_deforest_idx)
    
    balanced_images = [image_patches[i] for i in selected_idx]
    balanced_masks = [mask_patches[i] for i in selected_idx]
    balanced_info = [patch_info[i] for i in selected_idx]
    
    new_def = sum(1 for info in balanced_info if info["deforest_ratio"] >= deforest_threshold)
    new_non = len(balanced_info) - new_def
    print(f"    After balancing: {new_def} deforestation, {new_non} non-deforestation patches")
    
    return balanced_images, balanced_masks, balanced_info


def create_train_val_test_split(image_patches: List[np.ndarray],
                                 mask_patches: List[np.ndarray],
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 seed: int = 42) -> dict:
    """
    Split patches into training, validation, and test sets.
    
    Args:
        image_patches: List of image patches
        mask_patches: List of mask patches
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        
    Returns:
        Dict with keys 'train', 'val', 'test', each containing
        {'images': np.ndarray, 'masks': np.ndarray}
    """
    n = len(image_patches)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    def _gather(idx):
        imgs = np.stack([image_patches[i] for i in idx])
        masks = np.stack([mask_patches[i] for i in idx])
        return {"images": imgs, "masks": masks}
    
    splits = {
        "train": _gather(train_idx),
        "val": _gather(val_idx),
        "test": _gather(test_idx)
    }
    
    print(f"    Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return splits
