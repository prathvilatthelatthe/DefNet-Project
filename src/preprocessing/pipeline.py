"""
DeforestNet - Main Preprocessing Pipeline
Orchestrates the complete data preprocessing workflow:
  1. Read GeoTIFF images
  2. Noise removal (Lee filter for SAR, Gaussian for optical)
  3. Image normalization (percentile-based)
  4. Feature extraction (NDVI, EVI, SAVI, SAR indices)
  5. Convert to NumPy arrays
  6. Split into training patches
  7. Save preprocessed data
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.reader import (
    read_sentinel1, read_sentinel2, read_mask, 
    get_dataset_summary, load_patch_set
)
from src.preprocessing.noise_removal import (
    remove_noise_sentinel1, remove_noise_sentinel2
)
from src.preprocessing.normalization import (
    normalize_image, compute_global_stats
)
from src.preprocessing.feature_extraction import (
    compute_ndvi, compute_evi, compute_savi,
    compute_vv_vh_ratio, compute_rvi_sar, extract_all_features
)
from src.preprocessing.patch_extractor import (
    extract_patches, convert_mask_to_binary,
    balance_patches, create_train_val_test_split
)


def preprocess_pipeline(dataset_dir: str, output_dir: str, 
                         patch_size: int = 256, stride: int = 128):
    """
    Execute the complete preprocessing pipeline.
    
    Args:
        dataset_dir: Path to the raw dataset directory
        output_dir: Path to save preprocessed outputs
        patch_size: Size of training patches
        stride: Stride between patches
    """
    print("=" * 60)
    print("  DeforestNet - Data Preprocessing Pipeline")
    print("=" * 60)
    start_time = datetime.now()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ----------------------------------------------------------
    # STEP 1: Dataset Summary
    # ----------------------------------------------------------
    print("\n[STEP 1/7] Scanning dataset...")
    summary = get_dataset_summary(dataset_dir)
    print(f"  Total files: {summary['total_files']}")
    for name, info in summary["datasets"].items():
        print(f"  {name}: {info['num_files']} files | {info.get('sample_meta', {}).get('size', 'N/A')} | "
              f"{info.get('sample_meta', {}).get('bands', 'N/A')} bands")
    
    # ----------------------------------------------------------
    # STEP 2: Read all 16 grid patches
    # ----------------------------------------------------------
    print("\n[STEP 2/7] Reading GeoTIFF images...")
    
    all_s1_images = []
    all_s2_images = []
    all_masks = []
    
    for i in range(16):
        print(f"  Loading patch {i}/15...")
        patch_data = load_patch_set(dataset_dir, i)
        
        if "s1" in patch_data and "s2" in patch_data and "mask" in patch_data:
            all_s1_images.append(patch_data["s1"][0])
            all_s2_images.append(patch_data["s2"][0])
            all_masks.append(patch_data["mask"][0])
        else:
            print(f"  WARNING: Patch {i} missing data, skipping")
    
    print(f"  Loaded {len(all_s1_images)} complete patch sets")
    print(f"  S1 shape: {all_s1_images[0].shape} (bands, H, W)")
    print(f"  S2 shape: {all_s2_images[0].shape} (bands, H, W)")
    print(f"  Mask shape: {all_masks[0].shape} (H, W)")
    
    # ----------------------------------------------------------
    # STEP 3: Noise Removal
    # ----------------------------------------------------------
    print("\n[STEP 3/7] Removing noise...")
    
    denoised_s1 = []
    denoised_s2 = []
    
    for i in range(len(all_s1_images)):
        print(f"  Processing patch {i}/15...")
        print("  Sentinel-1 (SAR):")
        s1_clean = remove_noise_sentinel1(all_s1_images[i], lee_window=7, median_kernel=3)
        denoised_s1.append(s1_clean)
        
        print("  Sentinel-2 (Optical):")
        s2_clean = remove_noise_sentinel2(all_s2_images[i], gaussian_sigma=0.5)
        denoised_s2.append(s2_clean)
    
    print("  Noise removal complete!")
    
    # ----------------------------------------------------------
    # STEP 4: Image Normalization
    # ----------------------------------------------------------
    print("\n[STEP 4/7] Normalizing images...")
    
    # Compute global statistics across all patches
    print("  Computing global stats for Sentinel-1...")
    s1_global_stats = compute_global_stats(denoised_s1, method="percentile", 
                                            low_pct=2.0, high_pct=98.0)
    
    print("  Computing global stats for Sentinel-2...")
    s2_global_stats = compute_global_stats(denoised_s2, method="percentile",
                                            low_pct=2.0, high_pct=98.0)
    
    # Normalize each patch using global stats
    normed_s1 = []
    normed_s2 = []
    
    for i in range(len(denoised_s1)):
        s1_norm, _ = normalize_image(denoised_s1[i], method="percentile", stats=s1_global_stats)
        normed_s1.append(s1_norm)
        
        s2_norm, _ = normalize_image(denoised_s2[i], method="percentile", stats=s2_global_stats)
        normed_s2.append(s2_norm)
    
    print(f"  S1 normalized range: [{np.min(normed_s1[0]):.4f}, {np.max(normed_s1[0]):.4f}]")
    print(f"  S2 normalized range: [{np.min(normed_s2[0]):.4f}, {np.max(normed_s2[0]):.4f}]")
    
    # Save normalization stats
    norm_stats = {
        "s1_stats": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in s1_global_stats.items()},
        "s2_stats": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in s2_global_stats.items()}
    }
    stats_path = os.path.join(output_dir, "normalization_stats.json")
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved normalization stats to {stats_path}")
    
    # ----------------------------------------------------------
    # STEP 5: Feature Extraction
    # ----------------------------------------------------------
    print("\n[STEP 5/7] Extracting features...")
    
    all_features = []
    
    for i in range(len(normed_s1)):
        print(f"  Processing patch {i}/15...")
        features, band_names = extract_all_features(normed_s1[i], normed_s2[i])
        all_features.append(features)
    
    print(f"  Feature stack shape: {all_features[0].shape} ({len(band_names)} features)")
    print(f"  Features: {', '.join(band_names)}")
    
    # Save band names
    with open(os.path.join(output_dir, "band_names.json"), "w") as f:
        json.dump(band_names, f, indent=2)
    
    # ----------------------------------------------------------
    # STEP 6: Convert masks to binary & Extract patches
    # ----------------------------------------------------------
    print("\n[STEP 6/7] Extracting training patches...")
    
    all_image_patches = []
    all_mask_patches = []
    all_patch_info = []
    
    for i in range(len(all_features)):
        print(f"  Extracting from grid patch {i}/15...")
        binary_mask = convert_mask_to_binary(all_masks[i])
        
        img_patches, msk_patches, info = extract_patches(
            all_features[i], binary_mask,
            patch_size=patch_size, stride=stride,
            min_valid_ratio=0.9
        )
        
        all_image_patches.extend(img_patches)
        all_mask_patches.extend(msk_patches)
        all_patch_info.extend(info)
    
    print(f"\n  Total patches extracted: {len(all_image_patches)}")
    
    # Balance patches
    print("\n  Balancing dataset...")
    bal_images, bal_masks, bal_info = balance_patches(
        all_image_patches, all_mask_patches, all_patch_info,
        deforest_threshold=0.05
    )
    
    # ----------------------------------------------------------
    # STEP 7: Train/Val/Test Split & Save
    # ----------------------------------------------------------
    print("\n[STEP 7/7] Splitting and saving...")
    
    splits = create_train_val_test_split(
        bal_images, bal_masks,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    
    for split_name, split_data in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Save in compressed chunks using float16 to save disk space
        # float16 is sufficient for normalized [0,1] data
        images = split_data["images"].astype(np.float16)
        masks = split_data["masks"]
        chunk_size = 200  # Save 200 patches per file
        
        num_chunks = int(np.ceil(len(images) / chunk_size))
        for c in range(num_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, len(images))
            np.savez_compressed(
                os.path.join(split_dir, f"chunk{c}.npz"),
                images=images[start:end],
                masks=masks[start:end]
            )
        
        print(f"  {split_name}: images={split_data['images'].shape}, "
              f"masks={split_data['masks'].shape} ({num_chunks} compressed chunks)")
        print(f"    Saved to {split_dir}")
    
    # Save pipeline metadata
    metadata = {
        "timestamp": start_time.isoformat(),
        "dataset_dir": dataset_dir,
        "patch_size": patch_size,
        "stride": stride,
        "num_features": len(band_names),
        "feature_names": band_names,
        "total_patches_extracted": len(all_image_patches),
        "total_patches_after_balance": len(bal_images),
        "splits": {
            name: {"images_shape": list(data["images"].shape),
                   "masks_shape": list(data["masks"].shape)}
            for name, data in splits.items()
        },
        "deforestation_stats": {
            "mean_deforest_ratio": float(np.mean([info["deforest_ratio"] for info in bal_info])),
            "patches_with_deforestation": sum(1 for info in bal_info if info["deforest_ratio"] > 0.05)
        }
    }
    
    with open(os.path.join(output_dir, "pipeline_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    elapsed = datetime.now() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete! Elapsed: {elapsed}")
    print(f"  Output saved to: {output_dir}")
    print(f"{'=' * 60}")
    
    return splits, metadata


if __name__ == "__main__":
    # Default paths
    DATASET_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "dataset")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "preprocessed")
    
    splits, metadata = preprocess_pipeline(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        patch_size=256,
        stride=256  # Non-overlapping patches to keep dataset size manageable
    )
