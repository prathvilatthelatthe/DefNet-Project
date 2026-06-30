"""
DeforestNet - GeoTIFF Reader Module
Handles reading Sentinel-1, Sentinel-2, and mask GeoTIFF files.
Converts them to NumPy arrays while preserving metadata.
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Dict, List, Tuple, Optional


def read_geotiff(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Read a GeoTIFF file and return as NumPy array with metadata.
    
    Args:
        filepath: Path to the .tif file
        
    Returns:
        Tuple of (image_array [bands, H, W], metadata_dict)
    """
    with rasterio.open(filepath) as src:
        image = src.read()  # Shape: (bands, height, width)
        metadata = {
            "crs": str(src.crs),
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtypes": src.dtypes,
            "nodata": src.nodata,
            "filepath": filepath
        }
    return image, metadata


def read_sentinel1(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Read Sentinel-1 SAR GeoTIFF (2 bands: VV, VH).
    
    Args:
        filepath: Path to Sentinel-1 .tif file
        
    Returns:
        Tuple of (image [2, H, W] float32, metadata)
    """
    image, meta = read_geotiff(filepath)
    assert image.shape[0] == 2, f"Expected 2 bands (VV, VH), got {image.shape[0]}"
    meta["sensor"] = "Sentinel-1"
    meta["bands"] = ["VV", "VH"]
    return image.astype(np.float32), meta


def read_sentinel2(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Read Sentinel-2 optical GeoTIFF (4 bands: B2, B3, B4, B8).
    
    Args:
        filepath: Path to Sentinel-2 .tif file
        
    Returns:
        Tuple of (image [4, H, W] float32, metadata)
    """
    image, meta = read_geotiff(filepath)
    assert image.shape[0] == 4, f"Expected 4 bands (B2,B3,B4,B8), got {image.shape[0]}"
    meta["sensor"] = "Sentinel-2"
    meta["bands"] = ["B2_Blue", "B3_Green", "B4_Red", "B8_NIR"]
    return image.astype(np.float32), meta


def read_mask(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Read training mask GeoTIFF (1 band: class labels).
    Classes: 0=NoData, 1=Deforestation, 2=Non-Deforestation
    
    Args:
        filepath: Path to mask .tif file
        
    Returns:
        Tuple of (mask [H, W] uint8, metadata)
    """
    image, meta = read_geotiff(filepath)
    assert image.shape[0] == 1, f"Expected 1 band, got {image.shape[0]}"
    meta["sensor"] = "Mask"
    meta["classes"] = {0: "NoData", 1: "Deforestation", 2: "Non-Deforestation"}
    return image[0].astype(np.uint8), meta  # Squeeze band dim


def load_patch_set(dataset_dir: str, patch_index: int) -> Dict[str, Tuple[np.ndarray, dict]]:
    """
    Load a matched set of patches (S1, S2, mask) for a given patch index
    from the cloud-free dataset.
    
    Args:
        dataset_dir: Root dataset directory
        patch_index: Patch number (0-15)
        
    Returns:
        Dict with keys 's1', 's2', 'mask', each containing (array, metadata)
    """
    filename = f"RASTER_{patch_index}.tif"
    
    s1_path = os.path.join(dataset_dir, "1_CLOUD_FREE_DATASET", "1_SENTINEL1", 
                           "IMAGE_16_GRID", filename)
    s2_path = os.path.join(dataset_dir, "1_CLOUD_FREE_DATASET", "2_SENTINEL2", 
                           "IMAGE_16_GRID", filename)
    mask_path = os.path.join(dataset_dir, "3_TRAINING_MASKS", "MASK_16_GRID", 
                             filename)
    
    result = {}
    if os.path.exists(s1_path):
        result["s1"] = read_sentinel1(s1_path)
    if os.path.exists(s2_path):
        result["s2"] = read_sentinel2(s2_path)
    if os.path.exists(mask_path):
        result["mask"] = read_mask(mask_path)
    
    return result


def load_all_patches(dataset_dir: str, num_patches: int = 16) -> List[Dict]:
    """
    Load all 16 patch sets from the dataset.
    
    Args:
        dataset_dir: Root dataset directory
        num_patches: Number of patches (default 16)
        
    Returns:
        List of patch dictionaries
    """
    patches = []
    for i in range(num_patches):
        patch = load_patch_set(dataset_dir, i)
        patch["index"] = i
        patches.append(patch)
        print(f"  Loaded patch {i}/15")
    return patches


def get_dataset_summary(dataset_dir: str) -> dict:
    """
    Get a summary of the dataset including file counts, sizes, and band info.
    """
    summary = {"total_files": 0, "datasets": {}}
    
    for category in ["1_CLOUD_FREE_DATASET", "2_CLOUDY_DATASET", "3_TRAINING_MASKS"]:
        cat_path = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_path):
            continue
            
        tif_files = []
        for root, dirs, files in os.walk(cat_path):
            for f in files:
                if f.lower().endswith('.tif'):
                    tif_files.append(os.path.join(root, f))
        
        summary["datasets"][category] = {
            "num_files": len(tif_files),
            "files": [os.path.relpath(f, dataset_dir) for f in tif_files]
        }
        summary["total_files"] += len(tif_files)
        
        # Read one file for metadata
        if tif_files:
            _, meta = read_geotiff(tif_files[0])
            summary["datasets"][category]["sample_meta"] = {
                "crs": meta["crs"],
                "size": f"{meta['width']}x{meta['height']}",
                "bands": meta["count"],
                "dtype": str(meta["dtypes"])
            }
    
    return summary
