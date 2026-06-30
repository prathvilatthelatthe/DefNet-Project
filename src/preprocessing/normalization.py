"""
DeforestNet - Image Normalization Module
Normalizes satellite imagery to consistent value ranges for model training.
"""

import numpy as np
from typing import Tuple, Optional, Dict


def normalize_minmax(image: np.ndarray, 
                     band_min: Optional[np.ndarray] = None,
                     band_max: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Min-Max normalization to [0, 1] range per band.
    
    Args:
        image: Input image [bands, H, W]
        band_min: Pre-computed band minimums (for applying saved stats)
        band_max: Pre-computed band maximums (for applying saved stats)
        
    Returns:
        Tuple of (normalized_image, stats_dict)
    """
    num_bands = image.shape[0]
    normalized = np.zeros_like(image, dtype=np.float32)
    
    if band_min is None:
        band_min = np.array([np.nanmin(image[b]) for b in range(num_bands)])
    if band_max is None:
        band_max = np.array([np.nanmax(image[b]) for b in range(num_bands)])
    
    for b in range(num_bands):
        denom = band_max[b] - band_min[b]
        if denom > 0:
            normalized[b] = (image[b] - band_min[b]) / denom
        else:
            normalized[b] = 0.0
    
    normalized = np.clip(normalized, 0.0, 1.0)
    
    stats = {
        "method": "minmax",
        "band_min": band_min.tolist(),
        "band_max": band_max.tolist()
    }
    return normalized, stats


def normalize_standardize(image: np.ndarray,
                          band_mean: Optional[np.ndarray] = None,
                          band_std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Z-score standardization per band (mean=0, std=1).
    
    Args:
        image: Input image [bands, H, W]
        band_mean: Pre-computed band means
        band_std: Pre-computed band standard deviations
        
    Returns:
        Tuple of (normalized_image, stats_dict)
    """
    num_bands = image.shape[0]
    normalized = np.zeros_like(image, dtype=np.float32)
    
    if band_mean is None:
        band_mean = np.array([np.nanmean(image[b]) for b in range(num_bands)])
    if band_std is None:
        band_std = np.array([np.nanstd(image[b]) for b in range(num_bands)])
    
    for b in range(num_bands):
        if band_std[b] > 0:
            normalized[b] = (image[b] - band_mean[b]) / band_std[b]
        else:
            normalized[b] = 0.0
    
    stats = {
        "method": "standardize",
        "band_mean": band_mean.tolist(),
        "band_std": band_std.tolist()
    }
    return normalized, stats


def normalize_percentile(image: np.ndarray,
                         low_pct: float = 2.0,
                         high_pct: float = 98.0,
                         band_low: Optional[np.ndarray] = None,
                         band_high: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Percentile-based normalization to [0, 1] range.
    Clips extreme outliers using percentile thresholds, then scales.
    
    This is the RECOMMENDED method for satellite imagery because it
    handles outliers (clouds, shadows, sensor artifacts) gracefully.
    
    Args:
        image: Input image [bands, H, W]
        low_pct: Lower percentile for clipping (e.g., 2nd percentile)
        high_pct: Upper percentile for clipping (e.g., 98th percentile)
        band_low: Pre-computed low percentile values
        band_high: Pre-computed high percentile values
        
    Returns:
        Tuple of (normalized_image, stats_dict)
    """
    num_bands = image.shape[0]
    normalized = np.zeros_like(image, dtype=np.float32)
    
    if band_low is None:
        band_low = np.array([np.nanpercentile(image[b], low_pct) for b in range(num_bands)])
    if band_high is None:
        band_high = np.array([np.nanpercentile(image[b], high_pct) for b in range(num_bands)])
    
    for b in range(num_bands):
        denom = band_high[b] - band_low[b]
        if denom > 0:
            normalized[b] = (image[b] - band_low[b]) / denom
        else:
            normalized[b] = 0.0
    
    normalized = np.clip(normalized, 0.0, 1.0)
    
    stats = {
        "method": "percentile",
        "low_pct": low_pct,
        "high_pct": high_pct,
        "band_low": band_low.tolist(),
        "band_high": band_high.tolist()
    }
    return normalized, stats


def compute_global_stats(images: list, method: str = "percentile", **kwargs) -> Dict:
    """
    Compute normalization statistics across multiple images.
    This ensures consistent normalization across the entire dataset.
    
    Args:
        images: List of images [bands, H, W]
        method: Normalization method ("minmax", "standardize", "percentile")
        
    Returns:
        Dictionary of global statistics
    """
    num_bands = images[0].shape[0]
    
    if method == "minmax":
        band_min = np.array([min(np.nanmin(img[b]) for img in images) for b in range(num_bands)])
        band_max = np.array([max(np.nanmax(img[b]) for img in images) for b in range(num_bands)])
        return {"method": "minmax", "band_min": band_min, "band_max": band_max}
    
    elif method == "standardize":
        # Compute global mean and std using Welford's online algorithm
        all_pixels = [np.concatenate([img[b].ravel() for img in images]) for b in range(num_bands)]
        band_mean = np.array([np.nanmean(pixels) for pixels in all_pixels])
        band_std = np.array([np.nanstd(pixels) for pixels in all_pixels])
        return {"method": "standardize", "band_mean": band_mean, "band_std": band_std}
    
    elif method == "percentile":
        low_pct = kwargs.get("low_pct", 2.0)
        high_pct = kwargs.get("high_pct", 98.0)
        all_pixels = [np.concatenate([img[b].ravel() for img in images]) for b in range(num_bands)]
        band_low = np.array([np.nanpercentile(pixels, low_pct) for pixels in all_pixels])
        band_high = np.array([np.nanpercentile(pixels, high_pct) for pixels in all_pixels])
        return {"method": "percentile", "band_low": band_low, "band_high": band_high,
                "low_pct": low_pct, "high_pct": high_pct}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_image(image: np.ndarray, method: str = "percentile", 
                    stats: Optional[Dict] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
    """
    Unified normalization interface.
    
    Args:
        image: Input image [bands, H, W]
        method: "minmax", "standardize", or "percentile"
        stats: Pre-computed statistics (for applying saved stats)
        
    Returns:
        Tuple of (normalized_image, stats_dict)
    """
    if method == "minmax":
        return normalize_minmax(
            image, 
            band_min=stats.get("band_min") if stats else None,
            band_max=stats.get("band_max") if stats else None
        )
    elif method == "standardize":
        return normalize_standardize(
            image,
            band_mean=stats.get("band_mean") if stats else None,
            band_std=stats.get("band_std") if stats else None
        )
    elif method == "percentile":
        return normalize_percentile(
            image,
            low_pct=kwargs.get("low_pct", 2.0),
            high_pct=kwargs.get("high_pct", 98.0),
            band_low=stats.get("band_low") if stats else None,
            band_high=stats.get("band_high") if stats else None
        )
    else:
        raise ValueError(f"Unknown method: {method}")
