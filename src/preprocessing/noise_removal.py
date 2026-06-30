"""
DeforestNet - Noise Removal Module
Handles speckle noise removal for SAR (Sentinel-1) data 
and general noise filtering for optical (Sentinel-2) data.
"""

import numpy as np
from scipy.ndimage import median_filter, uniform_filter, gaussian_filter
from typing import Optional


def lee_filter(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Lee Speckle Filter for SAR imagery.
    Reduces speckle noise while preserving edges.
    
    The Lee filter works by computing local statistics (mean and variance)
    and applying an adaptive weighted combination of the pixel value and 
    the local mean.
    
    Args:
        image: Input SAR image (single band or multi-band [bands, H, W])
        window_size: Size of the filter window (odd number)
        
    Returns:
        Filtered image with same shape
    """
    if image.ndim == 3:
        # Apply per-band
        return np.stack([lee_filter(image[i], window_size) for i in range(image.shape[0])])
    
    img = image.astype(np.float64)
    
    # Local mean
    local_mean = uniform_filter(img, size=window_size)
    
    # Local variance  
    local_sq_mean = uniform_filter(img ** 2, size=window_size)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)  # Ensure non-negative
    
    # Overall variance (noise variance estimate)
    overall_var = np.var(img)
    
    # Weight factor
    # w = 1 - (noise_var / local_var)  clamped to [0, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = np.where(local_var > 0, 
                         1.0 - (overall_var / local_var),
                         0.0)
    weight = np.clip(weight, 0, 1)
    
    # Filtered = local_mean + weight * (original - local_mean)
    filtered = local_mean + weight * (img - local_mean)
    
    return filtered.astype(image.dtype)


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filter for impulse noise removal.
    Effective for salt-and-pepper noise in SAR imagery.
    
    Args:
        image: Input image [bands, H, W] or [H, W]
        kernel_size: Filter kernel size (odd number)
        
    Returns:
        Filtered image
    """
    if image.ndim == 3:
        return np.stack([
            median_filter(image[i], size=kernel_size) 
            for i in range(image.shape[0])
        ])
    return median_filter(image, size=kernel_size)


def apply_gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing filter.
    Good for general noise reduction in optical imagery.
    
    Args:
        image: Input image [bands, H, W] or [H, W]
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Smoothed image
    """
    if image.ndim == 3:
        return np.stack([
            gaussian_filter(image[i], sigma=sigma) 
            for i in range(image.shape[0])
        ])
    return gaussian_filter(image, sigma=sigma)


def remove_noise_sentinel1(image: np.ndarray, 
                            lee_window: int = 7, 
                            median_kernel: int = 3) -> np.ndarray:
    """
    Complete noise removal pipeline for Sentinel-1 SAR data.
    
    SAR imagery suffers from multiplicative speckle noise.
    Pipeline: Lee filter -> Median filter (for residual impulse noise)
    
    Args:
        image: Sentinel-1 image [2, H, W] (VV, VH bands)
        lee_window: Window size for Lee speckle filter
        median_kernel: Kernel size for median filter
        
    Returns:
        Denoised image [2, H, W]
    """
    print("    Applying Lee speckle filter (SAR)...")
    filtered = lee_filter(image, window_size=lee_window)
    
    print("    Applying median filter (residual noise)...")
    filtered = apply_median_filter(filtered, kernel_size=median_kernel)
    
    return filtered


def remove_noise_sentinel2(image: np.ndarray, 
                            gaussian_sigma: float = 0.5) -> np.ndarray:
    """
    Noise removal pipeline for Sentinel-2 optical data.
    
    Optical imagery has less severe noise than SAR.
    Light Gaussian smoothing to reduce sensor noise while preserving detail.
    
    Args:
        image: Sentinel-2 image [4, H, W] (B2, B3, B4, B8)
        gaussian_sigma: Sigma for Gaussian filter (keep small to preserve detail)
        
    Returns:
        Denoised image [4, H, W]
    """
    print("    Applying light Gaussian smoothing (optical)...")
    filtered = apply_gaussian_filter(image, sigma=gaussian_sigma)
    
    return filtered
