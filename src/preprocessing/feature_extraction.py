"""
DeforestNet - Feature Extraction Module
Computes vegetation indices (NDVI), texture features, and derived features
from Sentinel-1 and Sentinel-2 imagery for deforestation detection.
"""

import numpy as np
from scipy.ndimage import generic_filter
from typing import Tuple, Dict


# ============================================================
# VEGETATION INDICES (from Sentinel-2)
# ============================================================

def compute_ndvi(s2_image: np.ndarray, 
                 nir_band: int = 3, 
                 red_band: int = 2) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index (NDVI).
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Values range from -1 to 1:
      - Dense vegetation: 0.6 to 0.9
      - Sparse vegetation: 0.2 to 0.5
      - Bare soil/deforested: -0.1 to 0.2
      - Water: -1.0 to -0.1
    
    Args:
        s2_image: Sentinel-2 image [4, H, W] (B2, B3, B4, B8)
        nir_band: Index of NIR band (default: 3 = B8)
        red_band: Index of Red band (default: 2 = B4)
        
    Returns:
        NDVI array [H, W] with values in [-1, 1]
    """
    nir = s2_image[nir_band].astype(np.float64)
    red = s2_image[red_band].astype(np.float64)
    
    denominator = nir + red
    # Avoid division by zero
    ndvi = np.where(denominator != 0, (nir - red) / denominator, 0.0)
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def compute_evi(s2_image: np.ndarray,
                nir_band: int = 3,
                red_band: int = 2,
                blue_band: int = 0) -> np.ndarray:
    """
    Compute Enhanced Vegetation Index (EVI).
    
    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    
    EVI is more sensitive in high biomass regions (like tropical forests)
    and less affected by atmospheric conditions than NDVI.
    
    Args:
        s2_image: Sentinel-2 image [4, H, W]
        
    Returns:
        EVI array [H, W]
    """
    nir = s2_image[nir_band].astype(np.float64)
    red = s2_image[red_band].astype(np.float64)
    blue = s2_image[blue_band].astype(np.float64)
    
    denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = np.where(denominator != 0, 2.5 * (nir - red) / denominator, 0.0)
    evi = np.nan_to_num(evi, nan=0.0, posinf=0.0, neginf=0.0)
    
    return np.clip(evi, -1.0, 1.0).astype(np.float32)


def compute_savi(s2_image: np.ndarray,
                 nir_band: int = 3,
                 red_band: int = 2,
                 L: float = 0.5) -> np.ndarray:
    """
    Compute Soil-Adjusted Vegetation Index (SAVI).
    
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    
    Useful for areas with exposed soil (deforested regions).
    
    Args:
        s2_image: Sentinel-2 image [4, H, W]
        L: Soil brightness correction factor (0.5 for most cases)
        
    Returns:
        SAVI array [H, W]
    """
    nir = s2_image[nir_band].astype(np.float64)
    red = s2_image[red_band].astype(np.float64)
    
    denominator = nir + red + L
    savi = np.where(denominator != 0, 
                    ((nir - red) / denominator) * (1.0 + L), 0.0)
    savi = np.nan_to_num(savi, nan=0.0, posinf=0.0, neginf=0.0)
    
    return savi.astype(np.float32)


# ============================================================
# TEXTURE FEATURES (from Sentinel-1 SAR)
# ============================================================

def compute_glcm_contrast(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Compute local contrast (variance) as a texture measure.
    High contrast indicates heterogeneous areas (forest edges, deforestation boundaries).
    
    Args:
        image: Single-band image [H, W]
        window_size: Local window size
        
    Returns:
        Contrast feature [H, W]
    """
    local_mean = generic_filter(image, np.mean, size=window_size)
    local_var = generic_filter(image, np.var, size=window_size)
    return local_var.astype(np.float32)


def compute_local_entropy(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Compute local entropy as a texture measure.
    
    Higher entropy = more textural complexity (forest canopy).
    Lower entropy = more uniform (bare soil, water).
    
    Args:
        image: Single-band image [H, W] (should be normalized to [0, 1])
        window_size: Local window size
        
    Returns:
        Entropy feature [H, W]
    """
    # Quantize to 32 levels for histogram
    quantized = np.clip((image * 31).astype(np.int32), 0, 31)
    
    def _entropy(values):
        hist = np.bincount(values.astype(np.int32), minlength=32)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    entropy = generic_filter(quantized.astype(np.float64), _entropy, size=window_size)
    return entropy.astype(np.float32)


# ============================================================
# SAR-SPECIFIC FEATURES (from Sentinel-1)
# ============================================================

def compute_vv_vh_ratio(s1_image: np.ndarray) -> np.ndarray:
    """
    Compute VV/VH ratio from Sentinel-1.
    This ratio helps distinguish between forest and non-forest areas.
    
    Args:
        s1_image: Sentinel-1 image [2, H, W] (VV, VH)
        
    Returns:
        VV/VH ratio [H, W]
    """
    vv = s1_image[0].astype(np.float64)
    vh = s1_image[1].astype(np.float64)
    
    ratio = np.where(np.abs(vh) > 1e-10, vv / vh, 0.0)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    return ratio.astype(np.float32)


def compute_rvi_sar(s1_image: np.ndarray) -> np.ndarray:
    """
    Compute Radar Vegetation Index (RVI) from dual-pol SAR.
    
    RVI = 4 * VH / (VV + VH)
    
    Higher RVI indicates denser vegetation.
    
    Args:
        s1_image: Sentinel-1 image [2, H, W] (VV, VH)
        
    Returns:
        RVI array [H, W]
    """
    vv = s1_image[0].astype(np.float64)
    vh = s1_image[1].astype(np.float64)
    
    denominator = vv + vh
    rvi = np.where(denominator != 0, 4.0 * vh / denominator, 0.0)
    rvi = np.nan_to_num(rvi, nan=0.0, posinf=0.0, neginf=0.0)
    
    return np.clip(rvi, 0.0, 2.0).astype(np.float32)


# ============================================================
# COMBINED FEATURE STACK
# ============================================================

def extract_all_features(s1_image: np.ndarray, 
                         s2_image: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Extract all features from paired Sentinel-1 and Sentinel-2 images.
    
    Creates a feature stack combining:
    - Sentinel-1 bands (VV, VH)
    - Sentinel-2 bands (B2, B3, B4, B8)
    - NDVI, EVI, SAVI (vegetation indices)
    - VV/VH ratio, RVI (SAR features)
    
    Total: 2 + 4 + 3 + 2 = 11 feature bands
    
    Args:
        s1_image: Sentinel-1 [2, H, W]
        s2_image: Sentinel-2 [4, H, W]
        
    Returns:
        Tuple of (feature_stack [11, H, W], band_names list)
    """
    print("    Computing vegetation indices (NDVI, EVI, SAVI)...")
    ndvi = compute_ndvi(s2_image)
    evi = compute_evi(s2_image)
    savi = compute_savi(s2_image)
    
    print("    Computing SAR features (VV/VH ratio, RVI)...")
    vv_vh_ratio = compute_vv_vh_ratio(s1_image)
    rvi = compute_rvi_sar(s1_image)
    
    # Stack all features
    feature_stack = np.stack([
        s1_image[0],   # VV
        s1_image[1],   # VH
        s2_image[0],   # B2 Blue
        s2_image[1],   # B3 Green
        s2_image[2],   # B4 Red
        s2_image[3],   # B8 NIR
        ndvi,           # NDVI
        evi,            # EVI
        savi,           # SAVI
        vv_vh_ratio,    # VV/VH
        rvi             # RVI
    ], axis=0)
    
    band_names = [
        "S1_VV", "S1_VH",
        "S2_B2_Blue", "S2_B3_Green", "S2_B4_Red", "S2_B8_NIR",
        "NDVI", "EVI", "SAVI",
        "VV_VH_Ratio", "RVI"
    ]
    
    # Clean any remaining NaN/Inf values
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_stack.astype(np.float32), band_names
