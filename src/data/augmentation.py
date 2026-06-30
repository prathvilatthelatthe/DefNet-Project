"""
DeforestNet - Data Augmentation Module
Applies spatial and radiometric augmentations to satellite image patches
and their corresponding masks for training data expansion.

All transforms operate on:
  - image: np.ndarray of shape [C, H, W] (11 feature bands, float)
  - mask:  np.ndarray of shape [H, W] (binary uint8)

Spatial transforms are applied identically to both image and mask.
Radiometric transforms are applied only to the image.
"""

import numpy as np
from typing import Tuple


# ============================================================
# SPATIAL TRANSFORMS (applied to both image and mask)
# ============================================================

def random_horizontal_flip(image: np.ndarray, mask: np.ndarray,
                           p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Flip image and mask horizontally with probability p."""
    if np.random.random() < p:
        image = np.ascontiguousarray(image[:, :, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    return image, mask


def random_vertical_flip(image: np.ndarray, mask: np.ndarray,
                         p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Flip image and mask vertically with probability p."""
    if np.random.random() < p:
        image = np.ascontiguousarray(image[:, ::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])
    return image, mask


def random_rotate90(image: np.ndarray, mask: np.ndarray,
                    p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate image and mask by a random multiple of 90 degrees."""
    if np.random.random() < p:
        k = np.random.randint(1, 4)  # 1=90°, 2=180°, 3=270°
        # np.rot90 rotates axes (0,1) by default; for [C,H,W] we rotate axes (1,2)
        image = np.ascontiguousarray(np.rot90(image, k, axes=(1, 2)))
        mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
    return image, mask


def random_transpose(image: np.ndarray, mask: np.ndarray,
                     p: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Transpose (swap H and W axes) with probability p."""
    if np.random.random() < p:
        image = np.ascontiguousarray(image.transpose(0, 2, 1))
        mask = np.ascontiguousarray(mask.transpose(1, 0))
    return image, mask


# ============================================================
# RADIOMETRIC TRANSFORMS (applied to image only, NOT mask)
# ============================================================

def random_brightness(image: np.ndarray, mask: np.ndarray,
                      max_delta: float = 0.1,
                      p: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add random brightness shift to all bands.
    Only applied to the raw sensor bands (first 6), not derived indices.
    """
    if np.random.random() < p:
        delta = np.random.uniform(-max_delta, max_delta)
        # Apply only to sensor bands (S1_VV, S1_VH, S2_B2, S2_B3, S2_B4, S2_B8)
        image = image.copy()
        image[:6] = np.clip(image[:6] + delta, 0.0, 1.0)
    return image, mask


def random_contrast(image: np.ndarray, mask: np.ndarray,
                    low: float = 0.85, high: float = 1.15,
                    p: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale pixel values by a random contrast factor for sensor bands.
    """
    if np.random.random() < p:
        factor = np.random.uniform(low, high)
        image = image.copy()
        image[:6] = np.clip(image[:6] * factor, 0.0, 1.0)
    return image, mask


def random_band_noise(image: np.ndarray, mask: np.ndarray,
                      sigma: float = 0.02,
                      p: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add small Gaussian noise to sensor bands to simulate sensor variability.
    """
    if np.random.random() < p:
        image = image.copy()
        noise = np.random.normal(0, sigma, size=image[:6].shape).astype(image.dtype)
        image[:6] = np.clip(image[:6] + noise, 0.0, 1.0)
    return image, mask


def random_band_dropout(image: np.ndarray, mask: np.ndarray,
                        max_bands: int = 1,
                        p: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly zero out one sensor band to improve model robustness
    to missing data. Only drops from the 6 raw sensor bands.
    """
    if np.random.random() < p:
        image = image.copy()
        n_drop = np.random.randint(1, max_bands + 1)
        bands_to_drop = np.random.choice(6, size=n_drop, replace=False)
        for b in bands_to_drop:
            image[b] = 0.0
    return image, mask


# ============================================================
# MIXUP AUGMENTATION
# ============================================================

def mixup(image1: np.ndarray, mask1: np.ndarray,
          image2: np.ndarray, mask2: np.ndarray,
          alpha: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixup augmentation: linearly interpolate two samples.
    For segmentation, only mix the image; the mask uses the dominant sample.
    
    Args:
        image1, mask1: First sample
        image2, mask2: Second sample
        alpha: Beta distribution parameter (smaller = less mixing)
    
    Returns:
        Mixed image and the mask of the dominant sample
    """
    lam = np.random.beta(alpha, alpha)
    if lam < 0.5:
        lam = 1.0 - lam  # Ensure image1 is always dominant
    
    mixed_image = (lam * image1 + (1.0 - lam) * image2).astype(image1.dtype)
    # Use the dominant sample's mask
    return mixed_image, mask1


# ============================================================
# AUGMENTATION PIPELINE
# ============================================================

class TrainAugmentation:
    """
    Composed augmentation pipeline for training.
    Applies spatial + radiometric transforms with specified probabilities.
    """
    
    def __init__(self, 
                 flip_p: float = 0.5,
                 rotate_p: float = 0.5,
                 transpose_p: float = 0.3,
                 brightness_p: float = 0.3,
                 contrast_p: float = 0.3,
                 noise_p: float = 0.2,
                 band_dropout_p: float = 0.1):
        self.flip_p = flip_p
        self.rotate_p = rotate_p
        self.transpose_p = transpose_p
        self.brightness_p = brightness_p
        self.contrast_p = contrast_p
        self.noise_p = noise_p
        self.band_dropout_p = band_dropout_p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the full augmentation pipeline.
        
        Args:
            image: [C, H, W] float array
            mask: [H, W] uint8 array
            
        Returns:
            Augmented (image, mask) tuple
        """
        # Spatial transforms
        image, mask = random_horizontal_flip(image, mask, p=self.flip_p)
        image, mask = random_vertical_flip(image, mask, p=self.flip_p)
        image, mask = random_rotate90(image, mask, p=self.rotate_p)
        image, mask = random_transpose(image, mask, p=self.transpose_p)
        
        # Radiometric transforms (image only)
        image, mask = random_brightness(image, mask, p=self.brightness_p)
        image, mask = random_contrast(image, mask, p=self.contrast_p)
        image, mask = random_band_noise(image, mask, p=self.noise_p)
        image, mask = random_band_dropout(image, mask, p=self.band_dropout_p)
        
        return image, mask


class ValAugmentation:
    """
    No augmentation for validation/test — just pass through.
    """
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return image, mask
