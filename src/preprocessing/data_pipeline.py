"""
DeforestNet - Data Preprocessing Pipeline
Unified pipeline for processing satellite imagery.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.feature_extraction import (
    compute_ndvi, compute_evi, compute_savi,
    compute_vv_vh_ratio, compute_rvi_sar,
    extract_all_features
)
from src.preprocessing.normalization import (
    normalize_minmax, normalize_percentile,
    normalize_standardize, normalize_image,
    compute_global_stats
)
from src.utils.logger import get_logger
from src.utils.helpers import Timer
from configs.config import (
    BAND_NAMES, TOTAL_CHANNELS, NUM_CLASSES,
    NORMALIZATION_METHOD, PERCENTILE_LOW, PERCENTILE_HIGH
)

logger = get_logger("preprocessing")


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for satellite imagery.

    Steps:
    1. Load raw bands (6 bands: VV, VH, B2, B3, B4, B8)
    2. Compute derived indices (5 bands: NDVI, EVI, SAVI, VV/VH, RVI)
    3. Stack to 11 bands
    4. Normalize to [0, 1] range
    """

    def __init__(
        self,
        normalization_method: str = NORMALIZATION_METHOD,
        normalization_stats: Optional[Dict] = None,
        compute_stats: bool = True
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            normalization_method: 'minmax', 'percentile', or 'standardize'
            normalization_stats: Pre-computed normalization statistics
            compute_stats: Whether to compute stats from data
        """
        self.normalization_method = normalization_method
        self.normalization_stats = normalization_stats
        self.compute_stats = compute_stats

        self.band_names = BAND_NAMES
        self.num_channels = TOTAL_CHANNELS

        logger.info(f"PreprocessingPipeline initialized: {normalization_method}")

    def compute_derived_indices(
        self,
        raw_bands: np.ndarray
    ) -> np.ndarray:
        """
        Compute 5 derived indices from 6 raw bands.

        Args:
            raw_bands: Shape (6, H, W) - [VV, VH, B2, B3, B4, B8]

        Returns:
            Shape (5, H, W) - [NDVI, EVI, SAVI, VV/VH, RVI]
        """
        # Extract individual bands
        vv = raw_bands[0]
        vh = raw_bands[1]
        b2 = raw_bands[2]  # Blue
        b3 = raw_bands[3]  # Green
        b4 = raw_bands[4]  # Red
        b8 = raw_bands[5]  # NIR

        # Stack S2 bands for vegetation indices
        s2_stack = np.stack([b2, b3, b4, b8], axis=0)

        # Compute vegetation indices
        ndvi = compute_ndvi(s2_stack, nir_band=3, red_band=2)
        evi = compute_evi(s2_stack, nir_band=3, red_band=2, blue_band=0)
        savi = compute_savi(s2_stack, nir_band=3, red_band=2)

        # Stack S1 bands for SAR indices
        s1_stack = np.stack([vv, vh], axis=0)

        # Compute SAR indices
        vv_vh_ratio = compute_vv_vh_ratio(s1_stack)
        rvi = compute_rvi_sar(s1_stack)

        # Stack all derived indices
        derived = np.stack([ndvi, evi, savi, vv_vh_ratio, rvi], axis=0)

        return derived.astype(np.float32)

    def process_single(
        self,
        raw_bands: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Process a single image.

        Args:
            raw_bands: Shape (6, H, W) raw sensor bands
            normalize: Whether to normalize

        Returns:
            Shape (11, H, W) processed image
        """
        # Compute derived indices
        derived = self.compute_derived_indices(raw_bands)

        # Stack all bands
        full_stack = np.concatenate([raw_bands, derived], axis=0)

        # Normalize
        if normalize and self.normalization_stats is not None:
            full_stack, _ = normalize_image(
                full_stack,
                method=self.normalization_method,
                stats=self.normalization_stats
            )
        elif normalize:
            full_stack, _ = normalize_image(
                full_stack,
                method=self.normalization_method
            )

        # Handle NaN/Inf
        full_stack = np.nan_to_num(full_stack, nan=0.0, posinf=1.0, neginf=0.0)

        return full_stack.astype(np.float32)

    def process_batch(
        self,
        raw_batch: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Process a batch of images.

        Args:
            raw_batch: Shape (N, 6, H, W) raw sensor bands
            normalize: Whether to normalize

        Returns:
            Shape (N, 11, H, W) processed batch
        """
        processed = []

        for i in range(len(raw_batch)):
            proc = self.process_single(raw_batch[i], normalize=normalize)
            processed.append(proc)

        return np.stack(processed, axis=0)

    def fit(self, images: np.ndarray) -> Dict:
        """
        Compute normalization statistics from data.

        Args:
            images: Shape (N, 11, H, W) or list of (11, H, W) arrays

        Returns:
            Dictionary of normalization statistics
        """
        if isinstance(images, list):
            image_list = images
        else:
            image_list = [images[i] for i in range(len(images))]

        self.normalization_stats = compute_global_stats(
            image_list,
            method=self.normalization_method,
            low_pct=PERCENTILE_LOW,
            high_pct=PERCENTILE_HIGH
        )

        logger.info(f"Computed normalization stats: {self.normalization_method}")

        return self.normalization_stats

    def save_stats(self, path: Union[str, Path]):
        """Save normalization statistics to JSON."""
        path = Path(path)

        # Convert numpy arrays to lists for JSON
        stats_json = {}
        for key, value in self.normalization_stats.items():
            if isinstance(value, np.ndarray):
                stats_json[key] = value.tolist()
            else:
                stats_json[key] = value

        with open(path, 'w') as f:
            json.dump(stats_json, f, indent=2)

        logger.info(f"Saved normalization stats to {path}")

    def load_stats(self, path: Union[str, Path]):
        """Load normalization statistics from JSON."""
        path = Path(path)

        with open(path) as f:
            stats_json = json.load(f)

        # Convert lists back to numpy arrays
        for key in ['band_min', 'band_max', 'band_mean', 'band_std',
                    'band_low', 'band_high']:
            if key in stats_json:
                stats_json[key] = np.array(stats_json[key])

        self.normalization_stats = stats_json
        self.normalization_method = stats_json.get('method', self.normalization_method)

        logger.info(f"Loaded normalization stats from {path}")


class DataValidator:
    """Validate processed data quality."""

    @staticmethod
    def validate_image(image: np.ndarray) -> Dict:
        """
        Validate a single processed image.

        Args:
            image: Shape (C, H, W)

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "issues": []
        }

        # Check shape
        if image.ndim != 3:
            results["issues"].append(f"Expected 3D, got {image.ndim}D")
            results["valid"] = False

        if image.shape[0] != TOTAL_CHANNELS:
            results["issues"].append(
                f"Expected {TOTAL_CHANNELS} channels, got {image.shape[0]}"
            )
            results["valid"] = False

        # Check value range
        if image.min() < -0.1:
            results["issues"].append(f"Min value too low: {image.min():.4f}")

        if image.max() > 1.1:
            results["issues"].append(f"Max value too high: {image.max():.4f}")

        # Check for NaN/Inf
        if np.any(np.isnan(image)):
            results["issues"].append("Contains NaN values")
            results["valid"] = False

        if np.any(np.isinf(image)):
            results["issues"].append("Contains Inf values")
            results["valid"] = False

        # Check band statistics
        results["band_stats"] = {
            BAND_NAMES[i]: {
                "min": float(image[i].min()),
                "max": float(image[i].max()),
                "mean": float(image[i].mean()),
                "std": float(image[i].std())
            }
            for i in range(min(len(BAND_NAMES), image.shape[0]))
        }

        return results

    @staticmethod
    def validate_mask(mask: np.ndarray) -> Dict:
        """
        Validate a segmentation mask.

        Args:
            mask: Shape (H, W)

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "issues": []
        }

        # Check shape
        if mask.ndim != 2:
            results["issues"].append(f"Expected 2D, got {mask.ndim}D")
            results["valid"] = False

        # Check value range
        unique = np.unique(mask)
        if unique.min() < 0:
            results["issues"].append(f"Negative class index: {unique.min()}")
            results["valid"] = False

        if unique.max() >= NUM_CLASSES:
            results["issues"].append(
                f"Class index {unique.max()} >= NUM_CLASSES ({NUM_CLASSES})"
            )
            results["valid"] = False

        results["unique_classes"] = unique.tolist()
        results["class_counts"] = {
            int(c): int(np.sum(mask == c)) for c in unique
        }

        return results

    @staticmethod
    def validate_dataset(
        images: np.ndarray,
        masks: np.ndarray,
        sample_size: int = 10
    ) -> Dict:
        """
        Validate entire dataset (sample-based).

        Args:
            images: Shape (N, C, H, W)
            masks: Shape (N, H, W)
            sample_size: Number of samples to validate

        Returns:
            Dictionary with validation summary
        """
        n_samples = min(sample_size, len(images))
        indices = np.random.choice(len(images), n_samples, replace=False)

        results = {
            "total_samples": len(images),
            "validated_samples": n_samples,
            "valid_count": 0,
            "invalid_count": 0,
            "issues": []
        }

        for idx in indices:
            img_result = DataValidator.validate_image(images[idx])
            mask_result = DataValidator.validate_mask(masks[idx])

            if img_result["valid"] and mask_result["valid"]:
                results["valid_count"] += 1
            else:
                results["invalid_count"] += 1
                results["issues"].append({
                    "sample_idx": int(idx),
                    "image_issues": img_result["issues"],
                    "mask_issues": mask_result["issues"]
                })

        results["valid"] = results["invalid_count"] == 0

        return results


def preprocess_synthetic_data(
    data_dir: Path,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Preprocess synthetic dataset and compute statistics.

    Args:
        data_dir: Path to synthetic data
        output_dir: Output directory (default: same as data_dir)

    Returns:
        Dictionary with preprocessing results
    """
    if output_dir is None:
        output_dir = data_dir

    # Load training data
    logger.info("Loading training data...")
    train_images = np.load(data_dir / "train_images.npy")
    train_masks = np.load(data_dir / "train_masks.npy")

    # Initialize pipeline
    pipeline = PreprocessingPipeline()

    # Fit normalization statistics on training data
    with Timer("Computing normalization statistics") as t:
        pipeline.fit(train_images)

    logger.info(f"Normalization stats computed in {t.elapsed_ms:.0f}ms")

    # Save statistics
    stats_path = output_dir / "normalization_stats.json"
    pipeline.save_stats(stats_path)

    # Validate data
    logger.info("Validating dataset...")
    validator = DataValidator()
    validation_results = validator.validate_dataset(train_images, train_masks)

    if validation_results["valid"]:
        logger.info("Dataset validation passed!")
    else:
        logger.warning(f"Dataset has issues: {validation_results['issues']}")

    return {
        "normalization_stats": pipeline.normalization_stats,
        "stats_path": str(stats_path),
        "validation": validation_results
    }


if __name__ == "__main__":
    from configs.config import SYNTHETIC_DATA_DIR

    print("Testing preprocessing pipeline...")

    # Test with synthetic data
    try:
        # Load a sample
        images = np.load(SYNTHETIC_DATA_DIR / "train_images.npy")
        masks = np.load(SYNTHETIC_DATA_DIR / "train_masks.npy")

        print(f"  Loaded {len(images)} training samples")

        # Test pipeline
        pipeline = PreprocessingPipeline()

        # Fit on training data
        stats = pipeline.fit(images[:10])
        print(f"  Normalization stats: {list(stats.keys())}")

        # Validate
        validator = DataValidator()
        img_result = validator.validate_image(images[0])
        print(f"  Image validation: {img_result['valid']}")

        mask_result = validator.validate_mask(masks[0])
        print(f"  Mask validation: {mask_result['valid']}")

        print("\n[OK] Preprocessing pipeline tests passed!")

    except FileNotFoundError:
        print("\n[!] Dataset not found. Run 'python generate_dataset.py' first.")
