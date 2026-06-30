"""
DeforestNet - Preprocessing Package
Feature extraction, normalization, and data pipeline utilities.
"""

from .feature_extraction import (
    compute_ndvi,
    compute_evi,
    compute_savi,
    compute_vv_vh_ratio,
    compute_rvi_sar,
    extract_all_features
)

from .normalization import (
    normalize_minmax,
    normalize_percentile,
    normalize_standardize,
    normalize_image,
    compute_global_stats
)

from .data_pipeline import (
    PreprocessingPipeline,
    DataValidator,
    preprocess_synthetic_data
)

__all__ = [
    # Feature extraction
    "compute_ndvi",
    "compute_evi",
    "compute_savi",
    "compute_vv_vh_ratio",
    "compute_rvi_sar",
    "extract_all_features",
    # Normalization
    "normalize_minmax",
    "normalize_percentile",
    "normalize_standardize",
    "normalize_image",
    "compute_global_stats",
    # Pipeline
    "PreprocessingPipeline",
    "DataValidator",
    "preprocess_synthetic_data"
]
