"""
DeforestNet - Project Configuration
Central configuration for all paths, parameters, and hyperparameters.
Updated for 6-class classification with free notification system.
"""

import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

# Other directories
LOGS_DIR = PROJECT_ROOT / "logs"
DATABASE_DIR = PROJECT_ROOT / "database"

# Create directories if they don't exist
for dir_path in [SYNTHETIC_DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                 CHECKPOINTS_DIR, VISUALIZATION_DIR, PREDICTIONS_DIR,
                 GRADCAM_DIR, LOGS_DIR, DATABASE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# SATELLITE BANDS CONFIGURATION
# ============================================================
# Sentinel-1 SAR bands
S1_BANDS = {
    "VV": {"index": 0, "description": "Vertical-Vertical polarization", "range": (-25, 0)},
    "VH": {"index": 1, "description": "Vertical-Horizontal polarization", "range": (-30, -5)}
}

# Sentinel-2 Optical bands
S2_BANDS = {
    "B2": {"index": 2, "description": "Blue", "range": (0, 10000), "wavelength": "490nm"},
    "B3": {"index": 3, "description": "Green", "range": (0, 10000), "wavelength": "560nm"},
    "B4": {"index": 4, "description": "Red", "range": (0, 10000), "wavelength": "665nm"},
    "B8": {"index": 5, "description": "NIR", "range": (0, 10000), "wavelength": "842nm"}
}

# Derived indices (calculated from raw bands)
DERIVED_INDICES = {
    "NDVI": {"index": 6, "description": "Normalized Difference Vegetation Index", "range": (-1, 1)},
    "EVI": {"index": 7, "description": "Enhanced Vegetation Index", "range": (-1, 1)},
    "SAVI": {"index": 8, "description": "Soil Adjusted Vegetation Index", "range": (-1, 1)},
    "VV_VH_RATIO": {"index": 9, "description": "SAR polarization ratio", "range": (0, 20)},
    "RVI": {"index": 10, "description": "Radar Vegetation Index", "range": (0, 1)}
}

# Total channels
NUM_RAW_BANDS = 6   # VV, VH, B2, B3, B4, B8
NUM_DERIVED_BANDS = 5  # NDVI, EVI, SAVI, VV/VH, RVI
TOTAL_CHANNELS = NUM_RAW_BANDS + NUM_DERIVED_BANDS  # 11

# Band names in order
BAND_NAMES = ["VV", "VH", "B2", "B3", "B4", "B8", "NDVI", "EVI", "SAVI", "VV_VH_RATIO", "RVI"]

# ============================================================
# 6-CLASS CLASSIFICATION
# ============================================================
NUM_CLASSES = 6

CLASS_CONFIG = {
    0: {"name": "Forest", "color": (34, 139, 34), "rgb_hex": "#228B22", "description": "Healthy forest cover"},
    1: {"name": "Logging", "color": (139, 69, 19), "rgb_hex": "#8B4513", "description": "Illegal/legal logging activity"},
    2: {"name": "Mining", "color": (128, 0, 128), "rgb_hex": "#800080", "description": "Mining operations"},
    3: {"name": "Agriculture", "color": (255, 215, 0), "rgb_hex": "#FFD700", "description": "Agricultural expansion"},
    4: {"name": "Fire", "color": (255, 69, 0), "rgb_hex": "#FF4500", "description": "Fire/burnt areas"},
    5: {"name": "Infrastructure", "color": (128, 128, 128), "rgb_hex": "#808080", "description": "Roads, buildings, etc."}
}

CLASS_NAMES = [CLASS_CONFIG[i]["name"] for i in range(NUM_CLASSES)]
CLASS_COLORS = [CLASS_CONFIG[i]["color"] for i in range(NUM_CLASSES)]

# Class weights for imbalanced data (will be computed from training data)
# Default: equal weights
CLASS_WEIGHTS = [1.0, 2.0, 2.5, 1.5, 2.0, 1.5]  # Higher weight for rarer classes

# ============================================================
# IMAGE PARAMETERS
# ============================================================
IMAGE_SIZE = 256  # Input image size (256x256)
PATCH_SIZE = 256  # Patch size for training
PATCH_STRIDE = 128  # Overlap stride

# ============================================================
# PREPROCESSING PARAMETERS
# ============================================================
# Normalization
NORMALIZATION_METHOD = "minmax"  # Options: "minmax", "standardize", "percentile"
PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98

# Noise removal (for SAR)
LEE_FILTER_SIZE = 5  # Lee filter for SAR speckle noise

# ============================================================
# MODEL PARAMETERS
# ============================================================
MODEL_CONFIG = {
    "name": "UNet",
    "encoder": "resnet34",
    "in_channels": TOTAL_CHANNELS,  # 11
    "out_channels": NUM_CLASSES,     # 6
    "pretrained_encoder": False,     # Can't use pretrained with 11 channels
    "dropout": 0.2
}

# ============================================================
# TRAINING PARAMETERS
# ============================================================
TRAINING_CONFIG = {
    "batch_size": 16,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "optimizer": "adam",
    "scheduler": "reduce_on_plateau",
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "early_stopping_patience": 15,
    "num_workers": 0,  # 0 for Windows
    "pin_memory": True
}

# Loss function
LOSS_CONFIG = {
    "type": "cross_entropy",  # Options: "cross_entropy", "dice", "focal", "combined"
    "dice_weight": 0.5,
    "ce_weight": 0.5,
    "focal_gamma": 2.0,
    "label_smoothing": 0.1
}

# Data split
DATA_SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

RANDOM_SEED = 42

# Device configuration
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except (ImportError, OSError):
    DEVICE = "cpu"

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
SYNTHETIC_CONFIG = {
    "num_train_samples": 800,
    "num_val_samples": 200,
    "num_test_samples": 200,
    "image_size": IMAGE_SIZE,
    "noise_level": 0.1
}

# ============================================================
# ALERT SYSTEM CONFIGURATION
# ============================================================
ALERT_CONFIG = {
    "min_confidence": 0.7,           # Minimum confidence to trigger alert
    "min_affected_area": 5.0,        # Minimum hectares to trigger alert
    "pixel_to_hectare": 0.01,        # 10m resolution: 100 pixels = 1 hectare
    "alert_cooldown_hours": 6,       # Don't re-alert same location within hours
    "severity_thresholds": {
        "low": 10.0,       # 5 - 10 hectares
        "medium": 50.0,    # 10 - 50 hectares
        "high": 150.0,     # 50 - 150 hectares
        "critical": 300.0  # > 150 hectares (massive clearing)
    }
}

# ============================================================
# NOTIFICATION SYSTEM CONFIGURATION (ALL FREE)
# ============================================================
NOTIFICATION_CONFIG = {
    # Tier 1: Firebase Cloud Messaging (Free tier: 500k messages/month)
    "firebase": {
        "enabled": True,
        "credentials_file": str(PROJECT_ROOT / "configs" / "firebase_credentials.json"),
        "timeout_seconds": 10
    },

    # Tier 2: Telegram Bot (100% Free, Unlimited)
    "telegram": {
        "enabled": True,
        "bot_token": "",  # Set via environment variable TELEGRAM_BOT_TOKEN
        "timeout_seconds": 30
    },

    # Tier 3: Email via Gmail SMTP (Free: 500 emails/day)
    "email": {
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "",  # Set via environment variable EMAIL_SENDER
        "sender_password": "",  # Set via environment variable EMAIL_PASSWORD (App Password)
        "timeout_seconds": 30
    }
}

# ============================================================
# API CONFIGURATION
# ============================================================
API_CONFIG = {
    "host": os.environ.get("API_HOST", "0.0.0.0"),
    "port": int(os.environ.get("PORT", os.environ.get("API_PORT", 5000))),
    "debug": os.environ.get("API_DEBUG", "false").lower() == "true",
    "max_content_length": 50 * 1024 * 1024  # 50 MB max upload
}

# ============================================================
# DATABASE CONFIGURATION
# ============================================================
DATABASE_CONFIG = {
    "type": "sqlite",
    "path": str(DATABASE_DIR / "deforestnet.db"),
    "echo": False  # Set True for SQL query logging
}

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_file": str(LOGS_DIR / "deforestnet.log"),
    "max_bytes": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,
    "console_output": True
}

# ============================================================
# GRADCAM CONFIGURATION
# ============================================================
GRADCAM_CONFIG = {
    "target_layer": "decoder.blocks.0",  # Target layer for GradCAM
    "colormap": "jet",
    "alpha": 0.5  # Overlay transparency
}

# ============================================================
# COORDINATE SYSTEM
# ============================================================
GEO_CONFIG = {
    "crs": "EPSG:4326",  # WGS84 for lat/lon
    "resolution_meters": 10  # 10m per pixel (Sentinel-2)
}
