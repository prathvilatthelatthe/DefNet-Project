"""
DeforestNet - Utilities Package
Common utilities and helper functions.
"""

from .logger import get_logger, logger, DeforestNetLogger
from .database import Database, get_database
from .helpers import (
    set_seed,
    get_device,
    get_device_info,
    Timer,
    ensure_dir,
    load_json,
    save_json,
    format_bytes,
    calculate_area_hectares,
    pixels_to_coordinates,
    get_severity_level,
    format_timestamp,
    generate_alert_id,
    class_distribution_to_dict,
    detect_deforestation_change,
    normalize_coordinates,
    create_color_map,
    mask_to_rgb,
    print_project_info,
    PROJECT_INFO
)

__all__ = [
    # Logger
    "get_logger",
    "logger",
    "DeforestNetLogger",
    # Database
    "Database",
    "get_database",
    # Helpers
    "set_seed",
    "get_device",
    "get_device_info",
    "Timer",
    "ensure_dir",
    "load_json",
    "save_json",
    "format_bytes",
    "calculate_area_hectares",
    "pixels_to_coordinates",
    "get_severity_level",
    "format_timestamp",
    "generate_alert_id",
    "class_distribution_to_dict",
    "detect_deforestation_change",
    "normalize_coordinates",
    "create_color_map",
    "mask_to_rgb",
    "print_project_info",
    "PROJECT_INFO"
]
