"""
DeforestNet - Utility Functions
Common utilities used across the project.
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import torch for device detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """
    Get the best available device (CUDA, MPS, or CPU).

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if not TORCH_AVAILABLE:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed device information.

    Returns:
        Dictionary with device info
    """
    info = {
        "device": get_device(),
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None
    }

    if TORCH_AVAILABLE and torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return info


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000 if self.elapsed else 0

    def __str__(self):
        return f"{self.name}: {self.elapsed_ms:.2f}ms"


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Union[str, Path]) -> Dict:
    """
    Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary with JSON contents
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, path: Union[str, Path], indent: int = 2):
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        path: Output path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)


def format_bytes(size: int) -> str:
    """
    Format bytes to human-readable string.

    Args:
        size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def calculate_area_hectares(
    num_pixels: int,
    resolution_meters: float = 10.0
) -> float:
    """
    Calculate area in hectares from pixel count.

    Args:
        num_pixels: Number of pixels
        resolution_meters: Pixel resolution in meters (default: 10m for Sentinel-2)

    Returns:
        Area in hectares
    """
    # Area per pixel in square meters
    pixel_area_m2 = resolution_meters ** 2
    # Total area in square meters
    total_area_m2 = num_pixels * pixel_area_m2
    # Convert to hectares (1 hectare = 10,000 m²)
    return total_area_m2 / 10000


def pixels_to_coordinates(
    pixel_x: int,
    pixel_y: int,
    transform: Tuple[float, ...] = None,
    origin_lat: float = 0.0,
    origin_lon: float = 0.0,
    resolution: float = 10.0
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to lat/lon.

    Args:
        pixel_x: X pixel coordinate
        pixel_y: Y pixel coordinate
        transform: Geotransform tuple (if available)
        origin_lat: Origin latitude
        origin_lon: Origin longitude
        resolution: Pixel resolution in meters

    Returns:
        Tuple of (latitude, longitude)
    """
    if transform:
        # Use geotransform
        lon = transform[0] + pixel_x * transform[1]
        lat = transform[3] + pixel_y * transform[5]
    else:
        # Simple calculation (approximate)
        # 1 degree latitude ≈ 111km
        # 1 degree longitude ≈ 111km * cos(latitude)
        lat = origin_lat - (pixel_y * resolution / 111000)
        lon = origin_lon + (pixel_x * resolution / (111000 * np.cos(np.radians(origin_lat))))

    return (lat, lon)


def get_severity_level(area_hectares: float, confidence: float) -> str:
    """
    Determine alert severity level.

    Args:
        area_hectares: Affected area in hectares
        confidence: Model confidence (0-1)

    Returns:
        Severity level string
    """
    # Base severity on area
    if area_hectares < 0.5:
        base_severity = "low"
    elif area_hectares < 2.0:
        base_severity = "medium"
    elif area_hectares < 5.0:
        base_severity = "high"
    else:
        base_severity = "critical"

    # Adjust based on confidence
    severity_order = ["low", "medium", "high", "critical"]
    current_idx = severity_order.index(base_severity)

    if confidence > 0.95:
        current_idx = min(current_idx + 1, 3)
    elif confidence < 0.75:
        current_idx = max(current_idx - 1, 0)

    return severity_order[current_idx]


def format_timestamp(dt: datetime = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime to string.

    Args:
        dt: Datetime object (defaults to now)
        fmt: Format string

    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(fmt)


def generate_alert_id() -> str:
    """
    Generate a unique alert ID.

    Returns:
        Unique alert ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = random.randint(1000, 9999)
    return f"ALT-{timestamp}-{random_suffix}"


def class_distribution_to_dict(
    mask: np.ndarray,
    class_names: List[str]
) -> Dict[str, float]:
    """
    Calculate class distribution from segmentation mask.

    Args:
        mask: Segmentation mask (H, W) with class indices
        class_names: List of class names

    Returns:
        Dictionary with class percentages
    """
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size

    distribution = {}
    for class_idx, class_name in enumerate(class_names):
        if class_idx in unique:
            idx = np.where(unique == class_idx)[0][0]
            percentage = (counts[idx] / total_pixels) * 100
        else:
            percentage = 0.0
        distribution[class_name] = round(percentage, 2)

    return distribution


def detect_deforestation_change(
    current_mask: np.ndarray,
    previous_mask: np.ndarray,
    forest_class_idx: int = 0,
    min_change_pixels: int = 100
) -> Tuple[bool, int, List[int]]:
    """
    Detect if deforestation has occurred between two masks.

    Args:
        current_mask: Current segmentation mask
        previous_mask: Previous segmentation mask
        forest_class_idx: Index of forest class
        min_change_pixels: Minimum pixels changed to trigger detection

    Returns:
        Tuple of (change_detected, pixels_changed, new_classes)
    """
    # Find where forest was present before but isn't now
    was_forest = previous_mask == forest_class_idx
    is_not_forest = current_mask != forest_class_idx

    # Deforestation = was forest AND is not forest now
    deforested = was_forest & is_not_forest
    pixels_changed = np.sum(deforested)

    # Get what classes replaced forest
    if pixels_changed > 0:
        new_classes = np.unique(current_mask[deforested]).tolist()
    else:
        new_classes = []

    change_detected = pixels_changed >= min_change_pixels

    return (change_detected, int(pixels_changed), new_classes)


def normalize_coordinates(lat: float, lon: float) -> Tuple[float, float]:
    """
    Normalize coordinates to valid ranges.

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)

    Returns:
        Tuple of (normalized_lat, normalized_lon)
    """
    lat = max(-90, min(90, lat))

    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360

    return (lat, lon)


def create_color_map(num_classes: int = 6) -> np.ndarray:
    """
    Create color map for visualization.

    Args:
        num_classes: Number of classes

    Returns:
        Color map array (num_classes, 3)
    """
    from configs.config import CLASS_CONFIG

    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        if i in CLASS_CONFIG:
            colors[i] = CLASS_CONFIG[i]["color"]
        else:
            # Random color for undefined classes
            colors[i] = np.random.randint(0, 255, 3)

    return colors


def mask_to_rgb(mask: np.ndarray, color_map: np.ndarray = None) -> np.ndarray:
    """
    Convert class mask to RGB image.

    Args:
        mask: Segmentation mask (H, W)
        color_map: Color map array (num_classes, 3)

    Returns:
        RGB image (H, W, 3)
    """
    if color_map is None:
        color_map = create_color_map()

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx in range(len(color_map)):
        rgb[mask == class_idx] = color_map[class_idx]

    return rgb


# Project information
PROJECT_INFO = {
    "name": "DeforestNet",
    "version": "1.0.0",
    "description": "AI-powered deforestation detection and alert system",
    "author": "DeforestNet Team"
}


def print_project_info():
    """Print project information banner."""
    print("=" * 60)
    print(f"  {PROJECT_INFO['name']} v{PROJECT_INFO['version']}")
    print(f"  {PROJECT_INFO['description']}")
    print("=" * 60)
    print(f"  Device: {get_device()}")
    print(f"  Timestamp: {format_timestamp()}")
    print("=" * 60)


if __name__ == "__main__":
    # Test utilities
    print_project_info()

    # Test timer
    with Timer("Test operation") as t:
        time.sleep(0.1)
    print(f"✓ Timer test: {t}")

    # Test device detection
    device_info = get_device_info()
    print(f"✓ Device info: {device_info}")

    # Test area calculation
    area = calculate_area_hectares(1000)  # 1000 pixels at 10m resolution
    print(f"✓ Area calculation: 1000 pixels = {area} hectares")

    # Test severity
    severity = get_severity_level(2.5, 0.94)
    print(f"✓ Severity level: {severity}")

    print("\n✓ All utility tests passed!")
