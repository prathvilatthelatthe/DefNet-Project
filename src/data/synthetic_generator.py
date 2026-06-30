"""
DeforestNet - Synthetic Dataset Generator
Generates realistic 11-band satellite-like data with 6-class labels.

Classes:
    0: Forest - Healthy green vegetation
    1: Logging - Brown patches with low NDVI
    2: Mining - Blue/purple patches with pit patterns
    3: Agriculture - Regular grid patterns
    4: Fire - Black/burnt areas
    5: Infrastructure - Gray linear features (roads, buildings)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from configs.config import (
    SYNTHETIC_DATA_DIR, IMAGE_SIZE, NUM_CLASSES, CLASS_NAMES,
    TOTAL_CHANNELS, BAND_NAMES, RANDOM_SEED
)

logger = get_logger("synthetic_generator")


class SyntheticDataGenerator:
    """
    Generates synthetic satellite imagery with realistic spectral signatures.

    Each class has characteristic spectral signatures:
    - Forest: High NIR (B8), high NDVI, high VV/VH ratio
    - Logging: High Red (B4), low NDVI, moderate VV/VH
    - Mining: High Blue (B2), water features, low VV/VH
    - Agriculture: Regular patterns, moderate NDVI
    - Fire: Low all bands, very low NDVI, burnt signature
    - Infrastructure: Gray values, linear patterns
    """

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        seed: int = RANDOM_SEED
    ):
        """
        Initialize the generator.

        Args:
            image_size: Size of generated images (square)
            seed: Random seed for reproducibility
        """
        self.image_size = image_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Spectral signatures for each class (normalized 0-1)
        # Order: VV, VH, B2, B3, B4, B8
        self.spectral_signatures = {
            0: {  # Forest
                "VV": (0.6, 0.15), "VH": (0.3, 0.1),
                "B2": (0.1, 0.03), "B3": (0.15, 0.04),
                "B4": (0.12, 0.03), "B8": (0.7, 0.15)
            },
            1: {  # Logging
                "VV": (0.4, 0.1), "VH": (0.35, 0.1),
                "B2": (0.25, 0.05), "B3": (0.3, 0.06),
                "B4": (0.6, 0.1), "B8": (0.25, 0.08)
            },
            2: {  # Mining
                "VV": (0.3, 0.1), "VH": (0.4, 0.12),
                "B2": (0.5, 0.1), "B3": (0.35, 0.08),
                "B4": (0.3, 0.07), "B8": (0.15, 0.05)
            },
            3: {  # Agriculture
                "VV": (0.5, 0.12), "VH": (0.25, 0.08),
                "B2": (0.15, 0.04), "B3": (0.25, 0.06),
                "B4": (0.2, 0.05), "B8": (0.55, 0.12)
            },
            4: {  # Fire/Burnt
                "VV": (0.25, 0.08), "VH": (0.2, 0.07),
                "B2": (0.08, 0.03), "B3": (0.07, 0.02),
                "B4": (0.1, 0.03), "B8": (0.12, 0.04)
            },
            5: {  # Infrastructure
                "VV": (0.55, 0.15), "VH": (0.45, 0.12),
                "B2": (0.4, 0.08), "B3": (0.4, 0.08),
                "B4": (0.4, 0.08), "B8": (0.35, 0.08)
            }
        }

        logger.info(f"SyntheticDataGenerator initialized: {image_size}x{image_size}")

    def _generate_base_noise(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate Perlin-like noise for natural-looking textures."""
        noise = np.zeros(shape, dtype=np.float32)

        # Multi-scale noise
        for scale in [4, 8, 16, 32]:
            small = self.rng.rand(shape[0] // scale + 1, shape[1] // scale + 1)
            # Simple bilinear interpolation
            x = np.linspace(0, small.shape[1] - 1, shape[1])
            y = np.linspace(0, small.shape[0] - 1, shape[0])
            from scipy.ndimage import zoom
            zoomed = zoom(small, (shape[0] / small.shape[0], shape[1] / small.shape[1]), order=1)
            noise += zoomed[:shape[0], :shape[1]] / scale

        # Normalize
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise

    def _generate_forest_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate forest pattern - continuous with natural edges."""
        mask = np.ones(shape, dtype=np.float32)

        # Add some natural variation
        noise = self._generate_base_noise(shape)
        mask = mask * (0.8 + 0.2 * noise)

        return mask

    def _generate_logging_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate logging pattern - irregular patches."""
        mask = np.zeros(shape, dtype=np.float32)

        # Random irregular patches
        num_patches = self.rng.randint(1, 5)
        for _ in range(num_patches):
            cx = self.rng.randint(shape[1] // 4, 3 * shape[1] // 4)
            cy = self.rng.randint(shape[0] // 4, 3 * shape[0] // 4)

            # Create irregular shape using multiple ellipses
            for _ in range(self.rng.randint(2, 5)):
                offset_x = self.rng.randint(-30, 30)
                offset_y = self.rng.randint(-30, 30)
                rx = self.rng.randint(20, 60)
                ry = self.rng.randint(20, 60)

                y, x = np.ogrid[:shape[0], :shape[1]]
                ellipse = ((x - cx - offset_x) / rx) ** 2 + ((y - cy - offset_y) / ry) ** 2 <= 1
                mask[ellipse] = 1.0

        # Add noise for texture
        noise = self._generate_base_noise(shape)
        mask = mask * (0.7 + 0.3 * noise)

        return mask

    def _generate_mining_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate mining pattern - circular pits with water."""
        mask = np.zeros(shape, dtype=np.float32)

        # Create circular pit(s)
        num_pits = self.rng.randint(1, 3)
        for _ in range(num_pits):
            cx = self.rng.randint(shape[1] // 4, 3 * shape[1] // 4)
            cy = self.rng.randint(shape[0] // 4, 3 * shape[0] // 4)
            radius = self.rng.randint(30, 70)

            y, x = np.ogrid[:shape[0], :shape[1]]
            circle = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            mask[circle] = 1.0

            # Inner water circle (darker)
            inner_radius = radius * 0.6
            inner_circle = (x - cx) ** 2 + (y - cy) ** 2 <= inner_radius ** 2
            mask[inner_circle] = 0.7  # Water signature

        return mask

    def _generate_agriculture_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate agriculture pattern - regular grid fields."""
        mask = np.zeros(shape, dtype=np.float32)

        # Create rectangular fields
        field_size = self.rng.randint(30, 60)
        start_x = self.rng.randint(20, 50)
        start_y = self.rng.randint(20, 50)

        for i in range(3):
            for j in range(3):
                if self.rng.rand() > 0.3:  # 70% chance of field
                    x1 = start_x + j * (field_size + 5)
                    y1 = start_y + i * (field_size + 5)
                    x2 = min(x1 + field_size, shape[1])
                    y2 = min(y1 + field_size, shape[0])

                    if x2 > x1 and y2 > y1:
                        # Add row pattern within field
                        field = np.zeros((y2 - y1, x2 - x1))
                        row_width = self.rng.randint(3, 8)
                        for row in range(0, y2 - y1, row_width * 2):
                            field[row:min(row + row_width, y2 - y1), :] = 1.0
                        mask[y1:y2, x1:x2] = field

        return mask

    def _generate_fire_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate fire/burnt pattern - irregular with spread pattern."""
        mask = np.zeros(shape, dtype=np.float32)

        # Create burn scar
        cx = self.rng.randint(shape[1] // 4, 3 * shape[1] // 4)
        cy = self.rng.randint(shape[0] // 4, 3 * shape[0] // 4)

        # Irregular burnt area using noise
        noise = self._generate_base_noise(shape)
        y, x = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        base_radius = self.rng.randint(40, 80)
        threshold = base_radius + 20 * (noise - 0.5)
        mask[dist < threshold] = 1.0

        # Add some spread fingers
        for _ in range(self.rng.randint(2, 5)):
            angle = self.rng.rand() * 2 * np.pi
            length = self.rng.randint(20, 50)
            width = self.rng.randint(5, 15)

            for t in range(length):
                px = int(cx + (base_radius + t) * np.cos(angle))
                py = int(cy + (base_radius + t) * np.sin(angle))

                if 0 <= px < shape[1] and 0 <= py < shape[0]:
                    y_start = max(0, py - width)
                    y_end = min(shape[0], py + width)
                    x_start = max(0, px - width)
                    x_end = min(shape[1], px + width)
                    mask[y_start:y_end, x_start:x_end] = 1.0

        return mask

    def _generate_infrastructure_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate infrastructure pattern - roads and buildings."""
        mask = np.zeros(shape, dtype=np.float32)

        # Generate roads (lines)
        num_roads = self.rng.randint(1, 4)
        for _ in range(num_roads):
            if self.rng.rand() > 0.5:
                # Horizontal-ish road
                y = self.rng.randint(50, shape[0] - 50)
                width = self.rng.randint(3, 8)
                curve = self.rng.randint(-20, 20)
                for x in range(shape[1]):
                    y_pos = int(y + curve * np.sin(x * np.pi / shape[1]))
                    y_start = max(0, y_pos - width)
                    y_end = min(shape[0], y_pos + width)
                    mask[y_start:y_end, x] = 1.0
            else:
                # Vertical-ish road
                x = self.rng.randint(50, shape[1] - 50)
                width = self.rng.randint(3, 8)
                curve = self.rng.randint(-20, 20)
                for y in range(shape[0]):
                    x_pos = int(x + curve * np.sin(y * np.pi / shape[0]))
                    x_start = max(0, x_pos - width)
                    x_end = min(shape[1], x_pos + width)
                    mask[y, x_start:x_end] = 1.0

        # Add some buildings (rectangles)
        num_buildings = self.rng.randint(0, 5)
        for _ in range(num_buildings):
            bx = self.rng.randint(20, shape[1] - 40)
            by = self.rng.randint(20, shape[0] - 40)
            bw = self.rng.randint(10, 30)
            bh = self.rng.randint(10, 30)
            mask[by:by+bh, bx:bx+bw] = 1.0

        return mask

    def _apply_spectral_signature(
        self,
        mask: np.ndarray,
        class_idx: int,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """
        Apply spectral signature to a class mask.

        Args:
            mask: Binary or soft mask for the class
            class_idx: Class index (0-5)
            noise_level: Amount of noise to add

        Returns:
            6-band array with spectral values
        """
        h, w = mask.shape
        bands = np.zeros((6, h, w), dtype=np.float32)

        signature = self.spectral_signatures[class_idx]
        band_order = ["VV", "VH", "B2", "B3", "B4", "B8"]

        for i, band_name in enumerate(band_order):
            mean, std = signature[band_name]
            # Generate band with spatial correlation
            base_noise = self._generate_base_noise((h, w))
            band_values = mean + std * (base_noise - 0.5) * 2

            # Add random noise
            band_values += self.rng.randn(h, w) * noise_level * std

            # Apply mask
            bands[i] = band_values * mask

            # Clip to valid range
            bands[i] = np.clip(bands[i], 0, 1)

        return bands

    def _compute_derived_indices(self, raw_bands: np.ndarray) -> np.ndarray:
        """
        Compute 5 derived indices from 6 raw bands.

        Args:
            raw_bands: Shape (6, H, W) - VV, VH, B2, B3, B4, B8

        Returns:
            Shape (5, H, W) - NDVI, EVI, SAVI, VV/VH, RVI
        """
        VV, VH, B2, B3, B4, B8 = raw_bands

        # Small epsilon to avoid division by zero
        eps = 1e-8

        # NDVI = (B8 - B4) / (B8 + B4)
        ndvi = (B8 - B4) / (B8 + B4 + eps)
        ndvi = np.clip(ndvi, -1, 1)
        # Normalize to 0-1
        ndvi = (ndvi + 1) / 2

        # EVI = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1)
        evi = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + eps)
        evi = np.clip(evi, -1, 1)
        evi = (evi + 1) / 2

        # SAVI = 1.5 * (B8 - B4) / (B8 + B4 + 0.5)
        savi = 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps)
        savi = np.clip(savi, -1, 1)
        savi = (savi + 1) / 2

        # VV/VH ratio (normalized)
        vv_vh = VV / (VH + eps)
        vv_vh = np.clip(vv_vh / 20, 0, 1)  # Normalize assuming max ratio ~20

        # RVI = VV / (VV + VH)
        rvi = VV / (VV + VH + eps)
        rvi = np.clip(rvi, 0, 1)

        derived = np.stack([ndvi, evi, savi, vv_vh, rvi], axis=0)
        return derived.astype(np.float32)

    def generate_sample(
        self,
        primary_class: Optional[int] = None,
        multi_class: bool = True,
        forest_background: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single synthetic sample.

        Args:
            primary_class: Force a specific class (0-5), or None for random
            multi_class: Allow multiple classes in one image
            forest_background: Use forest as background

        Returns:
            Tuple of (image, mask)
            - image: Shape (11, H, W) float32, range [0, 1]
            - mask: Shape (H, W) int64, values 0-5
        """
        h, w = self.image_size, self.image_size

        # Initialize with forest background
        if forest_background:
            label_mask = np.zeros((h, w), dtype=np.int64)
            forest_mask = self._generate_forest_mask((h, w))
            combined_bands = self._apply_spectral_signature(forest_mask, 0)
        else:
            label_mask = np.zeros((h, w), dtype=np.int64)
            combined_bands = np.zeros((6, h, w), dtype=np.float32)

        # Select classes to add
        if primary_class is not None:
            classes_to_add = [primary_class]
        else:
            # Random selection
            if multi_class and self.rng.rand() > 0.5:
                # 2-3 classes
                num_classes = self.rng.randint(2, 4)
                classes_to_add = self.rng.choice([1, 2, 3, 4, 5], num_classes, replace=False).tolist()
            else:
                # Single class
                classes_to_add = [self.rng.randint(1, 6)]

        # Generate mask generators
        mask_generators = {
            1: self._generate_logging_mask,
            2: self._generate_mining_mask,
            3: self._generate_agriculture_mask,
            4: self._generate_fire_mask,
            5: self._generate_infrastructure_mask
        }

        # Add each class
        for class_idx in classes_to_add:
            if class_idx == 0:
                continue  # Skip forest (already background)

            class_mask = mask_generators[class_idx]((h, w))
            class_bands = self._apply_spectral_signature(class_mask, class_idx)

            # Blend with existing (class overwrites where mask > 0.5)
            blend_mask = class_mask > 0.5

            for b in range(6):
                combined_bands[b][blend_mask] = class_bands[b][blend_mask]

            label_mask[blend_mask] = class_idx

        # Compute derived indices
        derived_bands = self._compute_derived_indices(combined_bands)

        # Stack all 11 bands
        full_image = np.concatenate([combined_bands, derived_bands], axis=0)

        return full_image.astype(np.float32), label_mask.astype(np.int64)

    def generate_dataset(
        self,
        num_samples: int,
        output_dir: Optional[Path] = None,
        split: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a full dataset.

        Args:
            num_samples: Number of samples to generate
            output_dir: Directory to save files (optional)
            split: Dataset split name ('train', 'val', 'test')

        Returns:
            Tuple of (images, masks)
            - images: Shape (N, 11, H, W)
            - masks: Shape (N, H, W)
        """
        logger.info(f"Generating {num_samples} {split} samples...")

        images = []
        masks = []

        # Ensure class balance
        samples_per_class = num_samples // NUM_CLASSES

        for class_idx in range(NUM_CLASSES):
            for i in range(samples_per_class):
                # Occasionally allow multi-class
                multi_class = (i % 3 == 0) and class_idx != 0

                image, mask = self.generate_sample(
                    primary_class=class_idx if class_idx > 0 else None,
                    multi_class=multi_class
                )
                images.append(image)
                masks.append(mask)

                # Update seed for variety
                self.rng = np.random.RandomState(self.seed + len(images))

        # Generate remaining samples randomly
        remaining = num_samples - len(images)
        for _ in range(remaining):
            image, mask = self.generate_sample(multi_class=True)
            images.append(image)
            masks.append(mask)
            self.rng = np.random.RandomState(self.seed + len(images))

        images = np.stack(images, axis=0)
        masks = np.stack(masks, axis=0)

        # Shuffle
        indices = np.random.permutation(len(images))
        images = images[indices]
        masks = masks[indices]

        # Save if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            np.save(output_dir / f"{split}_images.npy", images)
            np.save(output_dir / f"{split}_masks.npy", masks)
            logger.info(f"Saved {split} data to {output_dir}")

        logger.info(f"Generated {len(images)} {split} samples")
        return images, masks


def generate_full_dataset(
    train_samples: int = 500,
    val_samples: int = 100,
    test_samples: int = 100,
    output_dir: Optional[Path] = None,
    seed: int = RANDOM_SEED
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate complete train/val/test dataset.

    Args:
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        output_dir: Output directory
        seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    if output_dir is None:
        output_dir = SYNTHETIC_DATA_DIR

    output_dir = Path(output_dir)

    generator = SyntheticDataGenerator(seed=seed)

    dataset = {}

    # Generate each split
    dataset["train"] = generator.generate_dataset(
        train_samples, output_dir, "train"
    )

    generator.seed = seed + 10000  # Different seed for val
    generator.rng = np.random.RandomState(generator.seed)
    dataset["val"] = generator.generate_dataset(
        val_samples, output_dir, "val"
    )

    generator.seed = seed + 20000  # Different seed for test
    generator.rng = np.random.RandomState(generator.seed)
    dataset["test"] = generator.generate_dataset(
        test_samples, output_dir, "test"
    )

    # Save metadata
    metadata = {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "image_size": IMAGE_SIZE,
        "num_channels": TOTAL_CHANNELS,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "band_names": BAND_NAMES,
        "seed": seed
    }

    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset generation complete. Saved to {output_dir}")

    return dataset


if __name__ == "__main__":
    # Quick test
    generator = SyntheticDataGenerator(seed=42)

    print("Testing single sample generation...")
    image, mask = generator.generate_sample()
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Unique classes in mask: {np.unique(mask)}")

    print("\nGenerator test passed!")
