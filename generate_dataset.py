"""
DeforestNet - Dataset Generation Script
Main script to generate synthetic training data.

Usage:
    python generate_dataset.py [--train N] [--val N] [--test N] [--visualize]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.synthetic_generator import SyntheticDataGenerator, generate_full_dataset
from src.data.visualization import (
    visualize_sample,
    visualize_batch,
    plot_class_distribution,
    plot_band_statistics
)
from src.utils.logger import get_logger
from src.utils.helpers import Timer, format_bytes
from configs.config import (
    SYNTHETIC_DATA_DIR, SYNTHETIC_CONFIG, VISUALIZATION_DIR,
    CLASS_NAMES, NUM_CLASSES, TOTAL_CHANNELS
)

logger = get_logger("generate_dataset")


def generate_and_save_dataset(
    train_samples: int = 500,
    val_samples: int = 100,
    test_samples: int = 100,
    seed: int = 42,
    visualize: bool = True
) -> dict:
    """
    Generate complete synthetic dataset.

    Args:
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        seed: Random seed
        visualize: Whether to create visualizations

    Returns:
        Dictionary with dataset info
    """
    output_dir = SYNTHETIC_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DeforestNet - Synthetic Dataset Generation")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Train samples: {train_samples}")
    print(f"  Val samples: {val_samples}")
    print(f"  Test samples: {test_samples}")
    print(f"  Random seed: {seed}")
    print("=" * 60)

    # Generate dataset
    with Timer("Dataset generation") as timer:
        dataset = generate_full_dataset(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            output_dir=output_dir,
            seed=seed
        )

    print(f"\n[OK] Generation completed in {timer.elapsed_ms/1000:.1f} seconds")

    # Dataset info
    info = {
        "train": {
            "images_shape": dataset["train"][0].shape,
            "masks_shape": dataset["train"][1].shape,
            "size_bytes": dataset["train"][0].nbytes + dataset["train"][1].nbytes
        },
        "val": {
            "images_shape": dataset["val"][0].shape,
            "masks_shape": dataset["val"][1].shape,
            "size_bytes": dataset["val"][0].nbytes + dataset["val"][1].nbytes
        },
        "test": {
            "images_shape": dataset["test"][0].shape,
            "masks_shape": dataset["test"][1].shape,
            "size_bytes": dataset["test"][0].nbytes + dataset["test"][1].nbytes
        }
    }

    # Print info
    print("\n" + "-" * 60)
    print("  Dataset Summary")
    print("-" * 60)
    for split_name, split_info in info.items():
        print(f"  {split_name.upper()}:")
        print(f"    Images: {split_info['images_shape']}")
        print(f"    Masks:  {split_info['masks_shape']}")
        print(f"    Size:   {format_bytes(split_info['size_bytes'])}")
    print("-" * 60)

    # Create visualizations
    if visualize:
        print("\n[...] Creating visualizations...")
        vis_dir = VISUALIZATION_DIR / "synthetic_dataset"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Sample visualization
        visualize_sample(
            dataset["train"][0][0],
            dataset["train"][1][0],
            title="Training Sample 1",
            save_path=vis_dir / "sample_1.png",
            show=False
        )
        print(f"  [OK] Saved sample visualization")

        # Batch visualization
        visualize_batch(
            dataset["train"][0][:4],
            dataset["train"][1][:4],
            num_samples=4,
            save_path=vis_dir / "batch_samples.png",
            show=False
        )
        print(f"  [OK] Saved batch visualization")

        # Class distribution for each split
        for split_name in ["train", "val", "test"]:
            plot_class_distribution(
                dataset[split_name][1],
                title=f"Class Distribution - {split_name.upper()}",
                save_path=vis_dir / f"class_dist_{split_name}.png",
                show=False
            )
        print(f"  [OK] Saved class distribution plots")

        # Band statistics
        plot_band_statistics(
            dataset["train"][0],
            save_path=vis_dir / "band_statistics.png",
            show=False
        )
        print(f"  [OK] Saved band statistics")

        print(f"\n  Visualizations saved to: {vis_dir}")

    # Verify saved files
    print("\n[...] Verifying saved files...")
    for split in ["train", "val", "test"]:
        img_path = output_dir / f"{split}_images.npy"
        mask_path = output_dir / f"{split}_masks.npy"

        if img_path.exists() and mask_path.exists():
            print(f"  [OK] {split}_images.npy ({format_bytes(img_path.stat().st_size)})")
            print(f"  [OK] {split}_masks.npy ({format_bytes(mask_path.stat().st_size)})")
        else:
            print(f"  [X] Missing files for {split} split!")

    # Metadata
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        print(f"  [OK] metadata.json")

    print("\n" + "=" * 60)
    print("  Dataset generation COMPLETE!")
    print("=" * 60)

    return info


def verify_dataset(data_dir: Path = None) -> bool:
    """
    Verify that generated dataset is valid.

    Args:
        data_dir: Path to data directory

    Returns:
        True if valid, False otherwise
    """
    if data_dir is None:
        data_dir = SYNTHETIC_DATA_DIR

    print("\n[...] Verifying dataset integrity...")

    try:
        # Load each split
        for split in ["train", "val", "test"]:
            images = np.load(data_dir / f"{split}_images.npy")
            masks = np.load(data_dir / f"{split}_masks.npy")

            # Check shapes
            assert images.ndim == 4, f"Images should be 4D, got {images.ndim}D"
            assert masks.ndim == 3, f"Masks should be 3D, got {masks.ndim}D"
            assert images.shape[0] == masks.shape[0], "Sample count mismatch"
            assert images.shape[1] == TOTAL_CHANNELS, f"Expected {TOTAL_CHANNELS} channels"
            assert images.shape[2] == images.shape[3], "Images should be square"

            # Check value ranges
            assert images.min() >= 0, "Image values should be >= 0"
            assert images.max() <= 1, "Image values should be <= 1"
            assert masks.min() >= 0, f"Mask values should be >= 0"
            assert masks.max() < NUM_CLASSES, f"Mask values should be < {NUM_CLASSES}"

            # Check for valid classes
            unique_classes = np.unique(masks)
            print(f"  [OK] {split}: {len(images)} samples, classes {unique_classes.tolist()}")

        print("[OK] Dataset verification passed!")
        return True

    except Exception as e:
        print(f"[X] Verification failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic deforestation detection dataset"
    )
    parser.add_argument(
        "--train", type=int, default=SYNTHETIC_CONFIG["num_train_samples"],
        help="Number of training samples"
    )
    parser.add_argument(
        "--val", type=int, default=SYNTHETIC_CONFIG["num_val_samples"],
        help="Number of validation samples"
    )
    parser.add_argument(
        "--test", type=int, default=SYNTHETIC_CONFIG["num_test_samples"],
        help="Number of test samples"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--visualize", action="store_true", default=True,
        help="Create visualizations"
    )
    parser.add_argument(
        "--no-visualize", action="store_true",
        help="Skip visualizations"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify existing dataset"
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_dataset()
        return

    # Generate dataset
    generate_and_save_dataset(
        train_samples=args.train,
        val_samples=args.val,
        test_samples=args.test,
        seed=args.seed,
        visualize=not args.no_visualize
    )

    # Verify
    verify_dataset()


if __name__ == "__main__":
    main()
