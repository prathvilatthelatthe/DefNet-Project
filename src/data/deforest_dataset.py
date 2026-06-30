"""
DeforestNet - PyTorch Dataset for Deforestation Detection
Flexible dataset class supporting multiple data formats.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.augmentation import TrainAugmentation, ValAugmentation
from src.utils.logger import get_logger
from configs.config import (
    SYNTHETIC_DATA_DIR, TRAINING_CONFIG, NUM_CLASSES,
    CLASS_NAMES, TOTAL_CHANNELS
)

logger = get_logger("dataset")


class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for deforestation detection.

    Supports loading from:
    - .npy files (synthetic data format)
    - .npz chunk files (preprocessed real data)
    - In-memory numpy arrays
    """

    def __init__(
        self,
        images: Union[np.ndarray, Path, str],
        masks: Union[np.ndarray, Path, str],
        transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        """
        Initialize dataset.

        Args:
            images: Either numpy array (N, C, H, W) or path to .npy file
            masks: Either numpy array (N, H, W) or path to .npy file
            transform: Optional transform function (image, mask) -> (image, mask)
            return_metadata: Whether to return sample index as third element
        """
        self.transform = transform
        self.return_metadata = return_metadata

        # Load images
        if isinstance(images, np.ndarray):
            self.images = images
        else:
            images_path = Path(images)
            if images_path.suffix == '.npy':
                self.images = np.load(images_path)
            else:
                raise ValueError(f"Unsupported file format: {images_path.suffix}")

        # Load masks
        if isinstance(masks, np.ndarray):
            self.masks = masks
        else:
            masks_path = Path(masks)
            if masks_path.suffix == '.npy':
                self.masks = np.load(masks_path)
            else:
                raise ValueError(f"Unsupported file format: {masks_path.suffix}")

        # Validate shapes
        assert len(self.images) == len(self.masks), \
            f"Image count ({len(self.images)}) != mask count ({len(self.masks)})"

        # Ensure correct types
        self.images = self.images.astype(np.float32)
        self.masks = self.masks.astype(np.int64)

        self.num_samples = len(self.images)
        self.num_channels = self.images.shape[1] if self.images.ndim == 4 else 1

        logger.info(f"Dataset initialized: {self.num_samples} samples, "
                   f"{self.num_channels} channels")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Returns:
            Tuple of (image_tensor, mask_tensor)
            - image: FloatTensor (C, H, W)
            - mask: LongTensor (H, W)
        """
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()

        # Handle NaN/Inf values
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        # Apply transform
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Convert to tensors
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()

        if self.return_metadata:
            return image_tensor, mask_tensor, idx

        return image_tensor, mask_tensor

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced data.

        Returns:
            Tensor of class weights (inverse frequency)
        """
        # Count pixels per class
        class_counts = np.zeros(NUM_CLASSES)
        for i in range(NUM_CLASSES):
            class_counts[i] = np.sum(self.masks == i)

        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)

        # Inverse frequency weighting
        total_pixels = self.masks.size
        class_weights = total_pixels / (NUM_CLASSES * class_counts)

        # Normalize
        class_weights = class_weights / class_weights.sum() * NUM_CLASSES

        return torch.FloatTensor(class_weights)

    def get_class_distribution(self) -> Dict[str, float]:
        """
        Get class distribution statistics.

        Returns:
            Dictionary mapping class names to percentages
        """
        total_pixels = self.masks.size
        distribution = {}

        for i, name in enumerate(CLASS_NAMES):
            count = np.sum(self.masks == i)
            distribution[name] = (count / total_pixels) * 100

        return distribution

    def get_sample_weights(self) -> np.ndarray:
        """
        Compute per-sample weights based on class frequencies.
        Useful for WeightedRandomSampler.

        Returns:
            Array of sample weights
        """
        class_weights = self.get_class_weights().numpy()

        # For each sample, compute weight based on dominant class
        sample_weights = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            mask = self.masks[i]
            # Use median class as representative (handles multi-class masks)
            unique, counts = np.unique(mask, return_counts=True)
            dominant_class = unique[np.argmax(counts)]
            sample_weights[i] = class_weights[dominant_class]

        return sample_weights


class SyntheticDataset(DeforestationDataset):
    """
    Convenience wrapper for loading synthetic dataset.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: Optional[Path] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize synthetic dataset.

        Args:
            split: One of 'train', 'val', 'test'
            data_dir: Path to synthetic data directory
            transform: Optional transform
        """
        if data_dir is None:
            data_dir = SYNTHETIC_DATA_DIR

        data_dir = Path(data_dir)

        images_path = data_dir / f"{split}_images.npy"
        masks_path = data_dir / f"{split}_masks.npy"

        if not images_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {images_path}. "
                f"Run 'python generate_dataset.py' first."
            )

        # Set default transform based on split
        if transform is None:
            if split == "train":
                transform = TrainAugmentation()
            else:
                transform = ValAugmentation()

        super().__init__(
            images=images_path,
            masks=masks_path,
            transform=transform
        )

        self.split = split
        self.data_dir = data_dir


def create_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = None,
    num_workers: int = None,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Path to data directory (default: SYNTHETIC_DATA_DIR)
        batch_size: Batch size (default from config)
        num_workers: Number of workers (default from config)
        pin_memory: Pin memory for GPU
        use_weighted_sampler: Use weighted random sampling for training

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    if data_dir is None:
        data_dir = SYNTHETIC_DATA_DIR

    if batch_size is None:
        batch_size = TRAINING_CONFIG["batch_size"]

    if num_workers is None:
        num_workers = TRAINING_CONFIG["num_workers"]

    # Create datasets
    train_dataset = SyntheticDataset("train", data_dir, TrainAugmentation())
    val_dataset = SyntheticDataset("val", data_dir, ValAugmentation())
    test_dataset = SyntheticDataset("test", data_dir, ValAugmentation())

    # Create sampler for training if requested
    train_sampler = None
    train_shuffle = True

    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_shuffle = False

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"DataLoaders created: "
               f"train={len(train_loader)} batches, "
               f"val={len(val_loader)} batches, "
               f"test={len(test_loader)} batches")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


def get_dataset_info(data_dir: Optional[Path] = None) -> Dict:
    """
    Get information about the dataset.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary with dataset information
    """
    if data_dir is None:
        data_dir = SYNTHETIC_DATA_DIR

    data_dir = Path(data_dir)

    # Load metadata if exists
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Compute additional stats
    info = {
        "data_dir": str(data_dir),
        "metadata": metadata
    }

    for split in ["train", "val", "test"]:
        images_path = data_dir / f"{split}_images.npy"
        masks_path = data_dir / f"{split}_masks.npy"

        if images_path.exists():
            images = np.load(images_path)
            masks = np.load(masks_path)

            info[split] = {
                "num_samples": len(images),
                "image_shape": list(images.shape),
                "mask_shape": list(masks.shape),
                "image_dtype": str(images.dtype),
                "mask_dtype": str(masks.dtype),
                "unique_classes": np.unique(masks).tolist()
            }

    return info


if __name__ == "__main__":
    # Test dataset
    print("Testing DeforestationDataset...")

    try:
        # Test SyntheticDataset
        train_ds = SyntheticDataset("train")
        print(f"  Train dataset: {len(train_ds)} samples")

        # Get a sample
        image, mask = train_ds[0]
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Mask dtype: {mask.dtype}")

        # Test class weights
        weights = train_ds.get_class_weights()
        print(f"  Class weights: {weights}")

        # Test distribution
        dist = train_ds.get_class_distribution()
        print(f"  Class distribution: {dist}")

        # Test dataloader
        dataloaders = create_dataloaders(batch_size=4)
        batch = next(iter(dataloaders["train"]))
        print(f"  Batch images: {batch[0].shape}")
        print(f"  Batch masks: {batch[1].shape}")

        print("\n[OK] Dataset tests passed!")

    except FileNotFoundError as e:
        print(f"\n[!] Dataset not found. Run 'python generate_dataset.py' first.")
        print(f"    Error: {e}")
