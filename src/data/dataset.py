"""
DeforestNet - PyTorch Dataset & DataLoader
Loads preprocessed .npz chunks and serves (image, mask) pairs for training.

Data format:
  - images: float16 arrays of shape [N, 11, 256, 256]  (11 feature bands)
  - masks:  uint8 arrays of shape [N, 256, 256]         (binary: 0/1)

Each .npz chunk file contains up to 200 samples.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.augmentation import TrainAugmentation, ValAugmentation


class DeforestNetDataset(Dataset):
    """
    PyTorch Dataset for DeforestNet preprocessed satellite imagery.
    
    Loads all .npz chunks from a split directory into memory (the full
    dataset is small enough: ~1 GB across all splits). Applies optional
    augmentation on-the-fly.
    """
    
    def __init__(self, split_dir: str, augmentation=None):
        """
        Args:
            split_dir: Path to preprocessed split directory
                       (e.g., outputs/preprocessed/train/)
            augmentation: Callable(image, mask) -> (image, mask) or None
        """
        self.augmentation = augmentation
        
        # Load all chunks and concatenate
        chunk_paths = sorted(glob.glob(os.path.join(split_dir, "chunk*.npz")))
        if not chunk_paths:
            raise FileNotFoundError(f"No chunk*.npz files found in {split_dir}")
        
        images_list = []
        masks_list = []
        for path in chunk_paths:
            data = np.load(path)
            images_list.append(data["images"])
            masks_list.append(data["masks"])
        
        # Concatenate all chunks: [total_samples, C, H, W] and [total_samples, H, W]
        self.images = np.concatenate(images_list, axis=0)
        self.masks = np.concatenate(masks_list, axis=0)
        
        self.num_samples = self.images.shape[0]
        self.num_bands = self.images.shape[1]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            image: torch.FloatTensor of shape [C, H, W]
            mask:  torch.LongTensor of shape [H, W]
        """
        image = np.nan_to_num(
            self.images[idx].astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        mask = self.masks[idx]
        
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)
        
        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        
        return image_tensor, mask_tensor


def get_dataloaders(preprocessed_dir: str,
                    batch_size: int = 16,
                    num_workers: int = 0,
                    pin_memory: bool = True) -> dict:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        preprocessed_dir: Root preprocessed directory containing
                          train/, val/, test/ subdirectories
        batch_size: Batch size for training (val/test use same size)
        num_workers: Number of worker processes for data loading.
                     Use 0 on Windows to avoid multiprocessing issues.
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to DataLoaders
    """
    train_dir = os.path.join(preprocessed_dir, "train")
    val_dir = os.path.join(preprocessed_dir, "val")
    test_dir = os.path.join(preprocessed_dir, "test")
    
    train_dataset = DeforestNetDataset(train_dir, augmentation=TrainAugmentation())
    val_dataset = DeforestNetDataset(val_dir, augmentation=ValAugmentation())
    test_dataset = DeforestNetDataset(test_dir, augmentation=ValAugmentation())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete last batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return {"train": train_loader, "val": val_loader, "test": test_loader}
