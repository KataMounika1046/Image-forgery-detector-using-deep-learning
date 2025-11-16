"""
Data loading and preprocessing utilities for Image Forgery Detection
"""
import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

from config import TrainingConfig, PROCESSED_DATA_DIR


class ImageForgeryDataset(Dataset):
    """
    Dataset class for image forgery detection
    Supports both binary classification (authentic vs forged) and multi-class
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        is_binary: bool = True
    ):
        """
        Args:
            data_dir: Root directory containing train/val/test subdirectories
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
            is_binary: If True, binary classification (0=authentic, 1=forged)
                      If False, multi-class (0=authentic, 1=splicing, 2=copy-move, etc.)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.is_binary = is_binary
        
        # Load data samples
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and labels"""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # For binary classification
        if self.is_binary:
            # Authentic images (label 0)
            authentic_dir = split_dir / 'authentic'
            if authentic_dir.exists():
                for img_path in authentic_dir.glob('*.jpg') + authentic_dir.glob('*.png'):
                    samples.append((str(img_path), 0))
            
            # Forged images (label 1)
            forged_dir = split_dir / 'forged'
            if forged_dir.exists():
                for img_path in forged_dir.glob('*.jpg') + forged_dir.glob('*.png'):
                    samples.append((str(img_path), 1))
        else:
            # Multi-class: authentic, splicing, copy-move, etc.
            class_dirs = {
                'authentic': 0,
                'splicing': 1,
                'copy-move': 2,
                'removal': 3,
            }
            
            for class_name, label in class_dirs.items():
                class_dir = split_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob('*.jpg') + class_dir.glob('*.png'):
                        samples.append((str(img_path), label))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            # Default transform
            image = Image.fromarray(image)
            image = transforms.ToTensor()(image)
        
        return image, label


def get_transforms(
    split: str = 'train',
    config: TrainingConfig = None
) -> A.Compose:
    """
    Get data augmentation transforms
    """
    if config is None:
        from config import TrainingConfig
        config = TrainingConfig()
    
    if split == 'train' and config.USE_AUGMENTATION:
        transform = A.Compose([
            A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
            A.HorizontalFlip(p=config.AUGMENTATION_PROB),
            A.RandomRotate90(p=config.AUGMENTATION_PROB),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=config.AUGMENTATION_PROB
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=config.AUGMENTATION_PROB
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            ),
            ToTensorV2()
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = A.Compose([
            A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
            A.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            ),
            ToTensorV2()
        ])
    
    return transform


def create_dataloaders(
    data_dir: str,
    config: TrainingConfig = None,
    is_binary: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    """
    if config is None:
        from config import TrainingConfig
        config = TrainingConfig()
    
    # Create datasets
    train_dataset = ImageForgeryDataset(
        data_dir=data_dir,
        split='train',
        transform=get_transforms('train', config),
        is_binary=is_binary
    )
    
    val_dataset = ImageForgeryDataset(
        data_dir=data_dir,
        split='val',
        transform=get_transforms('val', config),
        is_binary=is_binary
    )
    
    test_dataset = ImageForgeryDataset(
        data_dir=data_dir,
        split='test',
        transform=get_transforms('test', config),
        is_binary=is_binary
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader


def download_dataset_info():
    """
    Information about available datasets for image forgery detection
    """
    datasets_info = {
        "CASIA v1.0": {
            "description": "Classic dataset with splicing and copy-move forgeries",
            "url": "http://forensics.idealtest.org/",
            "size": "~1.2GB",
            "classes": ["authentic", "splicing", "copy-move"],
            "notes": "Requires registration"
        },
        "CASIA v2.0": {
            "description": "Extended version with more diverse forgeries",
            "url": "http://forensics.idealtest.org/",
            "size": "~4.5GB",
            "classes": ["authentic", "splicing", "copy-move"],
            "notes": "Requires registration"
        },
        "COVERAGE": {
            "description": "Copy-move forgery detection dataset",
            "url": "https://github.com/wenbihan/coverage",
            "size": "~500MB",
            "classes": ["authentic", "copy-move"],
            "notes": "GitHub repository"
        },
        "Columbia Uncompressed Image Splicing Detection": {
            "description": "High-quality uncompressed images",
            "url": "http://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm",
            "size": "~1GB",
            "classes": ["authentic", "splicing"],
            "notes": "Academic use"
        },
        "NIST Nimble 2017": {
            "description": "Large-scale dataset with various manipulations",
            "url": "https://www.nist.gov/itl/iad/mig/nimble-challenge-2017-evaluation",
            "size": "~10GB",
            "classes": ["authentic", "various forgeries"],
            "notes": "NIST challenge dataset"
        },
        "WildWeb": {
            "description": "Real-world web images",
            "url": "https://github.com/MKLab-ITI/image-forensics",
            "size": "~2GB",
            "classes": ["authentic", "forged"],
            "notes": "GitHub repository"
        }
    }
    
    return datasets_info

