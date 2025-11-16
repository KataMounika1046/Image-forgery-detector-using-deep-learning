"""
Utility functions for Image Forgery Detection
"""
import os
import shutil
import requests
from pathlib import Path
from typing import List, Optional
import zipfile
import tarfile
from tqdm import tqdm


def download_file(url: str, destination: Path, chunk_size: int = 8192):
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        destination: Path to save the file
        chunk_size: Size of chunks to download
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def extract_archive(archive_path: Path, extract_to: Path):
    """
    Extract archive file (zip or tar)
    
    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


def organize_dataset(
    source_dir: Path,
    target_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    """
    Organize dataset into train/val/test splits
    
    Args:
        source_dir: Source directory with images
        target_dir: Target directory for organized dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['authentic', 'forged']:
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Organize images
    for class_name in ['authentic', 'forged']:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            continue
        
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        n_images = len(images)
        
        # Calculate split indices
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Shuffle
        import random
        random.shuffle(images)
        
        # Split
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy files
        for img in tqdm(train_images, desc=f"Organizing {class_name} - train"):
            shutil.copy2(img, target_dir / 'train' / class_name / img.name)
        
        for img in tqdm(val_images, desc=f"Organizing {class_name} - val"):
            shutil.copy2(img, target_dir / 'val' / class_name / img.name)
        
        for img in tqdm(test_images, desc=f"Organizing {class_name} - test"):
            shutil.copy2(img, target_dir / 'test' / class_name / img.name)
    
    print(f"Dataset organized successfully in {target_dir}")


def create_synthetic_dataset(
    output_dir: Path,
    num_authentic: int = 1000,
    num_forged: int = 1000,
    image_size: tuple = (224, 224)
):
    """
    Create a synthetic dataset for testing (simple example)
    Note: This is a placeholder. Real synthetic datasets require more sophisticated methods.
    
    Args:
        output_dir: Directory to save synthetic images
        num_authentic: Number of authentic images to generate
        num_forged: Number of forged images to generate
        image_size: Size of generated images
    """
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    import random
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'authentic').mkdir(exist_ok=True)
    (output_dir / 'forged').mkdir(exist_ok=True)
    
    print("Generating synthetic dataset...")
    print("Note: This is a simple example. For real training, use proper datasets.")
    
    # Generate authentic images (simple patterns)
    for i in tqdm(range(num_authentic), desc="Generating authentic images"):
        img = Image.new('RGB', image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add random shapes
        for _ in range(random.randint(5, 15)):
            x1 = random.randint(0, image_size[0])
            y1 = random.randint(0, image_size[1])
            x2 = random.randint(0, image_size[0])
            y2 = random.randint(0, image_size[1])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x1, y1, x2, y2], fill=color)
        
        img.save(output_dir / 'authentic' / f'authentic_{i:04d}.jpg')
    
    # Generate forged images (with copy-move patterns)
    for i in tqdm(range(num_forged), desc="Generating forged images"):
        img = Image.new('RGB', image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add base pattern
        for _ in range(random.randint(5, 10)):
            x1 = random.randint(0, image_size[0])
            y1 = random.randint(0, image_size[1])
            x2 = random.randint(0, image_size[0])
            y2 = random.randint(0, image_size[1])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x1, y1, x2, y2], fill=color)
        
        # Add copy-move region (duplicate a region)
        region_size = 50
        src_x = random.randint(0, image_size[0] - region_size)
        src_y = random.randint(0, image_size[1] - region_size)
        region = img.crop((src_x, src_y, src_x + region_size, src_y + region_size))
        
        dst_x = random.randint(0, image_size[0] - region_size)
        dst_y = random.randint(0, image_size[1] - region_size)
        img.paste(region, (dst_x, dst_y))
        
        img.save(output_dir / 'forged' / f'forged_{i:04d}.jpg')
    
    print(f"Synthetic dataset created in {output_dir}")


def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model, input_size=(1, 3, 224, 224)):
    """Print model summary"""
    total_params, trainable_params = count_parameters(model)
    
    print("\n" + "="*50)
    print("Model Summary")
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*50 + "\n")

