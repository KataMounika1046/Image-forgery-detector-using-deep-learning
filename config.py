"""
Configuration file for Image Forgery Detection System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset paths
DATASET_DIR = DATA_DIR / "datasets"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

# Training configuration
class TrainingConfig:
    # Model selection: 'cnn', 'resnet', 'efficientnet', 'xception', 'vit'
    MODEL_NAME = 'resnet'
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Data augmentation
    USE_AUGMENTATION = True
    AUGMENTATION_PROB = 0.5
    
    # Transfer learning
    USE_PRETRAINED = True
    FREEZE_BACKBONE = False
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Model saving
    SAVE_BEST_MODEL = True
    SAVE_CHECKPOINTS = True
    
    # Device
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    # Logging
    USE_WANDB = False
    WANDB_PROJECT = "image-forgery-detection"

# Model-specific configurations
class ModelConfig:
    # ResNet
    RESNET_DEPTH = 50  # 18, 34, 50, 101, 152
    
    # EfficientNet
    EFFICIENTNET_VARIANT = 'b0'  # b0-b7
    
    # Vision Transformer
    VIT_PATCH_SIZE = 16
    VIT_DIM = 768
    VIT_DEPTH = 12
    VIT_HEADS = 12
    
    # CNN
    CNN_FILTERS = [32, 64, 128, 256]
    CNN_DROPOUT = 0.5

