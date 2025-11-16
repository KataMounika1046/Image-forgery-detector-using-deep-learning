"""
Training script for Image Forgery Detection models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from config import TrainingConfig, MODELS_DIR, LOGS_DIR
from models import create_model
from data_loader import create_dataloaders
from metrics import calculate_metrics, plot_confusion_matrix, plot_training_history


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current, best):
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle models that return attention maps
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Handle models that return attention maps
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_probs, all_labels


def train(
    model,
    train_loader,
    val_loader,
    config: TrainingConfig,
    num_classes: int = 2
):
    """Main training function"""
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_probs, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Calculate metrics
        metrics = calculate_metrics(
            val_labels, val_preds, val_probs, num_classes=num_classes
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(metrics['f1'])
        history['val_auc'].append(metrics['auc'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val F1: {metrics['f1']:.4f}, Val AUC: {metrics['auc']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
            if config.SAVE_BEST_MODEL:
                model_path = MODELS_DIR / f"best_model_{config.MODEL_NAME}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': config.__dict__
                }, model_path)
                print(f"Saved best model to {model_path}")
        
        # Save checkpoint
        if config.SAVE_CHECKPOINTS:
            checkpoint_path = MODELS_DIR / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, checkpoint_path)
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save training history
    history_path = LOGS_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training history
    plot_training_history(history, save_path=LOGS_DIR / "training_history.png")
    
    return model, history


def main():
    """Main training function"""
    from config import TrainingConfig, ModelConfig
    
    config = TrainingConfig()
    model_config = ModelConfig()
    
    # Dataset path (update this to your dataset path)
    data_dir = "data/datasets/processed"  # Update this path
    
    # Check if dataset exists
    if not Path(data_dir).exists():
        print(f"Warning: Dataset directory {data_dir} does not exist.")
        print("Please download and organize your dataset first.")
        print("See README.md for dataset information.")
        return
    
    # Create model
    num_classes = 2  # Binary classification (authentic vs forged)
    model = create_model(
        model_name=config.MODEL_NAME,
        num_classes=num_classes,
        config=config,
        model_config=model_config
    )
    
    print(f"Created {config.MODEL_NAME} model")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        config=config,
        is_binary=True
    )
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val: {len(val_loader.dataset)}")
    print(f"Test: {len(test_loader.dataset)}")
    
    # Train
    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        num_classes=num_classes
    )
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on Test Set")
    print("="*50)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_acc, test_preds, test_probs, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    test_metrics = calculate_metrics(
        test_labels, test_preds, test_probs, num_classes=num_classes
    )
    
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds,
        class_names=['Authentic', 'Forged'],
        save_path=LOGS_DIR / "test_confusion_matrix.png"
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

