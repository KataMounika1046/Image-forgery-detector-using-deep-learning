"""
Evaluation metrics for Image Forgery Detection
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from pathlib import Path
from typing import List, Optional


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 2
) -> dict:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (for AUC calculation)
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics for binary classification
    if num_classes == 2:
        metrics['precision_authentic'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics['recall_authentic'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics['f1_authentic'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        
        metrics['precision_forged'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['recall_forged'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['f1_forged'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # AUC-ROC
    if y_probs is not None:
        if num_classes == 2:
            # Binary classification
            metrics['auc'] = roc_auc_score(y_true, y_probs[:, 1])
            metrics['ap'] = average_precision_score(y_true, y_probs[:, 1])
        else:
            # Multi-class
            try:
                metrics['auc'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='weighted'
                )
            except:
                metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
        metrics['ap'] = 0.0
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6)
):
    """
    Plot confusion matrix
    """
    if class_names is None:
        class_names = ['Authentic', 'Forged']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6)
):
    """
    Plot ROC curve
    """
    if y_probs.shape[1] == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        auc_score = roc_auc_score(y_true, y_probs[:, 1])
    else:
        # Multi-class (one-vs-rest)
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1], pos_label=1)
        auc_score = roc_auc_score(y_true, y_probs[:, 1])
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6)
):
    """
    Plot Precision-Recall curve
    """
    if y_probs.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        ap_score = average_precision_score(y_true, y_probs[:, 1])
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1], pos_label=1)
        ap_score = average_precision_score(y_true, y_probs[:, 1])
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history: dict,
    save_path: Optional[Path] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot training history (loss and accuracy curves)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 and AUC
    if 'val_f1' in history and 'val_auc' in history:
        axes[2].plot(epochs, history['val_f1'], 'g-', label='Val F1')
        axes[2].plot(epochs, history['val_auc'], 'm-', label='Val AUC')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Score')
        axes[2].set_title('Validation F1 and AUC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
):
    """
    Print detailed classification report
    """
    if class_names is None:
        class_names = ['Authentic', 'Forged']
    
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=False
    )
    print(report)
    return report

