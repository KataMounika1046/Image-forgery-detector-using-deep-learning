"""
Forgery detection and visualization utilities
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import TrainingConfig, ModelConfig
from models import create_model


class ForgeryDetector:
    """
    Class for detecting forgeries in images
    """
    def __init__(
        self,
        model_path: str,
        model_name: str = 'resnet',
        device: str = 'cuda',
        num_classes: int = 2
    ):
        """
        Initialize the forgery detector
        
        Args:
            model_path: Path to trained model checkpoint
            model_name: Name of the model architecture
            device: Device to run inference on
            num_classes: Number of classes
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup transforms
        config = TrainingConfig()
        self.transform = A.Compose([
            A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
            A.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            ),
            ToTensorV2()
        ])
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        config = TrainingConfig()
        model_config = ModelConfig()
        
        # Create model
        model = create_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            config=config,
            model_config=model_config
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        return model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return image_tensor, image, original_shape
    
    def detect(
        self,
        image_path: str,
        return_attention: bool = False
    ) -> dict:
        """
        Detect forgery in an image
        
        Args:
            image_path: Path to image file
            return_attention: Whether to return attention map (if available)
        
        Returns:
            Dictionary with detection results
        """
        # Preprocess
        image_tensor, original_image, original_shape = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # Handle models that return attention maps
            attention_map = None
            if isinstance(output, tuple):
                output, attention_map = output
                if not return_attention:
                    attention_map = None
            
            # Get predictions
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
        
        # Prepare results
        results = {
            'image_path': image_path,
            'predicted_class': pred_class,
            'class_name': 'Forged' if pred_class == 1 else 'Authentic',
            'confidence': float(confidence),
            'probabilities': {
                'authentic': float(probs[0]),
                'forged': float(probs[1]) if len(probs) > 1 else 0.0
            },
            'original_image': original_image,
            'original_shape': original_shape,
            'attention_map': attention_map
        }
        
        return results
    
    def visualize_detection(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        show_attention: bool = True
    ) -> np.ndarray:
        """
        Visualize forgery detection results
        
        Args:
            image_path: Path to image file
            save_path: Optional path to save visualization
            show_attention: Whether to overlay attention map
        
        Returns:
            Visualization image
        """
        # Detect
        results = self.detect(image_path, return_attention=show_attention)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2 if show_attention and results['attention_map'] is not None else 1, 
                                figsize=(15, 5))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Original image with prediction
        ax = axes[0]
        image = results['original_image']
        ax.imshow(image)
        ax.axis('off')
        
        # Add prediction text
        pred_text = f"{results['class_name']} ({results['confidence']:.2%})"
        color = 'red' if results['predicted_class'] == 1 else 'green'
        ax.text(
            10, 30, pred_text,
            fontsize=16, color='white',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
        )
        ax.set_title('Forgery Detection Result', fontsize=14, fontweight='bold')
        
        # Attention map if available
        if show_attention and results['attention_map'] is not None and len(axes) > 1:
            ax = axes[1]
            attention = results['attention_map']
            
            # Resize attention map to original image size
            attention_np = attention[0, 0].cpu().numpy()
            attention_resized = cv2.resize(
                attention_np,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Overlay attention on original image
            ax.imshow(image)
            im = ax.imshow(attention_resized, alpha=0.5, cmap='hot')
            ax.axis('off')
            ax.set_title('Attention Map (Manipulated Regions)', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return results


def batch_detect(
    detector: ForgeryDetector,
    image_paths: List[str],
    output_dir: Optional[str] = None
) -> List[dict]:
    """
    Detect forgeries in multiple images
    
    Args:
        detector: ForgeryDetector instance
        image_paths: List of image paths
        output_dir: Optional directory to save visualizations
    
    Returns:
        List of detection results
    """
    results = []
    
    for image_path in image_paths:
        try:
            result = detector.detect(image_path)
            results.append(result)
            
            if output_dir:
                save_path = Path(output_dir) / f"detection_{Path(image_path).stem}.png"
                detector.visualize_detection(image_path, save_path=str(save_path))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect forgeries in images')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image to analyze')
    parser.add_argument('--model_name', type=str, default='resnet',
                       choices=['cnn', 'resnet', 'efficientnet', 'xception', 'vit'],
                       help='Model architecture name')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Create detector
    detector = ForgeryDetector(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device
    )
    
    # Detect
    results = detector.detect(args.image_path)
    
    print("\n" + "="*50)
    print("Forgery Detection Results")
    print("="*50)
    print(f"Image: {results['image_path']}")
    print(f"Prediction: {results['class_name']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"Probabilities:")
    print(f"  Authentic: {results['probabilities']['authentic']:.2%}")
    print(f"  Forged: {results['probabilities']['forged']:.2%}")
    
    # Visualize
    detector.visualize_detection(
        args.image_path,
        save_path=args.output_path
    )


if __name__ == "__main__":
    main()

