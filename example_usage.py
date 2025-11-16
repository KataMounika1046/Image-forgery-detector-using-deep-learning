"""
Example usage script for Image Forgery Detection System
This script demonstrates how to use the system for training and detection
"""
import torch
from pathlib import Path
from config import TrainingConfig, ModelConfig
from models import create_model
from data_loader import create_dataloaders
from detect import ForgeryDetector
from utils import print_model_summary


def example_training():
    """Example: Train a model"""
    print("="*60)
    print("Example: Training a Model")
    print("="*60)
    
    # Load configuration
    config = TrainingConfig()
    model_config = ModelConfig()
    
    # Set model name
    config.MODEL_NAME = 'resnet'  # Options: 'cnn', 'resnet', 'efficientnet', 'xception', 'vit'
    config.BATCH_SIZE = 32
    config.NUM_EPOCHS = 10  # Reduced for example
    config.USE_PRETRAINED = True
    config.USE_AUGMENTATION = True
    
    # Dataset path (update this to your dataset)
    data_dir = "data/datasets/processed"
    
    if not Path(data_dir).exists():
        print(f"\n⚠️  Dataset directory '{data_dir}' not found!")
        print("Please organize your dataset first. See README.md for instructions.")
        return
    
    # Create model
    num_classes = 2
    model = create_model(
        model_name=config.MODEL_NAME,
        num_classes=num_classes,
        config=config,
        model_config=model_config
    )
    
    # Print model summary
    print_model_summary(model)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        config=config,
        is_binary=True
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Train (uncomment to actually train)
    # from train import train
    # trained_model, history = train(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     config=config,
    #     num_classes=num_classes
    # )
    
    print("\n✅ Training setup complete!")
    print("Uncomment the training code in example_training() to start training.")


def example_detection():
    """Example: Detect forgeries in images"""
    print("\n" + "="*60)
    print("Example: Detecting Forgeries")
    print("="*60)
    
    # Model path (update this to your trained model)
    model_path = "models/best_model_resnet.pth"
    
    if not Path(model_path).exists():
        print(f"\n⚠️  Model file '{model_path}' not found!")
        print("Please train a model first or provide a valid model path.")
        return
    
    # Create detector
    detector = ForgeryDetector(
        model_path=model_path,
        model_name='resnet',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example image path (update this)
    image_path = "test_image.jpg"
    
    if not Path(image_path).exists():
        print(f"\n⚠️  Image file '{image_path}' not found!")
        print("Please provide a valid image path.")
        return
    
    # Detect
    print(f"\nAnalyzing image: {image_path}")
    results = detector.detect(image_path, return_attention=True)
    
    # Print results
    print("\n" + "-"*60)
    print("Detection Results:")
    print("-"*60)
    print(f"Prediction: {results['class_name']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"\nProbabilities:")
    print(f"  Authentic: {results['probabilities']['authentic']:.2%}")
    print(f"  Forged: {results['probabilities']['forged']:.2%}")
    
    # Visualize
    output_path = "results/example_detection.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    detector.visualize_detection(
        image_path,
        save_path=output_path,
        show_attention=True
    )
    
    print(f"\n✅ Visualization saved to: {output_path}")


def example_batch_detection():
    """Example: Batch detection on multiple images"""
    print("\n" + "="*60)
    print("Example: Batch Detection")
    print("="*60)
    
    model_path = "models/best_model_resnet.pth"
    
    if not Path(model_path).exists():
        print(f"\n⚠️  Model file '{model_path}' not found!")
        return
    
    detector = ForgeryDetector(
        model_path=model_path,
        model_name='resnet'
    )
    
    # List of image paths
    image_paths = [
        "test_image1.jpg",
        "test_image2.jpg",
        "test_image3.jpg"
    ]
    
    # Filter existing images
    existing_images = [p for p in image_paths if Path(p).exists()]
    
    if not existing_images:
        print("\n⚠️  No valid image files found!")
        return
    
    print(f"\nProcessing {len(existing_images)} images...")
    
    # Batch detect
    from detect import batch_detect
    
    results = batch_detect(
        detector=detector,
        image_paths=existing_images,
        output_dir="results/batch_detections"
    )
    
    # Print summary
    print("\n" + "-"*60)
    print("Batch Detection Summary:")
    print("-"*60)
    for result in results:
        if 'error' not in result:
            print(f"\n{Path(result['image_path']).name}:")
            print(f"  Prediction: {result['class_name']}")
            print(f"  Confidence: {result['confidence']:.2%}")
    
    print(f"\n✅ Results saved to: results/batch_detections/")


def example_model_comparison():
    """Example: Compare different model architectures"""
    print("\n" + "="*60)
    print("Example: Model Comparison")
    print("="*60)
    
    config = TrainingConfig()
    model_config = ModelConfig()
    
    models_to_compare = ['cnn', 'resnet', 'efficientnet']
    
    print("\nModel Architectures:")
    print("-"*60)
    
    for model_name in models_to_compare:
        model = create_model(
            model_name=model_name,
            num_classes=2,
            config=config,
            model_config=model_config
        )
        
        total_params, trainable_params = sum(p.numel() for p in model.parameters()), \
                                         sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n{model_name.upper()}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Image Forgery Detection System - Example Usage")
    print("="*60)
    
    # Example 1: Training setup
    example_training()
    
    # Example 2: Single image detection
    # example_detection()
    
    # Example 3: Batch detection
    # example_batch_detection()
    
    # Example 4: Model comparison
    example_model_comparison()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nFor more information, see README.md")
    print("To train a model, run: python train.py")
    print("To detect forgeries, run: python detect.py --model_path <path> --image_path <path>")
    print("To launch web interface, run: python app_gradio.py --model_path <path>")


if __name__ == "__main__":
    main()

