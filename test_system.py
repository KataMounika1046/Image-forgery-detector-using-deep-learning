"""
Quick test script to verify the system is working
Tests model creation without requiring a dataset
"""
import torch
from models import create_model
from config import TrainingConfig, ModelConfig
from utils import print_model_summary

def test_model_creation():
    """Test creating different models"""
    print("="*60)
    print("Testing Model Creation")
    print("="*60)
    
    config = TrainingConfig()
    model_config = ModelConfig()
    
    models_to_test = ['cnn', 'resnet', 'efficientnet']
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name.upper()}...")
        try:
            model = create_model(
                model_name=model_name,
                num_classes=2,
                config=config,
                model_config=model_config
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  [OK] Model created successfully!")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
                if isinstance(output, tuple):
                    output = output[0]
            
            print(f"  [OK] Forward pass successful!")
            print(f"  Output shape: {output.shape}")
            
        except Exception as e:
            print(f"  [ERROR] Failed: {str(e)}")
    
    print("\n" + "="*60)
    print("Model Creation Test Complete!")
    print("="*60)

def test_resnet_detailed():
    """Test ResNet model in detail"""
    print("\n" + "="*60)
    print("Detailed ResNet Test")
    print("="*60)
    
    config = TrainingConfig()
    model_config = ModelConfig()
    
    model = create_model('resnet', 2, config, model_config)
    print_model_summary(model)
    
    # Test with dummy batch
    print("\nTesting with dummy batch...")
    dummy_batch = torch.randn(4, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_batch)
        if isinstance(output, tuple):
            output, attention = output
            print(f"[OK] Output shape: {output.shape}")
            print(f"[OK] Attention map shape: {attention.shape}")
        else:
            print(f"[OK] Output shape: {output.shape}")
        
        # Check predictions
        probs = torch.softmax(output, dim=1)
        preds = torch.argmax(output, dim=1)
        print(f"[OK] Predictions: {preds.tolist()}")
        print(f"[OK] Probabilities shape: {probs.shape}")
    
    print("\n[SUCCESS] All tests passed!")
    print("="*60)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Image Forgery Detection System - System Test")
    print("="*60)
    
    # Test model creation
    test_model_creation()
    
    # Detailed ResNet test
    test_resnet_detailed()
    
    print("\n" + "="*60)
    print("System is ready to use!")
    print("="*60)
    print("\nNext steps:")
    print("1. Organize your dataset in: data/datasets/processed/")
    print("2. Run: python train.py")
    print("3. After training, run: python detect.py --model_path models/best_model_resnet.pth --image_path your_image.jpg")
    print("="*60)

