"""
Setup script for Image Forgery Detection System
"""
from pathlib import Path
import os


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/datasets/raw",
        "data/datasets/processed",
        "models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created directory: {directory}")


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'PIL',
        'matplotlib',
        'sklearn',
        'albumentations'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'opencv-python':
                __import__('cv2')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"[OK] {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"[X] {package} is NOT installed")
    
    if missing_packages:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("\n[SUCCESS] All required packages are installed!")
        return True


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n[SUCCESS] CUDA is available!")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("\n[WARNING] CUDA is not available. Training will use CPU (slower).")
    except ImportError:
        print("\n[WARNING] PyTorch is not installed. Cannot check CUDA.")


def main():
    """Main setup function"""
    print("="*60)
    print("Image Forgery Detection System - Setup")
    print("="*60)
    
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n3. Checking CUDA...")
    check_cuda()
    
    print("\n" + "="*60)
    if deps_ok:
        print("[SUCCESS] Setup completed successfully!")
        print("\nNext steps:")
        print("1. Download and organize your dataset (see README.md)")
        print("2. Configure training settings in config.py")
        print("3. Run: python train.py")
        print("4. Or run: python example_usage.py to see examples")
    else:
        print("[WARNING] Setup completed with warnings.")
        print("Please install missing dependencies before proceeding.")
    print("="*60)


if __name__ == "__main__":
    main()

