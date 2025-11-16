# 🚀 Execution Guide - Quick Reference

## 📋 Table of Contents
1. [Setup & Installation](#setup--installation)
2. [Training Commands](#training-commands)
3. [Detection Commands](#detection-commands)
4. [Web Interface Commands](#web-interface-commands)
5. [Example Commands](#example-commands)

---

## 🔧 Setup & Installation

### Initial Setup
```bash
# Install all dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🎓 Training Commands

### Basic Training
```bash
# Train with default settings (ResNet)
python train.py
```

### Training with Custom Configuration
Edit `config.py` first, then:
```bash
python train.py
```

### Training Specific Models
Edit `config.py` and set:
```python
MODEL_NAME = 'resnet'      # Options: 'cnn', 'resnet', 'efficientnet', 'xception', 'vit'
```

Then run:
```bash
python train.py
```

### Training Output
- Model saved to: `models/best_model_<model_name>.pth`
- Training history: `logs/training_history.json`
- Training plots: `logs/training_history.png`
- Confusion matrix: `logs/test_confusion_matrix.png`

---

## 🔍 Detection Commands

### Single Image Detection
```bash
python detect.py \
    --model_path models/best_model_resnet.pth \
    --image_path path/to/your/image.jpg \
    --output_path results/detection_result.png \
    --model_name resnet
```

### Python Script Detection
```python
from detect import ForgeryDetector

# Initialize detector
detector = ForgeryDetector(
    model_path="models/best_model_resnet.pth",
    model_name="resnet"
)

# Detect forgery
results = detector.detect("your_image.jpg")

# Print results
print(f"Prediction: {results['class_name']}")
print(f"Confidence: {results['confidence']:.2%}")

# Visualize
detector.visualize_detection(
    "your_image.jpg",
    save_path="results/result.png"
)
```

### Batch Detection
```python
from detect import batch_detect, ForgeryDetector

detector = ForgeryDetector("models/best_model_resnet.pth", "resnet")

results = batch_detect(
    detector=detector,
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    output_dir="results/batch_detections"
)
```

---

## 🌐 Web Interface Commands

### Gradio Interface
```bash
# Basic launch
python app_gradio.py --model_path models/best_model_resnet.pth

# With custom port
python app_gradio.py --model_path models/best_model_resnet.pth --port 7860

# With shareable link (for remote access)
python app_gradio.py --model_path models/best_model_resnet.pth --share

# With specific model
python app_gradio.py \
    --model_path models/best_model_resnet.pth \
    --model_name resnet \
    --port 7860 \
    --share
```

**Access URL:** `http://localhost:7860`

### Streamlit Interface
```bash
# Launch Streamlit app
streamlit run app_streamlit.py

# With custom port
streamlit run app_streamlit.py --server.port 8501
```

**Access URL:** `http://localhost:8501`

**Note:** Update model path in `app_streamlit.py` or use sidebar to configure.

---

## 📝 Example Commands

### Run Example Usage Script
```bash
python example_usage.py
```

### Model Comparison
```python
python -c "
from models import create_model
from config import TrainingConfig, ModelConfig

config = TrainingConfig()
model_config = ModelConfig()

for model_name in ['cnn', 'resnet', 'efficientnet']:
    model = create_model(model_name, 2, config, model_config)
    params = sum(p.numel() for p in model.parameters())
    print(f'{model_name}: {params:,} parameters')
"
```

### Quick Test (No Dataset Required)
```python
# Test model creation
python -c "
from models import create_model
from config import TrainingConfig, ModelConfig

model = create_model('resnet', 2, TrainingConfig(), ModelConfig())
print('Model created successfully!')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

---

## 🎯 Common Workflows

### Complete Training Workflow
```bash
# 1. Setup
pip install -r requirements.txt
python setup.py

# 2. Organize dataset (see README.md for structure)
# Place images in: data/datasets/processed/train/authentic/ and data/datasets/processed/train/forged/

# 3. Train
python train.py

# 4. Detect
python detect.py --model_path models/best_model_resnet.pth --image_path test.jpg
```

### Quick Detection Workflow
```bash
# 1. Ensure model exists
ls models/best_model_resnet.pth

# 2. Run detection
python detect.py \
    --model_path models/best_model_resnet.pth \
    --image_path your_image.jpg \
    --output_path result.png
```

### Web Interface Workflow
```bash
# 1. Train model (if not already done)
python train.py

# 2. Launch Gradio
python app_gradio.py --model_path models/best_model_resnet.pth --share

# 3. Open browser to: http://localhost:7860
# 4. Upload image and click "Detect Forgery"
```

---

## 🔧 Troubleshooting Commands

### Check Dependencies
```bash
python setup.py
```

### Verify Model File
```bash
python -c "
import torch
checkpoint = torch.load('models/best_model_resnet.pth', map_location='cpu')
print('Model keys:', list(checkpoint.keys()))
"
```

### Test Data Loading
```python
python -c "
from data_loader import create_dataloaders
from config import TrainingConfig

config = TrainingConfig()
train_loader, val_loader, test_loader = create_dataloaders(
    'data/datasets/processed',
    config
)
print(f'Train: {len(train_loader.dataset)} samples')
"
```

### Check CUDA
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 📊 Command-Line Arguments Reference

### train.py
No arguments - uses `config.py` settings

### detect.py
```bash
python detect.py \
    --model_path <path>      # Required: Path to model checkpoint
    --image_path <path>      # Required: Path to image file
    --model_name <name>      # Optional: Model architecture (default: resnet)
    --output_path <path>     # Optional: Path to save visualization
    --device <device>        # Optional: cuda or cpu (default: cuda)
```

### app_gradio.py
```bash
python app_gradio.py \
    --model_path <path>      # Required: Path to model checkpoint
    --model_name <name>      # Optional: Model architecture (default: resnet)
    --port <number>          # Optional: Port number (default: 7860)
    --share                   # Optional: Create shareable link
```

### app_streamlit.py
```bash
streamlit run app_streamlit.py \
    --server.port <number>   # Optional: Port number (default: 8501)
```

---

## 🎨 Quick Start Examples

### Example 1: Train and Detect
```bash
# Train
python train.py

# Wait for training to complete, then:
python detect.py \
    --model_path models/best_model_resnet.pth \
    --image_path test_image.jpg
```

### Example 2: Web Interface Only
```bash
# If you have a trained model:
python app_gradio.py \
    --model_path models/best_model_resnet.pth \
    --share
```

### Example 3: Batch Processing
```python
# Create batch_detect.py
from detect import ForgeryDetector, batch_detect

detector = ForgeryDetector("models/best_model_resnet.pth", "resnet")
image_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = batch_detect(detector, image_list, "results/")
```

---

## 📌 Important Notes

1. **Model Path**: Always use the full path or relative path from project root
2. **Image Formats**: Supports JPG, JPEG, PNG
3. **GPU/CPU**: Automatically detects CUDA, falls back to CPU
4. **Port Conflicts**: Change port if 7860 or 8501 are in use
5. **Dataset**: Must be organized correctly (see README.md)

---

## 🔗 Quick Links

- **Full Documentation**: See `README.md`
- **Quick Start**: See `QUICKSTART.md`
- **Project Summary**: See `PROJECT_SUMMARY.md`
- **Example Code**: See `example_usage.py`

---

**Happy Detecting! 🔍**

