# 🔍 Image Forgery Detection System

A state-of-the-art deep learning system for detecting manipulated images, including splicing, copy-move, and other common forgeries.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset Information](#dataset-information)
- [Quick Start](#quick-start)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Evaluation](#evaluation)
- [Detection](#detection)
- [Web Interface](#web-interface)
- [Performance Improvement Techniques](#performance-improvement-techniques)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Multiple Model Architectures**: CNN, ResNet, EfficientNet, Xception, Vision Transformer
- **Transfer Learning**: Pre-trained models from ImageNet
- **Attention Mechanisms**: Visualize manipulated regions
- **Data Augmentation**: Robust training with various augmentations
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Web Interfaces**: Gradio and Streamlit GUIs
- **Easy to Use**: Simple API for detection and visualization

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Step 1: Clone or Download the Project

```bash
cd aimlproject
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 Dataset Information

### Recommended Datasets

#### 1. **CASIA v1.0 & v2.0**
- **Description**: Classic datasets with splicing and copy-move forgeries
- **URL**: http://forensics.idealtest.org/
- **Size**: v1.0 ~1.2GB, v2.0 ~4.5GB
- **Classes**: Authentic, Splicing, Copy-Move
- **Note**: Requires registration
- **Preprocessing**: Organize into `train/val/test` splits with `authentic` and `forged` subdirectories

#### 2. **COVERAGE**
- **Description**: Copy-move forgery detection dataset
- **URL**: https://github.com/wenbihan/coverage
- **Size**: ~500MB
- **Classes**: Authentic, Copy-Move
- **Preprocessing**: 
  ```bash
  git clone https://github.com/wenbihan/coverage.git
  # Organize images into authentic/forged folders
  ```

#### 3. **Columbia Uncompressed Image Splicing Detection**
- **Description**: High-quality uncompressed images
- **URL**: http://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm
- **Size**: ~1GB
- **Classes**: Authentic, Splicing
- **Note**: Academic use

#### 4. **NIST Nimble 2017**
- **Description**: Large-scale dataset with various manipulations
- **URL**: https://www.nist.gov/itl/iad/mig/nimble-challenge-2017-evaluation
- **Size**: ~10GB
- **Classes**: Authentic, Various Forgeries

#### 5. **WildWeb**
- **Description**: Real-world web images
- **URL**: https://github.com/MKLab-ITI/image-forensics
- **Size**: ~2GB
- **Classes**: Authentic, Forged

### Dataset Organization

Organize your dataset in the following structure:

```
data/datasets/processed/
├── train/
│   ├── authentic/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── forged/
│       ├── img1.jpg
│       └── img2.jpg
├── val/
│   ├── authentic/
│   └── forged/
└── test/
    ├── authentic/
    └── forged/
```

### Dataset Preprocessing

Use the utility function to organize your dataset:

```python
from utils import organize_dataset
from pathlib import Path

# Organize dataset into train/val/test splits
organize_dataset(
    source_dir=Path("data/datasets/raw"),
    target_dir=Path("data/datasets/processed"),
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

## 🏃 Quick Start

### 1. Prepare Your Dataset

```bash
# Download and organize your dataset
# See Dataset Information section above
```

### 2. Configure Training

Edit `config.py` to adjust hyperparameters:

```python
class TrainingConfig:
    MODEL_NAME = 'resnet'  # Options: 'cnn', 'resnet', 'efficientnet', 'xception', 'vit'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    USE_PRETRAINED = True
    USE_AUGMENTATION = True
    # ... more options
```

### 3. Train the Model

```bash
python train.py
```

### 4. Detect Forgeries

```bash
python detect.py --model_path models/best_model_resnet.pth --image_path path/to/image.jpg
```

### 5. Launch Web Interface

**Gradio:**
```bash
python app_gradio.py --model_path models/best_model_resnet.pth --model_name resnet
```

**Streamlit:**
```bash
streamlit run app_streamlit.py
```

## 🏗️ Model Architectures

### 1. Simple CNN
- Lightweight architecture
- Good for quick experiments
- Customizable filter sizes

### 2. ResNet
- Deep residual networks
- Excellent feature extraction
- Attention mechanism included
- Variants: ResNet-18, 34, 50, 101, 152

### 3. EfficientNet
- Efficient and accurate
- Multiple variants (B0-B7)
- Good balance of speed and accuracy

### 4. Xception
- Extreme Inception architecture
- Strong feature representation
- Good for fine-grained detection

### 5. Vision Transformer (ViT)
- Transformer-based architecture
- State-of-the-art performance
- Requires more data and computation

## 🎓 Training

### Basic Training

```python
from train import train
from models import create_model
from data_loader import create_dataloaders
from config import TrainingConfig, ModelConfig

# Load configuration
config = TrainingConfig()
model_config = ModelConfig()

# Create model
model = create_model(
    model_name='resnet',
    num_classes=2,
    config=config,
    model_config=model_config
)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="data/datasets/processed",
    config=config
)

# Train
trained_model, history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)
```

### Training Options

- **Transfer Learning**: Use `USE_PRETRAINED=True` in config
- **Freeze Backbone**: Set `FREEZE_BACKBONE=True` for fine-tuning only classifier
- **Data Augmentation**: Enable with `USE_AUGMENTATION=True`
- **Early Stopping**: Configured with `EARLY_STOPPING_PATIENCE`
- **Learning Rate Scheduling**: Automatic reduction on plateau

### Monitoring Training

Training history is saved to `logs/training_history.json` and visualized in `logs/training_history.png`.

## 📈 Evaluation

### Calculate Metrics

```python
from metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve
import numpy as np

# After validation/test
metrics = calculate_metrics(
    y_true=test_labels,
    y_pred=test_preds,
    y_probs=test_probs,
    num_classes=2
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
print(f"AUC-ROC: {metrics['auc']:.4f}")

# Plot confusion matrix
plot_confusion_matrix(
    test_labels, test_preds,
    class_names=['Authentic', 'Forged'],
    save_path="results/confusion_matrix.png"
)

# Plot ROC curve
plot_roc_curve(
    test_labels, test_probs,
    save_path="results/roc_curve.png"
)
```

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Average Precision**: Area under the precision-recall curve

## 🔍 Detection

### Single Image Detection

```python
from detect import ForgeryDetector

# Initialize detector
detector = ForgeryDetector(
    model_path="models/best_model_resnet.pth",
    model_name="resnet"
)

# Detect forgery
results = detector.detect("path/to/image.jpg")

print(f"Prediction: {results['class_name']}")
print(f"Confidence: {results['confidence']:.2%}")
print(f"Probabilities: {results['probabilities']}")

# Visualize
detector.visualize_detection(
    "path/to/image.jpg",
    save_path="results/detection_result.png"
)
```

### Batch Detection

```python
from detect import batch_detect

results = batch_detect(
    detector=detector,
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    output_dir="results/batch_detections"
)
```

### Command Line

```bash
python detect.py \
    --model_path models/best_model_resnet.pth \
    --image_path test_image.jpg \
    --model_name resnet \
    --output_path result.png
```

## 🌐 Web Interface

### Gradio Interface

```bash
python app_gradio.py \
    --model_path models/best_model_resnet.pth \
    --model_name resnet \
    --port 7860 \
    --share  # Optional: create shareable link
```

Access at: `http://localhost:7860`

### Streamlit Interface

```bash
streamlit run app_streamlit.py
```

Access at: `http://localhost:8501`

Features:
- Upload images via drag-and-drop
- Real-time detection
- Visualization with attention maps
- Download results
- Model selection in sidebar

## 🚀 Performance Improvement Techniques

### 1. Transfer Learning

```python
# Use pretrained models
config.USE_PRETRAINED = True
config.FREEZE_BACKBONE = False  # Fine-tune entire model
# or
config.FREEZE_BACKBONE = True   # Fine-tune only classifier
```

### 2. Data Augmentation

Already enabled by default. Customize in `data_loader.py`:

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(...),
    A.RandomBrightnessContrast(...),
    # Add more augmentations
])
```

### 3. Attention Mechanisms

ResNet model includes attention mechanism. Visualize attention maps:

```python
results = detector.detect(image_path, return_attention=True)
# Attention map available in results['attention_map']
```

### 4. Ensemble Methods

Combine predictions from multiple models:

```python
models = ['resnet', 'efficientnet', 'xception']
predictions = []

for model_name in models:
    detector = ForgeryDetector(f"models/best_model_{model_name}.pth", model_name)
    results = detector.detect(image_path)
    predictions.append(results['probabilities'])

# Average predictions
ensemble_probs = np.mean(predictions, axis=0)
```

### 5. Hyperparameter Tuning

Experiment with:
- Learning rates: [0.0001, 0.001, 0.01]
- Batch sizes: [16, 32, 64]
- Model depths: ResNet-50, ResNet-101
- Dropout rates: [0.3, 0.5, 0.7]

### 6. Advanced Techniques

- **Focal Loss**: For imbalanced datasets
- **Mixup/CutMix**: Advanced augmentation
- **Test Time Augmentation**: Multiple predictions per image
- **Multi-scale Training**: Train on multiple image sizes

## 📁 Project Structure

```
aimlproject/
├── config.py                 # Configuration settings
├── data_loader.py            # Dataset loading and preprocessing
├── models.py                 # Model architectures
├── train.py                  # Training script
├── metrics.py                # Evaluation metrics
├── detect.py                 # Detection and visualization
├── app_gradio.py             # Gradio web interface
├── app_streamlit.py          # Streamlit web interface
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/
│   └── datasets/
│       ├── raw/              # Raw dataset
│       └── processed/        # Processed dataset (train/val/test)
│
├── models/                   # Saved model checkpoints
│   ├── best_model_resnet.pth
│   └── checkpoint_epoch_*.pth
│
├── results/                  # Detection results and visualizations
│   ├── detection_result.png
│   └── batch_detections/
│
└── logs/                     # Training logs and plots
    ├── training_history.json
    ├── training_history.png
    └── test_confusion_matrix.png
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `config.py`
   - Use smaller model (ResNet-18 instead of ResNet-50)
   - Enable gradient checkpointing

2. **Dataset Not Found**
   - Check dataset path in `train.py`
   - Ensure proper directory structure
   - Verify image file formats (JPG, PNG)

3. **Poor Performance**
   - Increase training epochs
   - Use data augmentation
   - Try different model architectures
   - Check dataset quality and balance

4. **Model Loading Errors**
   - Verify model path
   - Check model architecture matches
   - Ensure PyTorch version compatibility

### Getting Help

- Check logs in `logs/` directory
- Review training history plots
- Verify dataset organization
- Test with single image first

## 📝 Example Usage

### Complete Training Pipeline

```python
# 1. Organize dataset
from utils import organize_dataset
organize_dataset(Path("raw_data"), Path("processed_data"))

# 2. Train model
python train.py

# 3. Evaluate
# Metrics automatically calculated during training

# 4. Detect forgeries
python detect.py --model_path models/best_model_resnet.pth --image_path test.jpg

# 5. Launch web interface
python app_gradio.py --model_path models/best_model_resnet.pth
```

## 🎯 Best Practices

1. **Data Quality**: Use high-quality, diverse datasets
2. **Data Balance**: Ensure balanced authentic/forged samples
3. **Validation**: Always use separate validation set
4. **Regularization**: Use dropout and weight decay
5. **Monitoring**: Track training metrics closely
6. **Experimentation**: Try multiple architectures
7. **Ensemble**: Combine multiple models for better accuracy

## 📚 References

- CASIA Dataset: http://forensics.idealtest.org/
- COVERAGE Dataset: https://github.com/wenbihan/coverage
- PyTorch Documentation: https://pytorch.org/docs/
- Albumentations: https://albumentations.ai/

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements.

---

**Happy Detecting! 🔍**

