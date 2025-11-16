# Quick Start Guide

Get started with Image Forgery Detection in 5 minutes!

## Step 1: Installation (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

## Step 2: Prepare Dataset (2 minutes)

### Option A: Use Your Own Dataset

Organize your images in this structure:

```
data/datasets/processed/
├── train/
│   ├── authentic/  (put authentic images here)
│   └── forged/      (put forged images here)
├── val/
│   ├── authentic/
│   └── forged/
└── test/
    ├── authentic/
    └── forged/
```

### Option B: Download Public Dataset

1. **CASIA v2.0** (Recommended):
   - Visit: http://forensics.idealtest.org/
   - Register and download
   - Organize into the structure above

2. **COVERAGE** (Copy-Move):
   ```bash
   git clone https://github.com/wenbihan/coverage.git
   # Organize images into authentic/forged folders
   ```

## Step 3: Configure (30 seconds)

Edit `config.py` if needed:

```python
class TrainingConfig:
    MODEL_NAME = 'resnet'  # Start with ResNet
    BATCH_SIZE = 32        # Reduce if out of memory
    NUM_EPOCHS = 50
    USE_PRETRAINED = True  # Use transfer learning
```

## Step 4: Train (Variable time)

```bash
python train.py
```

Training will:
- Load your dataset
- Train the model
- Save best model to `models/best_model_resnet.pth`
- Generate training plots in `logs/`

**Tip**: Start with fewer epochs (e.g., 10) to test your setup first.

## Step 5: Detect Forgeries (1 minute)

```bash
python detect.py \
    --model_path models/best_model_resnet.pth \
    --image_path your_image.jpg \
    --output_path result.png
```

Or use Python:

```python
from detect import ForgeryDetector

detector = ForgeryDetector(
    model_path="models/best_model_resnet.pth",
    model_name="resnet"
)

results = detector.detect("your_image.jpg")
print(f"Prediction: {results['class_name']}")
print(f"Confidence: {results['confidence']:.2%}")
```

## Step 6: Launch Web Interface (Optional)

**Gradio:**
```bash
python app_gradio.py --model_path models/best_model_resnet.pth
```

**Streamlit:**
```bash
streamlit run app_streamlit.py
```

## Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: Reduce `BATCH_SIZE` in `config.py` (try 16 or 8)

### Issue: "Dataset not found"
**Solution**: Check your dataset path in `train.py` or use absolute path

### Issue: "Model file not found"
**Solution**: Train a model first or download a pretrained model

### Issue: Poor accuracy
**Solutions**:
- Use more training data
- Train for more epochs
- Try different model (EfficientNet, Xception)
- Enable data augmentation
- Check dataset quality

## Next Steps

1. **Experiment with models**: Try EfficientNet, Xception, or ViT
2. **Tune hyperparameters**: Learning rate, batch size, etc.
3. **Add more data**: More data = better performance
4. **Try ensemble**: Combine multiple models
5. **Read full README**: For advanced features

## Example Workflow

```python
# 1. Organize dataset
from utils import organize_dataset
from pathlib import Path
organize_dataset(Path("raw_data"), Path("processed_data"))

# 2. Train
# python train.py

# 3. Detect
from detect import ForgeryDetector
detector = ForgeryDetector("models/best_model_resnet.pth", "resnet")
results = detector.detect("test.jpg")
detector.visualize_detection("test.jpg", "result.png")
```

## Need Help?

- Check `README.md` for detailed documentation
- Run `python example_usage.py` for code examples
- Review training logs in `logs/` directory

---

**Happy Detecting! 🔍**

