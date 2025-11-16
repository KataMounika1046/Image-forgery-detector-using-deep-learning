# 🌐 Web Interface Guide

## 🚀 How to Launch

### Option 1: Double-click (Easiest)
1. Double-click `START_WEB.bat` in the project folder
2. Wait for browser to open automatically
3. If browser doesn't open, go to: **http://localhost:7860**

### Option 2: Command Line
```bash
python launch_web.py
```
Then open your browser and go to: **http://localhost:7860**

---

## 📤 How to Use

1. **Upload Image**
   - Click the upload area or drag & drop an image
   - Supports: JPG, JPEG, PNG formats

2. **Detect Forgery**
   - Click the "🔍 Detect Forgery" button
   - Or the detection runs automatically when you upload

3. **View Results**
   - See the prediction: **Authentic** or **Forged**
   - Check confidence percentage
   - View detailed analysis

---

## ⚠️ Demo Mode vs Real Detection

### Demo Mode (Current)
- Works immediately without training
- Shows interface structure
- Uses basic heuristics (not accurate)
- Shows "DEMO MODE" in results

### Real Detection (After Training)
1. Train a model first:
   ```bash
   python train.py
   ```
2. Launch the interface:
   ```bash
   python launch_web.py
   ```
3. The interface will automatically use your trained model
4. Shows "REAL MODEL DETECTION" in results

---

## 🔧 Troubleshooting

### Port Already in Use
If port 7860 is busy:
- Close other applications using that port
- Or edit `launch_web.py` and change `server_port=7860` to another number

### Browser Doesn't Open
- Manually go to: **http://localhost:7860**
- Or try: **http://127.0.0.1:7860**

### Interface Not Loading
1. Check if Python is running the script
2. Look for error messages in the terminal
3. Make sure Gradio is installed: `pip install gradio`

---

## 📝 Quick Commands

```bash
# Launch web interface
python launch_web.py

# Train model (for real detection)
python train.py

# Stop the server
Press Ctrl+C in the terminal
```

---

## 🎯 What You'll See

- **Left Side**: Upload area for images
- **Right Side**: Detection results with overlay
- **Bottom**: Detailed analysis text

Just upload any image and click "Detect Forgery" to test!

---

**Happy Detecting! 🔍**

