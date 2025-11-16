"""
Launch the Image Forgery Detection Web Interface
This will open in your browser automatically
"""
import gradio as gr
import numpy as np
from PIL import Image
import cv2
import torch
from pathlib import Path
import tempfile

def detect_forgery_demo(image):
    """
    Detect forgery in uploaded image
    This is a demo version that works without a trained model
    """
    if image is None:
        return None, "Please upload an image first"
    
    # Convert to numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Create result image
    result_img = img_array.copy()
    h, w = result_img.shape[:2]
    
    # Check if we have a trained model
    model_path = Path("models/best_model_resnet.pth")
    
    if model_path.exists():
        # Use real model
        try:
            from detect import ForgeryDetector
            
            # Save temp image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cv2.imwrite(tmp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            
            # Detect
            detector = ForgeryDetector(str(model_path), 'resnet')
            results = detector.detect(tmp_path, return_attention=True)
            
            # Create visualization
            pred_class = results['class_name']
            confidence = results['confidence']
            
            # Convert to simple Real/Fake
            if pred_class == "Authentic":
                display_text = "REAL"
                color = (0, 255, 0)  # Green
            else:
                display_text = "FAKE"
                color = (0, 0, 255)  # Red
            
            # Add text overlay
            cv2.rectangle(result_img, (10, 10), (400, 120), color, -1)
            cv2.putText(result_img, display_text, (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            cv2.putText(result_img, f"{confidence:.1%}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add attention map if available
            if results['attention_map'] is not None:
                attention = results['attention_map'][0, 0].cpu().numpy()
                attention_resized = cv2.resize(attention, (w, h))
                attention_colored = cv2.applyColorMap(
                    (attention_resized * 255).astype(np.uint8),
                    cv2.COLORMAP_HOT
                )
                result_img = cv2.addWeighted(
                    result_img, 0.7,
                    cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB), 0.3, 0
                )
            
            # Simple Real/Fake display
            if pred_class == "Authentic":
                result_text = "REAL"
            else:
                result_text = "FAKE"
            
            detail_text = f"""
RESULT: {result_text}
Confidence: {confidence:.1%}

Real: {results['probabilities']['authentic']:.1%}
Fake: {results['probabilities']['forged']:.1%}

Analyzed using trained deep learning model.
            """.strip()
            
            return result_img, detail_text
            
        except Exception as e:
            # Fall back to demo if model fails
            pass
    
    # DEMO MODE - Simulate detection
    # Simple heuristic-based demo (not real detection)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Simple demo: check for common forgery signs (very basic)
    variance = np.var(img_gray)
    mean_brightness = np.mean(img_gray)
    
    # Demo logic (not real detection - just for interface demo)
    # Simple heuristic for demo
    if variance < 500 or mean_brightness < 50 or mean_brightness > 200:
        result_text = "FAKE"
        conf = 0.65
        color = (0, 0, 255)  # Red
    else:
        result_text = "REAL"
        conf = 0.75
        color = (0, 255, 0)  # Green
    
    # Add overlay - BIG and CLEAR
    cv2.rectangle(result_img, (10, 10), (400, 120), color, -1)
    cv2.putText(result_img, result_text, (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(result_img, f"{conf:.1%}", (20, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    detail_text = f"""
RESULT: {result_text}
Confidence: {conf:.1%}

DEMO MODE - For accurate results, train a model first:
python train.py

Image Info: {w}x{h} pixels
    """.strip()
    
    return result_img, detail_text

# Create the interface
with gr.Blocks(title="Image Forgery Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Image Forgery Detection System
    
    Upload an image to check if it's **REAL** or **FAKE**.
    
    The system uses deep learning to detect image manipulations.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Image")
            image_input = gr.Image(
                label="Upload Image Here",
                type="numpy",
                height=400
            )
            detect_btn = gr.Button(
                "Check if Real or Fake", 
                variant="primary", 
                size="lg",
                scale=1
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Detection Result")
            image_output = gr.Image(
                label="Result with Detection Overlay",
                type="numpy",
                height=400
            )
            text_output = gr.Textbox(
                label="Detection Details",
                lines=8,
                interactive=False,
                placeholder="Upload an image and click 'Detect Forgery' to see results..."
            )
    
    gr.Markdown("""
    ---
    ### 📚 How It Works:
    1. **Upload**: Click or drag an image into the upload area
    2. **Detect**: Click the "Detect Forgery" button
    3. **View Results**: See the prediction and confidence score
    
    ### ⚙️ Status:
    - If you see "DEMO MODE": Train a model first with `python train.py`
    - If you see "REAL MODEL": The system is using your trained model
    
    ### 🎯 Supported Formats:
    - JPG, JPEG, PNG images
    - Any image size (will be resized automatically)
    """)
    
    # Connect the button
    detect_btn.click(
        fn=detect_forgery_demo,
        inputs=image_input,
        outputs=[image_output, text_output]
    )
    
    # Also detect on image upload
    image_input.upload(
        fn=detect_forgery_demo,
        inputs=image_input,
        outputs=[image_output, text_output]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Launching Image Forgery Detection Web Interface")
    print("="*60)
    print("\nThe interface will open in your browser automatically")
    print("If it doesn't open, go to: http://localhost:7860")
    print("\nTips:")
    print("   - Upload any image to test")
    print("   - For real detection, train a model first: python train.py")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Launch with auto-open browser
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True  # Automatically open browser
    )

