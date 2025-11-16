"""
Demo Gradio web interface for Image Forgery Detection
Works without a trained model - shows the interface structure
"""
import gradio as gr
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

def demo_detect_forgery(image):
    """
    Demo forgery detection (without actual model)
    Shows the interface structure
    """
    if image is None:
        return None, "Please upload an image"
    
    # Convert to numpy if PIL Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Create a demo result (simulating detection)
    result_image = image_np.copy()
    
    # Add demo overlay
    h, w = result_image.shape[:2]
    cv2.rectangle(result_image, (10, 10), (400, 100), (0, 255, 0), -1)
    cv2.putText(result_image, "DEMO MODE", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_image, "Authentic (Demo)", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Demo text
    detail_text = """
DEMO MODE - No trained model loaded

To use the full system:
1. Train a model: python train.py
2. Launch with model: python app_gradio.py --model_path models/best_model_resnet.pth

Current Status:
- Image uploaded successfully
- Interface is working
- Ready for model integration

Image Info:
- Shape: {}x{}
- Format: {}
    """.format(w, h, image_np.dtype).strip()
    
    return result_image, detail_text

# Create Gradio interface
with gr.Blocks(title="Image Forgery Detection - Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Image Forgery Detection System - DEMO MODE
    
    **This is a demo interface.** To use the full detection system:
    1. Train a model: `python train.py`
    2. Launch with model: `python app_gradio.py --model_path models/best_model_resnet.pth`
    
    Upload an image to see the interface structure.
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Image",
                type="numpy",
                height=400
            )
            detect_btn = gr.Button("Detect Forgery (Demo)", variant="primary", size="lg")
        
        with gr.Column():
            image_output = gr.Image(
                label="Detection Result",
                type="numpy",
                height=400
            )
            text_output = gr.Textbox(
                label="Detection Details",
                lines=10,
                interactive=False
            )
    
    gr.Markdown("""
    ### 📚 How to Use the Full System:
    
    1. **Prepare Dataset**: Organize images in `data/datasets/processed/`
    2. **Train Model**: Run `python train.py`
    3. **Launch Full Interface**: Run `python app_gradio.py --model_path models/best_model_resnet.pth`
    
    ### 🎯 Features:
    - Deep Learning-based detection
    - Multiple model architectures (ResNet, EfficientNet, Xception, ViT)
    - Attention map visualization
    - Real-time detection
    """)
    
    detect_btn.click(
        fn=demo_detect_forgery,
        inputs=image_input,
        outputs=[image_output, text_output]
    )

if __name__ == "__main__":
    print("="*60)
    print("Launching Demo Web Interface")
    print("="*60)
    print("\nAccess the interface at: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

