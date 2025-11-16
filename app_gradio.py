"""
Gradio web interface for Image Forgery Detection
"""
import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import tempfile

from detect import ForgeryDetector
from config import TrainingConfig, MODELS_DIR


class GradioForgeryDetector:
    """Gradio interface for forgery detection"""
    
    def __init__(self, model_path: str, model_name: str = 'resnet'):
        """Initialize the detector"""
        self.detector = ForgeryDetector(
            model_path=model_path,
            model_name=model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def detect_forgery(self, image):
        """
        Detect forgery in uploaded image
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Tuple of (result_image, prediction_text)
        """
        if image is None:
            return None, "Please upload an image"
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            if isinstance(image, Image.Image):
                image.save(tmp_path)
            else:
                cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        try:
            # Detect
            results = self.detector.detect(tmp_path, return_attention=True)
            
            # Create visualization
            original_image = results['original_image']
            
            # Create result image with overlay
            result_image = original_image.copy()
            
            # Add text overlay
            pred_text = f"{results['class_name']}\nConfidence: {results['confidence']:.2%}"
            color = (255, 0, 0) if results['predicted_class'] == 1 else (0, 255, 0)
            
            # Draw text background
            cv2.rectangle(
                result_image,
                (10, 10),
                (400, 100),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                result_image,
                pred_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            # Overlay attention map if available
            if results['attention_map'] is not None:
                attention = results['attention_map'][0, 0].cpu().numpy()
                attention_resized = cv2.resize(
                    attention,
                    (result_image.shape[1], result_image.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                attention_colored = cv2.applyColorMap(
                    (attention_resized * 255).astype(np.uint8),
                    cv2.COLORMAP_HOT
                )
                result_image = cv2.addWeighted(
                    result_image, 0.7,
                    cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB), 0.3,
                    0
                )
            
            # Create detailed text
            detail_text = f"""
Prediction: {results['class_name']}
Confidence: {results['confidence']:.2%}

Probabilities:
  • Authentic: {results['probabilities']['authentic']:.2%}
  • Forged: {results['probabilities']['forged']:.2%}
            """.strip()
            
            return result_image, detail_text
        
        except Exception as e:
            return None, f"Error: {str(e)}"
        
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


def create_gradio_interface(model_path: str, model_name: str = 'resnet'):
    """Create and launch Gradio interface"""
    
    detector = GradioForgeryDetector(model_path, model_name)
    
    # Create interface
    with gr.Blocks(title="Image Forgery Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Image Forgery Detection System
        
        Upload an image to detect if it has been manipulated or forged.
        The system uses deep learning to identify splicing, copy-move, and other common forgeries.
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=400
                )
                detect_btn = gr.Button("Detect Forgery", variant="primary", size="lg")
            
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
        ### How it works:
        1. Upload an image (JPG, PNG formats supported)
        2. Click "Detect Forgery" to analyze the image
        3. View the results with highlighted manipulated regions (if detected)
        
        ### Model Information:
        - Architecture: Deep Learning-based CNN/ResNet/EfficientNet
        - Capabilities: Detects splicing, copy-move, and other image manipulations
        """)
        
        detect_btn.click(
            fn=detector.detect_forgery,
            inputs=image_input,
            outputs=[image_output, text_output]
        )
        
        gr.Examples(
            examples=[],
            inputs=image_input,
            label="Example Images (add your own examples)"
        )
    
    return demo


def main():
    """Main function to launch Gradio app"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Gradio web interface')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, default='resnet',
                       choices=['cnn', 'resnet', 'efficientnet', 'xception', 'vit'],
                       help='Model architecture name')
    parser.add_argument('--share', action='store_true',
                       help='Create a shareable link')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the server on')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file {args.model_path} not found")
        print("Please train a model first or provide a valid model path")
        return
    
    # Create and launch interface
    demo = create_gradio_interface(args.model_path, args.model_name)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()

