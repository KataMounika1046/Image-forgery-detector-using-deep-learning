"""
Streamlit web interface for Image Forgery Detection
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import tempfile
import io

from detect import ForgeryDetector
from config import TrainingConfig, MODELS_DIR


@st.cache_resource
def load_detector(model_path: str, model_name: str = 'resnet'):
    """Load detector model (cached)"""
    return ForgeryDetector(
        model_path=model_path,
        model_name=model_name,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Image Forgery Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Image Forgery Detection System")
    st.markdown("""
    Upload an image to detect if it has been manipulated or forged.
    The system uses deep learning to identify splicing, copy-move, and other common forgeries.
    """)
    
    # Sidebar for model selection
    st.sidebar.title("Model Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        value=str(MODELS_DIR / "best_model_resnet.pth"),
        help="Path to trained model checkpoint"
    )
    
    model_name = st.sidebar.selectbox(
        "Model Architecture",
        options=['resnet', 'efficientnet', 'xception', 'cnn', 'vit'],
        index=0,
        help="Select the model architecture"
    )
    
    # Check if model exists
    if not Path(model_path).exists():
        st.sidebar.error(f"Model file not found: {model_path}")
        st.sidebar.info("Please train a model first or provide a valid model path")
        return
    
    # Load detector
    try:
        detector = load_detector(model_path, model_name)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image file to analyze"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_path = tmp_file.name
                image.save(tmp_path)
            
            # Detect button
            if st.button("🔍 Detect Forgery", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        # Detect
                        results = detector.detect(tmp_path, return_attention=True)
                        
                        # Display results
                        with col2:
                            st.subheader("Detection Results")
                            
                            # Prediction
                            pred_class = results['class_name']
                            confidence = results['confidence']
                            
                            if pred_class == "Forged":
                                st.error(f"⚠️ **FORGED** (Confidence: {confidence:.2%})")
                            else:
                                st.success(f"✅ **AUTHENTIC** (Confidence: {confidence:.2%})")
                            
                            # Probabilities
                            st.markdown("### Probabilities")
                            prob_authentic = results['probabilities']['authentic']
                            prob_forged = results['probabilities']['forged']
                            
                            col_a, col_f = st.columns(2)
                            with col_a:
                                st.metric("Authentic", f"{prob_authentic:.2%}")
                            with col_f:
                                st.metric("Forged", f"{prob_forged:.2%}")
                            
                            # Progress bars
                            st.progress(prob_authentic, text="Authentic Probability")
                            st.progress(prob_forged, text="Forged Probability")
                            
                            # Visualization
                            st.markdown("### Visualization")
                            
                            # Create visualization
                            original_image = results['original_image']
                            result_image = original_image.copy()
                            
                            # Add text overlay
                            pred_text = f"{pred_class} ({confidence:.2%})"
                            color = (255, 0, 0) if results['predicted_class'] == 1 else (0, 255, 0)
                            
                            cv2.rectangle(
                                result_image,
                                (10, 10),
                                (400, 80),
                                color,
                                -1
                            )
                            
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
                            
                            st.image(result_image, caption="Detection Result", use_container_width=True)
                            
                            # Download result
                            result_pil = Image.fromarray(result_image)
                            buf = io.BytesIO()
                            result_pil.save(buf, format='PNG')
                            st.download_button(
                                label="Download Result",
                                data=buf.getvalue(),
                                file_name="forgery_detection_result.png",
                                mime="image/png"
                            )
                    
                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
                    
                    finally:
                        # Clean up
                        Path(tmp_path).unlink(missing_ok=True)
    
    # Information section
    st.markdown("---")
    st.markdown("""
    ### 📚 How it works:
    1. **Upload**: Select an image file (JPG, PNG formats)
    2. **Analyze**: Click "Detect Forgery" to process the image
    3. **Results**: View prediction, confidence scores, and highlighted regions
    
    ### 🎯 Model Capabilities:
    - Detects image splicing (combining parts from different images)
    - Identifies copy-move forgeries (duplicated regions)
    - Recognizes other common image manipulations
    - Highlights suspicious regions using attention mechanisms
    
    ### ⚙️ Technical Details:
    - Deep Learning-based detection using CNN/ResNet/EfficientNet architectures
    - Transfer learning from ImageNet pretrained models
    - Attention mechanisms for region localization
    - Binary classification: Authentic vs Forged
    """)


if __name__ == "__main__":
    main()

