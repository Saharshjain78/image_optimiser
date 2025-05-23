import streamlit as st
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import requests
import time
import cv2
from io import BytesIO

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.noise_unet import NoiseAwareUNet
from utils.simple_noise_injector import NeuralNoiseInjector
from utils.data_utils import preprocess_image
from utils.image_utils import get_sample_image, create_sample_images, image_to_bytes

# Set page configuration
st.set_page_config(
    page_title="Neural Noise Segmentation App",
    page_icon="🧠",
    layout="wide"
)

# Function to convert PIL Image to bytes
def pil_to_bytes(img):
    return image_to_bytes(img, format="PNG")

# Function to apply segmentation
@st.cache_data
def apply_segmentation(image, noise_mode, noise_strength, _model):
    """Apply segmentation using the loaded model with enhanced image handling"""
    # Process image and get the adapted size
    image_tensor, adapted_size = preprocess_image(image)
    
    # Configure noise parameters
    noise_injector = NeuralNoiseInjector(noise_scale=noise_strength)
    _model.noise_injector = noise_injector
    
    # Generate segmentation with specified noise mode
    start_time = time.time()
    with torch.no_grad():
        output = _model(image_tensor, noise_mode=noise_mode)
    inference_time = time.time() - start_time
    
    # Apply sigmoid activation for binary segmentation
    output = torch.sigmoid(output)
    
    # Convert output to numpy array for display
    if isinstance(output, torch.Tensor):
        mask = output.detach().cpu().numpy()
        mask = mask[0, 0]  # First image, first channel
        mask = (mask * 255).astype(np.uint8)
    
    # Resize mask back to original image size for overlay
    img_np = np.array(image)
    original_h, original_w = img_np.shape[:2]
    mask_resized = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a colored overlay for visualization
    colored_mask = np.zeros((original_h, original_w, 3), dtype=np.uint8)
    colored_mask[mask_resized > 127] = [0, 255, 0]  # Green for segmented areas
    
    # Create overlay with transparency
    alpha = 0.5
    overlay = (alpha * colored_mask + (1 - alpha) * img_np).astype(np.uint8)
    
    # Generate heatmap visualization of mask for confidence display
    heatmap = np.uint8(plt.cm.jet(mask_resized / 255.0) * 255)
    
    # Calculate uncertainty based on multiple runs if in ensemble mode
    uncertainty_map = None
    if noise_mode == "ensemble":
        # Run multiple iterations with different noise patterns
        outputs = []
        iterations = 5
        for _ in range(iterations):
            with torch.no_grad():                
                # Reset noise injector iteration counter
                noise_injector.reset_iteration()
                # Run inference with single noise pattern
                output_iter = _model(image_tensor, noise_mode="single")
                output_iter = torch.sigmoid(output_iter)
                outputs.append(output_iter)
        
        # Calculate variance (uncertainty)
        stacked_outputs = torch.stack(outputs)
        variance = torch.var(stacked_outputs, dim=0)
        
        # Convert variance to numpy for display
        variance_np = variance.detach().cpu().numpy()
        variance_np = variance_np[0, 0]  # First image, first channel
        
        # Resize variance map to original image size
        variance_np = cv2.resize(variance_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize variance to [0, 1] for visualization
        min_val = variance_np.min()
        max_val = variance_np.max()
        if max_val > min_val:
            variance_np = (variance_np - min_val) / (max_val - min_val)
        
        # Convert to heatmap
        uncertainty_map = np.uint8(plt.cm.plasma(variance_np) * 255)
    
    return {
        'mask': mask_resized,
        'overlay': overlay,
        'heatmap': heatmap,
        'uncertainty_map': uncertainty_map,
        'inference_time': round(inference_time * 1000, 2),  # ms
        'adapted_size': adapted_size
    }

def main():
    st.title("Neural Noise-Driven Segmentation System")
    st.markdown("""
    This app demonstrates how controlled neural noise can enhance image segmentation. 
    Upload an image and experiment with different noise modes and strengths.
    """)
    
    # Sidebar for model selection and loading
    st.sidebar.header("Model Settings")
    
    # Local model settings
    use_local_model = st.sidebar.checkbox("Use Local Model", value=True)
    
    if use_local_model:
        model_path = st.sidebar.text_input(
            "Model Path", 
            value="models/noise_unet_model.pth",
            help="Path to the trained model file"
        )
        
        # Load model button
        if st.sidebar.button("Load Model"):
            try:
                with st.spinner("Loading model..."):
                    # Load the model
                    model = NoiseAwareUNet(3, 1)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    st.session_state['model'] = model
                    st.session_state['model_loaded'] = True
                    st.sidebar.success("Model loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading model: {str(e)}")
                st.session_state['model_loaded'] = False
    else:
        # API endpoint settings
        api_url = st.sidebar.text_input(
            "API Endpoint", 
            value="http://localhost:8000",
            help="URL of the FastAPI server"
        )
        st.session_state['api_url'] = api_url
        st.session_state['model_loaded'] = True  # Assume API is available
    
    # Check if we should show the main app
    if not st.session_state.get('model_loaded', False):
        if use_local_model:
            st.warning("Please load a model first to use the app.")
        else:
            st.warning("Make sure the API server is running at the specified URL.")
        return
    
    # Create columns for input image and settings
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Input Image")
        # File uploader for image input
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"], help="Upload an image for segmentation")
        use_sample_image = st.checkbox("Use a sample image instead", value=False)
        
        if use_sample_image:
            # Provide some sample images for testing
            sample_option = st.selectbox(
                "Select a sample image",
                ["Nature", "Portrait", "Urban", "Medical", "Satellite"]
            )
            
            # Ensure sample images are created if they don't exist
            create_sample_images()
            
            # Get the sample image (real or placeholder)
            image = get_sample_image(sample_option)
            st.image(image, caption=f"Sample Image: {sample_option}", use_container_width=True)
            st.session_state['input_image'] = image
        
        elif uploaded_file is not None:
            try:
                # Open and convert image
                image = Image.open(uploaded_file)
                
                # Display information about the image
                st.info(f"Image info: {image.format} image, Size: {image.width}x{image.height}, Mode: {image.mode}")
                
                # Convert to RGB for consistent processing
                image = image.convert("RGB")
                
                # Display the uploaded image
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Save the image in session state
                st.session_state['input_image'] = image
                
                # Clear any previous errors
                if 'image_error' in st.session_state:
                    del st.session_state['image_error']
                    
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")
                st.session_state['image_error'] = str(e)
                
        elif 'input_image' in st.session_state:
            # Display previously uploaded image
            st.image(st.session_state['input_image'], caption="Previously Uploaded Image", use_container_width=True)
    with col2:
        st.header("Segmentation Settings")
        
        # Target size selection
        target_size_option = st.radio(
            "Image Processing Resolution",
            ["Standard (256×256)", "Medium (384×384)", "High (512×512)", "Ultra (768×768)"],
            help="Higher resolution provides better details but slower processing"
        )
        
        # Map option to actual size
        target_size_map = {
            "Standard (256×256)": (256, 256),
            "Medium (384×384)": (384, 384), 
            "High (512×512)": (512, 512),
            "Ultra (768×768)": (768, 768)
        }
        target_size = target_size_map[target_size_option]
        
        # Noise mode selection
        noise_mode = st.radio(
            "Noise Mode",
            ["disabled", "single", "ensemble"],
            help="disabled: No noise, single: One noise pattern, ensemble: Multiple noise patterns averaged"
        )
        
        # Noise strength slider
        noise_strength = st.slider(
            "Noise Strength",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Higher values increase noise influence"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            show_confidence = st.checkbox("Show Confidence Map", value=True)
            show_uncertainty = st.checkbox("Show Uncertainty Map (Ensemble mode only)", value=True)
            
            if noise_mode == "ensemble":
                ensemble_iterations = st.slider(
                    "Ensemble Iterations",
                    min_value=3,
                    max_value=10,
                    value=5,
                    step=1,
                    help="More iterations provide better uncertainty estimation but slower processing"
                )
                # Store in session state to use in processing
                st.session_state['ensemble_iterations'] = ensemble_iterations
            else:
                # Default value stored in session state
                st.session_state['ensemble_iterations'] = 5
          # Process button
        process_button = st.button("Apply Segmentation")
          # Error handling
        if 'image_error' in st.session_state:
            st.error("Please upload a valid image before processing.")
            
    # Process image if button is clicked and image is loaded
    if process_button and 'input_image' in st.session_state and 'image_error' not in st.session_state:
        with st.spinner("Processing image..."):
            try:
                if use_local_model and 'model' in st.session_state:
                    # Update preprocess_image function to use selected target size
                    from functools import partial
                    preprocess_with_size = partial(preprocess_image, target_size=target_size)
                    
                    # Store original preprocess_image function
                    import utils.data_utils
                    original_preprocess = utils.data_utils.preprocess_image
                    
                    try:
                        # Temporarily replace with our version
                        utils.data_utils.preprocess_image = preprocess_with_size
                        
                        # Use local model for segmentation
                        results = apply_segmentation(
                            st.session_state['input_image'],
                            noise_mode,
                            noise_strength,
                            st.session_state['model']
                        )
                    finally:
                        # Always restore original function, even if an error occurs
                        utils.data_utils.preprocess_image = original_preprocess
                    
                    st.session_state['results'] = results
                    
                    # Add extra information
                    st.session_state['processing_details'] = {
                        'target_size': target_size,
                        'noise_mode': noise_mode,
                        'noise_strength': noise_strength,
                        'adapted_size': results.get('adapted_size', target_size)
                    }
                else:
                    # Use API for segmentation
                    try:
                        # Prepare the image for API upload
                        img_byte_arr = io.BytesIO()
                        st.session_state['input_image'].save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        # API call
                        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
                        data = {
                            'noise_mode': noise_mode,
                            'noise_strength': str(noise_strength),
                            'confidence_map': 'true',
                            'target_size_x': str(target_size[0]),
                            'target_size_y': str(target_size[1])
                        }
                        
                        response = requests.post(
                            f"{st.session_state['api_url']}/upload",
                            files=files,
                            data=data
                        )
                        
                        if response.status_code == 200:
                            api_result = response.json()
                            
                            # Download segmentation result
                            seg_response = requests.get(f"{st.session_state['api_url']}{api_result['segmentation_url']}")
                            mask = Image.open(io.BytesIO(seg_response.content))
                            
                            # Download confidence map if available
                            heatmap = None
                            if api_result.get('confidence_map_url'):
                                conf_response = requests.get(f"{st.session_state['api_url']}{api_result['confidence_map_url']}")
                                heatmap = Image.open(io.BytesIO(conf_response.content))
                            
                            # Download uncertainty map if available
                            uncertainty_map = None
                            if api_result.get('uncertainty_map_url'):
                                uncertainty_response = requests.get(f"{st.session_state['api_url']}{api_result['uncertainty_map_url']}")
                                uncertainty_map = Image.open(io.BytesIO(uncertainty_response.content))
                            
                            # Download overlay if available
                            overlay = None
                            if api_result.get('overlay_url'):
                                overlay_response = requests.get(f"{st.session_state['api_url']}{api_result['overlay_url']}")
                                overlay = Image.open(io.BytesIO(overlay_response.content))
                            
                            # Store in session state
                            st.session_state['api_results'] = {
                                'mask': mask,
                                'heatmap': heatmap,
                                'uncertainty_map': uncertainty_map,
                                'overlay': overlay,
                                'inference_time': api_result.get('inference_time_ms', 0),
                                'adapted_size': api_result.get('adapted_size', target_size)
                            }
                            
                            # Add extra information
                            st.session_state['processing_details'] = {
                                'target_size': target_size,
                                'noise_mode': noise_mode,
                                'noise_strength': noise_strength,
                                'adapted_size': api_result.get('adapted_size', target_size)
                            }
                        else:
                            st.error(f"API Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error calling API: {str(e)}")
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                
    # Display results if available
    if 'results' in st.session_state or 'api_results' in st.session_state:
        st.header("Segmentation Results")
        
        # Create columns for different visualizations
        col1, col2 = st.columns(2)
        
        # Get processing details if available
        processing_details = st.session_state.get('processing_details', {})
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            with col1:
                st.subheader("Segmentation Mask")
                st.image(results['mask'], caption="Segmentation Mask", use_column_width=True)
                
                # Add download button for mask
                mask_pil = Image.fromarray(results['mask'])
                mask_bytes = pil_to_bytes(mask_pil)
                st.download_button(
                    "Download Mask", 
                    mask_bytes,
                    "segmentation_mask.png",
                    "image/png"
                )
                
                st.text(f"Inference Time: {results['inference_time']} ms")
                
                if processing_details:
                    st.text(f"Processed at resolution: {processing_details.get('adapted_size', 'Unknown')}")
            
            with col2:
                st.subheader("Overlay Visualization")
                st.image(results['overlay'], caption="Segmentation Overlay", use_column_width=True)
                
                # Add download button for overlay
                overlay_pil = Image.fromarray(results['overlay'])
                overlay_bytes = pil_to_bytes(overlay_pil)
                st.download_button(
                    "Download Overlay", 
                    overlay_bytes,
                    "segmentation_overlay.png",
                    "image/png"
                )
                
                # Show processing metrics
                if processing_details:
                    st.text(f"Noise mode: {processing_details.get('noise_mode', 'Unknown')}")
                    st.text(f"Noise strength: {processing_details.get('noise_strength', 'Unknown')}")
            
            # Additional visualizations section
            st.subheader("Advanced Visualizations")
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                # Show confidence heatmap
                if show_confidence and results['heatmap'] is not None:
                    st.image(results['heatmap'], caption="Confidence Heatmap", use_column_width=True)
                    
                    # Add download button for heatmap
                    heatmap_pil = Image.fromarray(results['heatmap'])
                    heatmap_bytes = pil_to_bytes(heatmap_pil)
                    st.download_button(
                        "Download Heatmap", 
                        heatmap_bytes,
                        "confidence_heatmap.png",
                        "image/png"
                    )
            
            with adv_col2:
                # Show uncertainty map if available (ensemble mode)
                if show_uncertainty and results['uncertainty_map'] is not None:
                    st.image(results['uncertainty_map'], caption="Uncertainty Map", use_column_width=True)
                    
                    # Add download button for uncertainty map
                    uncertainty_pil = Image.fromarray(results['uncertainty_map'])
                    uncertainty_bytes = pil_to_bytes(uncertainty_pil)
                    st.download_button(
                        "Download Uncertainty Map", 
                        uncertainty_bytes,
                        "uncertainty_map.png",
                        "image/png"
                    )
            
            # Add a section for batch processing multiple images
            with st.expander("Batch Processing Options"):
                st.write("To process multiple images with the same settings, use one of these options:")
                
                batch_option = st.radio(
                    "Batch Processing Method",
                    ["Upload Directory (API only)", "Use Current Settings"]
                )
                
                if batch_option == "Upload Directory (API only)":
                    st.info("This feature requires the API server to be running.")
                    st.write("Configure a directory for batch processing through the API endpoint.")
                    
                    api_batch_dir = st.text_input(
                        "API Batch Directory Path",
                        placeholder="Enter directory path on the server"
                    )
                    
                    if st.button("Start Batch Processing via API") and api_batch_dir:
                        st.warning("Batch processing started on the server. Check the API logs for progress.")
                else:
                    st.info("Apply the current settings to new images by uploading them one at a time.")
                    st.write("Current settings will be preserved for new uploads.")
        else:
            # Display API results
            api_results = st.session_state['api_results']
            
            with col1:
                st.subheader("Segmentation Mask")
                st.image(api_results['mask'], caption="Segmentation Mask", use_column_width=True)
                
                # Add download button for mask
                mask_bytes = pil_to_bytes(api_results['mask'])
                st.download_button(
                    "Download Mask", 
                    mask_bytes,
                    "segmentation_mask.png",
                    "image/png"
                )
                
                st.text(f"Inference Time: {api_results.get('inference_time', 0)} ms")
                
                if processing_details:
                    st.text(f"Processed at resolution: {processing_details.get('adapted_size', 'Unknown')}")
            
            with col2:
                if api_results.get('overlay'):
                    st.subheader("Overlay Visualization")
                    st.image(api_results['overlay'], caption="Segmentation Overlay", use_column_width=True)
                    
                    # Add download button for overlay
                    overlay_bytes = pil_to_bytes(api_results['overlay'])
                    st.download_button(
                        "Download Overlay", 
                        overlay_bytes,
                        "segmentation_overlay.png",
                        "image/png"
                    )
                elif api_results.get('heatmap'):
                    st.subheader("Confidence Heatmap")
                    st.image(api_results['heatmap'], caption="Confidence Heatmap", use_column_width=True)
                    
                    # Add download button for heatmap
                    heatmap_bytes = pil_to_bytes(api_results['heatmap'])
                    st.download_button(
                        "Download Heatmap", 
                        heatmap_bytes,
                        "confidence_heatmap.png",
                        "image/png"
                    )
                
                # Show processing metrics
                if processing_details:
                    st.text(f"Noise mode: {processing_details.get('noise_mode', 'Unknown')}")
                    st.text(f"Noise strength: {processing_details.get('noise_strength', 'Unknown')}")
            
            # Additional visualizations in separate row
            if api_results.get('uncertainty_map') and show_uncertainty:
                st.subheader("Uncertainty Map")
                st.image(api_results['uncertainty_map'], caption="Uncertainty from Multiple Noise Patterns", use_column_width=True)
                
                # Add download button for uncertainty map
                uncertainty_bytes = pil_to_bytes(api_results['uncertainty_map'])
                st.download_button(
                    "Download Uncertainty Map", 
                    uncertainty_bytes,
                    "uncertainty_map.png",
                    "image/png"
                )
    
    # Additional information about the system
    with st.expander("Learn More About Neural Noise in Segmentation"):
        st.markdown("""
        ### How Neural Noise Enhances Segmentation
        
        This system deliberately introduces controlled neural noise as a feature rather than a bug in image segmentation. The noise:
        
        1. **Breaks symmetry** in ambiguous segmentation regions
        2. **Escapes local optima** during inference
        3. **Creates ensemble-like effects** with a single model
        4. **Generates probabilistic segmentation maps**
        
        ### Noise Modes
        
        - **Disabled**: Standard segmentation without noise
        - **Single**: Applies one noise pattern to enhance edge detection
        - **Ensemble**: Uses multiple noise patterns and averages the results, providing uncertainty estimation
        
        ### Benefits
        
        - **Improved Edge Detection**: Noise helps distinguish ambiguous boundaries
        - **Self-Regularizing**: The system learns which layers benefit from noise injection
        - **Uncertainty Quantification**: Noise variations provide natural uncertainty estimates
        - **Memory-Efficient Ensembling**: Single model produces multiple outputs via different noise patterns
        - **Adaptability**: Noise patterns help generalize to unseen data distributions
        """)

if __name__ == "__main__":
    # Initialize session state variables if they don't exist
    if 'model_loaded' not in st.session_state:
        st.session_state['model_loaded'] = False
    if 'input_image' not in st.session_state:
        st.session_state['input_image'] = None
    
    main()