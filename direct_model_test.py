import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import traceback

print("Starting direct model test script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Add the project root to the path
sys.path.append(os.getcwd())

try:
    print("Importing project modules...")
    from models.noise_unet import NoiseAwareUNet
    from utils.simple_noise_injector import NeuralNoiseInjector
    from utils.data_utils import preprocess_image
    print("Modules imported successfully")
except Exception as e:
    print(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

def load_model(model_path):
    """Load the segmentation model"""
    print(f"Loading model from {model_path}...")
    try:
        model = NoiseAwareUNet(3, 1)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise

def segment_image(model, image_path, noise_mode, noise_strength=0.1):
    """Segment an image using the model with specified noise mode"""
    print(f"Segmenting image {image_path} with noise_mode={noise_mode}, strength={noise_strength}...")
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    
    # Configure noise injector
    noise_injector = NeuralNoiseInjector(noise_scale=noise_strength)
    model.noise_injector = noise_injector
    
    # Generate segmentation
    with torch.no_grad():
        output = model(image_tensor, noise_mode=noise_mode)
        output = torch.sigmoid(output)
    
    # Convert to numpy
    segmentation = output.detach().cpu().numpy()[0, 0]
    
    return image, segmentation

def create_comparison(model, image_path, output_dir="direct_test_results"):
    """Create a comparison of different noise modes for a single image"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Test different noise configurations
    image, seg_no_noise = segment_image(model, image_path, "disabled")
    _, seg_low_noise = segment_image(model, image_path, "single", 0.1)
    _, seg_med_noise = segment_image(model, image_path, "single", 0.2)
    _, seg_high_noise = segment_image(model, image_path, "single", 0.3)
    _, seg_ensemble = segment_image(model, image_path, "ensemble", 0.1)
    
    # Create and save visualization
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(np.array(image))
    plt.title("Original Image")
    plt.axis("off")
    
    # No noise
    plt.subplot(2, 3, 2)
    plt.imshow(seg_no_noise, cmap="gray")
    plt.title("No Noise")
    plt.axis("off")
    
    # Low noise (0.1)
    plt.subplot(2, 3, 3)
    plt.imshow(seg_low_noise, cmap="gray")
    plt.title("Low Noise (0.1)")
    plt.axis("off")
    
    # Medium noise (0.2)
    plt.subplot(2, 3, 4)
    plt.imshow(seg_med_noise, cmap="gray")
    plt.title("Medium Noise (0.2)")
    plt.axis("off")
    
    # High noise (0.3)
    plt.subplot(2, 3, 5)
    plt.imshow(seg_high_noise, cmap="gray")
    plt.title("High Noise (0.3)")
    plt.axis("off")
    
    # Ensemble noise
    plt.subplot(2, 3, 6)
    plt.imshow(seg_ensemble, cmap="gray")
    plt.title("Ensemble Noise (0.1)")
    plt.axis("off")
    
    # Add a main title
    plt.suptitle(f"Neural Noise-Driven Segmentation: {os.path.basename(image_path)}", fontsize=16)
    plt.tight_layout()
    
    # Save the comparison
    output_path = os.path.join(output_dir, f"noise_comparison_{os.path.basename(image_path)}")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved comparison to {output_path}")
    
    return output_path

def test_all_images():
    """Test all images in the test_images directory"""
    try:
        # Load the model
        model = load_model("models/noise_unet_model.pth")
        
        # Get all test images
        test_dir = "test_images"
        print(f"Looking for test images in {test_dir}")
        print(f"Directory exists: {os.path.exists(test_dir)}")
        
        if not os.path.exists(test_dir):
            print("Test directory not found")
            return
            
        test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
        print(f"Found {len(test_images)} test images: {test_images}")
        
        # Create comparisons for each test image
        for image_path in test_images:
            try:
                print(f"Processing {image_path}")
                output_path = create_comparison(model, image_path)
                print(f"Successfully processed {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"Error in test_all_images: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting direct model testing...")
    test_all_images()
    print("Testing completed!")
