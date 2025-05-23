import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys

# Add the project root to the path
sys.path.append(os.getcwd())

# Import model and utilities
from models.noise_unet import NoiseAwareUNet
from utils.simple_noise_injector import NeuralNoiseInjector
from utils.data_utils import preprocess_image

# Path to the Naruto image
image_path = r"C:\Users\Asus\Documents\Projects\image_optimisation\test\HD-wallpaper-anime-naruto-running-naruto-grass-anime.jpg"

# Create output directory
output_dir = "naruto_direct_test"
os.makedirs(output_dir, exist_ok=True)

print(f"Testing with image: {image_path}")
print(f"Image exists: {os.path.exists(image_path)}")

def load_model():
    """Load the segmentation model"""
    print("Loading model...")
    model_path = "models/noise_unet_model.pth"
    model = NoiseAwareUNet(3, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def segment_image(model, image, noise_mode, noise_strength=0.1):
    """Segment an image using the model"""
    print(f"Segmenting with noise_mode={noise_mode}, strength={noise_strength}")
    
    # Configure noise injector
    noise_injector = NeuralNoiseInjector(noise_scale=noise_strength)
    model.noise_injector = noise_injector
    
    # Process the image
    image_tensor = preprocess_image(image)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor, noise_mode=noise_mode)
        output = torch.sigmoid(output)
    
    # Convert to numpy
    mask = output.detach().cpu().numpy()[0, 0]
    
    # Scale to [0, 255]
    mask = (mask * 255).astype(np.uint8)
    
    return mask

# Load image
try:
    image = Image.open(image_path)
    print(f"Image loaded successfully. Size: {image.size}")
    
    # Resize if too large (for faster processing)
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        print(f"Image resized to {new_size}")
    
    # Save original
    image.save(os.path.join(output_dir, "original_naruto.jpg"))
    
    # Load model
    model = load_model()
    
    # Test different noise modes
    noise_configs = [
        {"mode": "disabled", "strength": 0.1, "label": "no_noise"},
        {"mode": "single", "strength": 0.1, "label": "low_noise"},
        {"mode": "single", "strength": 0.3, "label": "high_noise"},
        {"mode": "ensemble", "strength": 0.1, "label": "ensemble"}
    ]
    
    results = []
    
    for config in noise_configs:
        try:
            # Segment the image
            mask = segment_image(model, image, config["mode"], config["strength"])
            
            # Save the segmentation mask
            mask_path = os.path.join(output_dir, f"naruto_seg_{config['label']}.png")
            Image.fromarray(mask).save(mask_path)
            print(f"Saved segmentation mask to {mask_path}")
            
            results.append({
                "config": config,
                "mask_path": mask_path
            })
        except Exception as e:
            print(f"Error with {config['label']}: {e}")
    
    # Create visualization
    if results:
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(np.array(image))
        plt.title("Original Naruto Image")
        plt.axis("off")
        
        # Display segmentation results
        for i, result in enumerate(results):
            plt.subplot(2, 3, i+2)
            mask_img = Image.open(result["mask_path"])
            plt.imshow(np.array(mask_img), cmap="gray")
            plt.title(f"{result['config']['mode'].capitalize()} {result['config']['strength']}")
            plt.axis("off")
        
        plt.suptitle("Naruto Image Segmentation with Neural Noise", fontsize=16)
        plt.tight_layout()
        
        # Save the comparison
        comparison_path = os.path.join(output_dir, "naruto_comparison.png")
        plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
        print(f"Saved comparison to {comparison_path}")
    
    print("Direct testing complete!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
