import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse
from pathlib import Path
import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.noise_unet import NoiseAwareUNet
from utils.advanced_noise_injector import EnhancedNoiseInjector

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess an image for the model"""
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Save a copy of the resized image
    resized_path = os.path.join(os.path.dirname(image_path), "resized_" + os.path.basename(image_path))
    cv2.imwrite(resized_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Convert to tensor
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) to (C, H, W)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    return image, resized_path

def segment_image(model, image, noise_mode="disabled", noise_strength=0.1):
    """Segment an image using the model with specified noise mode"""
    model.eval()
    
    # Update noise scale
    model.noise_injector.noise_scale = noise_strength
    
    # Move to device
    image = image.to(device)
    
    # Forward pass
    with torch.no_grad():
        start_time = time.time()
        output = model(image, noise_mode=noise_mode)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output)
        
        # Binarize the output
        mask = (probs > 0.5).float()
        
    return mask.cpu(), probs.cpu(), inference_time

def visualize_results(image, masks, probs, noise_modes, output_dir, noise_strength):
    """Create visualization of segmentation results"""
    # Convert image tensor to numpy
    image_np = image.squeeze(0).permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(len(noise_modes) + 1, 3, figsize=(15, 5 * (len(noise_modes) + 1)))
    
    # Plot original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    # Empty middle plot for original row
    axes[0, 1].axis("off")
    
    # Add information text
    info_text = """
    Neural Noise Injection: Enhanced Model
    
    This visualization shows how different noise modes affect segmentation.
    
    The model uses controlled noise patterns to enhance segmentation
    quality, especially at object boundaries and in ambiguous regions.
    
    Stronger noise can help with difficult boundaries but may
    cause over-segmentation if too high.
    """
    axes[0, 2].text(0.1, 0.5, info_text, fontsize=12)
    axes[0, 2].axis("off")
    
    # Plot each noise mode result
    for i, mode in enumerate(noise_modes):
        idx = i + 1  # Skip first row which is the original image
        
        # Binary mask
        mask_np = masks[i].squeeze(0).squeeze(0).numpy()
        axes[idx, 0].imshow(mask_np, cmap="gray")
        axes[idx, 0].set_title(f"Mask: {mode}")
        axes[idx, 0].axis("off")
        
        # Confidence map
        prob_np = probs[i].squeeze(0).squeeze(0).numpy()
        axes[idx, 1].imshow(prob_np, cmap="jet")
        axes[idx, 1].set_title(f"Confidence: {mode}")
        axes[idx, 1].axis("off")
        
        # Overlay
        overlay = image_np.copy()
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[:, :, 0] = mask_np * 255  # Red channel for visibility
        
        # Blend with alpha
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1, mask_rgb, alpha, 0)
        
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f"Overlay: {mode}")
        axes[idx, 2].axis("off")
    
    plt.suptitle(f"Neural Noise-Driven Segmentation (Strength: {noise_strength})", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"naruto_enhanced_noise_{noise_strength}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    
    return output_path

def test_on_naruto(model_path, image_path, output_dir, noise_strength=0.15):
    """Test the enhanced model on the Naruto image"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    image, resized_path = preprocess_image(image_path)
    print(f"Preprocessed image saved to: {resized_path}")
    
    # Create noise injector
    noise_injector = EnhancedNoiseInjector(
        noise_scale=noise_strength,
        noise_decay=0.98,
        noise_patterns=["gaussian", "perlin", "simplex", "structured", "adaptive"]
    )
    
    # Create and load model
    model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
    
    # Load model weights
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Define noise modes to test
    noise_modes = ["disabled", "single", "ensemble", "adaptive"]
    
    # Process with each noise mode
    masks = []
    probs = []
    inference_times = []
    
    for mode in noise_modes:
        print(f"Processing with noise mode: {mode}")
        mask, prob, inference_time = segment_image(model, image, mode, noise_strength)
        masks.append(mask)
        probs.append(prob)
        inference_times.append(inference_time)
        print(f"Inference time: {inference_time:.2f} ms")
    
    # Create visualization
    output_path = visualize_results(image, masks, probs, noise_modes, output_dir, noise_strength)
    
    # Create comparison table
    comparison_path = os.path.join(output_dir, "noise_comparison.md")
    with open(comparison_path, "w") as f:
        f.write("# Neural Noise Mode Comparison\n\n")
        f.write("| Noise Mode | Inference Time (ms) | Characteristics |\n")
        f.write("|------------|---------------------|------------------|\n")
        
        for mode, time_ms in zip(noise_modes, inference_times):
            if mode == "disabled":
                desc = "Standard segmentation, no noise enhancement"
            elif mode == "single":
                desc = "Single noise pattern for edge enhancement"
            elif mode == "ensemble":
                desc = "Multiple noise patterns averaged for robustness"
            elif mode == "adaptive":
                desc = "Content-adaptive noise based on image features"
            
            f.write(f"| {mode} | {time_ms:.2f} | {desc} |\n")
    
    print(f"Comparison table saved to {comparison_path}")
    
    return output_path, comparison_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test enhanced model on Naruto image")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--image_path", type=str, 
                        default="test/HD-wallpaper-anime-naruto-running-naruto-grass-anime.jpg", 
                        help="Path to the Naruto test image")
    parser.add_argument("--output_dir", type=str, default="naruto_enhanced_results", 
                        help="Directory to save test results")
    parser.add_argument("--noise_strength", type=float, default=0.15, 
                        help="Strength of noise to apply")
    
    args = parser.parse_args()
    
    test_on_naruto(args.model_path, args.image_path, args.output_dir, args.noise_strength)
