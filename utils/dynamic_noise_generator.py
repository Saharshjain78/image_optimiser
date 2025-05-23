import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.noise_unet import NoiseAwareUNet
from utils.advanced_noise_injector import EnhancedNoiseInjector

class DynamicNoiseGenerator:
    """
    Dynamically adjusts noise parameters based on image characteristics
    """
    def __init__(self, model, base_noise_scale=0.15):
        """
        Initialize the dynamic noise generator
        
        Args:
            model: NoiseAwareUNet model
            base_noise_scale: Base scale for noise amplitude
        """
        self.model = model
        self.base_noise_scale = base_noise_scale
        
    def analyze_image(self, image):
        """
        Analyze image characteristics to determine optimal noise parameters
        
        Args:
            image: Input image tensor [1, C, H, W]
            
        Returns:
            Dictionary of analysis results
        """
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if image_np.shape[2] == 1:
                image_np = np.repeat(image_np, 3, axis=2)
        else:
            image_np = image
            
        # Convert to 0-255 uint8 if needed
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Convert to grayscale for analysis
        if image_np.shape[2] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.squeeze()
            
        # Calculate gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize gradient magnitude
        gradient_magnitude = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
        
        # Calculate statistics
        avg_gradient = np.mean(gradient_magnitude)
        std_gradient = np.std(gradient_magnitude)
        
        # Calculate texture complexity (using Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_complexity = np.var(laplacian) / 100.0  # Normalize to reasonable range
        
        # Calculate contrast
        contrast = np.std(gray) / 128.0  # Normalize to about 0-1 range
        
        # Calculate noise floor (estimate of existing noise in image)
        noise_floor = estimate_noise(gray) / 255.0  # Normalize to 0-1
        
        # Calculate edge density
        edge_threshold = 100
        edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        return {
            'avg_gradient': float(avg_gradient),
            'std_gradient': float(std_gradient),
            'texture_complexity': float(texture_complexity),
            'contrast': float(contrast),
            'noise_floor': float(noise_floor),
            'edge_density': float(edge_density)
        }
    
    def determine_optimal_noise(self, image):
        """
        Determine optimal noise parameters based on image analysis
        
        Args:
            image: Input image tensor
            
        Returns:
            Dictionary of optimal noise parameters
        """
        # Analyze image
        analysis = self.analyze_image(image)
        
        # Determine optimal noise mode
        # Low edge density and contrast -> adaptive mode (helps find subtle edges)
        # High texture complexity -> structured mode (better for textured regions)
        # High gradient std -> perlin mode (good for varying boundaries)
        # Default -> gaussian mode (general purpose)
        
        if analysis['edge_density'] < 0.05 and analysis['contrast'] < 0.2:
            primary_mode = 'adaptive'
        elif analysis['texture_complexity'] > 0.5:
            primary_mode = 'structured'
        elif analysis['std_gradient'] > 0.2:
            primary_mode = 'perlin'
        else:
            primary_mode = 'gaussian'
            
        # Determine optimal noise scale
        # Higher for low contrast images (need more help finding boundaries)
        # Lower for already noisy images (avoid overwhelming the signal)
        # Adjust based on texture complexity
        
        noise_scale = self.base_noise_scale
        
        # Scale up for low contrast images
        if analysis['contrast'] < 0.1:
            noise_scale *= 1.5
        
        # Scale down for already noisy images
        if analysis['noise_floor'] > 0.1:
            noise_scale *= 0.7
        
        # Adjust based on texture complexity
        noise_scale *= (1.0 + 0.5 * analysis['texture_complexity'])
        
        # Cap the noise scale at reasonable bounds
        noise_scale = max(0.05, min(0.4, noise_scale))
        
        # Determine optimal ensemble size based on image complexity
        # More complex images benefit from larger ensembles
        complexity_score = (analysis['texture_complexity'] + 
                           analysis['std_gradient'] + 
                           analysis['edge_density'])
        
        if complexity_score > 0.6:
            ensemble_size = 5
        elif complexity_score > 0.3:
            ensemble_size = 3
        else:
            ensemble_size = 2
            
        return {
            'primary_mode': primary_mode,
            'noise_scale': float(noise_scale),
            'ensemble_size': int(ensemble_size),
            'image_analysis': analysis
        }
    
    def apply_dynamic_noise(self, image):
        """
        Apply dynamic noise based on optimal parameters
        
        Args:
            image: Input image tensor
            
        Returns:
            Segmentation output with dynamic noise, parameters used
        """
        # Get optimal parameters
        params = self.determine_optimal_noise(image)
        
        # Apply noise with optimal parameters
        self.model.noise_injector.noise_scale = params['noise_scale']
        
        if params['ensemble_size'] > 1:
            # Use ensemble mode
            with torch.no_grad():
                noisy_outputs = []
                
                # Always include primary mode
                output = self.model(image, noise_mode=params['primary_mode'])
                noisy_outputs.append(torch.sigmoid(output))
                
                # Add additional noise patterns
                noise_patterns = self.model.noise_injector.noise_patterns
                for i in range(params['ensemble_size'] - 1):
                    # Use different noise patterns
                    pattern_idx = i % len(noise_patterns)
                    pattern = noise_patterns[pattern_idx]
                    output = self.model(image, noise_mode=pattern)
                    noisy_outputs.append(torch.sigmoid(output))
                
                # Average the outputs
                final_output = torch.stack(noisy_outputs).mean(dim=0)
        else:
            # Use single noise mode
            with torch.no_grad():
                output = self.model(image, noise_mode=params['primary_mode'])
                final_output = torch.sigmoid(output)
        
        return final_output, params

def estimate_noise(gray_img):
    """
    Estimate the noise level in an image using the Laplacian operator
    
    Args:
        gray_img: Grayscale image
        
    Returns:
        Estimated noise level
    """
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    
    # Calculate standard deviation of Laplacian
    noise_estimate = np.std(laplacian)
    
    return noise_estimate

def visualize_dynamic_noise_results(image, output, params, save_path):
    """
    Visualize results with dynamic noise parameters
    
    Args:
        image: Input image tensor
        output: Segmentation output tensor
        params: Noise parameters used
        save_path: Path to save visualization
    """
    # Convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    else:
        image_np = image
        
    if isinstance(output, torch.Tensor):
        output_np = output.squeeze(0).squeeze(0).cpu().numpy()
    else:
        output_np = output
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    # Plot segmentation mask
    mask_np = (output_np > 0.5).astype(np.float32)
    axes[0, 1].imshow(mask_np, cmap="gray")
    axes[0, 1].set_title("Segmentation Mask")
    axes[0, 1].axis("off")
    
    # Plot confidence map
    axes[1, 0].imshow(output_np, cmap="jet")
    axes[1, 0].set_title("Confidence Map")
    axes[1, 0].axis("off")
    
    # Plot parameter info
    axes[1, 1].axis("off")
    info_text = f"""
    Dynamic Noise Parameters:
    
    Primary Mode: {params['primary_mode']}
    Noise Scale: {params['noise_scale']:.3f}
    Ensemble Size: {params['ensemble_size']}
    
    Image Analysis:
    Edge Density: {params['image_analysis']['edge_density']:.3f}
    Texture Complexity: {params['image_analysis']['texture_complexity']:.3f}
    Contrast: {params['image_analysis']['contrast']:.3f}
    Gradient Std: {params['image_analysis']['std_gradient']:.3f}
    Noise Floor: {params['image_analysis']['noise_floor']:.3f}
    """
    axes[1, 1].text(0.05, 0.95, info_text, fontsize=12, va="top")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test dynamic noise generator")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--test_image", type=str, required=True,
                        help="Path to test image")
    parser.add_argument("--output_dir", type=str, default="dynamic_noise_results",
                        help="Directory to save test results")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read and process image
    image = cv2.imread(args.test_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    target_size = (256, 256)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to tensor
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) to (C, H, W)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Create noise injector
    noise_injector = EnhancedNoiseInjector(
        noise_scale=0.15,
        noise_decay=0.98,
        noise_patterns=["gaussian", "perlin", "simplex", "structured", "adaptive"]
    )
    
    # Create and load model
    model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Create dynamic noise generator
    dynamic_generator = DynamicNoiseGenerator(model)
    
    # Apply dynamic noise
    output, params = dynamic_generator.apply_dynamic_noise(image_tensor)
    
    # Visualize results
    save_path = os.path.join(args.output_dir, "dynamic_noise_results.png")
    visualize_dynamic_noise_results(image_tensor, output, params, save_path)
    
    # Save parameters as text file
    with open(os.path.join(args.output_dir, "dynamic_noise_params.md"), "w") as f:
        f.write("# Dynamic Noise Parameters\n\n")
        f.write(f"## Image: {args.test_image}\n\n")
        f.write("### Optimal Parameters\n\n")
        f.write(f"- **Primary Noise Mode**: {params['primary_mode']}\n")
        f.write(f"- **Noise Scale**: {params['noise_scale']:.3f}\n")
        f.write(f"- **Ensemble Size**: {params['ensemble_size']}\n\n")
        
        f.write("### Image Analysis\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, value in params['image_analysis'].items():
            f.write(f"| {key} | {value:.3f} |\n")
            
    print(f"Results saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    main()
