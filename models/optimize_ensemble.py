import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.noise_unet import NoiseAwareUNet
from utils.advanced_noise_injector import EnhancedNoiseInjector

def optimize_ensemble(model, image_tensor, noise_strength=0.15, num_trials=5):
    """
    Find the optimal number of ensemble elements to maximize performance vs time
    
    Args:
        model: NoiseAwareUNet model
        image_tensor: Input image tensor
        noise_strength: Strength of noise to apply
        num_trials: Number of trials for each ensemble size
        
    Returns:
        Dictionary with performance metrics for different ensemble sizes
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    model.eval()
    
    # Test different ensemble sizes
    ensemble_sizes = [1, 2, 3, 5, 7, 10]
    results = {}
    
    for size in ensemble_sizes:
        # Skip size 1 as it's equivalent to single mode
        if size == 1:
            continue
            
        print(f"Testing ensemble size: {size}")
        times = []
        similarities = []  # IoU with the largest ensemble
        
        for _ in range(num_trials):
            # Reset noise scale
            model.noise_injector.noise_scale = noise_strength
            
            # Measure time
            start_time = time.time()
            
            with torch.no_grad():
                # Custom ensemble generation with specified size
                noisy_outputs = []
                for i in range(size):
                    # Generate noise modes randomly
                    noise_mode = np.random.choice(['gaussian', 'perlin', 'simplex', 'structured'])
                    output = model(image_tensor, noise_mode=noise_mode)
                    noisy_outputs.append(torch.sigmoid(output))
                
                # Average the outputs
                ensemble_output = torch.stack(noisy_outputs).mean(dim=0)
                
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(inference_time)
            
            # Store the result for comparison with largest ensemble
            if size == ensemble_sizes[-1]:
                reference_output = ensemble_output.clone()
            
            # Calculate IoU with reference (largest ensemble)
            if 'reference_output' in locals():
                # Binarize outputs
                binary_output = (ensemble_output > 0.5).float()
                binary_reference = (reference_output > 0.5).float()
                
                # Calculate IoU
                intersection = (binary_output * binary_reference).sum().item()
                union = (binary_output + binary_reference).clamp(0, 1).sum().item()
                iou = intersection / (union + 1e-8)
                similarities.append(iou)
        
        # Record average results
        results[size] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'similarity': np.mean(similarities) if similarities else 1.0  # Set to 1.0 for reference
        }
    
    # Analyze and visualize results
    plot_ensemble_analysis(results, ensemble_sizes)
    
    # Determine optimal ensemble size
    optimal_size = find_optimal_ensemble_size(results, ensemble_sizes)
    print(f"Optimal ensemble size: {optimal_size}")
    
    return results, optimal_size

def find_optimal_ensemble_size(results, ensemble_sizes):
    """Find optimal ensemble size based on time/accuracy tradeoff"""
    # Skip size 1 which is equivalent to single mode
    valid_sizes = [s for s in ensemble_sizes if s > 1]
    
    # Calculate efficiency metric (similarity / log(time))
    efficiency = {}
    for size in valid_sizes:
        if size in results:
            # Higher is better: we want high similarity with low time
            efficiency[size] = results[size]['similarity'] / np.log10(results[size]['avg_time'])
    
    # Find size with best efficiency
    optimal_size = max(efficiency.items(), key=lambda x: x[1])[0]
    return optimal_size

def plot_ensemble_analysis(results, ensemble_sizes):
    """Plot analysis of ensemble size vs performance"""
    valid_sizes = [s for s in ensemble_sizes if s > 1 and s in results]
    
    times = [results[s]['avg_time'] for s in valid_sizes]
    similarities = [results[s]['similarity'] for s in valid_sizes]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time plot
    ax1.errorbar(
        valid_sizes, 
        times, 
        yerr=[results[s]['std_time'] for s in valid_sizes],
        marker='o',
        capsize=5
    )
    ax1.set_xlabel('Ensemble Size')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Ensemble Size vs. Inference Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Similarity plot
    ax2.plot(valid_sizes, similarities, marker='o')
    ax2.set_xlabel('Ensemble Size')
    ax2.set_ylabel('Similarity to Full Ensemble')
    ax2.set_title('Ensemble Size vs. Similarity')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('ensemble_optimization.png')
    plt.close()

def load_model_and_test(model_path, test_image_path, output_dir='optimized_results'):
    """Load model and test with optimized ensemble"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test image
    import cv2
    from PIL import Image
    
    # Read image
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    target_size = (256, 256)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to tensor
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) to (C, H, W)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Create noise injector
    noise_injector = EnhancedNoiseInjector(
        noise_scale=0.15,
        noise_decay=0.98,
        noise_patterns=["gaussian", "perlin", "simplex", "structured", "adaptive"]
    )
    
    # Create and load model
    model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
    
    # Load model weights
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Run ensemble optimization
    os.makedirs(output_dir, exist_ok=True)
    results, optimal_size = optimize_ensemble(model, image_tensor)
    
    return results, optimal_size

def implement_optimized_ensemble_mode(model, image_tensor, optimal_size=3, noise_strength=0.15):
    """
    Implement an optimized ensemble mode with the optimal number of noise patterns
    
    Args:
        model: NoiseAwareUNet model
        image_tensor: Input image tensor
        optimal_size: Number of ensemble members to use
        noise_strength: Strength of noise to apply
        
    Returns:
        Optimized ensemble prediction
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    model.eval()
    
    # Reset noise scale
    model.noise_injector.noise_scale = noise_strength
    
    with torch.no_grad():
        # Generate outputs with different noise patterns
        noisy_outputs = []
        
        # Always include no-noise output for stability
        output = model(image_tensor, noise_mode="disabled")
        noisy_outputs.append(torch.sigmoid(output))
        
        # Add remaining ensemble members
        for i in range(optimal_size - 1):
            # Use different noise patterns
            pattern_idx = i % len(model.noise_injector.noise_patterns)
            pattern = model.noise_injector.noise_patterns[pattern_idx]
            output = model(image_tensor, noise_mode=pattern)
            noisy_outputs.append(torch.sigmoid(output))
        
        # Average the outputs
        ensemble_output = torch.stack(noisy_outputs).mean(dim=0)
    
    return ensemble_output

def main():
    parser = argparse.ArgumentParser(description="Optimize ensemble mode for image segmentation")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--test_image", type=str, required=True,
                        help="Path to test image")
    parser.add_argument("--output_dir", type=str, default="optimized_results",
                        help="Directory to save optimization results")
    
    args = parser.parse_args()
    
    results, optimal_size = load_model_and_test(
        args.model_path, 
        args.test_image,
        args.output_dir
    )
    
    # Write results to file
    with open(os.path.join(args.output_dir, "ensemble_optimization_results.md"), "w") as f:
        f.write("# Ensemble Mode Optimization Results\n\n")
        f.write("## Performance Analysis\n\n")
        f.write("| Ensemble Size | Inference Time (ms) | Similarity to Full Ensemble |\n")
        f.write("|---------------|---------------------|-----------------------------|\n")
        
        for size in sorted(results.keys()):
            f.write(f"| {size} | {results[size]['avg_time']:.2f} ± {results[size]['std_time']:.2f} | {results[size]['similarity']:.4f} |\n")
        
        f.write(f"\n## Conclusion\n\n")
        f.write(f"The optimal ensemble size is **{optimal_size}**, which provides the best balance of accuracy and inference speed.\n\n")
        f.write(f"![Ensemble Optimization](ensemble_optimization.png)\n")
    
    print(f"Optimization results saved to {args.output_dir}/ensemble_optimization_results.md")
    
    return 0

if __name__ == "__main__":
    main()
