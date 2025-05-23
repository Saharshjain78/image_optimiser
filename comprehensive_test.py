import os
import sys
import torch
import numpy as np
import cv2
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.noise_unet import NoiseAwareUNet
from utils.advanced_noise_injector import EnhancedNoiseInjector
from utils.dynamic_noise_generator import DynamicNoiseGenerator

def preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess an image for the model"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to tensor
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) to (C, H, W)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    return image

def segment_image_with_noise_modes(model, image, noise_modes, noise_strength=0.15, optimal_ensemble_size=3):
    """Segment an image using multiple noise modes with metrics"""
    device = next(model.parameters()).device
    image = image.to(device)
    model.eval()
    
    results = {}
    
    for mode in noise_modes:
        # Reset noise injector
        model.noise_injector.noise_scale = noise_strength
        
        # Segment with current mode
        start_time = time.time()
        
        with torch.no_grad():
            if mode == "disabled":
                output = model(image, noise_mode="disabled")
            elif mode == "single":
                output = model(image, noise_mode="single")
            elif mode == "ensemble":
                # Generate multiple outputs and average
                noisy_outputs = []
                
                # Always include no-noise output for stability
                output = model(image, noise_mode="disabled")
                noisy_outputs.append(torch.sigmoid(output))
                
                # Add remaining ensemble members with different noise patterns
                for i in range(optimal_ensemble_size - 1):
                    # Use different noise patterns
                    pattern_idx = i % len(model.noise_injector.noise_patterns)
                    pattern = model.noise_injector.noise_patterns[pattern_idx]
                    output = model(image, noise_mode=pattern)
                    noisy_outputs.append(torch.sigmoid(output))
                
                # Average the outputs
                ensemble_output = torch.stack(noisy_outputs).mean(dim=0)
                # Prevent double sigmoid
                output = torch.log(ensemble_output / (1 - ensemble_output + 1e-8))  # Inverse sigmoid
            elif mode == "adaptive":
                output = model(image, noise_mode="adaptive")
            elif mode == "dynamic":
                # Use dynamic noise generator
                dynamic_generator = DynamicNoiseGenerator(model, base_noise_scale=noise_strength)
                ensemble_output, params = dynamic_generator.apply_dynamic_noise(image)
                # Prevent double sigmoid
                output = torch.log(ensemble_output / (1 - ensemble_output + 1e-8))  # Inverse sigmoid
                # Store dynamic parameters
                results[mode] = {"dynamic_params": params}
            else:
                output = model(image, noise_mode="single")
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Apply sigmoid to output for visualization (if not already done)
        if mode != "ensemble" and mode != "dynamic":
            probs = torch.sigmoid(output)
        else:
            probs = ensemble_output if mode == "ensemble" else ensemble_output
            
        # Binarize output
        mask = (probs > 0.5).float()
        
        # Store results
        if mode not in results:
            results[mode] = {}
            
        results[mode].update({
            "mask": mask.cpu(),
            "probs": probs.cpu(),
            "inference_time": inference_time
        })
        
    return results

def calculate_metrics(results, reference_mode="ensemble"):
    """Calculate comparison metrics between different modes"""
    # Use ensemble mode as reference by default
    reference_mask = results[reference_mode]["mask"]
    reference_probs = results[reference_mode]["probs"]
    
    metrics = {}
    
    for mode, data in results.items():
        if mode == reference_mode:
            metrics[mode] = {
                "iou": 1.0,  # Self-comparison
                "dice": 1.0,  # Self-comparison
                "ssim": 1.0,  # Self-comparison
                "psnr": float('inf'),  # Self-comparison
                "inference_time": data["inference_time"]
            }
            continue
            
        mask = data["mask"]
        probs = data["probs"]
        
        # Convert tensors to numpy
        mask_np = mask.squeeze().numpy()
        probs_np = probs.squeeze().numpy()
        ref_mask_np = reference_mask.squeeze().numpy()
        ref_probs_np = reference_probs.squeeze().numpy()
        
        # Calculate IoU
        intersection = np.logical_and(mask_np > 0.5, ref_mask_np > 0.5).sum()
        union = np.logical_or(mask_np > 0.5, ref_mask_np > 0.5).sum()
        iou = intersection / (union + 1e-8)
        
        # Calculate Dice coefficient
        dice = (2 * intersection) / (mask_np.sum() + ref_mask_np.sum() + 1e-8)
        
        # Calculate SSIM between probability maps
        ssim_val = ssim(probs_np, ref_probs_np, data_range=1.0)
        
        # Calculate PSNR between probability maps
        psnr_val = psnr(ref_probs_np, probs_np, data_range=1.0)
        
        metrics[mode] = {
            "iou": float(iou),
            "dice": float(dice),
            "ssim": float(ssim_val),
            "psnr": float(psnr_val),
            "inference_time": float(data["inference_time"])
        }
        
        # Add dynamic parameters if available
        if "dynamic_params" in data:
            metrics[mode]["dynamic_params"] = data["dynamic_params"]
    
    return metrics

def visualize_comparison(image, results, metrics, save_path):
    """Create visualization of segmentation results with different noise modes"""
    # Convert image tensor to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.squeeze(0).permute(1, 2, 0).numpy()
    else:
        image_np = image
        
    # Image should be 0-1 range
    if image_np.max() > 1.1:
        image_np = image_np / 255.0
        
    modes = list(results.keys())
    
    # Create main figure
    fig = plt.figure(figsize=(15, 5 + 5 * ((len(modes) + 1) // 2)))
    
    # Plot original image
    ax = fig.add_subplot(len(modes) + 1, 3, 1)
    ax.imshow(image_np)
    ax.set_title("Original Image")
    ax.axis("off")
    
    # Plot metrics table
    ax = fig.add_subplot(len(modes) + 1, 3, 2)
    ax.axis("off")
    table_data = [
        ["Mode", "IoU", "SSIM", "Time (ms)"]
    ]
    for mode in modes:
        if mode in metrics:
            m = metrics[mode]
            table_data.append([
                mode,
                f"{m.get('iou', 0):.3f}",
                f"{m.get('ssim', 0):.3f}",
                f"{m.get('inference_time', 0):.1f}"
            ])
    
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("Performance Metrics")
    
    # Plot noise info
    ax = fig.add_subplot(len(modes) + 1, 3, 3)
    ax.axis("off")
    info_text = """
    Neural Noise Modes:
    
    disabled: No noise injection
    single: Single noise pattern
    ensemble: Multiple noise patterns averaged
    adaptive: Content-adaptive noise
    dynamic: Auto-tuned noise parameters
    
    Higher IoU/SSIM indicates greater similarity
    to the ensemble mode (reference).
    """
    ax.text(0.1, 0.5, info_text, fontsize=10)
    
    # Plot each mode result
    for i, mode in enumerate(modes):
        row = i + 1  # Skip first row for original image
        
        # Get data
        mask = results[mode]["mask"]
        probs = results[mode]["probs"]
        
        # Convert to numpy
        mask_np = mask.squeeze().numpy()
        probs_np = probs.squeeze().numpy()
        
        # Binary mask
        ax = fig.add_subplot(len(modes) + 1, 3, (row + 1) * 3 - 2)
        ax.imshow(mask_np, cmap="gray")
        ax.set_title(f"Mask: {mode}")
        ax.axis("off")
        
        # Confidence map
        ax = fig.add_subplot(len(modes) + 1, 3, (row + 1) * 3 - 1)
        im = ax.imshow(probs_np, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"Confidence: {mode}")
        ax.axis("off")
        
        # Add colorbar (only for first confidence map)
        if i == 0:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Overlay
        ax = fig.add_subplot(len(modes) + 1, 3, (row + 1) * 3)
        overlay = image_np.copy()
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay] * 3, axis=2)
            
        # Create visible overlay
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[:, :, 0] = mask_np * 255  # Red channel
        
        # Blend with alpha
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1, mask_rgb, alpha, 0)
        
        ax.imshow(overlay)
        ax.set_title(f"Overlay: {mode}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return save_path

def save_metrics_report(metrics, save_path):
    """Save metrics report as markdown"""
    with open(save_path, "w") as f:
        f.write("# Neural Noise Mode Comparison Metrics\n\n")
        
        # Create main metrics table
        f.write("## Performance Metrics\n\n")
        f.write("| Noise Mode | IoU | Dice | SSIM | PSNR | Inference Time (ms) |\n")
        f.write("|------------|-----|------|------|------|---------------------|\n")
        
        for mode, m in metrics.items():
            if "dynamic_params" in m:
                # Skip dynamic params here, will show separately
                f.write(f"| {mode} | {m.get('iou', 0):.4f} | {m.get('dice', 0):.4f} | {m.get('ssim', 0):.4f} | {m.get('psnr', 0):.2f} | {m.get('inference_time', 0):.2f} |\n")
            else:
                f.write(f"| {mode} | {m.get('iou', 0):.4f} | {m.get('dice', 0):.4f} | {m.get('ssim', 0):.4f} | {m.get('psnr', 0):.2f} | {m.get('inference_time', 0):.2f} |\n")
        
        # Add dynamic parameters if available
        for mode, m in metrics.items():
            if "dynamic_params" in m:
                f.write(f"\n## Dynamic Parameters for {mode} Mode\n\n")
                
                # Main parameters
                f.write("### Optimal Parameters\n\n")
                params = m["dynamic_params"]
                f.write(f"- **Primary Noise Mode**: {params['primary_mode']}\n")
                f.write(f"- **Noise Scale**: {params['noise_scale']:.3f}\n")
                f.write(f"- **Ensemble Size**: {params['ensemble_size']}\n\n")
                
                # Image analysis parameters
                f.write("### Image Analysis\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                
                for key, value in params['image_analysis'].items():
                    f.write(f"| {key} | {value:.3f} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("- **IoU (Intersection over Union)**: Measures the overlap between segmentation masks. Higher is better, with 1.0 being perfect overlap.\n")
        f.write("- **Dice Coefficient**: Similar to IoU but gives more weight to true positives. Higher is better, with 1.0 being perfect overlap.\n")
        f.write("- **SSIM (Structural Similarity Index)**: Measures the structural similarity between confidence maps. Higher is better, with 1.0 being perfect similarity.\n")
        f.write("- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality between confidence maps. Higher generally indicates better quality.\n")
        f.write("- **Inference Time**: Time taken for model to produce segmentation output in milliseconds.\n\n")

        f.write("The ensemble mode is used as the reference for comparing other modes.\n")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive testing of Neural Noise-Driven Segmentation")
    parser.add_argument("--model_path", type=str, default="models/noise_unet_model.pth",
                        help="Path to the trained model")
    parser.add_argument("--test_image", type=str, required=True,
                        help="Path to test image")
    parser.add_argument("--output_dir", type=str, default="comprehensive_results",
                        help="Directory to save test results")
    parser.add_argument("--noise_strength", type=float, default=0.15,
                        help="Strength of noise to apply")
    parser.add_argument("--optimal_ensemble_size", type=int, default=3,
                        help="Optimal ensemble size (from optimization)")
                        
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocess image
    print(f"Processing image: {args.test_image}")
    image = preprocess_image(args.test_image)
    
    # Initialize noise injector
    noise_injector = EnhancedNoiseInjector(
        noise_scale=args.noise_strength,
        noise_decay=0.98,
        noise_patterns=["gaussian", "perlin", "simplex", "structured", "adaptive"]
    )
    
    # Create and load model
    print(f"Loading model from: {args.model_path}")
    model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Define noise modes to test
    noise_modes = ["disabled", "single", "ensemble", "adaptive", "dynamic"]
    
    # Segment with each mode
    print("Segmenting with different noise modes...")
    results = segment_image_with_noise_modes(
        model, 
        image, 
        noise_modes, 
        noise_strength=args.noise_strength,
        optimal_ensemble_size=args.optimal_ensemble_size
    )
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(results)
    
    # Create visualizations
    print("Creating visualizations...")
    vis_path = os.path.join(args.output_dir, "comprehensive_comparison.png")
    visualize_comparison(image, results, metrics, vis_path)
    
    # Save metrics report
    metrics_path = os.path.join(args.output_dir, "noise_metrics_report.md")
    save_metrics_report(metrics, metrics_path)
    
    # Save metrics as JSON
    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        # Convert any non-serializable objects to strings
        json_metrics = {}
        for mode, m in metrics.items():
            json_metrics[mode] = {}
            for k, v in m.items():
                if k == "dynamic_params":
                    # Handle nested dict
                    json_metrics[mode][k] = {}
                    for pk, pv in v.items():
                        if pk == "image_analysis":
                            # Handle nested dict
                            json_metrics[mode][k][pk] = {}
                            for ak, av in pv.items():
                                json_metrics[mode][k][pk][ak] = av
                        else:
                            json_metrics[mode][k][pk] = pv
                else:
                    json_metrics[mode][k] = v
                    
        json.dump(json_metrics, f, indent=2)
    
    print(f"Results saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    main()
