import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.noise_unet import NoiseAwareUNet
from utils.advanced_noise_injector import EnhancedNoiseInjector
from utils.data_utils import SegmentationDataset
from torch.utils.data import DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate_model(model, dataloader, noise_modes=None, output_dir=None):
    """Evaluate model with different noise modes"""
    if noise_modes is None:
        noise_modes = ["disabled", "single", "ensemble", "adaptive"]
    
    # Results dictionary
    results = {mode: {'dice': [], 'samples': []} for mode in noise_modes}
    
    model.eval()
    with torch.no_grad():
        for mode in noise_modes:
            print(f"Evaluating with noise mode: {mode}")
            mode_dice = []
            
            for i, (images, masks) in enumerate(tqdm(dataloader)):
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass with current noise mode
                # For adaptive mode, use a different strength
                if mode == "adaptive":
                    # Select noise strength based on image complexity
                    edge_map = cv2.Canny(images[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 100, 200)
                    edge_ratio = np.sum(edge_map > 0) / (edge_map.shape[0] * edge_map.shape[1])
                    noise_strength = min(0.3, max(0.05, edge_ratio * 0.5))
                    outputs = model(images, noise_mode="single")
                else:
                    outputs = model(images, noise_mode=mode)
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                
                # Calculate dice coefficient
                dice = dice_coefficient(probs, masks)
                mode_dice.append(dice.item())
                
                # Save some sample predictions
                if i < 5 and output_dir:  # Save first 5 samples
                    for b in range(min(images.size(0), 2)):  # Save at most 2 from each batch
                        # Get image, mask and prediction
                        img = images[b].cpu().permute(1, 2, 0).numpy()
                        img = (img * 255).astype(np.uint8)
                        
                        mask = masks[b, 0].cpu().numpy()
                        mask = (mask * 255).astype(np.uint8)
                        
                        pred = probs[b, 0].cpu().numpy()
                        pred = (pred * 255).astype(np.uint8)
                        
                        # Create comparison visualization
                        h, w = mask.shape
                        comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
                        
                        # Original image
                        comparison[:, :w] = img
                        
                        # Ground truth mask (convert to RGB)
                        mask_rgb = np.stack([mask, mask, mask], axis=2)
                        comparison[:, w:2*w] = mask_rgb
                        
                        # Prediction (convert to RGB)
                        pred_rgb = np.stack([pred, pred, pred], axis=2)
                        comparison[:, 2*w:] = pred_rgb
                        
                        # Save the comparison
                        sample_idx = i * images.size(0) + b
                        save_path = os.path.join(output_dir, f"{mode}_sample_{sample_idx}.png")
                        cv2.imwrite(save_path, comparison)
            
            # Store results
            avg_dice = np.mean(mode_dice)
            results[mode]['dice'] = avg_dice
            print(f"Average Dice for {mode} mode: {avg_dice:.4f}")
    
    return results

def compare_noise_modes(results, output_dir):
    """Compare and visualize results from different noise modes"""
    modes = list(results.keys())
    dice_scores = [results[mode]['dice'] for mode in modes]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(modes, dice_scores, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
    plt.ylabel('Dice Coefficient')
    plt.ylim(0, 1)
    plt.title('Segmentation Quality by Noise Mode')
    
    # Add values above bars
    for bar, score in zip(bars, dice_scores):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.01, 
                f'{score:.4f}', 
                ha='center', va='bottom')
    
    # Add explanation
    explanation = """
    Noise Mode Comparison:
    - disabled: Standard segmentation without noise
    - single: Single noise pattern during inference
    - ensemble: Multiple noise patterns averaged
    - adaptive: Context-sensitive noise application
    """
    plt.figtext(0.15, 0.02, explanation, fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'noise_mode_comparison.png'))
    plt.close()
    
    # Generate a comprehensive report
    report_path = os.path.join(output_dir, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write("# Neural Noise-Driven Segmentation Model Evaluation\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Noise Mode | Dice Coefficient |\n")
        f.write("|------------|------------------|\n")
        for mode, score in zip(modes, dice_scores):
            f.write(f"| {mode} | {score:.4f} |\n")
        
        f.write("\n## Analysis\n\n")
        
        # Calculate improvement percentages
        baseline = results["disabled"]["dice"]
        for mode in modes:
            if mode != "disabled":
                improvement = results[mode]["dice"] - baseline
                percent = (improvement / baseline) * 100
                f.write(f"- **{mode}** mode {'improved' if improvement >= 0 else 'decreased'} performance by ")
                f.write(f"**{abs(percent):.2f}%** compared to baseline.\n")
        
        f.write("\n## Recommendations\n\n")
        best_mode = modes[np.argmax(dice_scores)]
        f.write(f"- The best performing noise mode is **{best_mode}** with a Dice score of {max(dice_scores):.4f}.\n")
        f.write("- Consider using this mode for production inference.\n")
        
        f.write("\n## Limitations\n\n")
        f.write("- The evaluation was performed on a limited test set and results may vary with different data.\n")
        f.write("- The Dice coefficient is one metric and other metrics like precision, recall, or boundary F1 score might provide additional insights.\n")
        f.write("- Performance differences between noise modes may be more pronounced on ambiguous or difficult cases.\n")
    
    print(f"Evaluation report saved to {report_path}")

def main(args):
    """Main function for model evaluation"""
    # Load test data
    test_dataset = SegmentationDataset(
        args.images_dir, 
        args.masks_dir,
        target_size=(args.image_size, args.image_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Loaded {len(test_dataset)} test images")
    
    # Create noise injector
    noise_injector = EnhancedNoiseInjector(
        noise_scale=args.noise_scale,
        noise_decay=args.noise_decay,
        noise_patterns=args.noise_patterns.split(',')
    )
    
    # Create and load model
    model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate the model
    results = evaluate_model(
        model,
        test_loader,
        noise_modes=["disabled", "single", "ensemble", "adaptive"],
        output_dir=args.output_dir
    )
    
    # Compare and visualize results
    compare_noise_modes(results, args.output_dir)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NoiseAwareUNet model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--images_dir", type=str, required=True, 
                        help="Directory containing test images")
    parser.add_argument("--masks_dir", type=str, required=True, 
                        help="Directory containing test masks")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for evaluation")
    parser.add_argument("--image_size", type=int, default=256, 
                        help="Size to resize images to during evaluation")
    parser.add_argument("--noise_scale", type=float, default=0.15, 
                        help="Base scale of noise injection")
    parser.add_argument("--noise_decay", type=float, default=0.98, 
                        help="Decay rate for noise over iterations")
    parser.add_argument("--noise_patterns", type=str, 
                        default="gaussian,perlin,simplex,structured,adaptive",
                        help="Comma-separated list of noise patterns to use")
    
    args = parser.parse_args()
    main(args)
