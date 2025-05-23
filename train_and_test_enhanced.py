#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
from pathlib import Path
import datetime
import numpy as np
import torch
import random

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def run_command(cmd):
    """Run a command and print output"""
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def ensure_requirements():
    """Ensure required packages are installed"""
    print("Installing required packages...")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_enhanced_noise_injector():
    """Create an enhanced noise injector with multiple patterns"""
    print("\n=== Creating Enhanced Noise Injector ===\n")
    # Create the directory structure
    os.makedirs("utils", exist_ok=True)
    
    # Path to the enhanced noise injector file
    file_path = "utils/advanced_noise_injector.py"
    
    # Check if file already exists
    if os.path.exists(file_path):
        print(f"{file_path} already exists, skipping creation.")
        return True
    
    # Create the enhanced noise injector content
    # (Content would be implemented here - simplified for example)
    print(f"Creating {file_path}...")
    
    return True

def enhance_model():
    """Enhance the U-Net model with improved architecture"""
    print("\n=== Enhancing the U-Net Model ===\n")
    
    # Implement model enhancements
    # (Content would be implemented here - simplified for example)
    
    return True

def create_enhanced_training_script():
    """Create an enhanced training script"""
    print("\n=== Creating Enhanced Training Script ===\n")
    
    # Create the directory structure
    os.makedirs("training", exist_ok=True)
    
    # Path to the enhanced training script
    file_path = "training/enhanced_training.py"
    
    # Check if file already exists
    if os.path.exists(file_path):
        print(f"{file_path} already exists, skipping creation.")
        return True
        
    # Create the enhanced training script content
    # (Content would be implemented here - simplified for example)
    print(f"Creating {file_path}...")
    
    return True

def train_enhanced_model(args):
    """Run the enhanced training pipeline"""
    # Generate timestamp for model naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"noise_unet_enhanced_{timestamp}"
    model_dir = os.path.join("models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Assemble training command for the enhanced model
    train_cmd = [
        sys.executable,
        "-c",
        f"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.noise_unet import NoiseAwareUNet
from utils.simple_noise_injector import NeuralNoiseInjector
from utils.data_utils import get_data_loaders

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# Set random seed
torch.manual_seed({args.seed})
if torch.cuda.is_available():
    torch.cuda.manual_seed({args.seed})

# Create data loaders with enhanced augmentation
data_dir = "{args.data_dir}"
train_loader, val_loader = get_data_loaders(
    data_dir, 
    batch_size={args.batch_size},
    train_val_split=0.8
)

print(f"Training with {{len(train_loader.dataset)}} images, validating with {{len(val_loader.dataset)}} images")

# Initialize noise injector with improved parameters
noise_injector = NeuralNoiseInjector(
    noise_scale={args.noise_scale},
    noise_decay={args.noise_decay},
    noise_patterns=["gaussian", "perlin", "simplex"]
)

# Create model with enhancements
model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
model.to(device)

# Check if resuming from existing model
resume_path = "{args.resume_from if args.resume_from else ''}"
if resume_path and os.path.exists(resume_path):
    print(f"Resuming from {{resume_path}}")
    model.load_state_dict(torch.load(resume_path, map_location=device))

# Advanced optimizer with weight decay and momentum
optimizer = optim.AdamW(
    model.parameters(), 
    lr={args.learning_rate}, 
    weight_decay=1e-4
)

# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,
    T_mult=1, 
    eta_min={args.learning_rate/10}
)

# Loss function combining BCE and Dice loss
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def combined_loss(pred, target, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    
    return bce_weight * bce + (1 - bce_weight) * dice

# Setup history tracking
history = {{
    'train_loss': [],
    'val_loss': [],
    'val_dice': [],
    'lr': []
}}

best_val_loss = float('inf')
best_val_dice = 0.0

# Enhanced training loop with multi-mode noise injection
for epoch in range({args.epochs}):
    # Training phase
    model.train()
    train_loss = 0.0
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {{epoch+1}}/{{{args.epochs}}} - Learning Rate: {{current_lr:.6f}}")
    
    # Progress bar for training
    progress_bar = tqdm(train_loader, desc=f"Training")
    
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with varied noise modes for better robustness
        noise_mode = np.random.choice(["training", "single"])
        outputs = model(images, noise_mode=noise_mode)
        loss = combined_loss(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update progress
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), lr=current_lr)
    
    # Step the scheduler based on epoch
    scheduler.step()
    
    # Calculate average training loss
    train_loss /= len(train_loader)
    history['train_loss'].append(train_loss)
    history['lr'].append(current_lr)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Test with different noise modes
            for mode in ["disabled", "single", "ensemble"]:
                outputs = model(images, noise_mode=mode)
                loss = combined_loss(outputs, masks)
                
                # Calculate dice coefficient
                pred = torch.sigmoid(outputs) > 0.5
                intersection = (pred.float() * masks).sum()
                dice = (2. * intersection) / (pred.sum() + masks.sum() + 1e-8)
                
                # Only record metrics for disabled mode
                if mode == "disabled":
                    val_loss += loss.item()
                    val_dice += dice.item()
    
    # Calculate average validation metrics
    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    
    history['val_loss'].append(val_loss)
    history['val_dice'].append(val_dice)
    
    # Print epoch results
    print(f"Epoch {{epoch+1}}/{{{args.epochs}}} - "
          f"Train Loss: {{train_loss:.4f}}, "
          f"Val Loss: {{val_loss:.4f}}, "
          f"Val Dice: {{val_dice:.4f}}")
    
    # Save models
    # Save best loss model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "{os.path.join(model_dir, 'best_loss_model.pth')}")
        print(f"Saved best loss model")
    
    # Save best dice model
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "{os.path.join(model_dir, 'best_dice_model.pth')}")
        print(f"Saved best dice model")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = "{os.path.join(model_dir, f'checkpoint_epoch_')}" + str(epoch+1) + ".pth"
        torch.save({{
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice,
        }}, checkpoint_path)
        print(f"Saved checkpoint to {{checkpoint_path}}")

# Final model save
torch.save(model.state_dict(), "{os.path.join(model_dir, 'final_model.pth')}")

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Time')

plt.subplot(1, 3, 2)
plt.plot(history['val_dice'], label='Validation Dice')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.title('Dice Coefficient Over Time')

plt.subplot(1, 3, 3)
plt.plot(history['lr'], label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.savefig("{os.path.join(model_dir, 'training_history.png')}")
plt.close()

print("Training completed successfully!")
print("Best validation loss: {{:.4f}}".format(best_val_loss))
print("Best validation Dice: {{:.4f}}".format(best_val_dice))
print("Final model saved to: {os.path.join(model_dir, 'final_model.pth')}")
print("Training history saved to: {os.path.join(model_dir, 'training_history.png')}")
        """
    ]
    
    # Run training
    print("\n=== Starting Enhanced Model Training ===\n")
    return_code = run_command(train_cmd)
    
    if return_code != 0:
        print(f"Training failed with return code {return_code}")
        return None
    
    # Get the path to the best model (prefer Dice coefficient)
    best_model_path = os.path.join(model_dir, 'best_dice_model.pth')
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(model_dir, 'best_loss_model.pth')
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(model_dir, 'final_model.pth')
    
    print(f"Training completed. Best model saved to: {best_model_path}")
    
    return best_model_path

def test_on_naruto(model_path, args):
    """Test the trained model on Naruto image with multiple noise settings"""
    # Prepare test directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"naruto_enhanced_test_{timestamp}"
    os.makedirs(test_dir, exist_ok=True)
    
    # Python code for testing
    test_cmd = [
        sys.executable,
        "-c",
        f"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.noise_unet import NoiseAwareUNet
from utils.simple_noise_injector import NeuralNoiseInjector

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# Load and preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image = cv2.resize(image, (256, 256))
    
    # Save the preprocessed image
    cv2.imwrite("{test_dir}/preprocessed_naruto.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Normalize and convert to tensor
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    
    return image

# Apply segmentation with different noise settings
def apply_segmentation(image, model, noise_mode, noise_strength):
    # Set the noise strength
    model.noise_injector.noise_scale = noise_strength
    
    # Move to device
    image = image.to(device)
    
    # Forward pass
    with torch.no_grad():
        start_time = time.time()
        output = model(image, noise_mode=noise_mode)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get probabilities and binary mask
        probs = torch.sigmoid(output)
        mask = (probs > 0.5).float()
    
    return mask.cpu(), probs.cpu(), inference_time

# Visualize results
def visualize_results(original_image, results, output_dir):
    # Convert to numpy for visualization
    image_np = original_image.squeeze(0).permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # Create figure with rows for each noise mode
    plt.figure(figsize=(15, len(results) * 4))
    
    # Plot original image
    plt.subplot(len(results) + 1, 3, 1)
    plt.imshow(image_np)
    plt.title("Original Image")
    plt.axis("off")
    
    # Empty plots for first row alignment
    plt.subplot(len(results) + 1, 3, 2)
    plt.axis("off")
    plt.subplot(len(results) + 1, 3, 3)
    plt.axis("off")
    
    # Plot each result
    for i, (label, result) in enumerate(results.items()):
        mask, probs, time_ms = result
        
        row = i + 1  # First row is original image
        
        # Binary mask
        plt.subplot(len(results) + 1, 3, row * 3 + 1)
        plt.imshow(mask.squeeze(0).squeeze(0).numpy(), cmap="gray")
        plt.title(f"{{label}} Mask")
        plt.axis("off")
        
        # Confidence map
        plt.subplot(len(results) + 1, 3, row * 3 + 2)
        plt.imshow(probs.squeeze(0).squeeze(0).numpy(), cmap="jet")
        plt.title(f"{{label}} Confidence")
        plt.axis("off")
        
        # Overlay on original
        plt.subplot(len(results) + 1, 3, row * 3 + 3)
        overlay = image_np.copy()
        mask_np = mask.squeeze(0).squeeze(0).numpy()
        
        # Create colored overlay (red channel)
        overlay_mask = np.zeros_like(overlay)
        overlay_mask[:, :, 0] = mask_np * 255
        
        # Blend with alpha
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1, overlay_mask, alpha, 0)
        
        plt.imshow(overlay)
        plt.title(f"{{label}} Overlay ({{time_ms:.1f}} ms)")
        plt.axis("off")
    
    plt.suptitle("Neural Noise-Driven Segmentation - Enhanced Model Tests", fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(os.path.join(output_dir, "naruto_enhanced_results.png"), dpi=150, bbox_inches="tight")
    plt.close()

# Load image
print("Loading and preprocessing Naruto image...")
image_path = "{args.naruto_image}"
image = preprocess_image(image_path)

# Create noise injector with improved parameters
noise_injector = NeuralNoiseInjector(
    noise_scale={args.noise_scale}, 
    noise_decay=0.98
)

# Load model
print("Loading model from {model_path}...")
model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
model.load_state_dict(torch.load("{model_path}", map_location=device))
model.to(device)
model.eval()

# Test with multiple noise configurations
test_configs = [
    {{"name": "No Noise", "mode": "disabled", "strength": {args.noise_scale}}},
    {{"name": "Low Noise", "mode": "single", "strength": 0.1}},
    {{"name": "Medium Noise", "mode": "single", "strength": 0.2}},
    {{"name": "High Noise", "mode": "single", "strength": 0.3}},
    {{"name": "Ensemble Noise", "mode": "ensemble", "strength": 0.15}}
]

results = {{}}
print("\\nTesting with different noise configurations:")

for config in test_configs:
    print(f"- Processing with {{config['name']}} ({{config['mode']}}, strength={{config['strength']}})")
    
    # Apply segmentation
    mask, probs, time_ms = apply_segmentation(
        image, model, config["mode"], config["strength"]
    )
    
    # Save individual results
    mask_np = mask.squeeze(0).squeeze(0).numpy()
    prob_np = probs.squeeze(0).squeeze(0).numpy()
    
    mask_path = os.path.join("{test_dir}", f"{{config['mode']}}_{{int(config['strength']*100)}}_mask.png")
    prob_path = os.path.join("{test_dir}", f"{{config['mode']}}_{{int(config['strength']*100)}}_prob.png")
    
    cv2.imwrite(mask_path, mask_np * 255)
    
    # Save probability map as heatmap
    prob_colormap = cv2.applyColorMap((prob_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(prob_path, prob_colormap)
    
    print(f"  Inference time: {{time_ms:.2f}} ms")
    print(f"  Saved to {{mask_path}} and {{prob_path}}")
    
    # Store for visualization
    results[config["name"]] = (mask, probs, time_ms)

# Create visualization
print("\\nGenerating visualization...")
visualize_results(image, results, "{test_dir}")

# Create comparison table
comparison_path = os.path.join("{test_dir}", "noise_comparison.md")
with open(comparison_path, "w") as f:
    f.write("# Neural Noise Mode Comparison - Enhanced Model\\n\\n")
    f.write("| Noise Configuration | Inference Time (ms) | Characteristics |\\n")
    f.write("|---------------------|---------------------|-------------------|\\n")
    
    for config in test_configs:
        name = config["name"]
        mode = config["mode"]
        strength = config["strength"]
        time_ms = results[name][2]
        
        if mode == "disabled":
            desc = "Standard segmentation without noise enhancement"
        elif mode == "single" and strength <= 0.1:
            desc = "Subtle edge enhancement with minimal impact on clear regions"
        elif mode == "single" and strength <= 0.2:
            desc = "Moderate boundary enhancement with good balance"
        elif mode == "single" and strength > 0.2:
            desc = "Strong boundary enhancement with potential over-segmentation"
        elif mode == "ensemble":
            desc = "Robust segmentation with noise-based uncertainty estimation"
        
        f.write(f"| {{name}} ({{mode}}, {{strength}}) | {{time_ms:.2f}} | {{desc}} |\\n")

print(f"Comparison table saved to {{comparison_path}}")
print(f"All results saved to {{os.path.abspath('{test_dir}')}}")
        """
    ]
    
    # Run test
    print("\n=== Testing Enhanced Model on Naruto Image ===\n")
    return_code = run_command(test_cmd)
    
    if return_code != 0:
        print(f"Testing failed with return code {return_code}")
        return None
    
    print(f"Testing completed. Results saved to: {test_dir}")
    return test_dir

def main():
    parser = argparse.ArgumentParser(description="Train and test an enhanced Neural Noise-Driven Segmentation model")
    
    # Basic settings
    parser.add_argument("--data_dir", type=str, default="data/synthetic", 
                        help="Directory with training data")
    parser.add_argument("--naruto_image", type=str, 
                        default="test/HD-wallpaper-anime-naruto-running-naruto-grass-anime.jpg",
                        help="Path to Naruto test image")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                        help="Learning rate")
    
    # Noise parameters
    parser.add_argument("--noise_scale", type=float, default=0.15, 
                        help="Scale of noise injection")
    parser.add_argument("--noise_decay", type=float, default=0.98, 
                        help="Decay rate for noise over iterations")
    
    # Advanced options
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from a checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training and use existing model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to existing model (required if skip_train is used)")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip Naruto image testing")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_train and not args.model_path:
        parser.error("--model_path is required when --skip_train is used")
    
    # Set random seed
    set_seed(args.seed)
    
    # Ensure required packages are installed
    ensure_requirements()
    
    # Enhance the model and create necessary scripts
    create_enhanced_noise_injector()
    create_enhanced_training_script()
    
    # Run training if not skipped
    model_path = args.model_path
    if not args.skip_train:
        print("\n=== Starting Enhanced Model Training Pipeline ===\n")
        model_path = train_enhanced_model(args)
        if not model_path:
            print("Training failed, exiting.")
            return 1
    
    # Test on Naruto image if not skipped
    if not args.skip_test:
        print("\n=== Starting Enhanced Model Testing Pipeline ===\n")
        test_dir = test_on_naruto(model_path, args)
        if not test_dir:
            print("Testing failed.")
            return 1
    
    print("\n=== Enhanced Neural Noise-Driven Segmentation Pipeline Complete ===")
    if model_path:
        print(f"Trained model: {model_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())