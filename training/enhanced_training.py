import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import random
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.noise_unet import NoiseAwareUNet
from utils.simple_noise_injector import NeuralNoiseInjector
from utils.data_utils import SegmentationDataset, get_data_loaders

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def dice_coefficient(pred, target, smooth=1.0):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

def dice_loss(pred, target, smooth=1.0):
    """Calculate Dice loss"""
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def focal_loss(pred, target, alpha=0.25, gamma=2):
    """Focal loss for addressing class imbalance"""
    pred = torch.sigmoid(pred)
    
    # Binary case
    p_t = torch.where(target == 1, pred, 1-pred)
    alpha_t = torch.where(target == 1, alpha, 1-alpha)
    
    loss = -alpha_t * (1 - p_t) ** gamma * torch.log(p_t + 1e-10)
    return loss.mean()

def combined_loss(pred, target, bce_weight=0.3, dice_weight=0.5, focal_weight=0.2):
    """Combination of BCE, Dice, and Focal loss"""
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    
    return bce_weight * bce + dice_weight * dice + focal_weight * focal

def get_augmentations(mode='train'):
    """Get augmentation pipeline"""
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.4),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:  # validation
        return A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def create_datasets(data_dir, train_val_split=0.8, target_size=(256, 256)):
    """Create datasets with augmentations"""
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    # List all images
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle and split into train/val
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_val_split)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")
    
    # Get augmentations
    train_transforms = get_augmentations('train')
    val_transforms = get_augmentations('val')
    
    # Create custom datasets with specific image lists and transforms
    class TrainDataset(SegmentationDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.images = train_images
            self.transform = train_transforms
            
    class ValDataset(SegmentationDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.images = val_images
            self.transform = val_transforms
    
    # Create the datasets
    train_dataset = TrainDataset(images_dir, masks_dir, target_size=target_size)
    val_dataset = ValDataset(images_dir, masks_dir, target_size=target_size)
    
    return train_dataset, val_dataset

def create_data_loaders(train_dataset, val_dataset, batch_size=8):
    """Create data loaders"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    return train_loader, val_loader

def evaluate_model(model, val_loader, noise_modes=None):
    """Evaluate model on validation set"""
    if noise_modes is None:
        noise_modes = ["disabled", "single", "ensemble"]
    
    # Results dictionary
    results = {mode: {'loss': 0.0, 'dice': 0.0} for mode in noise_modes}
    
    model.eval()
    with torch.no_grad():
        for mode in noise_modes:
            print(f"Evaluating with noise mode: {mode}")
            total_loss, total_dice = 0.0, 0.0
            
            for images, masks in tqdm(val_loader, desc=f"Validation ({mode})"):
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass with current noise mode
                outputs = model(images, noise_mode=mode)
                loss = combined_loss(outputs, masks)
                
                # Calculate dice coefficient
                dice = dice_coefficient(outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice
                
            # Calculate average metrics
            avg_loss = total_loss / len(val_loader)
            avg_dice = total_dice / len(val_loader)
            
            results[mode]['loss'] = avg_loss
            results[mode]['dice'] = avg_dice
            
            print(f"Noise mode {mode} - Val Loss: {avg_loss:.4f}, Val Dice: {avg_dice:.4f}")
    
    return results

def save_noise_analysis(eval_results, save_path):
    """Save noise analysis plot"""
    modes = list(eval_results.keys())
    dice_scores = [eval_results[mode]['dice'] for mode in modes]
    loss_values = [eval_results[mode]['loss'] for mode in modes]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(modes, dice_scores, color='skyblue')
    plt.ylabel('Dice Coefficient')
    plt.title('Segmentation Quality by Noise Mode')
    plt.ylim(0, 1)
    
    # Add values above bars
    for bar, score in zip(bars, dice_scores):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.01, 
                f'{score:.3f}', 
                ha='center', va='bottom')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(modes, loss_values, color='salmon')
    plt.ylabel('Loss Value')
    plt.title('Loss by Noise Mode')
    
    # Add values above bars
    for bar, loss in zip(bars, loss_values):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.01, 
                f'{loss:.3f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(data_dir, model_save_path, epochs=50, batch_size=8, lr=1e-4, 
                noise_scale=0.1, noise_decay=0.95, train_val_split=0.8,
                noise_patterns=None, save_every=10, resume_from=None):
    """Train the NoiseAwareUNet model with improved training loop"""
    if noise_patterns is None:
        noise_patterns = ["gaussian", "perlin", "simplex"]
    
    # Create datasets and loaders with augmentations
    train_dataset, val_dataset = create_datasets(
        data_dir, train_val_split=train_val_split
    )
    
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, batch_size=batch_size
    )
    
    # Print dataset info
    print(f"Training with {len(train_loader.dataset)} images, validating with {len(val_loader.dataset)} images")
    
    # Initialize noise injector with advanced patterns
    noise_injector = NeuralNoiseInjector(
        noise_scale=noise_scale,
        noise_decay=noise_decay,
        noise_patterns=noise_patterns
    )
    
    # Create model
    model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
    model.to(device)
    
    # Initialize model weights better (optional)
    if resume_from is None:
        # Initialize model weights better
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    else:
        # Resume training from checkpoint
        print(f"Resuming training from {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=device))
    
    # Setup optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler: cosine annealing with warm restarts
    # Effective for finding better local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, # Restart every 10 epochs
        T_mult=1,
        eta_min=lr/10
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_val_dice = 0.0
    
    # Create directories for model save path
    save_dir = os.path.dirname(model_save_path)
    checkpoints_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - Learning Rate: {current_lr:.6f}")
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with different noise modes
            # Randomly switch between training modes to improve robustness
            noise_mode = random.choice(["training", "single", "disabled"])
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
        
        # Step the scheduler
        scheduler.step()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['lr'].append(current_lr)
        
        # Validation phase using standard mode
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass with disabled noise for standard evaluation
                outputs = model(images, noise_mode="disabled")
                loss = combined_loss(outputs, masks)
                
                # Calculate dice coefficient
                dice = dice_coefficient(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Dice: {val_dice:.4f}")
        
        # Save models
        # Save best loss model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best loss model to {model_save_path}")
        
        # Save best dice model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_dice_path = os.path.join(save_dir, 'best_dice_model.pth')
            torch.save(model.state_dict(), best_dice_path)
            print(f"Saved best dice model to {best_dice_path}")
        
        # Save periodic checkpoints
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'history': history
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Evaluate final model with different noise modes
    print("\nEvaluating final model with different noise modes...")
    eval_results = evaluate_model(model, val_loader)
    
    # Save noise analysis plot
    noise_analysis_path = os.path.join(save_dir, 'noise_mode_analysis.png')
    save_noise_analysis(eval_results, noise_analysis_path)
    print(f"Saved noise mode analysis to {noise_analysis_path}")
    
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
    
    # Save plot
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training complete. Final model saved to {model_save_path}")
    print(f"Training history plot saved to {plot_path}")
    
    return model, history, eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NoiseAwareUNet model with enhancements")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory with 'images' and 'masks' subdirectories")
    parser.add_argument("--model_save_path", type=str, default="models/noise_unet_model.pth",
                        help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, 
                        help="Learning rate")
    parser.add_argument("--noise_scale", type=float, default=0.15, 
                        help="Scale of noise injection")
    parser.add_argument("--noise_decay", type=float, default=0.98, 
                        help="Decay rate for noise over iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Additional noise patterns
    noise_patterns = ["gaussian", "perlin", "simplex", "structured", "adaptive"]
    
    train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_scale=args.noise_scale,
        noise_decay=args.noise_decay,
        noise_patterns=noise_patterns,
        save_every=args.save_every,
        resume_from=args.resume_from
    )
