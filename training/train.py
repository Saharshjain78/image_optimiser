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

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.noise_unet import NoiseAwareUNet
from utils.simple_noise_injector import NeuralNoiseInjector
from utils.data_utils import get_data_loaders, SegmentationDataset

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dice_loss(pred, target, smooth=1.0):
    """Calculate Dice loss"""
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def combined_loss(pred, target, bce_weight=0.5):
    """Combination of BCE and Dice loss"""
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    
    return bce_weight * bce + (1 - bce_weight) * dice

def train_model(data_dir, model_save_path, epochs=50, batch_size=8, lr=1e-4, 
                noise_scale=0.1, noise_decay=0.95, train_val_split=0.8):
    """Train the NoiseAwareUNet model"""
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        data_dir, batch_size=batch_size, train_val_split=train_val_split
    )
    
    # Print dataset info
    print(f"Training with {len(train_loader.dataset)} images, validating with {len(val_loader.dataset)} images")
    
    # Initialize noise injector
    noise_injector = NeuralNoiseInjector(
        noise_scale=noise_scale,
        noise_decay=noise_decay,
        noise_patterns=["gaussian", "perlin", "simplex"]
    )
    
    # Create model
    model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
    model.to(device)
      # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': []
    }
    
    best_val_loss = float('inf')
    
    # Create directory for model save path if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, noise_mode="training")
            loss = combined_loss(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass 
                outputs = model(images, noise_mode="disabled")
                loss = combined_loss(outputs, masks)
                
                # Calculate dice coefficient
                preds = torch.sigmoid(outputs) > 0.5
                dice = 1 - dice_loss(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Time')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.title('Dice Coefficient Over Time')
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.dirname(model_save_path)
    plot_path = os.path.join(plot_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training complete. Final model saved to {model_save_path}")
    print(f"Training history plot saved to {plot_path}")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NoiseAwareUNet model")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory with 'images' and 'masks' subdirectories")
    parser.add_argument("--model_save_path", type=str, default="models/noise_unet_model.pth",
                        help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--noise_scale", type=float, default=0.1, 
                        help="Scale of noise injection")
    parser.add_argument("--noise_decay", type=float, default=0.95, 
                        help="Decay rate for noise over iterations")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_scale=args.noise_scale,
        noise_decay=args.noise_decay
    )