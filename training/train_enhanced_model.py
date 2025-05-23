import os
import sys
import argparse
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from training.enhanced_training import train_model, set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NoiseAwareUNet model with enhanced settings")
    
    # Data and model paths
    parser.add_argument("--data_dir", type=str, required=True, 
                      help="Directory with 'images' and 'masks' subdirectories")
    parser.add_argument("--model_save_dir", type=str, default="models",
                      help="Directory to save models and results")
    parser.add_argument("--model_name", type=str, default="enhanced_noise_unet",
                      help="Base name for the model files")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, 
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                      help="Base learning rate")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--save_every", type=int, default=10,
                      help="Save checkpoint every N epochs")
    
    # Noise parameters
    parser.add_argument("--noise_scale", type=float, default=0.15, 
                      help="Base scale of noise injection")
    parser.add_argument("--noise_decay", type=float, default=0.98, 
                      help="Decay rate for noise over iterations")
    parser.add_argument("--noise_patterns", type=str, default="gaussian,perlin,simplex,structured,adaptive",
                      help="Comma-separated list of noise patterns to use")
    
    # Advanced options
    parser.add_argument("--resume_from", type=str, default=None,
                      help="Resume training from checkpoint path")
    parser.add_argument("--image_size", type=int, default=256,
                      help="Size to resize images to during training")
    parser.add_argument("--augmentation_strength", type=str, default="medium",
                      choices=["none", "light", "medium", "heavy"],
                      help="Strength of data augmentation")
    
    args = parser.parse_args()
    
    # Setup paths
    timestamp = Path(__file__).stem
    model_save_dir = Path(args.model_save_dir) / f"{args.model_name}_{timestamp}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_path = model_save_dir / f"{args.model_name}.pth"
    
    # Parse noise patterns
    noise_patterns = args.noise_patterns.split(',')
    
    # Set random seed
    set_seed(args.seed)
    
    print(f"Starting enhanced training with the following settings:")
    print(f"- Data directory: {args.data_dir}")
    print(f"- Model save path: {model_save_path}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Noise scale: {args.noise_scale}")
    print(f"- Noise decay: {args.noise_decay}")
    print(f"- Noise patterns: {noise_patterns}")
    print(f"- Image size: {args.image_size}")
    print(f"- Augmentation strength: {args.augmentation_strength}")
    
    if args.resume_from:
        print(f"- Resuming from: {args.resume_from}")
    
    # Train the model
    train_model(
        data_dir=args.data_dir,
        model_save_path=str(model_save_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        noise_scale=args.noise_scale,
        noise_decay=args.noise_decay,
        noise_patterns=noise_patterns,
        save_every=args.save_every,
        resume_from=args.resume_from
    )
    
    print(f"Training complete. Model saved to {model_save_path}")
