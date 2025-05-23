#!/usr/bin/env python
"""
Quick Start Script for Neural Noise Segmentation App
This script validates dependencies, performs a quick model check, and launches the app
"""

import os
import sys
import subprocess
import time

# Get the project root directory
root_dir = os.path.dirname(os.path.abspath(__file__))

# Add project root to Python path
sys.path.append(root_dir)

def print_colored(message, color="reset"):
    """Print colored text in terminal"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['reset'])}{message}{colors['reset']}")

def check_requirements():
    """Check if all required packages are installed"""
    print_colored("Checking required packages...", "blue")
    
    try:
        import torch
        import streamlit
        import PIL
        import numpy
        import cv2
        import matplotlib
        print_colored("✓ All critical packages are installed!", "green")
        return True
    except ImportError as e:
        print_colored(f"✗ Missing package: {e}", "red")
        
        # Attempt to install requirements
        print_colored("Attempting to install required packages...", "yellow")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", os.path.join(root_dir, "requirements.txt")])
        
        # Verify installation
        try:
            import torch
            import streamlit
            import PIL
            import numpy
            import cv2
            import matplotlib
            print_colored("✓ Successfully installed packages!", "green")
            return True
        except ImportError as e:
            print_colored(f"✗ Failed to install package: {e}", "red")
            print_colored("Please run 'pip install -r requirements.txt' manually.", "yellow")
            return False

def validate_model():
    """Validate that the model loads correctly"""
    print_colored("Validating model...", "blue")
    
    try:
        from models.noise_unet import NoiseAwareUNet
        # Create a minimal model instance
        model = NoiseAwareUNet(n_channels=3, n_classes=1)
        print_colored("✓ Model validation successful!", "green")
        return True
    except Exception as e:
        print_colored(f"✗ Model validation failed: {e}", "red")
        return False

def run_app():
    """Run the Streamlit app"""
    print_colored("Starting the Neural Noise Segmentation App...", "blue")
    
    # Ensure sample images exist
    try:
        from utils.image_utils import create_sample_images
        create_sample_images()
        print_colored("✓ Sample images created successfully!", "green")
    except Exception as e:
        print_colored(f"! Sample image creation warning: {e}", "yellow")
    
    # Launch Streamlit app
    try:
        print_colored("Launching Streamlit interface (this may take a moment)...", "cyan")
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.join(root_dir, "streamlit_app.py")])
    except Exception as e:
        print_colored(f"✗ Failed to launch app: {e}", "red")
        return False
    
    return True

if __name__ == "__main__":
    print_colored("=" * 80, "magenta")
    print_colored("Neural Noise Segmentation - Quick Start", "magenta")
    print_colored("=" * 80, "magenta")
    
    # Run checks and start app
    if check_requirements() and validate_model():
        print_colored("All checks passed! Starting application...", "green")
        run_app()
    else:
        print_colored("Some checks failed. Please fix the issues before running the app.", "red")
        sys.exit(1)
