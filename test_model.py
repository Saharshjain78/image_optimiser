import os
import sys
import torch

# Add the project root directory to the Python path
print("Starting test script...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Current directory: {os.getcwd()}")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(f"Python path: {sys.path}")

try:
    print("Importing models...")
    from models.noise_unet import NoiseAwareUNet
    print("Importing utils...")
    from utils.noise_injector import NeuralNoiseInjector
    from PIL import Image
    import numpy as np
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_model_creation():
    try:
        print("Creating noise injector...")
        noise_injector = NeuralNoiseInjector(
            noise_scale=0.1,
            noise_decay=0.95,
            noise_patterns=["gaussian"]
        )
        
        print("Creating model...")
        model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
        
        print("Creating test input...")
        test_input = torch.randn(2, 3, 256, 256)  # Batch size 2, RGB, 256x256
        
        print("Running forward pass with noise mode 'training'...")
        output = model(test_input, noise_mode="training")
        
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
        
        print("Testing noise injection...")
        test_feature = torch.randn(2, 64, 128, 128)
        noise_result = noise_injector.inject_noise(test_feature)
        
        print("Noise injection successful!")
        print(f"Noise result shape: {noise_result.shape}")
        return True
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_creation()
