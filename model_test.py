import os
import sys
import torch

# Add the project root directory to the Python path
current_dir = os.path.abspath('.')
sys.path.append(current_dir)

try:
    from models.noise_unet import NoiseAwareUNet
    from utils.simple_noise_injector import NeuralNoiseInjector
    
    print("Modules imported successfully")
    
    # Create the model
    noise_injector = NeuralNoiseInjector(noise_scale=0.1, noise_decay=0.95, noise_patterns=["gaussian"])
    model = NoiseAwareUNet(n_channels=3, n_classes=1, noise_injector=noise_injector)
    
    print("Model created successfully")
    
    # Create test input
    test_input = torch.randn(2, 3, 256, 256)
    
    # Test forward pass
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("All tests passed!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
