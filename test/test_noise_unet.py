import unittest
import torch
from models.noise_unet import NoiseAwareUNet
from utils.simple_noise_injector import NeuralNoiseInjector

class TestNoiseUNet(unittest.TestCase):
    def setUp(self):
        self.model = NoiseAwareUNet(3, 1)
        self.input_tensor = torch.randn(1, 3, 256, 256)
        
    def test_forward_disabled(self):
        """Test forward pass with disabled noise"""
        output = self.model(self.input_tensor, noise_mode="disabled")
        self.assertEqual(output.shape, (1, 1, 256, 256))
        
    def test_forward_single(self):
        """Test forward pass with single noise pattern"""
        output = self.model(self.input_tensor, noise_mode="single")
        self.assertEqual(output.shape, (1, 1, 256, 256))
        
    def test_forward_ensemble(self):
        """Test forward pass with ensemble noise"""
        output = self.model(self.input_tensor, noise_mode="ensemble")
        self.assertEqual(output.shape, (1, 1, 256, 256))
        
    def test_noise_injector(self):
        """Test the noise injector component"""
        injector = NeuralNoiseInjector(noise_scale=0.1)
        feature_map = torch.ones(1, 64, 32, 32)
        
        # Test different noise types
        for noise_type in ["gaussian", "perlin", "structured"]:
            noisy_map = injector.inject_noise(feature_map, noise_type)
            self.assertEqual(noisy_map.shape, feature_map.shape)
            # Ensure noise was actually added
            self.assertFalse(torch.allclose(noisy_map, feature_map))
            
    def test_resize_handling(self):
        """Test that the model handles non-standard input sizes"""
        non_standard_tensor = torch.randn(1, 3, 300, 400)  # Non-standard size
        output = self.model(non_standard_tensor)
        self.assertEqual(output.shape, (1, 1, 300, 400))
        
if __name__ == '__main__':
    unittest.main()
