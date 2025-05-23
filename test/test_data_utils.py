import unittest
import torch
import numpy as np
from PIL import Image
from utils.data_utils import preprocess_image
from io import BytesIO

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        # Create a simple test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
    def test_preprocess_image(self):
        """Test image preprocessing function"""
        # Test with default size
        tensor, adapted_size = preprocess_image(self.test_image)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape[0], 1)  # Batch size
        self.assertEqual(tensor.shape[1], 3)  # RGB channels
        self.assertEqual(adapted_size[0], tensor.shape[2])
        self.assertEqual(adapted_size[1], tensor.shape[3])
        
        # Test with custom size
        target_size = (256, 256)
        tensor, adapted_size = preprocess_image(self.test_image, target_size=target_size)
        self.assertEqual(tensor.shape[2:], target_size)
        self.assertEqual(adapted_size, target_size)
        
        # Test normalization
        self.assertTrue(torch.max(tensor) <= 1.0)
        self.assertTrue(torch.min(tensor) >= -1.0)
        
    def test_image_formats(self):
        """Test preprocessing with different image formats"""
        formats = ['PNG', 'JPEG', 'BMP']
        for fmt in formats:
            # Convert to different format
            buffer = BytesIO()
            self.test_image.save(buffer, format=fmt)
            buffer.seek(0)
            img = Image.open(buffer)
            
            # Test processing
            tensor, _ = preprocess_image(img)
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape[1], 3)  # RGB channels
            
if __name__ == '__main__':
    unittest.main()
