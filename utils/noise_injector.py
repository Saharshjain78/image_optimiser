import numpy as np
import random
import torch
import torch.nn.functional as F
import cv2
from skimage.filters import sobel
# Removed dependency on external noise package

class NeuralNoiseInjector:
    def __init__(self, noise_scale=0.1, noise_decay=0.95, noise_patterns=["gaussian", "perlin", "simplex"]):
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.noise_patterns = noise_patterns
        self.current_iteration = 0
        
    def inject_noise(self, feature_map, pattern_type=None):
        """Inject calculated noise into feature maps based on current segmentation state"""
        if pattern_type is None:
            pattern_type = random.choice(self.noise_patterns)
            
        # Calculate effective noise scale with decay
        effective_scale = self.noise_scale * (self.noise_decay ** self.current_iteration)
        
        # Generate appropriate noise pattern
        if pattern_type == "gaussian":
            noise = self._generate_gaussian_noise(feature_map.shape, effective_scale)
        elif pattern_type == "perlin":
            noise = self._generate_perlin_noise(feature_map.shape, effective_scale)
        elif pattern_type == "simplex":
            noise = self._generate_simplex_noise(feature_map.shape, effective_scale)
            
        # Apply noise with feature map-dependent scaling
        edge_map = self._calculate_edge_strength(feature_map)
        
        # Scale noise impact based on edge uncertainty
        # Convert to PyTorch tensors if feature_map is a tensor
        if isinstance(feature_map, torch.Tensor):
            noise_tensor = torch.tensor(noise, dtype=feature_map.dtype, device=feature_map.device)
            edge_map_tensor = torch.tensor(edge_map, dtype=feature_map.dtype, device=feature_map.device)
            scaled_noise = noise_tensor * edge_map_tensor * effective_scale
        else:
            scaled_noise = noise * edge_map * effective_scale
        
        self.current_iteration += 1
        return feature_map + scaled_noise
    
    def _calculate_edge_strength(self, feature_map):
        """Calculate where edges are uncertain and need noise help"""
        # Convert tensor to numpy if needed
        if isinstance(feature_map, torch.Tensor):
            # Move to CPU and convert to numpy
            fm_np = feature_map.detach().cpu().numpy()
        else:
            fm_np = feature_map
            
        # For multi-channel feature maps, compute edge strength for each channel
        if len(fm_np.shape) == 4:  # [batch, channels, height, width]
            batch_size, channels, height, width = fm_np.shape
            edge_maps = np.zeros((batch_size, channels, height, width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    # Apply Sobel filter to get edge magnitudes
                    edge_maps[b, c] = sobel(fm_np[b, c])
                    
            # Normalize edge map to [0, 1]
            edge_maps = edge_maps / (np.max(edge_maps) + 1e-8)
            
            # Higher values at uncertain edges, lower values in certain regions
            uncertainty_weight = 0.5  # Balance between noise everywhere and only at edges
            edge_uncertainty = uncertainty_weight + (1 - uncertainty_weight) * edge_maps
            
            return edge_uncertainty
        else:
            # Handle single image case
            edge_map = sobel(fm_np)
            edge_map = edge_map / (np.max(edge_map) + 1e-8)
            uncertainty_weight = 0.5
            return uncertainty_weight + (1 - uncertainty_weight) * edge_map
        
    def _generate_gaussian_noise(self, shape, scale):
        """Generate Gaussian noise with the same shape as feature_map"""
        return np.random.normal(0, scale, shape).astype(np.float32)
    
    def _generate_perlin_noise(self, shape, scale):
        """Generate a simplified version of Perlin-like noise with the same shape as feature_map"""
        if len(shape) == 4:  # [batch, channels, height, width]
            batch_size, channels, height, width = shape
            noise = np.zeros((batch_size, channels, height, width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    # Generate a lower resolution noise
                    small_h, small_w = height // 8, width // 8
                    small_noise = np.random.randn(small_h, small_w).astype(np.float32)
                    
                    # Resize to create smooth Perlin-like noise
                    noise[b, c] = cv2.resize(small_noise, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # Normalize to [-scale, scale]
            noise = scale * noise / np.max(np.abs(noise) + 1e-8)
            return noise
        else:
            # Handle single image or different shape
            return np.random.normal(0, scale, shape).astype(np.float32)  # Fallback to Gaussian
    
    def _generate_simplex_noise(self, shape, scale):
        """Generate a simplified version of simplex-like noise with the same shape as feature_map"""
        if len(shape) == 4:  # [batch, channels, height, width]
            batch_size, channels, height, width = shape
            noise = np.zeros((batch_size, channels, height, width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    # Create multi-layered noise (similar to simplex with octaves)
                    base_noise = np.zeros((height, width), dtype=np.float32)
                    
                    # Add different frequency components
                    for octave in range(3):
                        octave_scale = 1.0 / (2 ** octave)
                        size = max(4, int(width * octave_scale)), max(4, int(height * octave_scale))
                        small_noise = np.random.randn(*size).astype(np.float32)
                        
                        # Resize to create smooth noise at different frequencies
                        resized = cv2.resize(small_noise, (width, height), interpolation=cv2.INTER_CUBIC)
                        base_noise += resized * (0.5 ** octave)  # Weight higher frequencies less
                    
                    noise[b, c] = base_noise
            
            # Normalize to [-scale, scale]
            noise = scale * noise / np.max(np.abs(noise) + 1e-8)
            return noise
        else:
            # Handle single image or different shape
            return np.random.normal(0, scale, shape).astype(np.float32)  # Fallback to Gaussian
    
    def reset_iteration(self):
        """Reset the iteration counter to restart noise decay"""
        self.current_iteration = 0