import numpy as np
import random
import torch
import cv2
from skimage.filters import sobel
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

class EnhancedNoiseInjector:
    def __init__(self, noise_scale=0.1, noise_decay=0.95, noise_patterns=None):
        """
        Enhanced noise injector with multiple pattern options and adaptive noise
        
        Args:
            noise_scale (float): Base scale for noise amplitude
            noise_decay (float): Rate at which noise decays over iterations
            noise_patterns (list): List of pattern types to use
        """
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.noise_patterns = noise_patterns or ["gaussian", "perlin", "simplex", "structured", "adaptive"]
        self.current_iteration = 0
        
        # Pre-generate some noise patterns for ensemble mode
        self.cached_patterns = {
            pattern: self._generate_pattern_seed(pattern) 
            for pattern in self.noise_patterns
        }
        
    def inject_noise(self, feature_map, pattern_type=None):
        """
        Inject noise into feature maps
        
        Args:
            feature_map (torch.Tensor): Input feature map tensor
            pattern_type (str, optional): Specific pattern to use, or random if None
            
        Returns:
            torch.Tensor: Feature map with injected noise
        """
        if pattern_type is None:
            pattern_type = random.choice(self.noise_patterns)
            
        # Calculate effective scale with decay
        effective_scale = self.noise_scale * (self.noise_decay ** self.current_iteration)
        
        # Generate noise based on pattern type
        if pattern_type == "gaussian":
            noise = self._generate_gaussian_noise(feature_map.shape, effective_scale)
        elif pattern_type == "perlin":
            noise = self._generate_perlin_noise(feature_map.shape, effective_scale)
        elif pattern_type == "simplex":
            noise = self._generate_simplex_noise(feature_map.shape, effective_scale)
        elif pattern_type == "structured":
            noise = self._generate_structured_noise(feature_map.shape, effective_scale)
        elif pattern_type == "adaptive":
            # Adaptive noise adjusts to feature map statistics
            noise = self._generate_adaptive_noise(feature_map, effective_scale)
        else:
            # Default to gaussian
            noise = self._generate_gaussian_noise(feature_map.shape, effective_scale)
            
        # Apply edge-aware scaling
        edge_map = self._calculate_edge_strength(feature_map)
        
        # Convert to PyTorch tensors if needed
        if isinstance(feature_map, torch.Tensor):
            noise_tensor = torch.tensor(noise, dtype=feature_map.dtype, device=feature_map.device)
            edge_map_tensor = torch.tensor(edge_map, dtype=feature_map.dtype, device=feature_map.device)
            scaled_noise = noise_tensor * edge_map_tensor * effective_scale
        else:
            scaled_noise = noise * edge_map * effective_scale
        
        self.current_iteration += 1
        return feature_map + scaled_noise
    
    def _generate_pattern_seed(self, pattern_type):
        """Generate a seed value for a specific pattern type"""
        return random.randint(0, 10000)
    
    def _calculate_edge_strength(self, feature_map):
        """
        Calculate edge strength map with enhanced detection
        
        Args:
            feature_map (torch.Tensor): Input feature map
            
        Returns:
            numpy.ndarray: Edge strength map
        """
        # Convert tensor to numpy if needed
        if isinstance(feature_map, torch.Tensor):
            fm_np = feature_map.detach().cpu().numpy()
        else:
            fm_np = feature_map
            
        # For multi-channel feature maps
        if len(fm_np.shape) == 4:  # [batch, channels, height, width]
            batch_size, channels, height, width = fm_np.shape
            edge_maps = np.zeros((batch_size, channels, height, width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    # Apply Sobel filter with pre-smoothing for better edge detection
                    smoothed = gaussian_filter(fm_np[b, c], sigma=0.5)
                    edge_maps[b, c] = sobel(smoothed)
                    
            # Normalize to [0, 1]
            max_val = np.max(edge_maps) + 1e-8
            edge_maps = edge_maps / max_val
            
            # Enhanced edge-aware uncertainty
            # Add base uncertainty level, higher near edges
            base_uncertainty = 0.3
            edge_uncertainty = base_uncertainty + (1 - base_uncertainty) * edge_maps
            edge_uncertainty = np.clip(edge_uncertainty, 0, 1)
            
            return edge_uncertainty
        else:
            # Handle single image
            smoothed = gaussian_filter(fm_np, sigma=0.5)
            edge_map = sobel(smoothed)
            edge_map = edge_map / (np.max(edge_map) + 1e-8)
            base_uncertainty = 0.3
            return base_uncertainty + (1 - base_uncertainty) * edge_map
    
    def _generate_gaussian_noise(self, shape, scale):
        """
        Generate Gaussian noise
        
        Args:
            shape (tuple): Shape of the noise tensor
            scale (float): Scale factor for noise amplitude
            
        Returns:
            numpy.ndarray: Gaussian noise array
        """
        return np.random.normal(0, scale, shape).astype(np.float32)
    
    def _generate_perlin_noise(self, shape, scale):
        """
        Generate Perlin noise
        
        Args:
            shape (tuple): Shape of the noise tensor
            scale (float): Scale factor for noise amplitude
            
        Returns:
            numpy.ndarray: Perlin noise array
        """
        # Simulate Perlin noise using a simple approach
        # For a more accurate implementation, consider using a dedicated library
        
        if len(shape) == 4:
            batch_size, channels, height, width = shape
            result = np.zeros(shape, dtype=np.float32)
            
            # Generate at multiple octaves and combine
            octaves = 4
            persistence = 0.5
            lacunarity = 2.0
            
            for b in range(batch_size):
                for c in range(channels):
                    total = np.zeros((height, width), dtype=np.float32)
                    
                    for octave in range(octaves):
                        freq = lacunarity ** octave
                        amp = persistence ** octave
                        
                        # Generate base noise
                        noise_base = np.random.normal(0, 1, (int(height/freq)+1, int(width/freq)+1))
                        # Smooth it
                        noise_base = gaussian_filter(noise_base, sigma=1.0)
                        # Resize to full size
                        noise = cv2.resize(noise_base, (width, height), interpolation=cv2.INTER_CUBIC)
                        
                        total += noise * amp
                    
                    # Normalize
                    total = total / np.max(np.abs(total) + 1e-8)
                    result[b, c] = total * scale
                    
            return result
        else:
            # Handle other shapes
            return self._generate_gaussian_noise(shape, scale)
    
    def _generate_simplex_noise(self, shape, scale):
        """
        Generate Simplex-like noise
        
        Args:
            shape (tuple): Shape of the noise tensor
            scale (float): Scale factor for noise amplitude
            
        Returns:
            numpy.ndarray: Simplex-like noise array
        """
        # Simulate simplex noise with a similar approach to perlin but different parameters
        
        if len(shape) == 4:
            batch_size, channels, height, width = shape
            result = np.zeros(shape, dtype=np.float32)
            
            # Generate at multiple frequencies
            frequencies = [1, 2, 4, 8]
            weights = [0.5, 0.25, 0.125, 0.0625]
            
            for b in range(batch_size):
                for c in range(channels):
                    total = np.zeros((height, width), dtype=np.float32)
                    
                    for freq, weight in zip(frequencies, weights):
                        size = (int(height/freq), int(width/freq))
                        noise_base = np.random.normal(0, 1, size)
                        noise_base = gaussian_filter(noise_base, sigma=1.0)
                        noise = cv2.resize(noise_base, (width, height), interpolation=cv2.INTER_LINEAR)
                        
                        total += noise * weight
                    
                    # Normalize
                    total = total / np.max(np.abs(total) + 1e-8)
                    result[b, c] = total * scale
                    
            return result
        else:
            # Handle other shapes
            return self._generate_gaussian_noise(shape, scale)
    
    def _generate_structured_noise(self, shape, scale):
        """
        Generate structured noise with patterns
        
        Args:
            shape (tuple): Shape of the noise tensor
            scale (float): Scale factor for noise amplitude
            
        Returns:
            numpy.ndarray: Structured noise array
        """
        if len(shape) == 4:
            batch_size, channels, height, width = shape
            result = np.zeros(shape, dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    # Create a base pattern
                    x = np.linspace(-3, 3, width)
                    y = np.linspace(-3, 3, height)
                    xx, yy = np.meshgrid(x, y)
                    
                    # Use combinations of trigonometric functions
                    z1 = np.sin(xx + random.random()) * np.cos(yy + random.random())
                    z2 = np.cos(2*xx + random.random()) * np.sin(2*yy + random.random())
                    z3 = np.sin(3*xx*yy + random.random())
                    
                    pattern = (z1 + z2 + z3) / 3
                    
                    # Add some randomness
                    noise = np.random.normal(0, 0.2, (height, width))
                    combined = pattern + noise
                    
                    # Normalize
                    combined = combined / np.max(np.abs(combined) + 1e-8)
                    result[b, c] = combined * scale
                    
            return result
        else:
            # Handle other shapes
            return self._generate_gaussian_noise(shape, scale)
    
    def _generate_adaptive_noise(self, feature_map, scale):
        """
        Generate adaptive noise based on feature map statistics
        
        Args:
            feature_map (torch.Tensor): Input feature map
            scale (float): Scale factor for noise amplitude
            
        Returns:
            numpy.ndarray: Adaptive noise array
        """
        if isinstance(feature_map, torch.Tensor):
            fm_np = feature_map.detach().cpu().numpy()
        else:
            fm_np = feature_map
            
        if len(fm_np.shape) == 4:
            batch_size, channels, height, width = fm_np.shape
            result = np.zeros(fm_np.shape, dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    # Calculate feature statistics
                    channel_data = fm_np[b, c]
                    mean = np.mean(channel_data)
                    std = np.std(channel_data) + 1e-8
                    
                    # Generate noise proportional to feature statistics
                    noise = np.random.normal(0, std * scale, (height, width))
                    
                    # Add gradient-based component
                    grad_x = np.gradient(channel_data, axis=0)
                    grad_y = np.gradient(channel_data, axis=1)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    gradient_magnitude = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
                    
                    # Final adaptive noise combines random and gradient components
                    adaptive_noise = noise * (0.5 + 0.5 * gradient_magnitude)
                    result[b, c] = adaptive_noise
                    
            return result
        else:
            # Handle other shapes
            return self._generate_gaussian_noise(fm_np.shape, scale)
    
    def reset_iteration(self):
        """Reset iteration counter"""
        self.current_iteration = 0
    
    def generate_ensemble_noise(self, feature_map, num_patterns=3, scale=None):
        """
        Generate multiple noise patterns for ensemble mode
        
        Args:
            feature_map (torch.Tensor): Input feature map
            num_patterns (int): Number of different patterns to generate
            scale (float, optional): Override the default noise scale
            
        Returns:
            list: List of feature maps with different noise patterns
        """
        if scale is None:
            scale = self.noise_scale
            
        ensemble = []
        # Always include the original
        ensemble.append(feature_map)
        
        # Add different noise patterns
        patterns = random.sample(self.noise_patterns, min(num_patterns, len(self.noise_patterns)))
        for pattern in patterns:
            noisy_map = self.inject_noise(feature_map, pattern)
            ensemble.append(noisy_map)
            
        return ensemble
