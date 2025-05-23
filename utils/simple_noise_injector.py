import numpy as np
import random
import torch
import cv2
from skimage.filters import sobel

class NeuralNoiseInjector:
    def __init__(self, noise_scale=0.1, noise_decay=0.95, noise_patterns=["gaussian", "perlin", "structured"]):
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.noise_patterns = noise_patterns
        self.current_iteration = 0
        self.noise_seed = random.randint(0, 100000)
        self.saved_noise_patterns = {}
        
    def inject_noise(self, feature_map, pattern_type=None):
        """Inject noise into feature maps with adaptive pattern selection"""
        if pattern_type is None:
            pattern_type = random.choice(self.noise_patterns)
            
        # Calculate effective scale with decay
        effective_scale = self.noise_scale * (self.noise_decay ** self.current_iteration)
        
        # Generate noise according to specified pattern
        if pattern_type == "gaussian" or pattern_type not in ["perlin", "structured"]:
            noise = self._generate_gaussian_noise(feature_map.shape, effective_scale)
        elif pattern_type == "perlin":
            noise = self._generate_perlin_like_noise(feature_map.shape, effective_scale)
        elif pattern_type == "structured":
            noise = self._generate_structured_noise(feature_map.shape, effective_scale)
            
        # Apply edge-aware scaling with adaptive thresholding
        edge_map = self._calculate_edge_strength(feature_map)
        
        # Convert to PyTorch tensors if needed
        if isinstance(feature_map, torch.Tensor):
            # Try to reuse tensor device and type for efficiency
            device = feature_map.device
            dtype = feature_map.dtype
            
            noise_tensor = torch.tensor(noise, dtype=dtype, device=device)
            edge_map_tensor = torch.tensor(edge_map, dtype=dtype, device=device)
            
            # Apply content-adaptive scaling
            content_factor = self._calculate_content_factor(feature_map)
            scaled_noise = noise_tensor * edge_map_tensor * effective_scale * content_factor
        else:
            # For numpy arrays
            content_factor = self._calculate_content_factor(feature_map)
            scaled_noise = noise * edge_map * effective_scale * content_factor
        
        # Cache the pattern for reproducibility in ensemble mode
        pattern_key = f"{pattern_type}_{self.current_iteration}"
        self.saved_noise_patterns[pattern_key] = scaled_noise
        
        self.current_iteration += 1
        return feature_map + scaled_noise
        
    def _calculate_content_factor(self, feature_map):
        """Calculate a content-dependent scaling factor"""
        # Compute the variance or complexity of the feature map
        if isinstance(feature_map, torch.Tensor):
            # Use torch statistics
            mean = torch.mean(feature_map)
            std = torch.std(feature_map)
            if std < 1e-5:  # Prevent division by zero
                return 1.0
                
            # Calculate normalized content complexity (higher for more complex features)
            complexity = torch.clamp(std / (mean + 1e-5), 0.5, 2.0)
            return complexity
        else:
            # Use numpy statistics
            mean = np.mean(feature_map)
            std = np.std(feature_map)
            if std < 1e-5:  # Prevent division by zero
                return 1.0
                  # Calculate normalized content complexity
            complexity = np.clip(std / (mean + 1e-5), 0.5, 2.0)
            return complexity
            
    def _calculate_edge_strength(self, feature_map):
        """Calculate edge strength map with improved robustness"""
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
                    # Apply Sobel filter with normalization
                    try:
                        edge_maps[b, c] = sobel(fm_np[b, c])
                    except Exception:
                        # Fallback to simple gradient if sobel fails
                        grad_x = np.gradient(fm_np[b, c], axis=0)
                        grad_y = np.gradient(fm_np[b, c], axis=1)
                        edge_maps[b, c] = np.sqrt(grad_x**2 + grad_y**2)
                    
            # Normalize to [0, 1]
            max_val = np.max(edge_maps)
            if max_val > 1e-8:
                edge_maps = edge_maps / max_val
            
            # Add base uncertainty with adaptive weight
            content_variance = np.mean(np.var(fm_np, axis=(2, 3)))
            uncertainty_weight = 0.3 + (0.4 * np.clip(content_variance / 0.1, 0, 1))
            edge_uncertainty = uncertainty_weight + (1 - uncertainty_weight) * edge_maps
            
            return edge_uncertainty
        else:
            # Handle single image
            try:
                edge_map = sobel(fm_np)
            except Exception:
                # Fallback to simple gradient if sobel fails
                grad_x = np.gradient(fm_np, axis=0)
                grad_y = np.gradient(fm_np, axis=1)
                edge_map = np.sqrt(grad_x**2 + grad_y**2)
                
            # Normalize
            max_val = np.max(edge_map)
            if max_val > 1e-8:
                edge_map = edge_map / max_val
                
            # Add base uncertainty
            uncertainty_weight = 0.5
            return uncertainty_weight + (1 - uncertainty_weight) * edge_map
        
    def _generate_gaussian_noise(self, shape, scale):
        """Generate Gaussian noise"""
        return np.random.normal(0, scale, shape).astype(np.float32)
    
    def _generate_perlin_like_noise(self, shape, scale):
        """Generate Perlin-like noise (simplified approach)"""
        # Start with random noise
        noise = np.random.normal(0, scale, shape).astype(np.float32)
        
        # Apply smoothing to create Perlin-like effect
        if len(shape) == 4:  # [batch, channels, height, width]
            for b in range(shape[0]):
                for c in range(shape[1]):
                    # Apply Gaussian blur for smoothing
                    try:
                        from scipy.ndimage import gaussian_filter
                        noise[b, c] = gaussian_filter(noise[b, c], sigma=2.0)
                    except ImportError:
                        # Fallback if scipy not available
                        pass
        else:
            try:
                from scipy.ndimage import gaussian_filter
                noise = gaussian_filter(noise, sigma=2.0)
            except ImportError:
                # Fallback if scipy not available
                pass
                
        return noise
    
    def _generate_structured_noise(self, shape, scale):
        """Generate structured noise with patterns"""
        if len(shape) == 4:  # [batch, channels, height, width]
            batch_size, channels, height, width = shape
            structured_noise = np.zeros(shape, dtype=np.float32)
            
            # Create a pattern based on the seed
            np.random.seed(self.noise_seed + self.current_iteration)
            
            for b in range(batch_size):
                for c in range(channels):
                    # Create a grid pattern
                    x = np.linspace(0, 4 * np.pi, width)
                    y = np.linspace(0, 4 * np.pi, height)
                    xx, yy = np.meshgrid(x, y)
                    
                    # Generate pattern with frequency variations
                    pattern = np.sin(xx) * np.cos(yy) + np.sin(2 * xx + 3) * np.cos(3 * yy)
                    
                    # Normalize and scale
                    pattern = pattern / np.max(np.abs(pattern)) * scale
                    
                    # Add some randomness
                    random_component = np.random.normal(0, scale * 0.5, (height, width))
                    structured_noise[b, c] = pattern + random_component
            
            # Reset random seed
            np.random.seed(None)
            return structured_noise
        else:
            # Single image case
            np.random.seed(self.noise_seed + self.current_iteration)
            
            height, width = shape[-2], shape[-1]
            x = np.linspace(0, 4 * np.pi, width)
            y = np.linspace(0, 4 * np.pi, height)
            xx, yy = np.meshgrid(x, y)
            
            pattern = np.sin(xx) * np.cos(yy) + np.sin(2 * xx + 3) * np.cos(3 * yy)
            pattern = pattern / np.max(np.abs(pattern)) * scale
            
            # Add randomness
            random_component = np.random.normal(0, scale * 0.5, (height, width))
            
            # Reset random seed
            np.random.seed(None)
            return pattern + random_component
    
    def reset_iteration(self):
        """Reset iteration counter"""
        self.current_iteration = 0
        # Generate new noise seed for variety
        self.noise_seed = random.randint(0, 100000)
