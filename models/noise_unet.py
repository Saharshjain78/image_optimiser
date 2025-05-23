import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.simple_noise_injector import NeuralNoiseInjector

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class NoiseAwareUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, noise_injector=None):
        super(NoiseAwareUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.noise_injector = noise_injector or NeuralNoiseInjector()
        
        # Standard U-Net components
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        # Noise injection gates - learnable parameters that control noise at each level
        self.noise_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(5)
        ])
        
        # Optional resizing layers for handling arbitrary input sizes
        self.resize_mode = 'bilinear'  # Options: 'bilinear', 'nearest'
    
    def forward(self, x, noise_mode="training"):
        # Get input dimensions
        input_shape = x.shape
        
        # Check for non-standard input sizes and make them divisible by 16
        # This is important for the skip connections to match properly
        h, w = x.size()[2:]
        h_new, w_new = ((h + 15) // 16) * 16, ((w + 15) // 16) * 16
        
        # Resize if dimensions are not divisible by 16
        if h != h_new or w != w_new:
            x = F.interpolate(x, size=(h_new, w_new), mode=self.resize_mode, align_corners=True if self.resize_mode == 'bilinear' else None)
            
        # Apply dynamic noise scale based on image dimensions
        if self.noise_injector and hasattr(self.noise_injector, 'noise_scale'):
            # Adjust noise scale inversely with image size
            original_scale = self.noise_injector.noise_scale
            size_factor = max(1.0, min(3.0, (512 * 512) / (h_new * w_new)))
            self.noise_injector.noise_scale = original_scale * size_factor

        # Encode path with noise injection
        x1 = self.inc(x)
        if noise_mode != "disabled":
            x1 = self.maybe_inject_noise(x1, 0, noise_mode)
            
        x2 = self.down1(x1)
        if noise_mode != "disabled":
            x2 = self.maybe_inject_noise(x2, 1, noise_mode)
        
        x3 = self.down2(x2)
        if noise_mode != "disabled":
            x3 = self.maybe_inject_noise(x3, 2, noise_mode)
        
        x4 = self.down3(x3)
        if noise_mode != "disabled":
            x4 = self.maybe_inject_noise(x4, 3, noise_mode)
        
        x5 = self.down4(x4)
        if noise_mode != "disabled":
            x5 = self.maybe_inject_noise(x5, 4, noise_mode)
        
        # Decode path (standard)
        x = self.up1(x5, x4)
        x = self.up2(x, x3) 
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # Restore original dimensions if they were changed
        if h != h_new or w != w_new:
            logits = F.interpolate(logits, size=(h, w), mode=self.resize_mode, align_corners=True if self.resize_mode == 'bilinear' else None)
        # Reset noise scale if it was adjusted
        if self.noise_injector and hasattr(self.noise_injector, 'noise_scale'):
            self.noise_injector.noise_scale = original_scale
        return logits
        
    def maybe_inject_noise(self, feature_map, gate_idx, noise_mode):
        """Apply noise conditionally based on learned gate values"""
        gate_value = torch.sigmoid(self.noise_gates[gate_idx])
        
        if noise_mode == "training":
            # During training, use gate value to determine noise application
            if random.random() < gate_value.item():
                return self.noise_injector.inject_noise(feature_map)
        elif noise_mode == "single":
            # Single noise pattern during inference
            # Use different noise patterns at different levels of the network
            # for better feature extraction
            if gate_idx == 0:
                # First level - use structured noise for edge detection
                noise_type = "structured"
            elif gate_idx == 1 or gate_idx == 2:
                # Middle levels - use perlin for texture enhancement
                noise_type = "perlin"
            else:
                # Deep levels - use gaussian for general regularization
                noise_type = "gaussian"
                
            return self.noise_injector.inject_noise(feature_map, noise_type)
        elif noise_mode == "ensemble":
            # During inference, run multiple noise patterns and average
            results = [feature_map]
            
            # Use stored patterns if available, otherwise generate new ones
            noise_patterns = []
            for pattern_type in ["gaussian", "perlin", "structured"]:
                pattern_key = f"{pattern_type}_{self.noise_injector.current_iteration}"
                if hasattr(self.noise_injector, 'saved_noise_patterns') and pattern_key in self.noise_injector.saved_noise_patterns:
                    pattern = self.noise_injector.saved_noise_patterns[pattern_key]
                    noise_patterns.append(pattern)
                else:
                    noise_patterns.append(pattern_type)
            
            # Apply all patterns
            for pattern in noise_patterns:
                noisy_map = self.noise_injector.inject_noise(feature_map, pattern)
                results.append(noisy_map)
                
            # Apply weighted average with more weight to the original feature map
            weights = [1.5] + [1.0] * (len(results) - 1)
            weighted_sum = sum(w * r for w, r in zip(weights, results))
            return weighted_sum / sum(weights)
            
        return feature_map