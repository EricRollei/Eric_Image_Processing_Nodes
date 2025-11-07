"""
Lightweight Progressive CNN for Film Grain Denoising

Efficient architecture using:
- Dense blocks for local feature extraction
- Progressive residual fusion
- Minimal parameters for fast inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class DenseBlock(nn.Module):
    """Dense block for local feature extraction"""
    
    def __init__(self, in_channels: int, growth_rate: int = 16, num_layers: int = 4):
        super(DenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.num_layers = num_layers
        self.growth_rate = growth_rate
    
    def forward(self, x):
        features = [x]
        
        for layer in self.layers:
            # Concatenate all previous features
            concat_features = torch.cat(features, dim=1)
            new_feature = layer(concat_features)
            features.append(new_feature)
        
        # Return concatenated features
        return torch.cat(features, dim=1)


class ProgressiveResidualBlock(nn.Module):
    """Progressive residual block that fuses shallow and deep features"""
    
    def __init__(self, in_channels: int, mid_channels: int):
        super(ProgressiveResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(mid_channels, in_channels, 3, padding=1)
        
        # Progressive fusion gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, shallow_feature=None):
        identity = x
        
        out = self.conv1(x)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        
        # Apply gating for progressive fusion
        if shallow_feature is not None:
            gate_weight = self.gate(x)
            out = out + identity * gate_weight + shallow_feature * (1 - gate_weight)
        else:
            out = out + identity
        
        return out


class LightweightAttention(nn.Module):
    """Lightweight attention module"""
    
    def __init__(self, channels: int):
        super(LightweightAttention, self).__init__()
        
        # Channel attention (lightweight version)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention (lightweight version)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_att(x)
        x = x * sa
        
        return x


class ProgressiveCNNModel(nn.Module):
    """
    Lightweight Progressive CNN for Film Grain Denoising
    
    Key features:
    - Dense blocks for comprehensive local feature extraction
    - Progressive residual fusion for global noise characteristics
    - Lightweight attention mechanisms
    - Minimal parameters (~500KB) for efficient inference
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_progressive_blocks: int = 3):
        super(ProgressiveCNNModel, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Dense block for local feature extraction
        self.dense_block = DenseBlock(base_channels, growth_rate=12, num_layers=4)
        dense_out_channels = base_channels + 12 * 4  # base + growth_rate * num_layers
        
        # Transition layer after dense block
        self.transition = nn.Sequential(
            nn.Conv2d(dense_out_channels, base_channels * 2, 1),
            nn.ReLU(inplace=True)
        )
        
        # Progressive residual blocks
        self.progressive_blocks = nn.ModuleList()
        for i in range(num_progressive_blocks):
            self.progressive_blocks.append(
                ProgressiveResidualBlock(base_channels * 2, base_channels * 2)
            )
        
        # Lightweight attention
        self.attention = LightweightAttention(base_channels * 2)
        
        # Final reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            denoised: Denoised image
        """
        # Initial features
        shallow_feature = self.initial_conv(x)
        
        # Dense block for local features
        dense_features = self.dense_block(shallow_feature)
        features = self.transition(dense_features)
        
        # Progressive residual processing with shallow feature fusion
        for i, block in enumerate(self.progressive_blocks):
            if i == 0:
                # First block: no shallow feature fusion yet
                features = block(features)
            else:
                # Subsequent blocks: fuse with initial shallow features
                features = block(features, shallow_feature=shallow_feature)
        
        # Apply attention
        features = self.attention(features)
        
        # Reconstruction
        noise = self.reconstruction(features)
        denoised = x - noise  # Residual learning: clean = noisy - noise
        
        return denoised
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ProgressiveCNNProcessor:
    """High-level processor for Progressive CNN model"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.model_loaded = False
    
    def initialize_model(self, in_channels: int = 3, pretrained_path: Optional[str] = None):
        """Initialize the model"""
        self.model = ProgressiveCNNModel(in_channels=in_channels).to(self.device)
        
        if pretrained_path is not None:
            try:
                self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                print(f"✓ Loaded Progressive CNN weights from {pretrained_path}")
            except Exception as e:
                print(f"⚠ Could not load weights: {e}. Using random initialization.")
        
        self.model.eval()
        self.model_loaded = True
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process image with Progressive CNN
        
        Args:
            image: Input image [H, W, C] or [H, W] in range [0, 1]
            
        Returns:
            denoised_image: Denoised result
            info: Processing information
        """
        if not self.model_loaded:
            self.initialize_model(in_channels=3 if len(image.shape) == 3 else 1)
        
        # Prepare input
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # Convert to tensor [1, C, H, W]
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        
        info = {}
        
        with torch.no_grad():
            denoised_tensor = self.model(img_tensor)
        
        # Convert back to numpy
        denoised = denoised_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        
        # Clip to valid range
        denoised = np.clip(denoised, 0, 1)
        
        if denoised.shape[2] == 1:
            denoised = denoised[:, :, 0]
        
        # Calculate quality metrics
        info['psnr'] = self._calculate_psnr(image, denoised)
        info['output_range'] = [float(denoised.min()), float(denoised.max())]
        
        return denoised, info
    
    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate PSNR between two images"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    def save_model(self, path: str):
        """Save model weights"""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            print(f"✓ Saved Progressive CNN weights to {path}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {"status": "Not initialized"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "device": str(self.device),
            "status": "Initialized"
        }