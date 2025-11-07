"""
FGA-NN: Film Grain Analysis Neural Network
Modern architecture for film grain removal and analysis

Based on recent research in film grain denoising with auto-regressive modeling
and multi-scale feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class ChannelAttention(nn.Module):
    """Channel attention module for feature recalibration"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention module for spatial feature recalibration"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction for film grain analysis"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Different kernel sizes for multi-scale
        self.conv1 = nn.Conv2d(in_channels, out_channels // 3, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 3, 5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 3, 7, padding=3)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        # Concatenate multi-scale features
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class ResidualBlock(nn.Module):
    """Residual block with attention"""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.cbam = CBAM(channels)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.cbam(out)
        out += residual
        out = self.relu(out)
        
        return out


class GrainParameterEstimator(nn.Module):
    """Estimate film grain parameters (auto-regressive coefficients)"""
    
    def __init__(self, feature_channels: int, num_params: int = 8):
        super(GrainParameterEstimator, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(feature_channels, feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels // 2, num_params),
            nn.Sigmoid()  # Normalize parameters to [0, 1]
        )
    
    def forward(self, x):
        """
        Returns grain parameters:
        - [0:3] AR coefficients
        - [3:6] Frequency band strengths
        - [6:8] Intensity-dependent scaling
        """
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        params = self.fc(x)
        return params


class FGANNModel(nn.Module):
    """
    Film Grain Analysis Neural Network
    
    Architecture for film grain removal with parameter estimation.
    Can be used for both denoising and grain analysis.
    """
    
    def __init__(self, in_channels: int = 3, feature_channels: int = 64, num_blocks: int = 6):
        super(FGANNModel, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = nn.Conv2d(in_channels, feature_channels, 3, padding=1)
        self.initial_relu = nn.ReLU(inplace=True)
        
        # Multi-scale feature extraction
        self.multiscale = MultiScaleFeatureExtractor(feature_channels, feature_channels)
        
        # Residual blocks with attention
        self.res_blocks = nn.ModuleList([
            ResidualBlock(feature_channels) for _ in range(num_blocks)
        ])
        
        # Grain parameter estimator branch
        self.param_estimator = GrainParameterEstimator(feature_channels, num_params=8)
        
        # Reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, in_channels, 3, padding=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_params: bool = False):
        """
        Forward pass
        
        Args:
            x: Input image tensor [B, C, H, W]
            return_params: Whether to return grain parameters
            
        Returns:
            denoised: Denoised image
            params: Grain parameters (if return_params=True)
        """
        # Initial features
        features = self.initial_conv(x)
        features = self.initial_relu(features)
        
        # Multi-scale extraction
        features = self.multiscale(features)
        
        # Residual processing
        for res_block in self.res_blocks:
            features = res_block(features)
        
        # Grain parameter estimation
        grain_params = None
        if return_params:
            grain_params = self.param_estimator(features)
        
        # Reconstruction
        denoised = self.reconstruction(features)
        denoised = x - denoised  # Residual learning: clean = noisy - noise
        
        if return_params:
            return denoised, grain_params
        return denoised
    
    def estimate_grain_strength(self, x):
        """Estimate grain strength from input image"""
        with torch.no_grad():
            features = self.initial_conv(x)
            features = self.initial_relu(features)
            features = self.multiscale(features)
            
            for res_block in self.res_blocks:
                features = res_block(features)
            
            params = self.param_estimator(features)
            
            # Average of AR coefficients and frequency bands
            grain_strength = params[:, :6].mean(dim=1)
            return grain_strength


class FGANNProcessor:
    """High-level processor for FGA-NN model"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.model_loaded = False
    
    def initialize_model(self, in_channels: int = 3, pretrained_path: Optional[str] = None):
        """Initialize the model"""
        self.model = FGANNModel(in_channels=in_channels).to(self.device)
        
        if pretrained_path is not None:
            try:
                self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                print(f"✓ Loaded FGA-NN weights from {pretrained_path}")
            except Exception as e:
                print(f"⚠ Could not load weights: {e}. Using random initialization.")
        
        self.model.eval()
        self.model_loaded = True
    
    def process_image(self, image: np.ndarray, return_params: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Process image with FGA-NN
        
        Args:
            image: Input image [H, W, C] or [H, W] in range [0, 1]
            return_params: Whether to return grain parameters
            
        Returns:
            denoised_image: Denoised result
            info: Processing information including grain parameters
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
            if return_params:
                denoised_tensor, params = self.model(img_tensor, return_params=True)
                info['grain_params'] = params.cpu().numpy()[0]
                
                # Interpret parameters
                info['ar_coefficients'] = info['grain_params'][:3]
                info['frequency_bands'] = info['grain_params'][3:6]
                info['intensity_scaling'] = info['grain_params'][6:8]
            else:
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
            print(f"✓ Saved FGA-NN weights to {path}")
    
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