"""
DnCNN (Denoising Convolutional Neural Network)
Classic and effective CNN architecture for image denoising

Based on: "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
Zhang et al., IEEE TIP 2017

Features:
- Simple but effective architecture
- Residual learning (predicts noise, not clean image)
- Batch normalization for stability
- Pre-trained weights available for multiple noise levels
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import os
from pathlib import Path
import urllib.request


class DnCNNModel(nn.Module):
    """
    DnCNN Model - Classic denoising CNN with residual learning
    
    Architecture:
    - First layer: Conv + ReLU
    - Middle layers: Conv + BN + ReLU (repeated)
    - Last layer: Conv only
    - Residual connection: output = input - noise_prediction
    """
    
    def __init__(self, in_channels: int = 1, depth: int = 17, num_filters: int = 64):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for color)
            depth: Number of convolutional layers
            num_filters: Number of filters in conv layers
        """
        super(DnCNNModel, self).__init__()
        
        self.depth = depth
        
        # First layer: Conv + ReLU
        layers = [
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]
        
        # Middle layers: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.extend([
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True)
            ])
        
        # Last layer: Conv only
        layers.append(
            nn.Conv2d(num_filters, in_channels, kernel_size=3, padding=1, bias=False)
        )
        
        self.dncnn = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with residual learning
        
        Args:
            x: Input noisy image [B, C, H, W]
            
        Returns:
            Denoised image (x - predicted_noise)
        """
        noise = self.dncnn(x)
        return x - noise  # Residual learning: clean = noisy - noise


class DnCNNProcessor:
    """High-level processor for DnCNN model with auto weight downloading"""
    
    # Pre-trained model URLs (from official DnCNN repository)
    PRETRAINED_URLS = {
        'grayscale_sigma15': 'https://github.com/cszn/DnCNN/raw/master/model/dncnn_15.pth',
        'grayscale_sigma25': 'https://github.com/cszn/DnCNN/raw/master/model/dncnn_25.pth',
        'grayscale_sigma50': 'https://github.com/cszn/DnCNN/raw/master/model/dncnn_50.pth',
        'grayscale_blind': 'https://github.com/cszn/DnCNN/raw/master/model/dncnn_gray_blind.pth',
        'color_blind': 'https://github.com/cszn/DnCNN/raw/master/model/dncnn_color_blind.pth',
    }
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.model_loaded = False
        self.weights_dir = Path(__file__).parent.parent / 'models' / 'pretrained_weights' / 'dncnn'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
    
    def download_weights(self, model_type: str = 'color_blind') -> str:
        """
        Download pre-trained weights if not already cached
        
        Args:
            model_type: Type of pre-trained model to download
            
        Returns:
            Path to downloaded weights file
        """
        if model_type not in self.PRETRAINED_URLS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.PRETRAINED_URLS.keys())}")
        
        weights_file = self.weights_dir / f'{model_type}.pth'
        
        # Check if already downloaded
        if weights_file.exists():
            print(f"✓ Using cached weights: {weights_file}")
            return str(weights_file)
        
        # Download weights
        url = self.PRETRAINED_URLS[model_type]
        print(f"⬇ Downloading DnCNN weights from: {url}")
        print(f"  Saving to: {weights_file}")
        
        try:
            urllib.request.urlretrieve(url, weights_file)
            print(f"✓ Download complete!")
            return str(weights_file)
        except Exception as e:
            print(f"✗ Download failed: {e}")
            print(f"  You can manually download from: {url}")
            raise
    
    def initialize_model(self, model_type: str = 'color_blind', in_channels: int = 3,
                        pretrained: bool = True):
        """
        Initialize the DnCNN model
        
        Args:
            model_type: Type of pre-trained model
            in_channels: Number of input channels (1=grayscale, 3=color)
            pretrained: Whether to load pre-trained weights
        """
        # Create model
        self.model = DnCNNModel(in_channels=in_channels).to(self.device)
        
        if pretrained:
            try:
                # Download weights if needed
                weights_path = self.download_weights(model_type)
                
                # Load weights
                state_dict = torch.load(weights_path, map_location=self.device)
                
                # Handle different state dict formats
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                self.model.load_state_dict(state_dict)
                print(f"✓ Loaded pre-trained DnCNN weights: {model_type}")
            except Exception as e:
                print(f"⚠ Could not load pre-trained weights: {e}")
                print("  Using random initialization")
        
        self.model.eval()
        self.model_loaded = True
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process image with DnCNN
        
        Args:
            image: Input image [H, W, C] or [H, W] in range [0, 1]
            
        Returns:
            denoised_image: Denoised result
            info: Processing information
        """
        if not self.model_loaded:
            # Auto-detect channels and initialize
            in_channels = 3 if len(image.shape) == 3 and image.shape[2] == 3 else 1
            model_type = 'color_blind' if in_channels == 3 else 'grayscale_blind'
            self.initialize_model(model_type=model_type, in_channels=in_channels)
        
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
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {"status": "Not initialized"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "device": str(self.device),
            "status": "Initialized",
            "depth": self.model.depth
        }
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available pre-trained models"""
        return {
            'color_blind': 'Color blind denoising (works for any noise level)',
            'grayscale_blind': 'Grayscale blind denoising (works for any noise level)',
            'grayscale_sigma15': 'Grayscale optimized for noise level σ=15',
            'grayscale_sigma25': 'Grayscale optimized for noise level σ=25',
            'grayscale_sigma50': 'Grayscale optimized for noise level σ=50',
        }
