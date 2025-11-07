"""
Noise-DA Model Integration for ComfyUI

Based on the official implementation from:
"Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration"
Paper: https://arxiv.org/abs/2406.18516
Code: https://github.com/KangLiao929/Noise-DA

Original Authors:
- Kang Liao (kang.liao@ntu.edu.sg)
- Zongsheng Yue
- Zhouxia Wang  
- Chen Change Loy

S-Lab, Nanyang Technological University

Citation:
@article{liao2024denoising,
    title={Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration},
    author={Liao, Kang and Yue, Zongsheng and Wang, Zhouxia and Loy, Chen Change},
    journal={arXiv preprint arXiv:2406.18516},
    year={2024}
}

This ComfyUI integration uses the RestorationNet architecture from the pretrained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

def conv3x3(in_chn, out_chn, bias=True):
    """3x3 convolution with padding"""
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

class ConvBlock(nn.Module):
    """Basic convolution block with LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, slope=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.LeakyReLU(slope, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class UpConvBlock(nn.Module):
    """Upsampling block with convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, slope=0.2):
        super().__init__()
        self.conv_up = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, slope)
    
    def forward(self, x, skip=None):
        # Upsample
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_up = self.conv_up(x_up)
        
        # Concatenate with skip connection if provided
        if skip is not None:
            # Crop skip connection to match upsampled size
            if skip.shape[2:] != x_up.shape[2:]:
                skip = self._center_crop(skip, x_up.shape[2:])
            x_up = torch.cat([x_up, skip], dim=1)
        
        # Apply convolution block
        return self.conv_block(x_up)
    
    def _center_crop(self, tensor, target_size):
        """Center crop tensor to target size"""
        _, _, h, w = tensor.size()
        target_h, target_w = target_size
        
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        
        return tensor[:, :, start_h:start_h + target_h, start_w:start_w + target_w]

class RestorationNet(nn.Module):
    """
    RestorationNet that matches the exact structure from Noise-DA models
    """
    def __init__(self, in_channels=3, out_channels=3, depth=4, wf=64, slope=0.2):
        super().__init__()
        self.depth = depth
        
        # Downsampling path
        self.down_path = nn.ModuleList()
        prev_channels = in_channels
        
        for i in range(depth):
            self.down_path.append(ConvBlock(prev_channels, (2**i) * wf, slope=slope))
            prev_channels = (2**i) * wf
        
        # Upsampling path
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UpConvBlock(prev_channels, (2**i) * wf, slope=slope))
            prev_channels = (2**i) * wf
        
        # Final output layer
        self.last = nn.Conv2d(prev_channels, out_channels, 3, padding=1, bias=True)
    
    def forward(self, x):
        # Store skip connections
        blocks = []
        
        # Downsampling
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        
        # Upsampling
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
        
        # Final output
        return self.last(x)

class NoiseDAProcessor:
    """Processor for Noise-DA models with proper RestorationNet loading"""
    
    def __init__(self, model_type: str = 'denoise', device: str = 'cuda'):
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the appropriate pretrained model"""
        model_path = Path(__file__).parent.parent / 'models' / f'noise_da_{self.model_type}.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Initialize model with correct architecture
        self.model = RestorationNet(in_channels=3, out_channels=3, depth=4, wf=64)
        
        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract RestorationNet weights
        restoration_weights = {}
        for key, value in checkpoint.items():
            if key.startswith('RestorationNet.'):
                # Remove the 'RestorationNet.' prefix
                clean_key = key[14:]
                # Remove leading dot if present
                if clean_key.startswith('.'):
                    clean_key = clean_key[1:]
                restoration_weights[clean_key] = value
        
        # Load the weights
        missing_keys, unexpected_keys = self.model.load_state_dict(restoration_weights, strict=False)
        
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print(f"✅ Successfully loaded all RestorationNet weights for {self.model_type}")
        else:
            print(f"⚠️  Partially loaded RestorationNet weights for {self.model_type}")
            if missing_keys:
                print(f"   Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"   Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
        
        self.model.to(self.device)
        self.model.eval()
        
    def process_image(self, image: np.ndarray, residual_mode: bool = True, residual_scale_factor: float = 0.5) -> np.ndarray:
        """
        Process image with Noise-DA model
        
        Args:
            image: Input image as numpy array (H, W, C) in [0, 255] range
            residual_mode: Whether to treat model output as residual (True) or direct output (False)
            residual_scale_factor: Multiplier for residual scaling
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # Store original image
        original_image = image.copy()
        
        # Convert to tensor with proper size handling
        input_tensor, original_size = self._preprocess(image)
        
        # Process with model
        with torch.no_grad():
            model_output = self.model(input_tensor)
            
        # Convert model output back to numpy (keep in normalized space)
        # The RestorationNet outputs RESIDUALS, not clean images
        model_residuals = self._postprocess_normalized(model_output, original_size)
        
        # Normalize original image to [0, 1] space
        if original_image.dtype == np.uint8:
            original_norm = original_image.astype(np.float32) / 255.0
        else:
            original_norm = original_image.astype(np.float32)
        
        if residual_mode:
            # Apply residuals with scaling factor
            # Residuals are typically small corrections that should be added to the input
            result_norm = original_norm + model_residuals * residual_scale_factor
            result_norm = np.clip(result_norm, 0, 1)
        else:
            # Direct output mode - still apply residuals but without scaling
            result_norm = original_norm + model_residuals
            result_norm = np.clip(result_norm, 0, 1)
        
        # Convert back to [0, 255] range
        result = (result_norm * 255).astype(np.uint8)
            
        return result
        
    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for RestorationNet input"""
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        # Convert to float and normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        
        # RestorationNet uses depth=4, so we need padding for 2^4 = 16
        pad_factor = 16
        
        # Calculate padding needed
        pad_h = (pad_factor - (original_height % pad_factor)) % pad_factor
        pad_w = (pad_factor - (original_width % pad_factor)) % pad_factor
        
        # Pad the image if needed
        if pad_h > 0 or pad_w > 0:
            # Use reflection padding to avoid edge artifacts
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Convert to tensor format (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        return image.to(self.device), (original_height, original_width)
        
    def _postprocess_normalized(self, tensor: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Convert model output back to normalized image [0, 1] space"""
        # Remove batch dimension and convert to numpy
        image = tensor.squeeze(0).cpu().detach().numpy()
        
        # Convert from (C, H, W) to (H, W, C)
        image = image.transpose(1, 2, 0)
        
        # Crop back to original size
        original_height, original_width = original_size
        image = image[:original_height, :original_width, :]
        
        # Keep in [0, 1] space, clamp to valid range
        image = np.clip(image, 0, 1)
        
        return image.astype(np.float32)
        
    def _postprocess(self, tensor: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Convert model output back to image"""
        # Remove batch dimension and convert to numpy
        image = tensor.squeeze(0).cpu().detach().numpy()
        
        # Convert from (C, H, W) to (H, W, C)
        image = image.transpose(1, 2, 0)
        
        # Crop back to original size
        original_height, original_width = original_size
        image = image[:original_height, :original_width, :]
        
        # Convert to [0, 255] and uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image

# Test function for standalone usage
def test_noise_da_processor():
    """Test the NoiseDAProcessor independently"""
    import cv2
    
    # Create a simple test image
    test_image = np.random.rand(256, 256, 3) * 255
    test_image = test_image.astype(np.uint8)
    
    # Test all models
    for model_type in ['denoise', 'deblur', 'derain']:
        try:
            processor = NoiseDAProcessor(model_type)
            result = processor.process_image(test_image)
            print(f"{model_type} model: SUCCESS - Output shape: {result.shape}")
        except Exception as e:
            print(f"{model_type} model: FAILED - {e}")

if __name__ == "__main__":
    test_noise_da_processor()
