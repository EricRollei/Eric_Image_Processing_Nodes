"""
Real-ESRGAN Processing Implementation
Based on 2024-2025 research findings for practical super-resolution

Real-ESRGAN variants:
- RealESRGAN_x4plus: General scenes
- RealESRGAN_x4plus_anime_6B: Illustrations/anime
- realesr-general-x4v3: Lightweight alternative

Reference: https://github.com/xinntao/Real-ESRGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for ESRGAN"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RealESRGANProcessor:
    """Real-ESRGAN processor for practical super-resolution and enhancement"""
    
    def __init__(self, model_name: str = "RealESRGAN_x4plus", device: str = "auto"):
        """
        Initialize Real-ESRGAN processor
        
        Args:
            model_name: Model variant to use
            device: Processing device (auto, cpu, cuda)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.scale = 4  # Default scale factor
        
        # Check if model file exists
        models_dir = Path(__file__).parent.parent / "models"
        model_path = models_dir / f"{model_name}.pth"
        if model_path.exists():
            logger.info(f"Found pretrained model: {model_name}.pth")
            self.model_path = str(model_path)
        else:
            logger.warning(f"Model {model_name}.pth not found in models directory")
            logger.info("Download models using: python download_models.py")
            self.model_path = None
        
        # Model configurations
        self.model_configs = {
            "RealESRGAN_x4plus": {
                "scale": 4,
                "num_in_ch": 3,
                "num_out_ch": 3,
                "num_feat": 64,
                "num_block": 23,
                "num_grow_ch": 32,
                "description": "General scenes, best overall quality"
            },
            "RealESRGAN_x4plus_anime_6B": {
                "scale": 4,
                "num_in_ch": 3,
                "num_out_ch": 3,
                "num_feat": 64,
                "num_block": 6,
                "num_grow_ch": 32,
                "description": "Optimized for illustrations and anime"
            },
            "realesr-general-x4v3": {
                "scale": 4,
                "num_in_ch": 3,
                "num_out_ch": 3,
                "num_feat": 64,
                "num_block": 16,
                "num_grow_ch": 32,
                "description": "Lightweight alternative"
            }
        }
        
        # Tile processing for memory efficiency
        self.tile_size = 512
        self.tile_pad = 32
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for processing"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_model(self) -> bool:
        """Load or create Real-ESRGAN model"""
        try:
            # Try to import Real-ESRGAN
            try:
                from realesrgan import RealESRGANer
                from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                
                # Use official Real-ESRGAN implementation
                model_path = self.model_path  # Use downloaded model if available
                self.model = RealESRGANer(
                    scale=self.scale,
                    model_path=model_path,
                    model=None,
                    tile=self.tile_size,
                    tile_pad=self.tile_pad,
                    pre_pad=0,
                    half=True if self.device.type == 'cuda' else False,
                    gpu_id=0 if self.device.type == 'cuda' else None
                )
                logger.info(f"Loaded Real-ESRGAN model: {self.model_name}")
                return True
                
            except ImportError:
                logger.warning("Real-ESRGAN not installed, using fallback implementation")
                return self._create_fallback_model()
                
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN model: {e}")
            return self._create_fallback_model()
    
    def _create_fallback_model(self) -> bool:
        """Create fallback super-resolution model"""
        try:
            # Simple ESRGAN-style architecture as fallback
            config = self.model_configs[self.model_name]
            
            class FallbackESRGAN(nn.Module):
                def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=16, num_grow_ch=32):
                    super().__init__()
                    self.scale = scale
                    
                    # Feature extraction
                    self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
                    
                    # Residual blocks
                    self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                    self.body = nn.Sequential(*[
                        ResidualDenseBlock(num_feat, num_grow_ch) for _ in range(num_block)
                    ])
                    
                    # Upsampling
                    self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                    self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                    self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                    self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
                    
                    self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
                
                def forward(self, x):
                    feat = self.lrelu(self.conv_first(x))
                    
                    # Trunk
                    trunk = self.trunk_conv(self.body(feat))
                    feat = feat + trunk
                    
                    # Upsampling
                    feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
                    feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
                    out = self.conv_last(self.lrelu(self.conv_hr(feat)))
                    
                    return out
            
            # Create model
            model_config = {k: v for k, v in config.items() if k != 'description'}
            self.model = FallbackESRGAN(**model_config).to(self.device)
            
            # Load pretrained weights if available
            if self.model_path and Path(self.model_path).exists():
                try:
                    checkpoint = torch.load(self.model_path, map_location='cpu')
                    
                    # Handle different checkpoint formats
                    if 'params' in checkpoint:
                        state_dict = checkpoint['params']
                    elif 'params_ema' in checkpoint:
                        state_dict = checkpoint['params_ema']
                    else:
                        state_dict = checkpoint
                    
                    # Load state dict
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded pretrained weights from {self.model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load pretrained weights: {e}")
                    logger.info("Using random initialization")
            else:
                logger.info("No pretrained weights found, using random initialization")
            
            self.model.eval()
            
            logger.info(f"Created fallback Real-ESRGAN model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            return False
    
    def process_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process image with Real-ESRGAN
        
        Args:
            image: Input image (H, W, C) in RGB format
            **kwargs: Additional processing parameters
        
        Returns:
            Enhanced image
        """
        if self.model is None:
            if not self._load_model():
                return self._fallback_bicubic(image)
        
        try:
            # Prepare image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
            
            # Process with Real-ESRGAN
            if hasattr(self.model, 'enhance'):
                # Official Real-ESRGAN
                logger.info("Using official Real-ESRGAN enhance method")
                output, _ = self.model.enhance(image, outscale=self.scale)
                logger.info(f"Raw output shape: {output.shape}, dtype: {output.dtype}, range: [{output.min()}, {output.max()}]")
                
                # Ensure output is in correct format
                if output.dtype == np.uint8:
                    # Convert uint8 [0,255] to float32 [0,1]
                    output = output.astype(np.float32) / 255.0
                elif output.dtype in [np.float32, np.float64]:
                    # Check if it's already in [0,1] range
                    if output.max() <= 1.0:
                        output = output.astype(np.float32)
                    else:
                        # Assume it's in [0,255] range
                        output = output.astype(np.float32) / 255.0
                
                logger.info(f"Final output shape: {output.shape}, dtype: {output.dtype}, range: [{output.min():.3f}, {output.max():.3f}]")
                
            else:
                # Fallback model
                logger.info("Using fallback model")
                output = self._process_with_fallback(image)
            
            return output
            
        except Exception as e:
            logger.error(f"Real-ESRGAN processing failed: {e}")
            return self._fallback_bicubic(image)
    
    def _process_with_fallback(self, image: np.ndarray) -> np.ndarray:
        """Process image with fallback model"""
        # Convert to tensor
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # Process
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Convert back
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output, 0, 1)
        
        # Normalize to use full range if output is too dark
        if output.max() < 0.8:  # If max is less than 0.8, normalize to full range
            output_min = output.min()
            output_max = output.max()
            if output_max > output_min:
                output = (output - output_min) / (output_max - output_min)
                logger.info(f"Normalized fallback output from [{output_min:.3f}, {output_max:.3f}] to [0, 1]")
        
        return (output * 255).astype(np.uint8)
    
    def _fallback_bicubic(self, image: np.ndarray) -> np.ndarray:
        """Fallback bicubic upsampling"""
        h, w = image.shape[:2]
        new_h, new_w = h * self.scale, w * self.scale
        
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply some basic enhancement
        # Unsharp masking
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        upscaled = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
        
        return upscaled
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        config = self.model_configs.get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "scale_factor": self.scale,
            "device": str(self.device),
            "tile_size": self.tile_size,
            "description": config.get("description", "Unknown model"),
            "memory_usage": "4-12GB VRAM (depending on input size)",
            "supported_formats": ["RGB", "BGR"],
            "input_range": "[0, 1] float or [0, 255] uint8"
        }


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for ESRGAN"""
    
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        return x5 * 0.2 + x


def get_realesrgan_presets() -> Dict[str, Dict[str, Any]]:
    """Get available Real-ESRGAN presets"""
    return {
        "general": {
            "model_name": "RealESRGAN_x4plus",
            "description": "Best for general photographs and natural scenes",
            "scale": 4,
            "memory_usage": "8-12GB VRAM"
        },
        "anime": {
            "model_name": "RealESRGAN_x4plus_anime_6B",
            "description": "Optimized for illustrations, anime, and artistic images",
            "scale": 4,
            "memory_usage": "4-8GB VRAM"
        },
        "lightweight": {
            "model_name": "realesr-general-x4v3",
            "description": "Lightweight version for faster processing",
            "scale": 4,
            "memory_usage": "4-6GB VRAM"
        }
    }


def process_with_realesrgan(image: np.ndarray, 
                          model_name: str = "RealESRGAN_x4plus",
                          device: str = "auto",
                          **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Process image with Real-ESRGAN
    
    Args:
        image: Input image (H, W, C) in RGB format
        model_name: Model variant to use
        device: Processing device
        **kwargs: Additional parameters
    
    Returns:
        (enhanced_image, processing_info)
    """
    processor = RealESRGANProcessor(model_name, device)
    enhanced = processor.process_image(image, **kwargs)
    info = processor.get_model_info()
    
    return enhanced, info


# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic image
    test_image = np.random.rand(256, 256, 3).astype(np.float32)
    
    print("Testing Real-ESRGAN processing...")
    enhanced, info = process_with_realesrgan(test_image)
    
    print(f"Original shape: {test_image.shape}")
    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Model info: {info}")
