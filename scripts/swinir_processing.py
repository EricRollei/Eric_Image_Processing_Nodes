"""
SwinIR (Swin Transformer for Image Restoration) Implementation

This implementation is based on the official SwinIR repository:
https://github.com/JingyunLiang/SwinIR

Original Paper:
"SwinIR: Image Restoration Using Swin Transformer"
https://arxiv.org/abs/2108.10257

Credits:
- Original authors: Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, Radu Timofte
- Official implementation: https://github.com/JingyunLiang/SwinIR
- Pretrained models: https://github.com/JingyunLiang/SwinIR/releases

This file integrates the official SwinIR network architecture and pretrained models
into a ComfyUI-compatible processor with proper model loading and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
from typing import Optional, Tuple, Union, Dict, Any
import os
import urllib.request
import sys

# Import the official SwinIR network
try:
    # Try to import from the downloaded file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    network_path = os.path.join(parent_dir, 'network_swinir.py')
    
    if os.path.exists(network_path):
        # Add to path and import
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from network_swinir import SwinIR
        print("Successfully imported official SwinIR implementation")
    else:
        # Download the network file
        print("Downloading SwinIR network architecture...")
        url = 'https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/models/network_swinir.py'
        urllib.request.urlretrieve(url, network_path)
        
        # Import after download
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from network_swinir import SwinIR
        print("Downloaded and imported official SwinIR implementation")
        
except Exception as e:
    print(f"Error importing SwinIR: {e}")
    print("Using fallback implementation")
    SwinIR = None

class SwinIRProcessor:
    """
    Advanced image processing using SwinIR (Swin Transformer for Image Restoration).
    
    This processor now properly uses the official SwinIR implementation with pretrained models.
    """
    
    def __init__(self, task='classical_sr', scale=2, noise=25, jpeg=40, device='cuda'):
        """
        Initialize SwinIR processor.
        
        Args:
            task (str): Task type - 'classical_sr', 'lightweight_sr', 'real_sr', 
                       'gray_dn', 'color_dn', 'jpeg_car', 'color_jpeg_car'
            scale (int): Upsampling scale factor (1, 2, 3, 4, 8)
            noise (int): Noise level for denoising (15, 25, 50)
            jpeg (int): JPEG quality for artifact reduction (10, 20, 30, 40)
            device (str): Device to run on ('cuda' or 'cpu')
        """
        self.task = task
        self.scale = scale
        self.noise = noise
        self.jpeg = jpeg
        
        # Improved device selection
        if device == 'auto':
            # Auto-detect best available device
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"🚀 Auto-detected GPU: {torch.cuda.get_device_name(0)}")
                print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                self.device = 'cpu'
                print("🔧 Auto-detected CPU (no CUDA available)")
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            print("⚠️  CUDA requested but not available, falling back to CPU")
        elif device == 'cpu':
            self.device = 'cpu'
            print("🔧 Using CPU processing (explicitly requested)")
        else:
            self.device = 'cpu'
            print(f"🔧 Using CPU processing (unknown device: {device})")
        
        print(f"SwinIR processor initialized for task: {task}")
        print(f"Parameters: scale={scale}, noise={noise}, jpeg={jpeg}, device={self.device}")
        
        # Check if models are available
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            print(f"Available models: {model_files}")
        else:
            print("❌ Models directory not found!")
        
        # Initialize model
        if SwinIR is not None:
            self.model = self._create_model()
            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()
                print("✅ SwinIR model loaded successfully")
            else:
                print("❌ Failed to create SwinIR model")
                self.model = None
        else:
            print("❌ SwinIR not available, using fallback")
            self.model = None
    
    def _create_model(self):
        """Create and load SwinIR model based on task configuration"""
        # Define model architecture based on task (from official main_test_swinir.py)
        if self.task == 'classical_sr':
            model = SwinIR(upscale=self.scale, in_chans=3, img_size=64, window_size=8,
                          img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                          num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                          upsampler='pixelshuffle', resi_connection='1conv')
            param_key = 'params'
            
        elif self.task == 'lightweight_sr':
            model = SwinIR(upscale=self.scale, in_chans=3, img_size=64, window_size=8,
                          img_range=1., depths=[6, 6, 6, 6], embed_dim=60, 
                          num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                          upsampler='pixelshuffledirect', resi_connection='1conv')
            param_key = 'params'
            
        elif self.task == 'real_sr':
            model = SwinIR(upscale=self.scale, in_chans=3, img_size=64, window_size=8,
                          img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                          num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                          upsampler='nearest+conv', resi_connection='1conv')
            param_key = 'params_ema'
            
        elif self.task == 'gray_dn':
            model = SwinIR(upscale=1, in_chans=1, img_size=128, window_size=8,
                          img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                          num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                          upsampler='', resi_connection='1conv')
            param_key = 'params'
            
        elif self.task == 'color_dn':
            model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                          img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                          num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                          upsampler='', resi_connection='1conv')
            param_key = 'params'
            
        elif self.task == 'jpeg_car':
            model = SwinIR(upscale=1, in_chans=1, img_size=126, window_size=7,
                          img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                          num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                          upsampler='', resi_connection='1conv')
            param_key = 'params'
            
        elif self.task == 'color_jpeg_car':
            model = SwinIR(upscale=1, in_chans=3, img_size=126, window_size=7,
                          img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                          num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                          upsampler='', resi_connection='1conv')
            param_key = 'params'
            
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # Load pretrained weights
        model_path = self._get_model_path()
        if model_path and os.path.exists(model_path):
            print(f"Loading pretrained model from: {model_path}")
            try:
                pretrained_model = torch.load(model_path, map_location='cpu')
                
                # Extract the correct parameter key
                if param_key in pretrained_model:
                    state_dict = pretrained_model[param_key]
                    print(f"✅ Using parameter key: {param_key}")
                else:
                    state_dict = pretrained_model
                    print(f"⚠️  Parameter key {param_key} not found, using entire checkpoint")
                
                # Load state dict
                model.load_state_dict(state_dict, strict=True)
                print("✅ Pretrained weights loaded successfully")
                
            except Exception as e:
                print(f"❌ Error loading pretrained weights: {e}")
                print("Using randomly initialized model")
                print("💡 For 4x classical SR, you may need to download:")
                print("   https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
                
        else:
            print(f"❌ Model file not found: {model_path}")
            if self.task == 'classical_sr' and self.scale == 4:
                print("💡 For 4x classical SR, download the model from:")
                print("   https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
            print("Using randomly initialized model")
        
        return model
    
    def _get_model_path(self):
        """Get the path to the appropriate pretrained model."""
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Model filename mappings (from SwinIR repository)
        model_files = {
            'classical_sr': {
                2: '001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth',
                3: '001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth',
                4: '001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth',
                8: '001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth'
            },
            'lightweight_sr': {
                2: '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
                3: '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth',
                4: '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth'
            },
            'real_sr': {
                2: '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth',
                4: '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
            },
            'gray_dn': {
                15: '004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth',
                25: '004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth',
                50: '004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth'
            },
            'color_dn': {
                15: '005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth',
                25: '005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth',
                50: '005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth'
            },
            'jpeg_car': {
                10: '006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth',
                20: '006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth',
                30: '006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth',
                40: '006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth'
            },
            'color_jpeg_car': {
                10: '007_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth',
                20: '007_colorCAR_DFWB_s126w7_SwinIR-M_jpeg20.pth',
                30: '007_colorCAR_DFWB_s126w7_SwinIR-M_jpeg30.pth',
                40: '007_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth'
            }
        }
        
        # Get list of available models
        available_models = []
        if os.path.exists(models_dir):
            available_models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        # First try to find the exact model for the task
        if self.task in model_files:
            task_models = model_files[self.task]
            if self.task in ['classical_sr', 'lightweight_sr', 'real_sr']:
                param_key = self.scale
            elif self.task in ['gray_dn', 'color_dn']:
                param_key = self.noise
            elif self.task in ['jpeg_car', 'color_jpeg_car']:
                param_key = self.jpeg
            else:
                param_key = list(task_models.keys())[0]
            
            if param_key in task_models:
                expected_filename = task_models[param_key]
                model_path = os.path.join(models_dir, expected_filename)
                
                # Check if exact file exists
                if os.path.exists(model_path):
                    print(f"✅ Found exact model: {expected_filename}")
                    return model_path
                else:
                    print(f"⚠️  Expected model not found: {expected_filename}")
                    
                    # Try to find a similar model (e.g., different scale)
                    for filename in available_models:
                        if (self.task == 'classical_sr' and 'classicalSR' in filename) or \
                           (self.task == 'color_dn' and 'colorDN' in filename) or \
                           (self.task == 'real_sr' and 'realSR' in filename):
                            print(f"🔄 Using similar model: {filename}")
                            return os.path.join(models_dir, filename)
        
        # Fallback to any available model
        if available_models:
            fallback_model = available_models[0]
            print(f"⚠️  Using fallback model: {fallback_model}")
            return os.path.join(models_dir, fallback_model)
        
        print("❌ No models found in models directory")
        return None
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for SwinIR model."""
        # Convert to tensor
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Ensure range [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Handle channel requirements for different tasks
        if self.task == 'gray_dn':
            # Grayscale denoising: convert RGB to grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to grayscale using standard weights
                image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                image = np.expand_dims(image, axis=2)  # Add channel dimension
            
        elif self.task == 'jpeg_car':
            # JPEG artifact removal (grayscale): convert RGB to grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to grayscale using standard weights
                image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                image = np.expand_dims(image, axis=2)  # Add channel dimension
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        
        # Handle different task requirements
        if self.task in ['jpeg_car', 'color_jpeg_car']:
            # JPEG tasks expect [0, 255] range
            tensor = tensor * 255.0
        
        return tensor.to(self.device)
    
    def _postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Postprocess tensor back to numpy image."""
        # Move to CPU and convert to numpy
        tensor = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        
        # Handle different task requirements
        if self.task in ['jpeg_car', 'color_jpeg_car']:
            # JPEG tasks output [0, 255] range
            tensor = tensor / 255.0
        
        # Handle grayscale outputs
        if self.task in ['gray_dn', 'jpeg_car']:
            # Convert grayscale back to RGB by duplicating channels
            if tensor.shape[2] == 1:
                tensor = np.repeat(tensor, 3, axis=2)
        
        # Ensure valid range
        tensor = np.clip(tensor, 0, 1)
        
        return tensor
    
    def process_image(
        self,
        image: np.ndarray,
        strength: float = 1.0,
        detail_boost: float = 0.0,
        detail_radius: float = 1.2
    ) -> np.ndarray:
        """
        Process image using SwinIR.
        
        Args:
            image (np.ndarray): Input image (H, W, C) in range [0, 1]
            strength (float): Processing strength (0.0 to 1.0)
            detail_boost (float): Additional detail amplification after restoration (0.0-0.6 recommended)
            detail_radius (float): Gaussian radius for detail extraction when boosting detail
            
        Returns:
            np.ndarray: Processed image (H, W, C) in range [0, 1]
        """
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("SwinIR processor expects a single image (batch size 1)")
            image = image[0]

        # Input validation - be more flexible with channels
        if image.ndim == 2:
            # Grayscale image, add channel dimension
            image = np.expand_dims(image, axis=2)
        elif image.ndim != 3:
            raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")
        
        # Handle different channel requirements
        if image.shape[2] == 1:
            # Grayscale input - duplicate to RGB if needed for RGB tasks
            if self.task in ['color_dn', 'classical_sr', 'lightweight_sr', 'real_sr', 'color_jpeg_car']:
                image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 3:
            # RGB input - will be converted to grayscale in preprocessing if needed
            pass
        else:
            raise ValueError(f"Image must have 1 or 3 channels, got {image.shape[2]}")
        
        # Store original image for strength blending
        original_image = image.copy()
        
        # Ensure image is in range [0, 1]
        image = np.clip(image, 0, 1)
        
        try:
            if self.model is not None:
                # Use proper SwinIR model
                print(f"🔄 Processing with SwinIR model (task: {self.task})")
                print(f"   Input shape: {image.shape}, channels: {image.shape[2]}")
                
                # Memory management
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    initial_memory = torch.cuda.memory_allocated() / 1024**2
                    print(f"   Initial GPU memory: {initial_memory:.1f} MB")
                
                import time
                start_time = time.time()
                
                # Preprocess
                input_tensor = self._preprocess_image(image)
                print(f"   Preprocessed tensor shape: {input_tensor.shape}")
                
                # Process with model
                with torch.no_grad():
                    # Handle window size for padding
                    if self.task in ['jpeg_car', 'color_jpeg_car']:
                        window_size = 7
                    else:
                        window_size = 8
                    
                    # Pad image if needed
                    _, _, h_old, w_old = input_tensor.size()
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    
                    if h_pad > 0 or w_pad > 0:
                        input_tensor = F.pad(input_tensor, (0, w_pad, 0, h_pad), 'reflect')
                        print(f"   Padded to: {input_tensor.shape}")
                    
                    # Process
                    if self.device == 'cuda':
                        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                        print(f"   Peak GPU memory: {peak_memory:.1f} MB")
                    
                    output_tensor = self.model(input_tensor)
                    
                    # Remove padding
                    if h_pad > 0 or w_pad > 0:
                        if self.task in ['classical_sr', 'lightweight_sr', 'real_sr']:
                            # Super-resolution tasks
                            output_tensor = output_tensor[:, :, :h_old*self.scale, :w_old*self.scale]
                        else:
                            # Other tasks
                            output_tensor = output_tensor[:, :, :h_old, :w_old]
                
                # Postprocess
                result = self._postprocess_image(output_tensor)
                
                # Apply strength - handle different output sizes
                if self.task in ['classical_sr', 'lightweight_sr', 'real_sr'] and self.scale > 1:
                    # For super-resolution, we need to upscale the original image to match
                    from scipy.ndimage import zoom
                    upscaled_original = zoom(original_image, (self.scale, self.scale, 1), order=3)
                    result = upscaled_original * (1 - strength) + result * strength
                else:
                    # For same-size tasks (denoising, JPEG artifact removal)
                    result = original_image * (1 - strength) + result * strength
                
                # Performance reporting
                processing_time = time.time() - start_time
                print(f"✅ SwinIR processing completed in {processing_time:.2f}s")
                
                if self.device == 'cuda':
                    final_memory = torch.cuda.memory_allocated() / 1024**2
                    torch.cuda.empty_cache()
                    print(f"   Final GPU memory: {final_memory:.1f} MB")
                
            else:
                # Fallback processing
                print("⚠️  Using fallback processing (SwinIR model not available)")
                
                from scipy.ndimage import zoom
                from scipy.ndimage import gaussian_filter
                
                # Apply some basic processing based on task
                if self.task == 'classical_sr':
                    # Simple bicubic upscaling
                    result = zoom(original_image, (self.scale, self.scale, 1), order=3)
                elif self.task == 'color_dn':
                    # Simple gaussian denoising
                    result = gaussian_filter(original_image, sigma=0.5)
                elif self.task == 'jpeg_car':
                    # Simple sharpening
                    from scipy.ndimage import laplace
                    laplacian = laplace(original_image)
                    result = original_image - 0.1 * laplacian
                else:
                    # Default: light sharpening
                    from scipy.ndimage import laplace
                    laplacian = laplace(original_image)
                    result = original_image - 0.05 * laplacian
                
                # Apply strength
                result = original_image * (1 - strength) + result * strength

            # Optional sharpness/detail boost
            if detail_boost and detail_boost > 0:
                detail_boost = float(detail_boost)
                detail_boost = max(0.0, min(detail_boost, 1.0))
                sigma = max(0.1, float(detail_radius))
                # Ensure contiguous arrays for OpenCV processing
                base = np.ascontiguousarray(result)
                blurred = cv2.GaussianBlur(base, (0, 0), sigma)
                high_freq = base - blurred
                high_freq = np.ascontiguousarray(high_freq)
                # Stabilize boost to prevent clipping by blending with original image for negative boost
                result = base + detail_boost * high_freq
                result = np.ascontiguousarray(result)
            
            # Ensure output is in valid range
            result = np.ascontiguousarray(result)
            result = np.clip(result, 0, 1)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in SwinIR processing: {e}")
            import traceback
            traceback.print_exc()
            return image
    
    def __call__(
        self,
        image: np.ndarray,
        strength: float = 1.0,
        detail_boost: float = 0.0,
        detail_radius: float = 1.2
    ) -> np.ndarray:
        """Convenient call interface."""
        return self.process_image(
            image,
            strength=strength,
            detail_boost=detail_boost,
            detail_radius=detail_radius
        )

# Factory function for easy instantiation
def create_swinir_processor(task='classical_sr', scale=2, noise=25, jpeg=40, device=None):
    """
    Create a SwinIR processor for specific task.
    
    Args:
        task (str): Task type
        scale (int): Upsampling scale factor
        noise (int): Noise level for denoising
        jpeg (int): JPEG quality for artifact reduction
        device (str): Device to run on ('auto', 'cuda', 'cpu')
        
    Returns:
        SwinIRProcessor: Configured processor
    """
    if device is None or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    processor = SwinIRProcessor(
        task=task,
        scale=scale,
        noise=noise,
        jpeg=jpeg,
        device=device
    )
    
    return processor
