"""
BM3D ComfyUI Nodes
Block-matching and 3D filtering for advanced image denoising in ComfyUI

Provides nodes for:
- BM3D Grayscale Denoising
- BM3D Color Denoising
- BM3D Deblurring + Denoising
- BM3D Profile Comparison
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
import warnings

try:
    from ..base_node import BaseImageProcessingNode
    # Try to import the processor - use fallback for circular import issues
    try:
        from Eric_Image_Processing_Nodes import BM3DProcessor
    except ImportError:
        try:
            from ..scripts.bm3d_denoise import BM3DProcessor
        except ImportError:
            print("Warning: BM3DProcessor not available")
            BM3DProcessor = None
    BASE_NODE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Base node or BM3D scripts not available: {e}")
    BASE_NODE_AVAILABLE = False
    BM3DProcessor = None
    # Fallback base class
    class BaseImageProcessingNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}}
        
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "process"
        CATEGORY = "Eric's Image Processing"


class BM3DDenoiseNode(BaseImageProcessingNode):
    """
    BM3D Denoising Node: State-of-the-art block-matching and 3D filtering
    
    Automatically handles both grayscale and color images using the BM3D algorithm,
    which excels at removing additive spatially correlated stationary Gaussian noise.
    
    Features:
    - Automatic grayscale/color detection
    - Multiple quality profiles
    - Automatic noise estimation
    - Professional-grade results
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "profile": (["np", "refilter", "vn", "vn_old", "high", "deb"], {
                    "default": "np",
                    "tooltip": "BM3D Profile:\n• np: Normal Profile (balanced)\n• refilter: Enhanced quality\n• vn: Very noisy images\n• high: Maximum quality\n• deb: Debug mode"
                }),
                "stage": (["all", "hard", "wiener"], {
                    "default": "all",
                    "tooltip": "Processing stage:\n• all: Complete BM3D (recommended)\n• hard: Hard thresholding only\n• wiener: Wiener filtering focus"
                }),
            },
            "optional": {
                "noise_sigma": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.001,
                    "tooltip": "Noise standard deviation (0 = auto-estimate)"
                }),
                "colorspace": (["YCbCr", "opp"], {
                    "default": "YCbCr",
                    "tooltip": "Color space for RGB processing:\n• YCbCr: Standard luminance/chrominance\n• opp: Opponent color space"
                }),
                "auto_estimate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically estimate noise level"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("denoised_image", "processing_info")
    FUNCTION = "process_bm3d"
    CATEGORY = "Eric's Image Processing/BM3D"
    
    def __init__(self):
        super().__init__()
        self.processor = None
    
    def process_bm3d(self, image: torch.Tensor, profile: str = "np", stage: str = "all",
                    noise_sigma: float = 0.0, colorspace: str = "YCbCr", 
                    auto_estimate: bool = True) -> Tuple[torch.Tensor, str]:
        """Process image with BM3D denoising"""
        
        if not BASE_NODE_AVAILABLE or BM3DProcessor is None:
            return image, "❌ BM3D components not available"
        
        try:
            # Convert ComfyUI tensor to numpy
            if len(image.shape) == 4:
                np_image = image[0].cpu().numpy()  # Take first image from batch
            else:
                np_image = image.cpu().numpy()
            
            # Validate dimensions
            if np_image.ndim not in [2, 3]:
                return (image, f"❌ Invalid image dimensions: {np_image.shape}. Expected 2D or 3D array.")
            
            if np_image.ndim == 3 and np_image.shape[2] not in [1, 3]:
                return (image, f"❌ Invalid channel count: {np_image.shape[2]}. Expected 1 or 3 channels.")
            
            # Check for reasonable image size
            h, w = np_image.shape[:2]
            if h < 16 or w < 16:
                return (image, f"❌ Image too small: {h}x{w}. Minimum size is 16x16.")
            
            if h > 8192 or w > 8192:
                return (image, f"⚠️ Warning: Large image {h}x{w} may cause memory issues.")
            
            # Ensure values are in [0, 1]
            if np_image.max() > 1.0:
                np_image = np_image / 255.0
            
            # Ensure contiguous array for processing
            np_image = np.ascontiguousarray(np_image)
            
            # Initialize processor
            if self.processor is None:
                self.processor = BM3DProcessor()
            
            # Prepare noise sigma
            sigma = None if (auto_estimate or noise_sigma == 0.0) else noise_sigma
            
            info_parts = []
            info_parts.append(f"🔬 BM3D Processing:")
            info_parts.append(f"  • Profile: {profile}")
            info_parts.append(f"  • Stage: {stage}")
            info_parts.append(f"  • Image shape: {np_image.shape}")
            
            if len(np_image.shape) == 3 and np_image.shape[2] == 3:
                info_parts.append(f"  • Color space: {colorspace}")
            
            if sigma is not None:
                info_parts.append(f"  • Manual noise σ: {sigma:.4f}")
            else:
                info_parts.append(f"  • Auto noise estimation: enabled")
            
            info_parts.append("")
            
            # Process image
            result, process_info = self.processor.process_image(
                np_image, 
                sigma=sigma, 
                profile=profile, 
                stage=stage,
                colorspace=colorspace
            )
            
            if result is not None:
                # Convert back to ComfyUI tensor
                if len(result.shape) == 2:
                    result = np.expand_dims(result, axis=2)
                
                result_tensor = torch.from_numpy(result).float()
                
                # Add batch dimension if needed
                if len(result_tensor.shape) == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                
                # Add processing results to info
                info_parts.append(f"✅ BM3D {process_info.get('mode', 'processing')} completed")
                info_parts.append(f"📊 Processing Details:")
                
                if 'sigma_estimated' in process_info:
                    if isinstance(process_info['sigma_estimated'], list):
                        sigma_str = f"[{', '.join([f'{s:.4f}' for s in process_info['sigma_estimated']])}]"
                    else:
                        sigma_str = f"{process_info['sigma_estimated']:.4f}"
                    info_parts.append(f"  • Estimated noise σ: {sigma_str}")
                
                if 'psnr' in process_info:
                    info_parts.append(f"  • PSNR: {process_info['psnr']:.2f} dB")
                
                if 'ssim' in process_info:
                    info_parts.append(f"  • SSIM: {process_info['ssim']:.4f}")
                
                output_range = process_info.get('output_range', [0, 1])
                info_parts.append(f"  • Output range: [{output_range[0]:.3f}, {output_range[1]:.3f}]")
                
                processing_info = "\n".join(info_parts)
                return (result_tensor, processing_info)
            else:
                error_info = "\n".join(info_parts + [f"❌ BM3D processing failed: {process_info.get('error', 'Unknown error')}"])
                return (image, error_info)
                
        except Exception as e:
            error_msg = f"❌ BM3D error: {str(e)}"
            print(error_msg)
            return (image, error_msg)


class BM3DDeblurNode(BaseImageProcessingNode):
    """
    BM3D Deblurring Node: Combined deblurring and denoising
    
    Performs joint deblurring and denoising using BM3D algorithm.
    Requires a point spread function (PSF) to model the blur.
    
    Perfect for:
    - Motion blur removal
    - Gaussian blur removal
    - Simultaneous blur and noise removal
    - Professional image restoration
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "profile": (["np", "refilter", "vn", "vn_old", "high", "deb"], {
                    "default": "high"
                }),
                "psf_type": (["gaussian", "motion"], {
                    "default": "gaussian",
                    "tooltip": "Type of blur to remove:\n• gaussian: Gaussian blur\n• motion: Motion blur"
                }),
                "psf_size": ("INT", {
                    "default": 15, 
                    "min": 3, 
                    "max": 51, 
                    "step": 2,
                    "tooltip": "PSF kernel size (odd numbers only)"
                }),
                "blur_sigma": ("FLOAT", {
                    "default": 2.0, 
                    "min": 0.1, 
                    "max": 10.0, 
                    "step": 0.1,
                    "tooltip": "Blur strength parameter"
                }),
            },
            "optional": {
                "noise_sigma": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.001,
                    "tooltip": "Noise standard deviation (0 = auto-estimate)"
                }),
                "auto_estimate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically estimate noise level"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("deblurred_image", "processing_info")
    FUNCTION = "process_deblur"
    CATEGORY = "Eric's Image Processing/BM3D"
    
    def __init__(self):
        super().__init__()
        self.processor = None
    
    def process_deblur(self, image: torch.Tensor, profile: str = "high", 
                      psf_type: str = "gaussian", psf_size: int = 15, 
                      blur_sigma: float = 2.0, noise_sigma: float = 0.0, 
                      auto_estimate: bool = True) -> Tuple[torch.Tensor, str]:
        """Process image with BM3D deblurring"""
        
        if not BASE_NODE_AVAILABLE or BM3DProcessor is None:
            return image, "❌ BM3D components not available"
        
        try:
            # Convert ComfyUI tensor to numpy
            if len(image.shape) == 4:
                np_image = image[0].cpu().numpy()
            else:
                np_image = image.cpu().numpy()
            
            # Validate: BM3D deblurring requires grayscale images
            if np_image.ndim == 3:
                if np_image.shape[2] == 3:
                    # Convert to grayscale
                    np_image = np_image.mean(axis=2)
                elif np_image.shape[2] == 1:
                    np_image = np_image[:, :, 0]
            
            if np_image.ndim != 2:
                return (image, f"❌ BM3D deblurring requires grayscale images. Got shape: {np_image.shape}")
            
            # Check for reasonable image size
            h, w = np_image.shape
            if h < 16 or w < 16:
                return (image, f"❌ Image too small: {h}x{w}. Minimum size is 16x16.")
            
            # Ensure values are in [0, 1]
            if np_image.max() > 1.0:
                np_image = np_image / 255.0
            
            # Ensure contiguous array
            np_image = np.ascontiguousarray(np_image)
            
            # Initialize processor
            if self.processor is None:
                self.processor = BM3DProcessor()
            
            # Create PSF
            if psf_type == "gaussian":
                psf = self.processor.create_gaussian_psf((psf_size, psf_size), blur_sigma)
            else:  # motion blur
                # Create simple motion blur PSF
                psf = np.zeros((psf_size, psf_size))
                center = psf_size // 2
                length = int(blur_sigma * 5)  # Motion blur length
                for i in range(max(0, center - length//2), min(psf_size, center + length//2 + 1)):
                    psf[center, i] = 1.0
                psf = psf / psf.sum()
            
            # Prepare noise sigma
            sigma = None if (auto_estimate or noise_sigma == 0.0) else noise_sigma
            
            info_parts = []
            info_parts.append(f"🎯 BM3D Deblurring:")
            info_parts.append(f"  • Profile: {profile}")
            info_parts.append(f"  • PSF type: {psf_type}")
            info_parts.append(f"  • PSF size: {psf_size}x{psf_size}")
            info_parts.append(f"  • Blur sigma: {blur_sigma}")
            
            if sigma is not None:
                info_parts.append(f"  • Manual noise σ: {sigma:.4f}")
            else:
                info_parts.append(f"  • Auto noise estimation: enabled")
            
            # Handle color vs grayscale processing
            is_color = len(np_image.shape) == 3 and np_image.shape[2] == 3
            info_parts.append(f"  • Image type: {'Color (LAB processing)' if is_color else 'Grayscale'}")
            info_parts.append("")
            
            if is_color:
                # Convert RGB to LAB color space
                from skimage import color
                lab_image = color.rgb2lab(np_image)
                
                # Extract L, a, b channels
                L_channel = lab_image[:, :, 0] / 100.0  # Normalize L to [0, 1]
                a_channel = lab_image[:, :, 1]
                b_channel = lab_image[:, :, 2]
                
                # Process only the luminance (L) channel with BM3D
                result_L, process_info = self.processor.denoise_with_deblurring(
                    L_channel, psf, sigma=sigma, profile=profile
                )
                
                if result_L is not None:
                    # Reconstruct LAB image
                    result_L_normalized = result_L * 100.0  # Denormalize L back to [0, 100]
                    reconstructed_lab = np.stack([result_L_normalized, a_channel, b_channel], axis=2)
                    
                    # Convert back to RGB
                    result = color.lab2rgb(reconstructed_lab)
                    result = np.clip(result, 0, 1)
                    
                    info_parts.append(f"✅ BM3D deblurring completed (LAB processing)")
                else:
                    result = None
                    info_parts.append(f"❌ BM3D deblurring failed on luminance channel")
            else:
                # Process grayscale image directly
                result, process_info = self.processor.denoise_with_deblurring(
                    np_image, psf, sigma=sigma, profile=profile
                )
                
                if result is not None:
                    info_parts.append(f"✅ BM3D deblurring completed (grayscale)")
                else:
                    info_parts.append(f"❌ BM3D deblurring failed")
            
            if result is not None:
                # Convert back to ComfyUI tensor
                if len(result.shape) == 2:
                    result = np.expand_dims(result, axis=2)
                
                result_tensor = torch.from_numpy(result).float()
                
                if len(result_tensor.shape) == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                
                # Add processing results to info
                info_parts.append(f"📊 Processing Details:")
                
                if 'sigma_estimated' in process_info:
                    info_parts.append(f"  • Estimated noise σ: {process_info['sigma_estimated']:.4f}")
                
                if 'psnr' in process_info:
                    info_parts.append(f"  • PSNR: {process_info['psnr']:.2f} dB")
                
                if 'ssim' in process_info:
                    info_parts.append(f"  • SSIM: {process_info['ssim']:.4f}")
                
                processing_info = "\n".join(info_parts)
                return (result_tensor, processing_info)
            else:
                error_info = "\n".join(info_parts + ["❌ BM3D deblurring failed"])
                return (image, error_info)
                
        except Exception as e:
            error_msg = f"❌ BM3D deblurring error: {str(e)}"
            print(error_msg)
            return (image, error_msg)


class BM3DComparisonNode(BaseImageProcessingNode):
    """
    BM3D Profile Comparison Node: Compare different BM3D profiles
    
    Processes the same image with multiple BM3D profiles and provides
    visual comparison with quality metrics.
    
    Profiles compared:
    - np: Normal Profile (balanced)
    - refilter: Enhanced quality
    - high: Maximum quality
    - vn: Very noisy images
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stage": (["all", "hard", "wiener"], {
                    "default": "all"
                }),
            },
            "optional": {
                "noise_sigma": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.001
                }),
                "colorspace": (["YCbCr", "opp"], {
                    "default": "YCbCr"
                }),
                "compare_profiles": ("STRING", {
                    "default": "np,refilter,high,vn",
                    "tooltip": "Comma-separated list of profiles to compare"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("profile_1", "profile_2", "profile_3", "profile_4", "comparison_report")
    FUNCTION = "compare_profiles"
    CATEGORY = "Eric's Image Processing/BM3D"
    
    def __init__(self):
        super().__init__()
        self.processor = None
    
    def compare_profiles(self, image: torch.Tensor, stage: str = "all",
                        noise_sigma: float = 0.0, colorspace: str = "YCbCr",
                        compare_profiles: str = "np,refilter,high,vn") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Compare BM3D profiles"""
        
        if not BASE_NODE_AVAILABLE:
            return image, image, image, image, "❌ BM3D comparison components not available"
        
        try:
            # Convert ComfyUI tensor to numpy
            if len(image.shape) == 4:
                np_image = image[0].cpu().numpy()
            else:
                np_image = image.cpu().numpy()
            
            # Ensure values are in [0, 1]
            if np_image.max() > 1.0:
                np_image = np_image / 255.0
            
            # Initialize processor
            if self.processor is None:
                self.processor = BM3DProcessor()
            
            # Parse profiles to compare
            profiles = [p.strip() for p in compare_profiles.split(",")]
            profiles = profiles[:4]  # Limit to 4 profiles max
            
            # Prepare noise sigma
            sigma = None if noise_sigma == 0.0 else noise_sigma
            
            report_parts = []
            report_parts.append("🔬 BM3D Profile Comparison")
            report_parts.append("=" * 50)
            
            # Image analysis
            if sigma is None:
                estimated_sigma = self.processor.estimate_noise_sigma(np_image)
                report_parts.append(f"📊 Estimated noise σ: {estimated_sigma:.4f}")
            else:
                report_parts.append(f"📊 Manual noise σ: {sigma:.4f}")
            
            report_parts.append(f"🎛️ Stage: {stage}")
            report_parts.append(f"🌈 Color space: {colorspace}")
            report_parts.append("")
            
            # Process with each profile
            results = []
            for i, profile in enumerate(profiles):
                report_parts.append(f"Processing profile {i+1}: {profile}")
                
                try:
                    result, info = self.processor.process_image(
                        np_image, sigma=sigma, profile=profile, 
                        stage=stage, colorspace=colorspace
                    )
                    
                    if result is not None:
                        # Convert to tensor
                        if len(result.shape) == 2:
                            result = np.expand_dims(result, axis=2)
                        
                        result_tensor = torch.from_numpy(result).float()
                        if len(result_tensor.shape) == 3:
                            result_tensor = result_tensor.unsqueeze(0)
                        
                        results.append(result_tensor)
                        
                        # Add metrics to report
                        profile_info = f"  • {profile.upper()}: "
                        if 'psnr' in info:
                            profile_info += f"PSNR={info['psnr']:.2f}dB, "
                        if 'ssim' in info:
                            profile_info += f"SSIM={info['ssim']:.4f}"
                        
                        report_parts.append(profile_info)
                    else:
                        results.append(image)
                        report_parts.append(f"  • {profile.upper()}: Processing failed")
                        
                except Exception as e:
                    results.append(image)
                    report_parts.append(f"  • {profile.upper()}: Error - {e}")
            
            # Pad results to 4 outputs
            while len(results) < 4:
                results.append(image)
            
            # Summary
            report_parts.append("")
            report_parts.append("📋 Profile Characteristics:")
            profile_descriptions = self.processor.get_profile_info()
            for profile in profiles:
                if profile in profile_descriptions:
                    report_parts.append(f"  • {profile}: {profile_descriptions[profile]}")
            
            comparison_report = "\n".join(report_parts)
            
            return (results[0], results[1], results[2], results[3], comparison_report)
            
        except Exception as e:
            error_msg = f"❌ BM3D comparison error: {str(e)}"
            print(error_msg)
            return (image, image, image, image, error_msg)


# Node class mappings for ComfyUI registration
BM3D_NODE_CLASS_MAPPINGS = {
    "BM3DDenoiseNode": BM3DDenoiseNode,
    "BM3DDeblurNode": BM3DDeblurNode,
    "BM3DComparisonNode": BM3DComparisonNode,
}

BM3D_NODE_DISPLAY_NAME_MAPPINGS = {
    "BM3DDenoiseNode": "BM3D Denoising",
    "BM3DDeblurNode": "BM3D Deblurring",
    "BM3DComparisonNode": "BM3D Profile Comparison",
}
