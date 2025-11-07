"""
ComfyUI node for Wiener filter restoration  
Optimal frequency domain restoration when noise characteristics are known
"""

import torch
import numpy as np
from ..base_node import BaseImageProcessingNode
from Eric_Image_Processing_Nodes import (
    wiener_filter_restoration,
    adaptive_wiener_filter,
    parametric_wiener_filter,
    get_wiener_presets
)



class WienerFilterNode(BaseImageProcessingNode):
    """
    Wiener filter for optimal image restoration in frequency domain
    
    Excellent for:
    - Optimal restoration when noise level is known
    - Frequency domain sharpening with noise control
    - Lens defocus correction with minimal artifacts
    - Scientific/medical image restoration
    
    Performance: Fast (FFT-based), CPU, good quality/noise balance
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["auto", "manual", "presets"], {
                    "default": "auto",
                    "tooltip": "Parameter selection mode:\n• auto: Automatic noise/signal estimation\n• manual: Full parameter control\n• presets: Choose from common scenarios"
                }),
            },
            "optional": {
                # Auto mode parameters
                "blur_strength": (["light", "medium", "heavy"], {
                    "default": "medium",
                    "tooltip": "Blur correction strength (auto mode):\n• light: Minor defocus (sigma~1.5)\n• medium: Moderate blur (sigma~3.0)\n• heavy: Heavy blur (sigma~5.0)"
                }),
                "noise_level": (["low", "medium", "high", "auto"], {
                    "default": "auto",
                    "tooltip": "Noise level (auto mode):\n• low: Clean images (SNR>30dB)\n• medium: Moderate noise (SNR~20-30dB)\n• high: Noisy images (SNR<20dB)\n• auto: Estimate from image"
                }),
                
                # Manual mode parameters
                "blur_type": (["gaussian", "motion"], {
                    "default": "gaussian",
                    "tooltip": "Degradation type (manual mode):\n• gaussian: Lens defocus, atmospheric blur\n• motion: Camera shake, object motion"
                }),
                "blur_size": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 25.0,
                    "step": 0.05,
                    "tooltip": "Blur parameter (manual mode):\n• 0.1-0.5: Ultra-fine sharpening for high-res images\n• 0.5-2.0: Light blur correction\n• 2.0-6.0: Moderate blur (recommended)\n• 6.0-15.0: Heavy blur\n• 15.0+: Extreme blur"
                }),
                "K_value": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0001,
                    "max": 2.0,
                    "step": 0.0001,
                    "tooltip": "Regularization K (manual mode):\n• 0.0001-0.001: Ultra-clean images, maximum sharpening\n• 0.001-0.01: Clean images, sharp results\n• 0.01-0.1: Moderate noise (recommended)\n• 0.1-0.5: Noisy images, smooth results\n• 0.5-2.0: Very noisy images, heavy smoothing"
                }),
                "motion_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Motion direction (manual, motion blur only)"
                }),
                "motion_length": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.5,
                    "max": 150.0,
                    "step": 0.5,
                    "tooltip": "Motion distance (manual, motion blur only):\n• 0.5-2.0: Micro-motion correction for high-res images\n• 2.0-10.0: Light motion blur\n• 10.0-30.0: Moderate motion (recommended)\n• 30.0-80.0: Heavy motion blur\n• 80.0+: Extreme motion"
                }),
                
                # Preset mode
                "preset": (["none"] + list(get_wiener_presets().keys()), {
                    "default": "none",
                    "tooltip": "Preset configurations:\n• none: Use other parameters\n• light_blur_clean: Light blur, clean image\n• medium_blur_noisy: Moderate blur with noise\n• motion_horizontal: Horizontal motion blur\n• etc."
                }),
                
                # Advanced options
                "estimate_noise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-estimate noise parameters (recommended)"
                }),
                "clip_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clip output to valid range [0,255]"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("filtered_image",)
    FUNCTION = "filter"
    CATEGORY = "Eric's Image Processing/Restoration"
    
    def filter(
        self,
        image,
        mode="auto",
        blur_strength="medium",
        noise_level="auto",
        blur_type="gaussian",
        blur_size=3.0,
        K_value=0.05,
        motion_angle=0.0,
        motion_length=15.0,
        preset="none",
        estimate_noise=True,
        clip_output=True
    ):
        """Apply Wiener filter to input image"""
        
        try:
            if mode == "presets" and preset != "none":
                # Use preset configuration
                presets = get_wiener_presets()
                if preset in presets:
                    preset_config = presets[preset]
                    print(f"Using preset: {preset} - {preset_config['description']}")
                    
                    def process_func(img_np, **kwargs):
                        # Apply preset parameters
                        if 'blur_type' in preset_config:
                            return wiener_filter_restoration(
                                img_np,
                                blur_type=preset_config['blur_type'],
                                motion_angle=preset_config.get('motion_angle', 0),
                                motion_length=preset_config.get('motion_length', 15),
                                K=preset_config['K'],
                                estimate_noise=False,
                                clip=clip_output
                            )
                        else:
                            return wiener_filter_restoration(
                                img_np,
                                blur_type='gaussian',
                                blur_size=preset_config['blur_size'],
                                K=preset_config['K'],
                                estimate_noise=False,
                                clip=clip_output
                            )
                else:
                    raise ValueError(f"Unknown preset: {preset}")
                    
            elif mode == "auto":
                # Automatic mode
                def process_func(img_np, **kwargs):
                    return adaptive_wiener_filter(
                        img_np,
                        blur_strength=blur_strength,
                        noise_level=noise_level
                    )
                print(f"Auto Wiener: blur_strength={blur_strength}, noise_level={noise_level}")
                
            else:  # manual mode
                def process_func(img_np, **kwargs):
                    return wiener_filter_restoration(
                        img_np,
                        blur_type=blur_type,
                        blur_size=blur_size,
                        motion_angle=motion_angle,
                        motion_length=motion_length,
                        K=K_value,
                        estimate_noise=estimate_noise,
                        clip=clip_output
                    )
                print(f"Manual Wiener: {blur_type}, blur_size={blur_size}, K={K_value}")
            
            # Process image safely
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in Wiener filter: {str(e)}")
            # Return original image on error
            return (image,)


class WienerFilterSNRNode(BaseImageProcessingNode):
    """Simplified Wiener filter using Signal-to-Noise Ratio specification"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "snr_db": ("FLOAT", {
                    "default": 25.0,
                    "min": 5.0,
                    "max": 60.0,
                    "step": 1.0,
                    "tooltip": "Signal-to-Noise Ratio (dB):\n• 5-15 dB: Very noisy images\n• 15-25 dB: Moderately noisy\n• 25-40 dB: Clean images (recommended)\n• 40+ dB: Very clean images"
                }),
                "blur_sigma": ("FLOAT", {
                    "default": 2.5,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Blur size (standard deviation):\n• 1.0-2.0: Light defocus\n• 2.0-4.0: Moderate blur\n• 4.0+ dB: Heavy blur"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("filtered_image",)
    FUNCTION = "filter_snr"
    CATEGORY = "Eric's Image Processing/Restoration"
    
    def filter_snr(self, image, snr_db=25.0, blur_sigma=2.5):
        """Apply Wiener filter with SNR-based parameterization"""
        
        try:
            def process_func(img_np, **kwargs):
                return parametric_wiener_filter(
                    img_np,
                    psf_sigma=blur_sigma,
                    snr_db=snr_db
                )
            
            print(f"SNR Wiener: {snr_db}dB SNR, sigma={blur_sigma}")
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in SNR Wiener filter: {str(e)}")
            return (image,)


class WienerFilterCompareNode(BaseImageProcessingNode):
    """Compare different Wiener filter configurations side by side"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_size": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.1,
                    "tooltip": "Blur size for comparison test"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("light_reg", "medium_reg", "heavy_reg", "adaptive")
    FUNCTION = "compare"
    CATEGORY = "Eric's Image Processing/Restoration"
    
    def compare(self, image, blur_size=3.0):
        """Compare different regularization levels"""
        
        try:
            img_np = self.tensor_to_numpy(image)
            
            # Light regularization (sharp)
            result1 = wiener_filter_restoration(
                img_np, blur_size=blur_size, K=0.01, estimate_noise=False
            )
            
            # Medium regularization (balanced)
            result2 = wiener_filter_restoration(
                img_np, blur_size=blur_size, K=0.05, estimate_noise=False
            )
            
            # Heavy regularization (smooth)
            result3 = wiener_filter_restoration(
                img_np, blur_size=blur_size, K=0.2, estimate_noise=False
            )
            
            # Adaptive
            result4 = adaptive_wiener_filter(img_np, 'medium', 'auto')
            
            # Convert back to tensors
            tensor1 = self.numpy_to_tensor(result1)
            tensor2 = self.numpy_to_tensor(result2)
            tensor3 = self.numpy_to_tensor(result3)
            tensor4 = self.numpy_to_tensor(result4)
            
            return (tensor1, tensor2, tensor3, tensor4)
            
        except Exception as e:
            print(f"Error in Wiener comparison: {str(e)}")
            return (image, image, image, image)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WienerFilter": WienerFilterNode,
    "WienerFilterSNR": WienerFilterSNRNode,
    "WienerFilterCompare": WienerFilterCompareNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WienerFilter": "Wiener Filter Restoration (Eric)",
    "WienerFilterSNR": "Wiener Filter SNR (Eric)",
    "WienerFilterCompare": "Wiener Filter Compare (Eric)"
}