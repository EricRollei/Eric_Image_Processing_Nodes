"""
Wavelet Denoising Node for ComfyUI
Advanced wavelet-based denoising with multiple thresholding methods

Author: Eric Hiss (GitHub: EricRollei)
License: See LICENSE file in repository root

Wavelet Denoising Algorithms:
    VisuShrink: Donoho & Johnstone (1994) - Universal threshold
    BayesShrink: Chang et al. (2000) - Adaptive Bayesian threshold
    SUREShrink: Donoho & Johnstone (1995) - Stein's Unbiased Risk Estimate
    
    Base Reference:
    @article{donoho1995adapting,
      title={Adapting to Unknown Smoothness via Wavelet Shrinkage},
      author={Donoho, David L and Johnstone, Iain M},
      journal={Journal of the American Statistical Association},
      year={1995}
    }

Dependencies:
    - PyTorch (BSD 3-Clause License)
    - PyWavelets (MIT License)
    - NumPy (BSD 3-Clause License)
"""

import torch
from ..base_node import BaseImageProcessingNode
from Eric_Image_Processing_Nodes import (
    wavelet_denoise, 
    get_available_wavelets, 
    estimate_noise_level
)


class WaveletDenoiseNode(BaseImageProcessingNode):
    """
    Advanced wavelet denoising node with multiple threshold selection methods
    
    Excellent for:
    - Natural photographs with Gaussian noise
    - Scanned images with grain
    - General purpose denoising with edge preservation
    
    Performance: Very fast, runs on CPU, scales well with image size
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        wavelet_choices = list(get_available_wavelets().keys())
        
        return {
            "required": {
                "image": ("IMAGE",),
                "wavelet": (wavelet_choices, {
                    "default": "db8", 
                    "tooltip": "Wavelet type:\n" + "\n".join([f"• {k}: {v}" for k, v in get_available_wavelets().items()])
                }),
                "method": (["BayesShrink", "VisuShrink", "SureShrink", "manual"], {
                    "default": "BayesShrink",
                    "tooltip": "Threshold selection:\n• BayesShrink: Adaptive, best for natural images\n• VisuShrink: Conservative, preserves details\n• SureShrink: Hybrid approach using SURE\n• manual: Use custom sigma value"
                }),
                "mode": (["soft", "hard"], {
                    "default": "soft",
                    "tooltip": "Thresholding mode:\n• soft: Gradual transition, smoother (recommended)\n• hard: Sharp cutoff, preserves edges but may create artifacts"
                }),
            },
            "optional": {
                "sigma": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 255.0, 
                    "step": 0.05,
                    "tooltip": "Noise level (0 = auto-estimate):\n• 0.1-2.0: Ultra-light noise for high-quality images\n• 2.0-10.0: Light noise\n• 10.0-25.0: Moderate noise\n• 25.0-50.0: Heavy noise\n• 50.0+: Extreme noise\n• For 'manual' method: represents noise standard deviation\n• Higher values = more aggressive denoising"
                }),
                "levels": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8, 
                    "step": 1,
                    "tooltip": "Decomposition levels (0 = auto):\n• 3-4: Preserve fine details\n• 5-6: Standard denoising\n• 7-8: Heavy denoising"
                }),
                "multichannel": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process color channels independently (recommended for color images)"
                }),
                "rescale_sigma": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Rescale noise estimate for wavelet domain (recommended)"
                }),
                "show_noise_estimate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print estimated noise level to console"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_image",)
    FUNCTION = "denoise"
    CATEGORY = "Eric's Image Processing/Denoising"
    
    def denoise(
        self, 
        image, 
        wavelet="db8", 
        method="BayesShrink", 
        mode="soft",
        sigma=0.0,
        levels=0,
        multichannel=True,
        rescale_sigma=True,
        show_noise_estimate=False
    ):
        """Apply wavelet denoising to input image"""
        
        try:
            # Convert parameters
            sigma_value = None if sigma == 0.0 else sigma
            levels_value = None if levels == 0 else levels
            
            # Show noise estimate if requested
            if show_noise_estimate:
                img_np = self.tensor_to_numpy(image)
                estimated_noise = estimate_noise_level(img_np)
                print(f"\n=== WAVELET DENOISING NOISE ANALYSIS ===")
                print(f"Estimated noise level: {estimated_noise:.2f} (0-255 scale)")
                print(f"Noise level (0-1 scale): {estimated_noise/255:.4f}")
                if estimated_noise < 5:
                    print("Assessment: Very low noise - consider using conservative denoising")
                elif estimated_noise < 15:
                    print("Assessment: Low noise - standard denoising recommended")
                elif estimated_noise < 30:
                    print("Assessment: Moderate noise - aggressive denoising may be needed")
                else:
                    print("Assessment: High noise - strong denoising recommended")
                
                if method == "manual" and sigma_value is None:
                    print(f"Manual mode suggestion: Set sigma to {estimated_noise:.1f}")
                print("==========================================\n")
            
            # Define processing function
            def process_func(img_np, **kwargs):
                return wavelet_denoise(
                    img_np,
                    wavelet=wavelet,
                    sigma=sigma_value,
                    mode=mode,
                    method=method,
                    levels=levels_value,
                    multichannel=multichannel,
                    rescale_sigma=rescale_sigma
                )
            
            # Process image safely
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in wavelet denoising: {str(e)}")
            # Return original image on error
            return (image,)


class StationaryWaveletDenoiseNode(BaseImageProcessingNode):
    """
    Stationary (translation-invariant) wavelet denoising node
    
    Excellent for:
    - Images with important structures at different positions
    - Reducing shift-sensitivity artifacts
    - Higher quality denoising with less artifacts
    
    Performance: Slower than regular wavelets but better quality
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        wavelet_choices = list(get_available_wavelets().keys())
        
        return {
            "required": {
                "image": ("IMAGE",),
                "wavelet": (wavelet_choices, {
                    "default": "db8", 
                    "tooltip": "Wavelet type:\n" + "\n".join([f"• {k}: {v}" for k, v in get_available_wavelets().items()])
                }),
                "method": (["BayesShrink", "VisuShrink", "SureShrink", "manual"], {
                    "default": "BayesShrink",
                    "tooltip": "Threshold selection:\n• BayesShrink: Adaptive, best for natural images\n• VisuShrink: Conservative, preserves details\n• SureShrink: Hybrid approach using SURE\n• manual: Use custom sigma value"
                }),
                "mode": (["soft", "hard"], {
                    "default": "soft",
                    "tooltip": "Thresholding mode:\n• soft: Gradual transition, smoother (recommended)\n• hard: Sharp cutoff, preserves edges but may create artifacts"
                }),
            },
            "optional": {
                "sigma": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 255.0, 
                    "step": 0.05,
                    "tooltip": "Noise level (0 = auto-estimate):\n• 0.1-2.0: Ultra-light noise for high-quality images\n• 2.0-10.0: Light noise\n• 10.0-25.0: Moderate noise\n• 25.0-50.0: Heavy noise\n• 50.0+: Extreme noise\n• For 'manual' method: represents noise standard deviation\n• Higher values = more aggressive denoising"
                }),
                "levels": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8, 
                    "step": 1,
                    "tooltip": "Decomposition levels (0 = auto):\n• 2-3: Preserve fine details\n• 4-5: Standard denoising\n• 6-8: Heavy denoising (slower)"
                }),
                "multichannel": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process color channels independently (recommended for color images)"
                }),
                "rescale_sigma": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Rescale noise estimate for wavelet domain (recommended)"
                }),
                "show_noise_estimate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print estimated noise level to console"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_image",)
    FUNCTION = "denoise"
    CATEGORY = "Eric's Image Processing/Denoising"
    
    def denoise(
        self, 
        image, 
        wavelet="db8", 
        method="BayesShrink", 
        mode="soft",
        sigma=0.0,
        levels=0,
        multichannel=True,
        rescale_sigma=True,
        show_noise_estimate=False
    ):
        """Apply stationary wavelet denoising to input image"""
        
        try:
            # Import the new function
            from Eric_Image_Processing_Nodes.scripts.wavelet_denoise import wavelet_denoise_stationary
            
            # Convert parameters
            sigma_value = None if sigma == 0.0 else sigma
            levels_value = None if levels == 0 else levels
            
            # Show noise estimate if requested
            if show_noise_estimate:
                img_np = self.tensor_to_numpy(image)
                estimated_noise = estimate_noise_level(img_np)
                print(f"\n=== STATIONARY WAVELET DENOISING NOISE ANALYSIS ===")
                print(f"Estimated noise level: {estimated_noise:.2f} (0-255 scale)")
                print(f"Noise level (0-1 scale): {estimated_noise/255:.4f}")
                if estimated_noise < 5:
                    print("Assessment: Very low noise - consider using conservative denoising")
                elif estimated_noise < 15:
                    print("Assessment: Low noise - standard denoising recommended")
                elif estimated_noise < 30:
                    print("Assessment: Moderate noise - aggressive denoising may be needed")
                else:
                    print("Assessment: High noise - strong denoising recommended")
                
                if method == "manual" and sigma_value is None:
                    print(f"Manual mode suggestion: Set sigma to {estimated_noise:.1f}")
                print("=====================================================\n")
                if method == "manual" and sigma_value is None:
                    print(f"Suggestion: Set sigma to {estimated_noise:.1f} for manual mode")
            
            # Define processing function
            def process_func(img_np, **kwargs):
                return wavelet_denoise_stationary(
                    img_np,
                    wavelet=wavelet,
                    sigma=sigma_value,
                    mode=mode,
                    method=method,
                    levels=levels_value,
                    multichannel=multichannel,
                    rescale_sigma=rescale_sigma
                )
            
            # Process image safely
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in stationary wavelet denoising: {str(e)}")
            # Return original image on error
            return (image,)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WaveletDenoise": WaveletDenoiseNode,
    "StationaryWaveletDenoise": StationaryWaveletDenoiseNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveletDenoise": "Wavelet Denoise (Eric)",
    "StationaryWaveletDenoise": "Stationary Wavelet Denoise (Eric)"
}