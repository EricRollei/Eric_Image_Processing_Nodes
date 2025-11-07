"""
ComfyUI nodes for advanced frequency domain enhancement
Homomorphic filtering, phase-preserving enhancement, and multi-scale processing
"""

import torch
import numpy as np
from ..base_node import BaseImageProcessingNode
from Eric_Image_Processing_Nodes import (
    homomorphic_filter,
    phase_preserving_enhancement,
    multiscale_fft_enhancement,
    adaptive_frequency_filter,
    get_frequency_enhancement_presets
)


class HomomorphicFilterNode(BaseImageProcessingNode):
    """
    Homomorphic filtering for illumination and reflectance separation
    
    Excellent for:
    - Correcting uneven lighting in scanned photos
    - Shadow removal and highlight recovery
    - Normalizing illumination across image regions
    - Scientific imaging with lighting variations
    
    Performance: Fast (FFT-based), dramatic lighting improvements
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "d0": ("FLOAT", {
                    "default": 40.0,
                    "min": 1.0,
                    "max": 200.0,
                    "step": 0.5,
                    "tooltip": "Cutoff frequency:\n• 1.0-10.0: Very strong illumination correction\n• 10.0-30.0: Strong illumination correction\n• 30.0-60.0: Moderate correction (recommended)\n• 60.0-100.0: Subtle correction\n• 100.0+: Very subtle correction"
                }),
            },
            "optional": {
                "gamma_h": ("FLOAT", {
                    "default": 1.8,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "High frequency gain (detail enhancement):\n• 0.1-0.8: Reduce high frequencies (smoothing)\n• 0.8-1.2: Subtle detail enhancement\n• 1.2-2.5: Enhance details (recommended)\n• 2.5-5.0: Strong detail enhancement\n• 5.0-10.0: Extreme detail enhancement"
                }),
                "gamma_l": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Low frequency gain (illumination suppression):\n• 0.01-0.3: Very strong illumination suppression\n• 0.3-0.6: Strong suppression\n• 0.6-0.8: Moderate suppression (recommended)\n• 0.8-1.0: Gentle suppression\n• 1.0-2.0: Boost low frequencies"
                }),
                "c": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Transition sharpness (1.0 recommended)"
                }),
                "filter_type": (["gaussian", "butterworth"], {
                    "default": "gaussian",
                    "tooltip": "Filter type:\n• gaussian: Smooth transitions\n• butterworth: Sharper transitions"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "enhance"
    CATEGORY = "Eric's Image Processing/Frequency Enhancement"
    
    def enhance(self, image, d0=40.0, gamma_h=1.8, gamma_l=0.6, c=1.0, filter_type="gaussian"):
        """Apply homomorphic filtering to correct illumination"""
        
        try:
            def process_func(img_np, **kwargs):
                return homomorphic_filter(
                    img_np,
                    d0=d0,
                    gamma_h=gamma_h,
                    gamma_l=gamma_l,
                    c=c,
                    filter_type=filter_type
                )
            
            print(f"Homomorphic filter: d0={d0}, γH={gamma_h}, γL={gamma_l}")
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in homomorphic filtering: {str(e)}")
            return (image,)


class PhasePreservingEnhancementNode(BaseImageProcessingNode):
    """
    Phase-preserving frequency domain enhancement
    
    Excellent for:
    - Natural-looking detail enhancement
    - Sharpening without ringing artifacts
    - Preserving image character while boosting details
    - Scientific and medical image enhancement
    
    Performance: Fast (FFT-based), natural results
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhancement_factor": ("FLOAT", {
                    "default": 1.4,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Enhancement strength:\n• 0.1-0.8: Reduce details (smoothing)\n• 0.8-1.2: Subtle enhancement\n• 1.2-2.0: Moderate enhancement (recommended)\n• 2.0-3.5: Strong enhancement\n• 3.5-5.0: Extreme enhancement"
                }),
            },
            "optional": {
                "frequency_range_low": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.8,
                    "step": 0.01,
                    "tooltip": "Low frequency cutoff:\n• 0.0-0.05: Include very low frequencies\n• 0.05-0.15: Include more low frequencies\n• 0.15-0.3: Standard range (recommended)\n• 0.3+: High frequencies only"
                }),
                "frequency_range_high": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "High frequency cutoff:\n• 0.1-0.3: Conservative (avoid noise)\n• 0.3-0.6: Moderate range\n• 0.6-0.9: Standard range (recommended)\n• 0.9-1.0: Include highest frequencies"
                }),
                "method": (["magnitude_scaling", "adaptive_scaling"], {
                    "default": "magnitude_scaling",
                    "tooltip": "Enhancement method:\n• magnitude_scaling: Uniform enhancement\n• adaptive_scaling: Content-aware enhancement"
                }),
                "preserve_dc": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve DC component (brightness) - recommended True"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "enhance"
    CATEGORY = "Eric's Image Processing/Frequency Enhancement"
    
    def enhance(
        self, 
        image, 
        enhancement_factor=1.4, 
        frequency_range_low=0.1, 
        frequency_range_high=0.7,
        method="magnitude_scaling",
        preserve_dc=True
    ):
        """Apply phase-preserving enhancement"""
        
        try:
            frequency_range = (frequency_range_low, frequency_range_high)
            
            def process_func(img_np, **kwargs):
                return phase_preserving_enhancement(
                    img_np,
                    enhancement_factor=enhancement_factor,
                    frequency_range=frequency_range,
                    method=method,
                    preserve_dc=preserve_dc
                )
            
            print(f"Phase-preserving: factor={enhancement_factor}, range=({frequency_range_low}, {frequency_range_high})")
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in phase-preserving enhancement: {str(e)}")
            return (image,)


class MultiscaleFFTEnhancementNode(BaseImageProcessingNode):
    """
    Multi-scale frequency domain enhancement using Laplacian pyramid
    
    Excellent for:
    - Sophisticated detail enhancement across multiple scales
    - Professional photo retouching
    - Scientific image analysis requiring scale-specific enhancement
    - Replacing traditional unsharp masking with superior results
    
    Performance: Medium speed, excellent quality
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scales": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 6,
                    "step": 1,
                    "tooltip": "Number of scales to process:\n• 2-3: Fast, basic enhancement\n• 4: Standard (recommended)\n• 5-6: Detailed, slower processing"
                }),
            },
            "optional": {
                "scale_1_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.05,
                    "tooltip": "Enhancement for finest details (scale 1):\n• 0.1-0.8: Reduce finest details\n• 0.8-1.2: Subtle enhancement\n• 1.2-2.0: Moderate enhancement\n• 2.0-4.0: Strong enhancement"
                }),
                "scale_2_factor": ("FLOAT", {
                    "default": 1.3,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.05,
                    "tooltip": "Enhancement for fine details (scale 2):\n• 0.1-0.8: Reduce fine details\n• 0.8-1.2: Subtle enhancement\n• 1.2-2.0: Moderate enhancement\n• 2.0-4.0: Strong enhancement"
                }),
                "scale_3_factor": ("FLOAT", {
                    "default": 1.6,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.05,
                    "tooltip": "Enhancement for medium details (scale 3):\n• 0.1-0.8: Reduce medium details\n• 0.8-1.2: Subtle enhancement\n• 1.2-2.0: Moderate enhancement\n• 2.0-4.0: Strong enhancement"
                }),
                "scale_4_factor": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.05,
                    "tooltip": "Enhancement for coarse details (scale 4):\n• 0.1-0.8: Reduce coarse details\n• 0.8-1.2: Subtle enhancement\n• 1.2-2.0: Moderate enhancement\n• 2.0-4.0: Strong enhancement"
                }),
                "scale_5_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.05,
                    "tooltip": "Enhancement for very coarse details (scale 5):\n• 0.1-0.8: Reduce very coarse details\n• 0.8-1.2: Subtle enhancement\n• 1.2-2.0: Moderate enhancement\n• 2.0-4.0: Strong enhancement"
                }),
                "scale_6_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.05,
                    "tooltip": "Enhancement for coarsest details (scale 6):\n• 0.1-0.8: Reduce coarsest details\n• 0.8-1.2: Subtle enhancement\n• 1.2-2.0: Moderate enhancement\n• 2.0-4.0: Strong enhancement"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "enhance"
    CATEGORY = "Eric's Image Processing/Frequency Enhancement"
    
    def enhance(
        self, 
        image, 
        scales=4,
        scale_1_factor=1.0,
        scale_2_factor=1.3,
        scale_3_factor=1.6,
        scale_4_factor=1.2,
        scale_5_factor=1.0,
        scale_6_factor=1.0
    ):
        """Apply multi-scale FFT enhancement"""
        
        try:
            # Collect enhancement factors
            enhancement_factors = [
                scale_1_factor, scale_2_factor, scale_3_factor,
                scale_4_factor, scale_5_factor, scale_6_factor
            ][:scales]
            
            def process_func(img_np, **kwargs):
                return multiscale_fft_enhancement(
                    img_np,
                    scales=scales,
                    enhancement_factors=enhancement_factors
                )
            
            print(f"Multi-scale FFT: {scales} scales, factors={enhancement_factors}")
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in multi-scale FFT enhancement: {str(e)}")
            return (image,)


class AdaptiveFrequencyFilterNode(BaseImageProcessingNode):
    """
    Adaptive frequency filtering based on local image statistics
    
    Excellent for:
    - Content-aware enhancement that adapts to image regions
    - Automatic detail enhancement with noise suppression
    - Professional photo retouching workflows
    - Images with mixed content (smooth areas + details)
    
    Performance: Medium speed, intelligent enhancement
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detail_enhancement": ("FLOAT", {
                    "default": 1.6,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Enhancement factor for detailed regions:\n• 0.1-0.8: Reduce details (smoothing)\n• 0.8-1.2: Subtle enhancement\n• 1.2-2.0: Standard enhancement (recommended)\n• 2.0-3.5: Strong enhancement\n• 3.5-5.0: Extreme enhancement"
                }),
            },
            "optional": {
                "smooth_enhancement": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.02,
                    "tooltip": "Enhancement factor for smooth regions:\n• 0.1-0.7: Strong smoothing\n• 0.7-1.0: Slight smoothing\n• 1.0-1.3: Gentle enhancement (recommended)\n• 1.3-2.0: Moderate enhancement\n• 2.0-3.0: Strong enhancement"
                }),
                "noise_reduction": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.2,
                    "step": 0.02,
                    "tooltip": "Reduction factor for likely noise regions:\n• 0.1-0.4: Very strong noise suppression\n• 0.4-0.7: Strong noise suppression\n• 0.7-0.9: Moderate suppression (recommended)\n• 0.9-1.0: Minimal suppression\n• 1.0-1.2: Slight enhancement even in noise regions"
                }),
                "variance_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0001,
                    "max": 0.5,
                    "step": 0.0001,
                    "tooltip": "Threshold for smooth vs detailed regions:\n• 0.0001-0.002: More regions treated as detailed\n• 0.002-0.01: Balanced (recommended)\n• 0.01-0.05: More regions treated as smooth\n• 0.05-0.5: Most regions treated as smooth"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "enhance"
    CATEGORY = "Eric's Image Processing/Frequency Enhancement"
    
    def enhance(
        self, 
        image, 
        detail_enhancement=1.6,
        smooth_enhancement=1.1,
        noise_reduction=0.8,
        variance_threshold=0.01
    ):
        """Apply adaptive frequency filtering"""
        
        try:
            def process_func(img_np, **kwargs):
                return adaptive_frequency_filter(
                    img_np,
                    local_variance_threshold=variance_threshold,
                    smooth_enhancement=smooth_enhancement,
                    detail_enhancement=detail_enhancement,
                    noise_reduction=noise_reduction
                )
            
            print(f"Adaptive frequency: detail={detail_enhancement}, smooth={smooth_enhancement}, noise={noise_reduction}")
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in adaptive frequency filtering: {str(e)}")
            return (image,)


class FrequencyEnhancementPresetsNode(BaseImageProcessingNode):
    """Preset configurations for common frequency enhancement scenarios"""
    
    @classmethod
    def INPUT_TYPES(cls):
        presets = list(get_frequency_enhancement_presets().keys())
        
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (presets, {
                    "default": "detail_enhancement",
                    "tooltip": "Preset configurations:\n" + "\n".join([
                        f"• {k}: {v['description']}" 
                        for k, v in get_frequency_enhancement_presets().items()
                    ])
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "enhance_preset"
    CATEGORY = "Eric's Image Processing/Frequency Enhancement"
    
    def enhance_preset(self, image, preset="detail_enhancement"):
        """Apply frequency enhancement using preset configuration"""
        
        try:
            presets = get_frequency_enhancement_presets()
            
            if preset not in presets:
                raise ValueError(f"Unknown preset: {preset}")
            
            preset_config = presets[preset]
            method = preset_config['method']
            params = preset_config['params']
            
            print(f"Frequency preset: {preset} - {preset_config['description']}")
            
            def process_func(img_np, **kwargs):
                if method == 'homomorphic':
                    return homomorphic_filter(img_np, **params)
                elif method == 'phase_preserving':
                    return phase_preserving_enhancement(img_np, **params)
                elif method == 'multiscale_fft':
                    return multiscale_fft_enhancement(img_np, **params)
                elif method == 'adaptive':
                    return adaptive_frequency_filter(img_np, **params)
                else:
                    raise ValueError(f"Unknown method in preset: {method}")
            
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in frequency enhancement preset: {str(e)}")
            return (image,)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "HomomorphicFilter": HomomorphicFilterNode,
    "PhasePreservingEnhancement": PhasePreservingEnhancementNode,
    "MultiscaleFFTEnhancement": MultiscaleFFTEnhancementNode,
    "AdaptiveFrequencyFilter": AdaptiveFrequencyFilterNode,
    "FrequencyEnhancementPresets": FrequencyEnhancementPresetsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HomomorphicFilter": "Homomorphic Filter (Eric)",
    "PhasePreservingEnhancement": "Phase-Preserving Enhancement (Eric)",
    "MultiscaleFFTEnhancement": "Multi-scale FFT Enhancement (Eric)",
    "AdaptiveFrequencyFilter": "Adaptive Frequency Filter (Eric)",
    "FrequencyEnhancementPresets": "Frequency Enhancement Presets (Eric)"
}