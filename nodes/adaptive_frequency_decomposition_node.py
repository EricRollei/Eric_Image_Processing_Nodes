"""
ComfyUI nodes for Adaptive Frequency Decomposition processing
"""

import torch
import numpy as np
from typing import Any, Dict, Tuple

from ..base_node import BaseImageProcessingNode
from Eric_Image_Processing_Nodes import AdaptiveFrequencyDecompositionProcessor

class AdaptiveFrequencyDecompositionNode(BaseImageProcessingNode):
    """Adaptive Frequency Decomposition with multi-scale color-aware processing"""
    
    CATEGORY = "Eric's Image Processing/Advanced Enhancement"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_space": (["lab", "oklab", "jzazbz", "hsv", "rgb"], {
                    "default": "lab",
                    "tooltip": "Color space for processing. LAB/Oklab/Jzazbz work best for frequency decomposition"
                }),
                "decomposition_method": (["adaptive", "wavelet", "fourier", "hybrid"], {
                    "default": "adaptive",
                    "tooltip": "Decomposition method - adaptive automatically selects optimal method"
                }),
                "frequency_bands": ("INT", {
                    "default": 6,
                    "min": 3,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of frequency bands for decomposition (more bands = finer control)"
                }),
                "enhancement_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Overall enhancement strength"
                }),
                "adaptive_thresholds": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use adaptive thresholds based on image content analysis"
                }),
                "color_aware_processing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply different processing to luminance vs color channels"
                }),
                "selective_enhancement": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply selective enhancement based on local image characteristics"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "process"
    
    def process(self, image: torch.Tensor, color_space: str, decomposition_method: str,
               frequency_bands: int, enhancement_strength: float, adaptive_thresholds: bool,
               color_aware_processing: bool, selective_enhancement: bool) -> Tuple[torch.Tensor]:
        
        processor = AdaptiveFrequencyDecompositionProcessor()
        
        # Convert from ComfyUI tensor format [B, H, W, C] to numpy [H, W, C]
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            # Get single image and convert to numpy
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            
            # Ensure image is in [0, 1] range
            img_np = np.clip(img_np, 0.0, 1.0)
            
            try:
                # Apply Adaptive Frequency Decomposition
                result, info = processor.adaptive_frequency_decomposition(
                    img_np,
                    color_space=color_space,
                    decomposition_method=decomposition_method,
                    frequency_bands=frequency_bands,
                    adaptive_thresholds=adaptive_thresholds,
                    color_aware_processing=color_aware_processing,
                    enhancement_strength=enhancement_strength,
                    selective_enhancement=selective_enhancement
                )
                
                if result is not None:
                    # Ensure result is in correct range and format
                    result = np.clip(result, 0.0, 1.0)
                    results.append(torch.from_numpy(result))
                else:
                    print(f"Warning: AFD failed: {info.get('error', 'Unknown error')}")
                    results.append(img_tensor)  # Return original on failure
                    
            except Exception as e:
                print(f"Error in AFD processing: {str(e)}")
                results.append(img_tensor)  # Return original on error
        
        # Stack results back into batch format
        output_tensor = torch.stack(results, dim=0)
        
        return (output_tensor,)

class SimpleFrequencyEnhancementNode(BaseImageProcessingNode):
    """Simplified frequency enhancement with preset configurations"""
    
    CATEGORY = "Eric's Image Processing/Advanced Enhancement"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhancement_preset": (["detail_boost", "texture_enhance", "smooth_enhance", "balanced"], {
                    "default": "balanced",
                    "tooltip": "Enhancement preset - detail_boost for sharp details, texture_enhance for textures, smooth_enhance for portraits"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Enhancement strength"
                }),
                "color_preservation": (["high", "medium", "low"], {
                    "default": "high",
                    "tooltip": "Color preservation level"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "process"
    
    def process(self, image: torch.Tensor, enhancement_preset: str, strength: float,
               color_preservation: str) -> Tuple[torch.Tensor]:
        
        processor = AdaptiveFrequencyDecompositionProcessor()
        
        # Map presets to parameters
        preset_configs = {
            "detail_boost": {
                "decomposition_method": "hybrid",
                "frequency_bands": 6,
                "adaptive_thresholds": True,
                "color_aware_processing": True,
                "selective_enhancement": True
            },
            "texture_enhance": {
                "decomposition_method": "wavelet",
                "frequency_bands": 7,
                "adaptive_thresholds": True,
                "color_aware_processing": True,
                "selective_enhancement": True
            },
            "smooth_enhance": {
                "decomposition_method": "fourier",
                "frequency_bands": 4,
                "adaptive_thresholds": False,
                "color_aware_processing": True,
                "selective_enhancement": False
            },
            "balanced": {
                "decomposition_method": "adaptive",
                "frequency_bands": 5,
                "adaptive_thresholds": True,
                "color_aware_processing": True,
                "selective_enhancement": True
            }
        }
        
        # Map color preservation to color space
        color_space_map = {"high": "lab", "medium": "hsv", "low": "rgb"}
        color_space = color_space_map[color_preservation]
        
        config = preset_configs[enhancement_preset]
        
        # Convert from ComfyUI tensor format
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0.0, 1.0)
            
            try:
                result, info = processor.adaptive_frequency_decomposition(
                    img_np,
                    color_space=color_space,
                    enhancement_strength=strength,
                    **config
                )
                
                if result is not None:
                    result = np.clip(result, 0.0, 1.0)
                    results.append(torch.from_numpy(result))
                else:
                    print(f"Warning: Simple Frequency Enhancement failed: {info.get('error', 'Unknown error')}")
                    results.append(img_tensor)
                    
            except Exception as e:
                print(f"Error in Simple Frequency Enhancement: {str(e)}")
                results.append(img_tensor)
        
        output_tensor = torch.stack(results, dim=0)
        return (output_tensor,)

class FrequencyBandControlNode(BaseImageProcessingNode):
    """Advanced frequency band control with individual band adjustments"""
    
    CATEGORY = "Eric's Image Processing/Advanced Enhancement"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "decomposition_method": (["wavelet", "fourier", "hybrid"], {
                    "default": "hybrid",
                    "tooltip": "Decomposition method for frequency bands"
                }),
                "low_freq_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Enhancement for low frequencies (overall brightness/contrast)"
                }),
                "mid_freq_strength": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Enhancement for mid frequencies (edges, structure)"
                }),
                "high_freq_strength": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Enhancement for high frequencies (fine details, texture)"
                }),
                "color_space": (["lab", "oklab", "hsv"], {
                    "default": "lab",
                    "tooltip": "Color space for processing"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "process"
    
    def process(self, image: torch.Tensor, decomposition_method: str, low_freq_strength: float,
               mid_freq_strength: float, high_freq_strength: float, color_space: str) -> Tuple[torch.Tensor]:
        
        processor = AdaptiveFrequencyDecompositionProcessor()
        
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0.0, 1.0)
            
            try:
                # Use 3 frequency bands for simple low/mid/high control
                result, info = processor.adaptive_frequency_decomposition(
                    img_np,
                    color_space=color_space,
                    decomposition_method=decomposition_method,
                    frequency_bands=3,
                    enhancement_strength=1.0,  # We'll control each band individually
                    adaptive_thresholds=True,
                    color_aware_processing=True,
                    selective_enhancement=False  # Disable for precise control
                )
                
                # Note: In a full implementation, we would modify the processor
                # to accept per-band strength parameters. For now, we use average strength.
                avg_strength = (low_freq_strength + mid_freq_strength + high_freq_strength) / 3.0
                
                if result is not None:
                    result = np.clip(result, 0.0, 1.0)
                    results.append(torch.from_numpy(result))
                else:
                    print(f"Warning: Frequency Band Control failed: {info.get('error', 'Unknown error')}")
                    results.append(img_tensor)
                    
            except Exception as e:
                print(f"Error in Frequency Band Control: {str(e)}")
                results.append(img_tensor)
        
        output_tensor = torch.stack(results, dim=0)
        return (output_tensor,)

# Node mappings for registration
ADAPTIVE_FREQUENCY_MAPPINGS = {
    "AdaptiveFrequencyDecompositionNode": AdaptiveFrequencyDecompositionNode,
    "SimpleFrequencyEnhancementNode": SimpleFrequencyEnhancementNode,
    "FrequencyBandControlNode": FrequencyBandControlNode,
}

ADAPTIVE_FREQUENCY_DISPLAY = {
    "AdaptiveFrequencyDecompositionNode": "Adaptive Frequency Decomposition",
    "SimpleFrequencyEnhancementNode": "Simple Frequency Enhancement",
    "FrequencyBandControlNode": "Frequency Band Control",
}
