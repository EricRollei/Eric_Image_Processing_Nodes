"""
ComfyUI nodes for Learning-Based CLAHE processing
"""

import torch
import numpy as np
from typing import Any, Dict, Tuple

from ..base_node import BaseImageProcessingNode
from Eric_Image_Processing_Nodes import LearningBasedCLAHEProcessor

class LearningBasedCLAHENode(BaseImageProcessingNode):
    """Learning-Based CLAHE with automatic parameter optimization"""
    
    CATEGORY = "Eric's Image Processing/Advanced Enhancement"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_space": (["lab", "oklab", "jzazbz", "hsv", "rgb"], {
                    "default": "lab",
                    "tooltip": "Color space for processing. LAB/Oklab/Jzazbz work on luminance only, preserving colors better"
                }),
                "ml_method": (["random_forest", "xgboost", "hybrid", "rule_based"], {
                    "default": "hybrid",
                    "tooltip": "Machine learning method for parameter optimization. Hybrid uses both RF and XGBoost"
                }),
                "base_clip_limit": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Base CLAHE clipping limit - ML will optimize from this starting point"
                }),
                "region_size": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 16,
                    "step": 2,
                    "tooltip": "Grid size for CLAHE regions (8x8 is standard)"
                }),
                "adaptive_regions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use adaptive region sizing based on local image complexity"
                }),
                "perceptual_weighting": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply perceptual importance weighting to focus enhancement on salient regions"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "process"
    
    def process(self, image: torch.Tensor, color_space: str, ml_method: str,
               base_clip_limit: float, region_size: int, adaptive_regions: bool,
               perceptual_weighting: bool) -> Tuple[torch.Tensor]:
        
        processor = LearningBasedCLAHEProcessor()
        
        # Convert from ComfyUI tensor format [B, H, W, C] to numpy [H, W, C]
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            # Get single image and convert to numpy
            img_tensor = image[i]
            # CRITICAL: Ensure tensor slice is contiguous before conversion
            img_tensor = img_tensor.contiguous()
            img_np = img_tensor.cpu().numpy()
            
            # Ensure image is in [0, 1] range
            img_np = np.clip(img_np, 0.0, 1.0)
            
            try:
                # Apply Learning-Based CLAHE
                result, info = processor.learning_based_clahe(
                    img_np,
                    color_space=color_space,
                    ml_method=ml_method,
                    region_size=(region_size, region_size),
                    base_clip_limit=base_clip_limit,
                    adaptive_regions=adaptive_regions,
                    perceptual_weighting=perceptual_weighting
                )
                
                if result is not None:
                    # Ensure result is in correct range and format
                    result = np.clip(result, 0.0, 1.0)
                    results.append(torch.from_numpy(result))
                else:
                    print(f"Warning: Learning-Based CLAHE failed: {info.get('error', 'Unknown error')}")
                    results.append(img_tensor)  # Return original on failure
                    
            except Exception as e:
                print(f"Error in Learning-Based CLAHE processing: {str(e)}")
                results.append(img_tensor)  # Return original on error
        
        # Stack results back into batch format
        output_tensor = torch.stack(results, dim=0)
        
        return (output_tensor,)

class SimpleLearningCLAHENode:
    """Simplified Learning-Based CLAHE with fewer parameters"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhancement_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Overall enhancement strength"
                }),
                "color_preservation": (["high", "medium", "low"], {
                    "default": "high",
                    "tooltip": "Color preservation level - high uses LAB space, medium uses HSV, low uses RGB"
                }),
                "detail_level": (["fine", "medium", "coarse"], {
                    "default": "medium",
                    "tooltip": "Detail level for region analysis - fine uses 4x4, medium 8x8, coarse 16x16"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "process"
    CATEGORY = "Eric's Image Processing/Advanced Enhancement"
    
    def process(self, image: torch.Tensor, enhancement_strength: float,
               color_preservation: str, detail_level: str) -> Tuple[torch.Tensor]:
        
        processor = LearningBasedCLAHEProcessor()
        
        # Map parameters
        color_space_map = {"high": "lab", "medium": "hsv", "low": "rgb"}
        region_size_map = {"fine": 4, "medium": 8, "coarse": 16}
        
        color_space = color_space_map[color_preservation]
        region_size = region_size_map[detail_level]
        
        # Convert base clip limit from enhancement strength
        base_clip_limit = 1.0 + enhancement_strength * 2.0
        
        # Convert from ComfyUI tensor format
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            # CRITICAL: Ensure tensor slice is contiguous before conversion
            img_tensor = img_tensor.contiguous()
            img_np = img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0.0, 1.0)
            
            try:
                result, info = processor.learning_based_clahe(
                    img_np,
                    color_space=color_space,
                    ml_method="hybrid",
                    region_size=(region_size, region_size),
                    base_clip_limit=base_clip_limit,
                    adaptive_regions=True,
                    perceptual_weighting=True
                )
                
                if result is not None:
                    result = np.clip(result, 0.0, 1.0)
                    results.append(torch.from_numpy(result))
                else:
                    print(f"Warning: Simple Learning CLAHE failed: {info.get('error', 'Unknown error')}")
                    results.append(img_tensor)
                    
            except Exception as e:
                print(f"Error in Simple Learning CLAHE: {str(e)}")
                results.append(img_tensor)
        
        output_tensor = torch.stack(results, dim=0)
        return (output_tensor,)

class AdvancedColorSpaceCLAHENode:
    """Advanced CLAHE with focus on modern perceptual color spaces"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_space": (["lab", "oklab", "jzazbz"], {
                    "default": "oklab",
                    "tooltip": "Perceptual color space - Oklab and Jzazbz are more perceptually uniform than LAB"
                }),
                "clip_limit": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "CLAHE clipping limit for contrast control"
                }),
                "luminance_focus": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 1.5,
                    "step": 0.1,
                    "tooltip": "Focus enhancement on luminance channel (higher = more luminance-focused)"
                }),
                "enable_ml_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable machine learning parameter optimization"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "process"
    CATEGORY = "Eric's Image Processing/Advanced Enhancement"
    
    def process(self, image: torch.Tensor, color_space: str, clip_limit: float,
               luminance_focus: float, enable_ml_optimization: bool) -> Tuple[torch.Tensor]:
        
        processor = LearningBasedCLAHEProcessor()
        
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            # CRITICAL: Ensure tensor slice is contiguous before conversion
            img_tensor = img_tensor.contiguous()
            img_np = img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0.0, 1.0)
            
            try:
                # Adjust parameters based on luminance focus
                adjusted_clip_limit = clip_limit * luminance_focus
                ml_method = "hybrid" if enable_ml_optimization else "rule_based"
                
                result, info = processor.learning_based_clahe(
                    img_np,
                    color_space=color_space,
                    ml_method=ml_method,
                    region_size=(8, 8),
                    base_clip_limit=adjusted_clip_limit,
                    adaptive_regions=True,
                    perceptual_weighting=True
                )
                
                if result is not None:
                    result = np.clip(result, 0.0, 1.0)
                    results.append(torch.from_numpy(result))
                else:
                    print(f"Warning: Advanced Color Space CLAHE failed: {info.get('error', 'Unknown error')}")
                    results.append(img_tensor)
                    
            except Exception as e:
                print(f"Error in Advanced Color Space CLAHE: {str(e)}")
                results.append(img_tensor)
        
        output_tensor = torch.stack(results, dim=0)
        return (output_tensor,)

# Node mappings for registration
LEARNING_CLAHE_MAPPINGS = {
    "LearningBasedCLAHENode": LearningBasedCLAHENode,
    "SimpleLearningCLAHENode": SimpleLearningCLAHENode,
    "AdvancedColorSpaceCLAHENode": AdvancedColorSpaceCLAHENode,
}

LEARNING_CLAHE_DISPLAY = {
    "LearningBasedCLAHENode": "Learning-Based CLAHE",
    "SimpleLearningCLAHENode": "Simple Learning CLAHE",
    "AdvancedColorSpaceCLAHENode": "Advanced Color Space CLAHE",
}
