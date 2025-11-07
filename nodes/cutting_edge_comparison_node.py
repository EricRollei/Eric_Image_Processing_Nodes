"""
Cutting-Edge Enhancement Comparison Node
Showcases and compares the three 2024-2025 research techniques:
1. Advanced Sharpening (Smart, HiRaLoAm, Edge-Directional, Multi-Scale, Guided Filter)
2. Learning-Based CLAHE (ML-guided parameter optimization)
3. Adaptive Frequency Decomposition (Color-aware multi-scale processing)
"""

import torch
import numpy as np
from typing import Any, Dict, Tuple, List

from ..base_node import BaseImageProcessingNode
from Eric_Image_Processing_Nodes import (
    AdvancedSharpeningProcessor,
    LearningBasedCLAHEProcessor,
    AdaptiveFrequencyDecompositionProcessor
)

class CuttingEdgeEnhancementComparisonNode(BaseImageProcessingNode):
    """Compare all three cutting-edge enhancement techniques"""
    
    CATEGORY = "Eric's Image Processing/Advanced Enhancement"
    
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
                    "tooltip": "Overall enhancement strength for all methods"
                }),
                "color_space": (["lab", "oklab", "hsv"], {
                    "default": "lab",
                    "tooltip": "Color space for processing (LAB recommended for best results)"
                }),
                "sharpening_method": (["smart", "hiraloam", "edge_directional", "multiscale", "guided"], {
                    "default": "smart",
                    "tooltip": "Advanced sharpening method to use"
                }),
                "enable_sharpening": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Advanced Sharpening processing"
                }),
                "enable_clahe": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Learning-Based CLAHE processing"
                }),
                "enable_afd": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Adaptive Frequency Decomposition processing"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("original", "advanced_sharpening", "learning_clahe", "adaptive_frequency")
    FUNCTION = "compare_enhancements"
    
    def compare_enhancements(self, image: torch.Tensor, enhancement_strength: float,
                           color_space: str, sharpening_method: str, enable_sharpening: bool,
                           enable_clahe: bool, enable_afd: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = image.shape[0]
        
        # Results storage
        original_results = []
        sharpening_results = []
        clahe_results = []
        afd_results = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0.0, 1.0)
            
            # Original image
            original_results.append(img_tensor)
            
            # Advanced Sharpening
            if enable_sharpening:
                try:
                    sharpening_processor = AdvancedSharpeningProcessor()
                    
                    sharpening_result, sharpening_info = sharpening_processor.apply_sharpening(
                        img_np, 
                        method=sharpening_method,
                        strength=enhancement_strength,
                        preserve_edges=True,
                        adaptive_radius=True
                    )
                    
                    if sharpening_result is not None:
                        sharpening_result = np.clip(sharpening_result, 0.0, 1.0)
                        sharpening_results.append(torch.from_numpy(sharpening_result))
                    else:
                        print(f"Advanced Sharpening failed: {sharpening_info.get('error', 'Unknown error')}")
                        sharpening_results.append(img_tensor)
                        
                except Exception as e:
                    print(f"Error in Advanced Sharpening: {str(e)}")
                    sharpening_results.append(img_tensor)
            else:
                sharpening_results.append(img_tensor)
            
            # Learning-Based CLAHE
            if enable_clahe:
                try:
                    clahe_processor = LearningBasedCLAHEProcessor()
                    
                    clahe_result, clahe_info = clahe_processor.learning_based_clahe(
                        img_np,
                        color_space=color_space,
                        ml_method="hybrid",
                        region_size=(8, 8),
                        base_clip_limit=1.0 + enhancement_strength,
                        adaptive_regions=True,
                        perceptual_weighting=True
                    )
                    
                    if clahe_result is not None:
                        clahe_result = np.clip(clahe_result, 0.0, 1.0)
                        clahe_results.append(torch.from_numpy(clahe_result))
                    else:
                        print(f"Learning-Based CLAHE failed: {clahe_info.get('error', 'Unknown error')}")
                        clahe_results.append(img_tensor)
                        
                except Exception as e:
                    print(f"Error in Learning-Based CLAHE: {str(e)}")
                    clahe_results.append(img_tensor)
            else:
                clahe_results.append(img_tensor)
            
            # Adaptive Frequency Decomposition
            if enable_afd:
                try:
                    afd_processor = AdaptiveFrequencyDecompositionProcessor()
                    
                    afd_result, afd_info = afd_processor.adaptive_frequency_decomposition(
                        img_np,
                        color_space=color_space,
                        decomposition_method="adaptive",
                        frequency_bands=6,
                        enhancement_strength=enhancement_strength,
                        adaptive_thresholds=True,
                        color_aware_processing=True,
                        selective_enhancement=True
                    )
                    
                    if afd_result is not None:
                        afd_result = np.clip(afd_result, 0.0, 1.0)
                        afd_results.append(torch.from_numpy(afd_result))
                    else:
                        print(f"Adaptive Frequency Decomposition failed: {afd_info.get('error', 'Unknown error')}")
                        afd_results.append(img_tensor)
                        
                except Exception as e:
                    print(f"Error in Adaptive Frequency Decomposition: {str(e)}")
                    afd_results.append(img_tensor)
            else:
                afd_results.append(img_tensor)
        
        # Stack results
        original_tensor = torch.stack(original_results, dim=0)
        sharpening_tensor = torch.stack(sharpening_results, dim=0)
        clahe_tensor = torch.stack(clahe_results, dim=0)
        afd_tensor = torch.stack(afd_results, dim=0)
        
        return (original_tensor, sharpening_tensor, clahe_tensor, afd_tensor)

class CuttingEdgePipelineNode(BaseImageProcessingNode):
    """Sequential pipeline applying all three cutting-edge techniques"""
    
    CATEGORY = "Eric's Image Processing/Advanced Enhancement"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pipeline_preset": (["balanced", "detail_focused", "contrast_focused", "frequency_focused", "custom"], {
                    "default": "balanced",
                    "tooltip": "Preset configuration for the enhancement pipeline"
                }),
                "overall_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Overall enhancement strength"
                }),
                "color_space": (["lab", "oklab", "hsv"], {
                    "default": "lab",
                    "tooltip": "Color space for all processing steps"
                }),
            },
            "optional": {
                "custom_sharpening_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Custom sharpening strength (only used with custom preset)"
                }),
                "custom_clahe_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Custom CLAHE strength (only used with custom preset)"
                }),
                "custom_afd_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Custom AFD strength (only used with custom preset)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "process_pipeline"
    
    def process_pipeline(self, image: torch.Tensor, pipeline_preset: str, overall_strength: float,
                        color_space: str, custom_sharpening_strength: float = 1.0,
                        custom_clahe_strength: float = 1.0, custom_afd_strength: float = 1.0) -> Tuple[torch.Tensor]:
        
        # Define preset configurations
        preset_configs = {
            "balanced": {
                "sharpening_strength": 1.0,
                "clahe_strength": 1.0,
                "afd_strength": 1.0,
                "order": ["clahe", "afd", "sharpening"]  # CLAHE first, then AFD, then sharpening
            },
            "detail_focused": {
                "sharpening_strength": 1.5,
                "clahe_strength": 0.8,
                "afd_strength": 1.2,
                "order": ["afd", "sharpening", "clahe"]  # Frequency first for detail
            },
            "contrast_focused": {
                "sharpening_strength": 0.8,
                "clahe_strength": 1.5,
                "afd_strength": 0.9,
                "order": ["clahe", "sharpening", "afd"]  # CLAHE first for contrast
            },
            "frequency_focused": {
                "sharpening_strength": 0.9,
                "clahe_strength": 0.7,
                "afd_strength": 1.6,
                "order": ["afd", "clahe", "sharpening"]  # AFD first for frequency enhancement
            },
            "custom": {
                "sharpening_strength": custom_sharpening_strength,
                "clahe_strength": custom_clahe_strength,
                "afd_strength": custom_afd_strength,
                "order": ["clahe", "afd", "sharpening"]  # Default order
            }
        }
        
        config = preset_configs.get(pipeline_preset, preset_configs["balanced"])
        
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            current_image = img_tensor.cpu().numpy()
            current_image = np.clip(current_image, 0.0, 1.0)
            
            # Apply enhancements in specified order
            for step in config["order"]:
                try:
                    if step == "sharpening":
                        sharpening_processor = AdvancedSharpeningProcessor()
                        
                        result, info = sharpening_processor.apply_sharpening(
                            current_image,
                            method="smart",
                            strength=config["sharpening_strength"] * overall_strength,
                            preserve_edges=True,
                            adaptive_radius=True
                        )
                        if result is not None:
                            current_image = np.clip(result, 0.0, 1.0)
                    
                    elif step == "clahe":
                        clahe_processor = LearningBasedCLAHEProcessor()
                        
                        result, info = clahe_processor.learning_based_clahe(
                            current_image,
                            color_space=color_space,
                            ml_method="hybrid",
                            region_size=(8, 8),
                            base_clip_limit=1.0 + config["clahe_strength"] * overall_strength,
                            adaptive_regions=True,
                            perceptual_weighting=True
                        )
                        if result is not None:
                            current_image = np.clip(result, 0.0, 1.0)
                    
                    elif step == "afd":
                        afd_processor = AdaptiveFrequencyDecompositionProcessor()
                        
                        result, info = afd_processor.adaptive_frequency_decomposition(
                            current_image,
                            color_space=color_space,
                            decomposition_method="adaptive",
                            frequency_bands=6,
                            enhancement_strength=config["afd_strength"] * overall_strength,
                            adaptive_thresholds=True,
                            color_aware_processing=True,
                            selective_enhancement=True
                        )
                        if result is not None:
                            current_image = np.clip(result, 0.0, 1.0)
                
                except Exception as e:
                    print(f"Warning: {step} step failed: {str(e)}")
                    continue
            
            results.append(torch.from_numpy(current_image))
        
        output_tensor = torch.stack(results, dim=0)
        return (output_tensor,)

# Node mappings for registration
CUTTING_EDGE_COMPARISON_MAPPINGS = {
    "CuttingEdgeEnhancementComparisonNode": CuttingEdgeEnhancementComparisonNode,
    "CuttingEdgePipelineNode": CuttingEdgePipelineNode,
}

CUTTING_EDGE_COMPARISON_DISPLAY = {
    "CuttingEdgeEnhancementComparisonNode": "🏆 Cutting-Edge Enhancement Comparison",
    "CuttingEdgePipelineNode": "🔗 Cutting-Edge Enhancement Pipeline",
}
