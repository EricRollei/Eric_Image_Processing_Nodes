"""
Advanced Image Enhancement Nodes for ComfyUI
Based on 2024-2025 research findings
"""

import torch
import numpy as np
from typing import Tuple

# Import our advanced processors
try:
    from Eric_Image_Processing_Nodes import BaseImageProcessingNode
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base_node import BaseImageProcessingNode

from ..scripts.advanced_traditional_processing import (
    LBCLAHEProcessor,
    MultiScaleRetinexProcessor,
    BM3DGTADProcessor,
    SmartSharpeningProcessor
)

class LBCLAHENode(BaseImageProcessingNode):
    """Learning-Based CLAHE Node with automatic parameter tuning"""
    
    def __init__(self):
        super().__init__()
        self.processor = LBCLAHEProcessor()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "auto_tune": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically tune parameters based on image analysis:\n• True: Use machine learning-based optimization\n• False: Use default parameters"
                }),
            },
            "optional": {
                "manual_clip_limit": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Manual clip limit (used when auto_tune is False):\n• 1.0: Conservative enhancement\n• 2.0: Balanced enhancement\n• 4.0: Strong enhancement"
                }),
                "manual_grid_size": (["fine_8x8", "balanced_16x16", "coarse_32x32", "auto_scale"], {
                    "default": "auto_scale",
                    "tooltip": "Manual grid size (used when auto_tune is False):\n• fine_8x8: Fine detail preservation (small tiles)\n• balanced_16x16: Balanced processing (medium tiles)\n• coarse_32x32: Global contrast enhancement (large tiles)\n• auto_scale: Automatically scale grid size based on image dimensions"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("enhanced_image", "processing_info")
    FUNCTION = "process_with_lb_clahe"
    CATEGORY = "Eric's Nodes/Advanced Enhancement"
    
    def process_with_lb_clahe(self, image: torch.Tensor, auto_tune: bool, 
                             manual_clip_limit: float = 2.0, manual_grid_size: str = "8x8") -> Tuple[torch.Tensor, str]:
        """Process image with Learning-Based CLAHE"""
        
        try:
            # Convert tensor to numpy and ensure C-contiguous for OpenCV
            np_image = self.tensor_to_numpy(image)
            np_image = np.ascontiguousarray(np_image)
            
            if auto_tune:
                # Use automatic parameter tuning
                result = self.processor.process_image(np_image, auto_tune=True)
                
                # Extract features for info
                features = self.processor.extract_features(np_image)
                params = self.processor.auto_tune_parameters(features)
                
                info_lines = [
                    "Learning-Based CLAHE (Auto-tuned)",
                    f"Auto Clip Limit: {params['clip_limit']:.2f}",
                    f"Auto Grid Size: {params['grid_size']}",
                    f"Image Contrast: {features['local_contrast']:.3f}",
                    f"Edge Density: {features['edge_density']:.3f}",
                    f"Brightness Ratio: {features['brightness_ratio']:.3f}"
                ]
            else:
                # Use manual parameters
                # Override processor parameters temporarily
                original_method = self.processor.process_image
                
                def manual_process(img, auto_tune_param):
                    # Handle scale-aware grid sizing
                    if manual_grid_size == "auto_scale":
                        # Use automatic scale calculation
                        height, width = img.shape[:2]
                        target_tile_size = 80
                        grid_h = max(4, min(32, height // target_tile_size))
                        grid_w = max(4, min(32, width // target_tile_size))
                        grid_size = (grid_h, grid_w)
                    else:
                        # Parse manual grid size with scale awareness
                        if manual_grid_size == "fine_8x8":
                            base_grid = 8
                        elif manual_grid_size == "balanced_16x16":
                            base_grid = 16
                        elif manual_grid_size == "coarse_32x32":
                            base_grid = 32
                        
                        # Scale the grid size based on image dimensions
                        height, width = img.shape[:2]
                        # For images larger than 1000px, increase grid size proportionally
                        scale_factor = max(1.0, min(height, width) / 1000.0)
                        scaled_grid = int(base_grid * scale_factor)
                        grid_size = (scaled_grid, scaled_grid)
                    
                    if len(img.shape) == 3:
                        import cv2
                        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                        l_channel = lab[:, :, 0]
                        clahe = cv2.createCLAHE(clipLimit=manual_clip_limit, tileGridSize=grid_size)
                        l_channel = clahe.apply(l_channel)
                        lab[:, :, 0] = l_channel
                        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    else:
                        import cv2
                        clahe = cv2.createCLAHE(clipLimit=manual_clip_limit, tileGridSize=grid_size)
                        return clahe.apply(img)
                
                result = manual_process(np_image, False)
                
                info_lines = [
                    "Learning-Based CLAHE (Manual)",
                    f"Manual Clip Limit: {manual_clip_limit}",
                    f"Manual Grid Size: {manual_grid_size}",
                    "Auto-tuning: Disabled"
                ]
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(result)
            
            # Generate info
            info_lines.extend([
                f"Input Shape: {np_image.shape}",
                f"Output Shape: {result.shape}"
            ])
            
            info = "\\n".join(info_lines)
            
            return (result_tensor, info)
            
        except Exception as e:
            error_msg = f"Error in LB-CLAHE processing: {str(e)}"
            print(error_msg)
            return (image, error_msg)

class MultiScaleRetinexNode(BaseImageProcessingNode):
    """Multi-scale Retinex Node for natural color enhancement"""
    
    def __init__(self):
        super().__init__()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["balanced", "fine_detail", "global_illumination", "custom"], {
                    "default": "balanced",
                    "tooltip": "Retinex processing preset:\n• balanced: Good for most images\n• fine_detail: Emphasizes local details\n• global_illumination: Corrects overall lighting\n• custom: Use manual scale settings"
                }),
            },
            "optional": {
                "color_restoration": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply color restoration (MSRCR):\n• True: Preserve natural colors\n• False: Standard MSR without color correction"
                }),
                "gain": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Output gain factor:\n• 0.5: Subtle enhancement\n• 1.0: Standard enhancement\n• 2.0: Strong enhancement"
                }),
                "offset": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Output offset:\n• -20: Darken result\n• 0: No offset\n• +20: Brighten result"
                }),
                "small_scale": ("FLOAT", {
                    "default": 15.0, "min": 5.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Small scale (fine details) - only used with 'custom' preset"
                }),
                "medium_scale": ("FLOAT", {
                    "default": 80.0, "min": 20.0, "max": 200.0, "step": 5.0,
                    "tooltip": "Medium scale (local contrast) - only used with 'custom' preset"
                }),
                "large_scale": ("FLOAT", {
                    "default": 250.0, "min": 100.0, "max": 500.0, "step": 10.0,
                    "tooltip": "Large scale (global illumination) - only used with 'custom' preset"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("enhanced_image", "processing_info")
    FUNCTION = "process_with_msr"
    CATEGORY = "Eric's Nodes/Advanced Enhancement"
    
    def process_with_msr(self, image: torch.Tensor, preset: str, color_restoration: bool = True,
                        gain: float = 1.0, offset: float = 0.0, small_scale: float = 15.0,
                        medium_scale: float = 80.0, large_scale: float = 250.0) -> Tuple[torch.Tensor, str]:
        """Process image with Multi-scale Retinex"""
        
        try:
            # Convert tensor to numpy and ensure C-contiguous for OpenCV
            np_image = self.tensor_to_numpy(image)
            np_image = np.ascontiguousarray(np_image)
            
            # Set scales based on preset
            if preset == "balanced":
                scales = [15, 80, 250]
                weights = [1/3, 1/3, 1/3]
            elif preset == "fine_detail":
                scales = [5, 25, 80]
                weights = [0.5, 0.3, 0.2]
            elif preset == "global_illumination":
                scales = [50, 150, 300]
                weights = [0.2, 0.3, 0.5]
            else:  # custom
                scales = [small_scale, medium_scale, large_scale]
                weights = [1/3, 1/3, 1/3]
            
            # Create processor with specified scales
            processor = MultiScaleRetinexProcessor(scales=scales, weights=weights)
            
            # Process image
            result = processor.process_image(np_image, color_restoration=color_restoration, 
                                           gain=gain, offset=offset)
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(result)
            
            # Generate processing info
            info_lines = [
                "Multi-scale Retinex Processing",
                f"Preset: {preset}",
                f"Base Scales: {[f'{s:.1f}' for s in processor.base_scales]}",
                f"Adapted Scales: {[f'{s:.1f}' for s in processor.scales]}",
                f"Weights: {[f'{w:.2f}' for w in weights]}",
                f"Color Restoration: {'Enabled' if color_restoration else 'Disabled'}",
                f"Gain: {gain:.1f}",
                f"Offset: {offset:.1f}",
                f"Input Shape: {np_image.shape}",
                f"Output Shape: {result.shape}"
            ]
            
            info = "\\n".join(info_lines)
            
            return (result_tensor, info)
            
        except Exception as e:
            error_msg = f"Error in MSR processing: {str(e)}"
            print(error_msg)
            return (image, error_msg)

class BM3DFilmGrainNode(BaseImageProcessingNode):
    """
    BM3D-GT&AD Node for film grain denoising with resolution-aware parameter scaling
    
    This node is optimized for high-resolution images (1-20MP) with automatic parameter
    adjustment based on image resolution:
    
    Resolution Scaling:
    - <= 2MP: Full strength (scale factor 1.0)
    - 2-4MP: Reduced strength (scale factor 0.9)
    - 4-8MP: Further reduced (scale factor 0.75)
    - > 8MP: Minimal processing (scale factor 0.6)
    
    Grain Strength Presets (base sigma values):
    - ultra_light: 5.0 (ideal for high-res images)
    - light: 8.0 (subtle grain removal)
    - medium: 15.0 (balanced denoising)
    - heavy: 25.0 (strong grain removal)
    - custom: User-defined sigma (1.0-80.0)
    
    Patch Size Adaptation:
    - Automatically scales patch size based on image dimensions
    - Larger images get larger patches for better performance
    - Conservative scaling prevents overly aggressive processing
    """
    
    def __init__(self):
        super().__init__()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grain_strength": (["ultra_light", "light", "medium", "heavy", "custom"], {
                    "default": "medium",
                    "tooltip": "Film grain strength preset:\n• ultra_light: Minimal grain removal (high-res images)\n• light: Subtle grain removal\n• medium: Balanced denoising\n• heavy: Strong grain removal\n• custom: Use manual sigma value"
                }),
            },
            "optional": {
                "custom_sigma": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 80.0, "step": 0.5,
                    "tooltip": "Custom noise standard deviation (only used with 'custom' preset):\n• 3-5: Very light denoising (high resolution)\n• 8-12: Light denoising\n• 15-20: Medium denoising\n• 25-35: Heavy denoising"
                }),
                "preserve_texture": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve film texture characteristics:\n• True: Maintain natural film look\n• False: Maximum noise removal"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("denoised_image", "processing_info")
    FUNCTION = "process_with_bm3d"
    CATEGORY = "Eric's Nodes/Advanced Enhancement"
    
    def process_with_bm3d(self, image: torch.Tensor, grain_strength: str, 
                         custom_sigma: float = 15.0, preserve_texture: bool = True) -> Tuple[torch.Tensor, str]:
        """Process image with BM3D-GT&AD for film grain"""
        
        try:
            # Convert tensor to numpy and ensure C-contiguous for OpenCV
            np_image = self.tensor_to_numpy(image)
            np_image = np.ascontiguousarray(np_image)
            
            # Calculate image scale factor for high-resolution adjustment
            height, width = np_image.shape[:2]
            total_pixels = height * width
            megapixels = total_pixels / (1024 * 1024)
            
            # Base sigma values adjusted for high-resolution images
            if grain_strength == "ultra_light":
                base_sigma = 5.0  # Very gentle for high-res images
            elif grain_strength == "light":
                base_sigma = 8.0  # Reduced from 15.0
            elif grain_strength == "medium":
                base_sigma = 15.0  # Reduced from 25.0
            elif grain_strength == "heavy":
                base_sigma = 25.0  # Reduced from 40.0
            else:  # custom
                base_sigma = custom_sigma
            
            # Scale sigma based on image resolution
            # For high-resolution images, reduce sigma to avoid over-processing
            if megapixels > 8.0:  # > 8MP
                sigma_scale = 0.6
            elif megapixels > 4.0:  # 4-8MP
                sigma_scale = 0.75
            elif megapixels > 2.0:  # 2-4MP
                sigma_scale = 0.9
            else:  # <= 2MP
                sigma_scale = 1.0
                
            sigma = base_sigma * sigma_scale
            
            # Adjust sigma if preserving texture
            if preserve_texture:
                sigma *= 0.8  # Slightly reduced from 0.7 for better texture preservation
            
            # Create processor with resolution-aware parameters
            processor = BM3DGTADProcessor(
                sigma=sigma,
                search_window=39,
                max_blocks=32,
                threshold=2500.0
            )
            
            # Process image
            result = processor.process_image(np_image)
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(result)
            
            # Generate processing info
            info_lines = [
                "BM3D-GT&AD Film Grain Denoising",
                f"Grain Strength: {grain_strength}",
                f"Image Resolution: {width}x{height} ({megapixels:.1f}MP)",
                f"Base Sigma: {base_sigma:.1f}",
                f"Resolution Scale Factor: {sigma_scale:.2f}",
                f"Effective Sigma: {sigma:.1f}",
                f"Adapted Patch Size: {processor.patch_size}x{processor.patch_size}",
                f"Adapted Patch Distance: {processor.patch_distance}",
                f"Texture Preservation: {'Enabled' if preserve_texture else 'Disabled'}",
                f"Input Shape: {np_image.shape}",
                f"Output Shape: {result.shape}"
            ]
            
            info = "\\n".join(info_lines)
            
            return (result_tensor, info)
            
        except Exception as e:
            error_msg = f"Error in BM3D processing: {str(e)}"
            print(error_msg)
            return (image, error_msg)

class SmartSharpeningNode(BaseImageProcessingNode):
    """Smart Sharpening Node with artifact detection"""
    
    def __init__(self):
        super().__init__()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpening_mode": (["auto_adaptive", "conservative", "aggressive", "custom"], {
                    "default": "auto_adaptive",
                    "tooltip": "Sharpening mode:\n• auto_adaptive: Automatic parameter selection\n• conservative: Gentle sharpening\n• aggressive: Strong sharpening\n• custom: Manual parameters"
                }),
            },
            "optional": {
                "custom_radius": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Custom radius (only used with 'custom' mode):\n• 0.5: Fine detail sharpening\n• 1.0: Balanced sharpening\n• 2.0: Broad sharpening"
                }),
                "custom_amount": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Custom amount (only used with 'custom' mode):\n• 0.3: Subtle sharpening\n• 0.5: Moderate sharpening\n• 1.0: Strong sharpening"
                }),
                "overshoot_protection": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable overshoot artifact protection:\n• True: Prevent halo artifacts\n• False: Maximum sharpening effect"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sharpened_image", "processing_info")
    FUNCTION = "process_with_smart_sharpening"
    CATEGORY = "Eric's Nodes/Advanced Enhancement"
    
    def process_with_smart_sharpening(self, image: torch.Tensor, sharpening_mode: str,
                                    custom_radius: float = 1.0, custom_amount: float = 0.5,
                                    overshoot_protection: bool = True) -> Tuple[torch.Tensor, str]:
        """Process image with Smart Sharpening"""
        
        try:
            # Convert tensor to numpy and ensure C-contiguous for OpenCV
            np_image = self.tensor_to_numpy(image)
            np_image = np.ascontiguousarray(np_image)

            # Normalize to batch-first array for consistent processing
            if np_image.ndim == 3:  # Single image [H, W, C]
                batch_imgs = np_image[np.newaxis, ...]
                single_input = True
            elif np_image.ndim == 4:  # Batch [N, H, W, C]
                batch_imgs = np_image
                single_input = False
            else:
                raise ValueError(f"Unexpected image shape for smart sharpening: {np_image.shape}")

            processor = SmartSharpeningProcessor()
            processed_frames = []
            info_blocks = []

            for frame_idx, frame in enumerate(batch_imgs):
                frame = np.ascontiguousarray(frame)

                # Set parameters based on mode
                if sharpening_mode == "auto_adaptive":
                    result = processor.process_image(frame, auto_params=True)
                    radius, amount = processor.adaptive_radius_control(frame)
                    info_params = f"Auto Radius: {radius:.2f}, Auto Amount: {amount:.2f}"
                elif sharpening_mode == "conservative":
                    result = processor.process_image(frame, auto_params=False, radius=0.8, amount=0.3)
                    info_params = "Conservative: Radius: 0.8, Amount: 0.3"
                elif sharpening_mode == "aggressive":
                    result = processor.process_image(frame, auto_params=False, radius=1.5, amount=0.8)
                    info_params = "Aggressive: Radius: 1.5, Amount: 0.8"
                else:  # custom
                    result = processor.process_image(frame, auto_params=False, radius=custom_radius, amount=custom_amount)
                    info_params = f"Custom: Radius: {custom_radius}, Amount: {custom_amount}"

                # Apply overshoot protection if enabled (skip for auto-adaptive which has internal control)
                if overshoot_protection and sharpening_mode != "auto_adaptive":
                    overshoot_mask = processor.detect_overshoot(frame, result, threshold=0.1)
                    if overshoot_mask.ndim == 2:
                        overshoot_mask = overshoot_mask[:, :, np.newaxis]

                    blended = np.where(
                        overshoot_mask,
                        frame.astype(np.float32) * 0.7 + result.astype(np.float32) * 0.3,
                        result.astype(np.float32)
                    )
                    result = np.clip(blended, 0, 255).astype(np.uint8)

                result = np.ascontiguousarray(result)
                processed_frames.append(result)

                info_lines = [
                    "Smart Sharpening with Artifact Detection",
                    f"Mode: {sharpening_mode}",
                    info_params,
                    f"Overshoot Protection: {'Enabled' if overshoot_protection else 'Disabled'}",
                    f"Input Shape: {frame.shape}",
                    f"Output Shape: {result.shape}"
                ]

                if batch_imgs.shape[0] > 1:
                    info_lines.insert(0, f"Frame {frame_idx + 1}/{batch_imgs.shape[0]}")

                info_blocks.append("\n".join(info_lines))

            result_stack = np.stack(processed_frames, axis=0)
            result_tensor = self.numpy_to_tensor(result_stack)

            # If original input was single image, drop batch dim for return consistency
            if single_input and len(result_tensor.shape) == 4 and result_tensor.shape[0] == 1:
                result_tensor = result_tensor[0]

            info = "\n\n".join(info_blocks)

            return (result_tensor, info)
            
        except Exception as e:
            error_msg = f"Error in Smart Sharpening: {str(e)}"
            print(error_msg)
            return (image, error_msg)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LB-CLAHE": LBCLAHENode,
    "Multi-Scale Retinex": MultiScaleRetinexNode,
    "BM3D Film Grain Denoising": BM3DFilmGrainNode,
    "Smart Sharpening": SmartSharpeningNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LB-CLAHE": "LB-CLAHE (Learning-Based)",
    "Multi-Scale Retinex": "Multi-Scale Retinex",
    "BM3D Film Grain Denoising": "BM3D Film Grain Denoising",
    "Smart Sharpening": "Smart Sharpening",
}
