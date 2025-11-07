"""
ComfyUI node for Richardson-Lucy deconvolution
Advanced restoration for lens blur, motion blur, and PSF correction
"""

import torch
import numpy as np
from ..base_node import BaseImageProcessingNode
from Eric_Image_Processing_Nodes import (
    richardson_lucy_deconvolution, 
    get_blur_presets,
    estimate_motion_blur,
    create_motion_psf,
    create_gaussian_psf
)

class RichardsonLucyNode(BaseImageProcessingNode):
    """
    Richardson-Lucy deconvolution for advanced image restoration
    
    Excellent for:
    - Lens blur correction (defocus, camera shake)
    - Motion blur removal (linear motion)
    - Astronomical image restoration
    - Microscopy deconvolution
    
    Performance: Medium speed, CPU-based, iterative algorithm
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        blur_types = ["gaussian", "motion", "custom"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_type": (blur_types, {
                    "default": "gaussian",
                    "tooltip": "Blur type to correct:\n• gaussian: Lens defocus, camera shake\n• motion: Linear motion blur\n• custom: User-provided PSF"
                }),
                "iterations": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "RL iterations:\n• 5-10: Fast, light restoration\n• 10-20: Balanced quality/speed (recommended)\n• 20-50: High quality, may amplify noise\n• 50+: Experimental, likely over-restoration"
                }),
            },
            "optional": {
                # Gaussian blur parameters
                "blur_size": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 25.0,
                    "step": 0.05,
                    "tooltip": "Gaussian blur size (std dev):\n• 0.1-0.5: Ultra-fine corrections for high-res images\n• 0.5-1.5: Light defocus, subtle sharpening\n• 1.5-4.0: Moderate blur (recommended)\n• 4.0-10.0: Heavy blur\n• 10.0+: Extreme blur (slow processing)"
                }),
                
                # Motion blur parameters  
                "motion_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Motion direction (degrees):\n• 0°: Horizontal motion\n• 45°: Diagonal motion\n• 90°: Vertical motion\n• Auto-estimation available in presets"
                }),
                "motion_length": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.5,
                    "max": 150.0,
                    "step": 0.5,
                    "tooltip": "Motion distance (pixels):\n• 0.5-2.0: Micro-motion correction for high-res images\n• 2.0-10.0: Light motion blur\n• 10.0-30.0: Moderate motion (recommended)\n• 30.0-80.0: Heavy motion blur\n• 80.0+: Extreme motion (slow processing)"
                }),
                
                # Advanced parameters
                "regularization": ("FLOAT", {
                    "default": 0.005,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.0005,
                    "tooltip": "TV regularization strength:\n• 0.0: No regularization (may amplify noise)\n• 0.0005-0.005: Ultra-light smoothing for high-quality images\n• 0.005-0.02: Light smoothing (recommended)\n• 0.02-0.1: Moderate smoothing\n• 0.1+: Heavy smoothing, reduces detail"
                }),
                
                "clip_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clip output to valid range [0,255] (recommended)"
                }),
                
                "use_preset": (["none"] + list(get_blur_presets().keys()), {
                    "default": "none",
                    "tooltip": "Preset configurations:\n• none: Use manual parameters\n• light_camera_shake: Slight defocus\n• moderate_defocus: Standard lens blur\n• horizontal_motion: Camera shake\n• etc."
                }),
                
                "estimate_motion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-estimate motion blur parameters (experimental)"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore"
    CATEGORY = "Eric's Image Processing/Restoration"
    
    def restore(
        self,
        image,
        blur_type="gaussian",
        iterations=15,
        blur_size=3.0,
        motion_angle=0.0,
        motion_length=15.0,
        regularization=0.005,
        clip_output=True,
        use_preset="none",
        estimate_motion=False
    ):
        """Apply Richardson-Lucy deconvolution to input image"""
        
        try:
            # Use preset if specified
            if use_preset != "none":
                presets = get_blur_presets()
                if use_preset in presets:
                    preset = presets[use_preset]
                    print(f"Using preset: {use_preset}")
                    
                    # Override parameters with preset values
                    blur_type = preset.get('blur_type', blur_type)
                    blur_size = preset.get('blur_size', blur_size)
                    motion_angle = preset.get('motion_angle', motion_angle)
                    motion_length = preset.get('motion_length', motion_length)
                    iterations = preset.get('iterations', iterations)
                    regularization = preset.get('regularization', regularization)
            
            # Auto-estimate motion parameters if requested
            if estimate_motion and blur_type == 'motion':
                img_np = self.tensor_to_numpy(image)
                estimated_angle, estimated_length = estimate_motion_blur(img_np)
                print(f"Estimated motion: angle={estimated_angle:.1f}°, length={estimated_length:.1f}px")
                motion_angle = estimated_angle
                motion_length = estimated_length
            
            # Define processing function
            def process_func(img_np, **kwargs):
                return richardson_lucy_deconvolution(
                    img_np,
                    psf=None,  # Generate PSF automatically
                    iterations=iterations,
                    clip=clip_output,
                    blur_type=blur_type,
                    blur_size=blur_size,
                    motion_angle=motion_angle,
                    motion_length=motion_length,
                    regularization=regularization
                )
            
            # Process image safely
            print(f"Richardson-Lucy: {blur_type} blur, {iterations} iterations, regularization={regularization}")
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in Richardson-Lucy deconvolution: {str(e)}")
            # Return original image on error
            return (image,)


class RichardsonLucySimpleNode(BaseImageProcessingNode):
    """Simplified Richardson-Lucy node with preset configurations"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "correction_type": (["camera_shake", "lens_defocus", "motion_blur"], {
                    "default": "lens_defocus",
                    "tooltip": "Type of blur to correct:\n• camera_shake: Light camera movement\n• lens_defocus: Out-of-focus blur\n• motion_blur: Linear object/camera motion"
                }),
                "strength": (["light", "medium", "strong"], {
                    "default": "medium",
                    "tooltip": "Correction strength:\n• light: Gentle restoration\n• medium: Balanced restoration\n• strong: Aggressive restoration"
                })
            },
            "optional": {
                "motion_direction": (["horizontal", "vertical", "diagonal"], {
                    "default": "horizontal",
                    "tooltip": "Motion direction (for motion_blur type only)"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore_simple"
    CATEGORY = "Eric's Image Processing/Restoration"
    
    def restore_simple(self, image, correction_type="lens_defocus", strength="medium", motion_direction="horizontal"):
        """Simple Richardson-Lucy restoration with automatic parameters"""
        
        try:
            # Parameter mapping
            strength_params = {
                "light": {"iterations": 8, "blur_factor": 0.7, "reg": 0.002},
                "medium": {"iterations": 15, "blur_factor": 1.0, "reg": 0.005},
                "strong": {"iterations": 25, "blur_factor": 1.5, "reg": 0.01}
            }
            
            direction_angles = {
                "horizontal": 0.0,
                "vertical": 90.0,
                "diagonal": 45.0
            }
            
            params = strength_params[strength]
            
            # Configure based on correction type
            if correction_type == "camera_shake":
                blur_type = "gaussian"
                blur_size = 2.0 * params["blur_factor"]
                
            elif correction_type == "lens_defocus":
                blur_type = "gaussian"  
                blur_size = 3.5 * params["blur_factor"]
                
            elif correction_type == "motion_blur":
                blur_type = "motion"
                blur_size = 0  # Not used for motion
                motion_angle = direction_angles[motion_direction]
                motion_length = 12.0 * params["blur_factor"]
            
            def process_func(img_np, **kwargs):
                if correction_type == "motion_blur":
                    return richardson_lucy_deconvolution(
                        img_np,
                        iterations=params["iterations"],
                        blur_type="motion",
                        motion_angle=motion_angle,
                        motion_length=motion_length,
                        regularization=params["reg"]
                    )
                else:
                    return richardson_lucy_deconvolution(
                        img_np,
                        iterations=params["iterations"],
                        blur_type="gaussian",
                        blur_size=blur_size,
                        regularization=params["reg"]
                    )
            
            print(f"Simple RL: {correction_type} ({strength})")
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in simple Richardson-Lucy: {str(e)}")
            return (image,)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "RichardsonLucy": RichardsonLucyNode,
    "RichardsonLucySimple": RichardsonLucySimpleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RichardsonLucy": "Richardson-Lucy Deconvolution (Eric)",
    "RichardsonLucySimple": "Richardson-Lucy Simple (Eric)"
}