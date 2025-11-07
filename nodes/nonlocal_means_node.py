"""
Non-Local Means Denoising Node for ComfyUI
Excellent for photographs and natural images with organic textures

Author: Eric Hiss (GitHub: EricRollei)
License: See LICENSE file in repository root

Non-Local Means Algorithm:
    Paper: "A Non-Local Algorithm for Image Denoising"
    Authors: Buades et al., CVPR 2005
    
    Citation:
    @inproceedings{buades2005non,
      title={A Non-Local Algorithm for Image Denoising},
      author={Buades, Antoni and Coll, Bartomeu and Morel, Jean-Michel},
      booktitle={IEEE Computer Society Conference on Computer Vision and Pattern Recognition},
      year={2005}
    }

Dependencies:
    - PyTorch (BSD 3-Clause License)
    - OpenCV (Apache 2.0 License)
    - NumPy (BSD 3-Clause License)
    - scikit-image (BSD 3-Clause License)
"""

import torch
from ..base_node import BaseImageProcessingNode
from Eric_Image_Processing_Nodes import (
    nonlocal_means_denoise, 
    adaptive_nonlocal_means, 
    get_recommended_parameters
)


class NonLocalMeansNode(BaseImageProcessingNode):
    """
    Non-Local Means denoising node - excellent for natural photographs
    
    Best for:
    - Digital photographs with organic noise
    - Images with textures and repetitive patterns  
    - Preserving fine details while removing noise
    - Scanned photographs with film grain
    
    Performance: Medium speed, CPU-based, good quality/speed balance
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "Parameter selection:\n• auto: Automatic parameter selection\n• manual: Full manual control"
                }),
            },
            "optional": {
                # Auto mode parameters
                "noise_level": (["auto", "low", "medium", "high"], {
                    "default": "auto",
                    "tooltip": "Noise level (auto mode):\n• auto: Detect automatically\n• low: Light noise (sigma < 10)\n• medium: Moderate noise (sigma 10-20)\n• high: Heavy noise (sigma > 20)"
                }),
                "quality": (["fast", "balanced", "high"], {
                    "default": "balanced",
                    "tooltip": "Quality vs speed (auto mode):\n• fast: ~3x faster, good quality\n• balanced: Good quality/speed balance\n• high: Best quality, slower"
                }),
                "preserve_textures": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Optimize parameters for texture preservation (auto mode)"
                }),
                
                # Manual mode parameters
                "h": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Filtering strength (manual mode):\n• 0.1-1.0: Ultra-light denoising for high-quality images\n• 1.0-5.0: Subtle denoising, preserve fine details\n• 5.0-15.0: Balanced denoising (recommended)\n• 15-30: Strong denoising\n• 30-100: Very strong, may blur details"
                }),
                "patch_size": ("INT", {
                    "default": 7,
                    "min": 3,
                    "max": 15,
                    "step": 2,
                    "tooltip": "Patch size (manual mode, odd numbers):\n• 5: Fast, small textures\n• 7: Standard choice (recommended)\n• 9-11: Large textures, slower\n• 13-15: Very large patterns, very slow"
                }),
                "patch_distance": ("INT", {
                    "default": 11,
                    "min": 5,
                    "max": 50,
                    "step": 2,
                    "tooltip": "Search window (manual mode):\n• 5-9: Fast, minimal search\n• 11-15: Standard choice (recommended)\n• 17-25: Better quality, slower\n• 27-50: Maximum quality, much slower\n• Must be > patch_size"
                }),
                "method": (["opencv", "skimage"], {
                    "default": "opencv",
                    "tooltip": "Implementation:\n• opencv: Faster, good quality (recommended)\n• skimage: Slower, potentially better quality"
                }),
                "fast_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fast mode (OpenCV only):\n• True: ~3x faster with slight quality loss\n• False: Higher quality, slower"
                }),
                "multichannel": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process color channels independently (recommended for color images)"
                }),
                "image_type_hint": (["photograph", "scanned_photo", "digital_art"], {
                    "default": "photograph",
                    "tooltip": "Image type for parameter suggestions (manual mode)"
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
        mode="auto",
        noise_level="auto",
        quality="balanced", 
        preserve_textures=True,
        h=15.0,
        patch_size=7,
        patch_distance=11,
        method="opencv",
        fast_mode=True,
        multichannel=True,
        image_type_hint="photograph"
    ):
        """Apply Non-Local Means denoising to input image"""
        
        try:
            # Validate patch_distance > patch_size
            if patch_distance <= patch_size:
                print(f"Warning: patch_distance ({patch_distance}) must be > patch_size ({patch_size}). Adjusting.")
                patch_distance = patch_size + 4
            
            # Define processing function based on mode
            if mode == "auto":
                def process_func(img_np, **kwargs):
                    return adaptive_nonlocal_means(
                        img_np,
                        noise_level=noise_level,
                        quality=quality,
                        preserve_textures=preserve_textures
                    )
                    
                print(f"Auto mode: noise_level={noise_level}, quality={quality}, preserve_textures={preserve_textures}")
                
            else:  # manual mode
                # Show parameter recommendations
                if noise_level != "auto":
                    recommended = get_recommended_parameters(image_type_hint, noise_level)
                    print(f"Recommended parameters for {image_type_hint}/{noise_level}:")
                    print(f"  h={recommended['h']}, patch_size={recommended['patch_size']}, patch_distance={recommended['patch_distance']}")
                
                def process_func(img_np, **kwargs):
                    return nonlocal_means_denoise(
                        img_np,
                        h=h,
                        patch_size=patch_size,
                        patch_distance=patch_distance,
                        multichannel=multichannel,
                        method=method,
                        fast_mode=fast_mode
                    )
                
                print(f"Manual mode: h={h}, patch_size={patch_size}, patch_distance={patch_distance}")
            
            # Process image safely
            result = self.process_image_safe(image, process_func)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in Non-Local Means denoising: {str(e)}")
            # Return original image on error
            return (image,)


# Alternative simplified node for quick use
class NonLocalMeansSimpleNode(BaseImageProcessingNode):
    """Simplified Non-Local Means node with automatic parameter selection"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": (["light", "medium", "strong"], {
                    "default": "medium",
                    "tooltip": "Denoising strength:\n• light: Preserve maximum detail\n• medium: Balanced denoising\n• strong: Heavy denoising"
                }),
                "speed": (["fast", "balanced", "quality"], {
                    "default": "balanced", 
                    "tooltip": "Speed vs quality trade-off"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_image",)
    FUNCTION = "denoise_simple"
    CATEGORY = "Eric's Image Processing/Denoising"
    
    def denoise_simple(self, image, strength="medium", speed="balanced"):
        """Simple Non-Local Means denoising with automatic parameters"""
        
        try:
            # Map user inputs to parameters
            noise_level_map = {"light": "low", "medium": "medium", "strong": "high"}
            quality_map = {"fast": "fast", "balanced": "balanced", "quality": "high"}
            
            def process_func(img_np, **kwargs):
                return adaptive_nonlocal_means(
                    img_np,
                    noise_level=noise_level_map[strength],
                    quality=quality_map[speed],
                    preserve_textures=True
                )
            
            result = self.process_image_safe(image, process_func)
            return (result,)
            
        except Exception as e:
            print(f"Error in simple Non-Local Means denoising: {str(e)}")
            return (image,)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "NonLocalMeans": NonLocalMeansNode,
    "NonLocalMeansSimple": NonLocalMeansSimpleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NonLocalMeans": "Non-Local Means Denoise (Eric)",
    "NonLocalMeansSimple": "Non-Local Means Simple (Eric)"
}