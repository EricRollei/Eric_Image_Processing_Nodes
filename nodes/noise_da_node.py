"""
Noise-DA Model Node for ComfyUI

Based on the official implementation from:
"Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration"
Paper: https://arxiv.org/abs/2406.18516
Code: https://github.com/KangLiao929/Noise-DA

Original Authors:
- Kang Liao (kang.liao@ntu.edu.sg)
- Zongsheng Yue
- Zhouxia Wang  
- Chen Change Loy

S-Lab, Nanyang Technological University

Citation:
@article{liao2024denoising,
    title={Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration},
    author={Liao, Kang and Yue, Zongsheng and Wang, Zhouxia and Loy, Chen Change},
    journal={arXiv preprint arXiv:2406.18516},
    year={2024}
}

This ComfyUI node provides access to pretrained Noise-DA models for:
- Image denoising
- Image deblurring  
- Image deraining
"""

import torch
import numpy as np
from typing import Tuple

# Import from parent package
try:
    from ..base_node import BaseImageProcessingNode
    from ..scripts.noise_da_processing import NoiseDAProcessor
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode
    from scripts.noise_da_processing import NoiseDAProcessor

class NoiseDANode(BaseImageProcessingNode):
    """ComfyUI Node for Noise-DA pretrained models"""
    
    def __init__(self):
        super().__init__()
        self.processors = {}
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (["denoise", "deblur", "derain"], {
                    "default": "denoise",
                    "tooltip": "Noise-DA model type:\n• denoise: Remove noise from images using domain adaptation\n• deblur: Remove blur from images\n• derain: Remove rain streaks from images\n\nBased on 'Denoising as Adaptation' (ICLR 2025)\nGitHub: https://github.com/KangLiao929/Noise-DA"
                }),
                "output_mode": (["residual", "direct"], {
                    "default": "residual",
                    "tooltip": "Output interpretation:\n• residual: Model outputs residual/correction (recommended)\n• direct: Model outputs direct result"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration if available"
                }),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Processing strength:\n• 0.0: No processing (original image)\n• 0.8: Strong processing (recommended)\n• 1.0: Full processing\n• 2.0: Enhanced processing"
                }),
                "residual_scale": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 50.0, "step": 0.5,
                    "tooltip": "Residual scaling factor:\n• 0.1: Very subtle corrections\n• 5.0: Moderate corrections (recommended)\n• 10.0: Strong corrections\n• 20.0+: Very strong corrections\n\nNote: Higher values may cause over-processing or artifacts"
                }),
                "blend_mode": (["normal", "soft_light", "overlay"], {
                    "default": "normal",
                    "tooltip": "Blending mode between original and processed:\n• normal: Linear blend\n• soft_light: Softer correction\n• overlay: Enhanced correction"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "processing_info")
    FUNCTION = "process_with_noise_da"
    CATEGORY = "Eric's Nodes/AI Models"
    
    def process_with_noise_da(self, image: torch.Tensor, model_type: str, output_mode: str, 
                             use_gpu: bool, strength: float = 0.8, residual_scale: float = 5.0, 
                             blend_mode: str = "normal") -> Tuple[torch.Tensor, str]:
        """Process image with Noise-DA model"""
        
        try:
            # Get or create processor
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            processor_key = f"{model_type}_{device}"
            
            if processor_key not in self.processors:
                print(f"Loading Noise-DA {model_type} model for {device.upper()}")
                self.processors[processor_key] = NoiseDAProcessor(model_type, device)
                
            processor = self.processors[processor_key]
            
            # Convert tensor to numpy
            np_image = self.tensor_to_numpy(image)
            original_image = np_image.copy()
            
            # Process image
            residual_mode = (output_mode == "residual")
            
            if residual_mode:
                # Get corrected image from model (original with residuals applied)
                corrected_image = processor.process_image(np_image, residual_mode=True, residual_scale_factor=residual_scale)
                
                # Apply blending between original and corrected image
                if blend_mode == "normal":
                    processed_image = self._blend_images(original_image, corrected_image, strength)
                elif blend_mode == "soft_light":
                    processed_image = self._apply_soft_light_blend(original_image, corrected_image, strength)
                elif blend_mode == "overlay":
                    processed_image = self._apply_overlay_blend(original_image, corrected_image, strength)
                    
            else:
                # Direct output from model
                processed_image = processor.process_image(np_image, residual_mode=False, residual_scale_factor=residual_scale)
                
                # Apply strength as blend between original and processed
                if strength != 1.0:
                    processed_image = self._blend_images(original_image, processed_image, strength)
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(processed_image)
            
            # Generate processing info
            info = self._generate_processing_info(model_type, output_mode, device, strength, residual_scale, blend_mode, np_image.shape)
            
            return (result_tensor, info)
            
        except Exception as e:
            error_msg = f"Error in Noise-DA processing: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return (image, error_msg)
    
    def _apply_soft_light_blend(self, original: np.ndarray, corrected: np.ndarray, strength: float) -> np.ndarray:
        """Apply soft light blending between original and corrected images"""
        orig_f = original.astype(np.float32) / 255.0
        corr_f = corrected.astype(np.float32) / 255.0
        
        # Soft light formula
        result = np.where(corr_f <= 0.5,
                         orig_f - (1 - 2 * corr_f) * orig_f * (1 - orig_f),
                         orig_f + (2 * corr_f - 1) * (np.sqrt(orig_f) - orig_f))
        
        # Apply strength
        result = orig_f * (1 - strength) + result * strength
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    def _apply_overlay_blend(self, original: np.ndarray, corrected: np.ndarray, strength: float) -> np.ndarray:
        """Apply overlay blending between original and corrected images"""
        orig_f = original.astype(np.float32) / 255.0
        corr_f = corrected.astype(np.float32) / 255.0
        
        # Overlay formula
        result = np.where(orig_f < 0.5,
                         2 * orig_f * corr_f,
                         1 - 2 * (1 - orig_f) * (1 - corr_f))
        
        # Apply strength
        result = orig_f * (1 - strength) + result * strength
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    def _blend_images(self, original: np.ndarray, processed: np.ndarray, strength: float) -> np.ndarray:
        """Blend two images with given strength"""
        result = original.astype(np.float32) * (1 - strength) + processed.astype(np.float32) * strength
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _generate_processing_info(self, model_type: str, output_mode: str, device: str, 
                                 strength: float, residual_scale: float, blend_mode: str, shape: tuple) -> str:
        """Generate processing information"""
        info_lines = []
        info_lines.append(f"Noise-DA {model_type.capitalize()} Processing")
        info_lines.append(f"Based on 'Denoising as Adaptation' (ICLR 2025)")
        info_lines.append(f"GitHub: https://github.com/KangLiao929/Noise-DA")
        info_lines.append("")
        info_lines.append(f"Model: {model_type}")
        info_lines.append(f"Device: {device.upper()}")
        info_lines.append(f"Output Mode: {output_mode}")
        info_lines.append(f"Strength: {strength:.1f}")
        info_lines.append(f"Residual Scale: {residual_scale:.1f}")
        info_lines.append(f"Blend Mode: {blend_mode}")
        info_lines.append(f"Image Shape: {shape}")
        
        return "\\n".join(info_lines)

class NoiseDABatchNode(BaseImageProcessingNode):
    """Batch processing node for Noise-DA models"""
    
    def __init__(self):
        super().__init__()
        self.processors = {}
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_type": (["denoise", "deblur", "derain"], {
                    "default": "denoise",
                    "tooltip": "Noise-DA model type for batch processing\nBased on 'Denoising as Adaptation' (ICLR 2025)\nGitHub: https://github.com/KangLiao929/Noise-DA"
                }),
                "output_mode": (["residual", "direct"], {"default": "residual"}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_images", "batch_info")
    FUNCTION = "process_batch_with_noise_da"
    CATEGORY = "Eric's Nodes/AI Models"
    
    def process_batch_with_noise_da(self, images: torch.Tensor, model_type: str, output_mode: str, 
                                   use_gpu: bool, strength: float = 1.0) -> Tuple[torch.Tensor, str]:
        """Process batch of images with Noise-DA model"""
        
        try:
            # Get or create processor
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            processor_key = f"{model_type}_{device}"
            
            if processor_key not in self.processors:
                print(f"Loading Noise-DA {model_type} model for batch processing")
                self.processors[processor_key] = NoiseDAProcessor(model_type, device)
                
            processor = self.processors[processor_key]
            
            # Process each image in batch
            processed_images = []
            batch_size = images.shape[0]
            
            print(f"Processing batch of {batch_size} images with Noise-DA {model_type}")
            
            for i in range(batch_size):
                # Get single image
                single_image = images[i:i+1]
                np_image = self.tensor_to_numpy(single_image)
                
                # Process image
                residual_mode = (output_mode == "residual")
                processed_image = processor.process_image(np_image, residual_mode)
                
                # Apply strength if needed
                if residual_mode and strength != 1.0:
                    processed_image = np_image.astype(np.float32) * (1 - strength) + processed_image.astype(np.float32) * strength
                    processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
                
                # Convert back to tensor
                processed_tensor = self.numpy_to_tensor(processed_image)
                processed_images.append(processed_tensor)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{batch_size} images")
            
            # Combine processed images
            result_tensor = torch.cat(processed_images, dim=0)
            
            # Generate batch info
            info = f"Batch Noise-DA {model_type.capitalize()} Processing\\n"
            info += f"Based on 'Denoising as Adaptation' (ICLR 2025)\\n"
            info += f"GitHub: https://github.com/KangLiao929/Noise-DA\\n"
            info += f"\\n"
            info += f"Processed Images: {batch_size}\\n"
            info += f"Device: {device.upper()}\\n"
            info += f"Mode: {output_mode}\\n"
            info += f"Strength: {strength:.1f}"
            
            return (result_tensor, info)
            
        except Exception as e:
            error_msg = f"Error in batch Noise-DA processing: {str(e)}"
            print(error_msg)
            return (images, error_msg)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "NoiseDANode": NoiseDANode,
    "NoiseDABatchNode": NoiseDABatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseDANode": "Noise-DA Processing (Eric)",
    "NoiseDABatchNode": "Noise-DA Batch Processing (Eric)",
}
