"""
NAFNet Denoising Node for ComfyUI
Simple Baselines for Image Restoration - state-of-the-art baseline-free network

Author: Eric Hiss (GitHub: EricRollei)
License: See LICENSE file in repository root

Original NAFNet Algorithm:
    Paper: "Simple Baselines for Image Restoration"
    Authors: Chen et al., ECCV 2022
    Source: https://github.com/megvii-research/NAFNet
    License: MIT License
    Pretrained weights: Auto-download from GitHub releases (NAFNet-width32.pth, NAFNet-width64.pth)
    
    Citation:
    @inproceedings{chen2022nafnet,
      title={Simple Baselines for Image Restoration},
      author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
      booktitle={European Conference on Computer Vision},
      year={2022}
    }

Dependencies:
    - PyTorch (BSD 3-Clause License)
    - NumPy (BSD 3-Clause License)
"""

import torch
import numpy as np
from typing import Tuple
import os

try:
    from ..base_node import BaseImageProcessingNode
    from ..models.nafnet_architecture import NAFNetProcessor
    BASE_NODE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Base node or NAFNet model not available: {e}")
    BASE_NODE_AVAILABLE = False
    
    class BaseImageProcessingNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}}
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "process"
        CATEGORY = "Eric's Image Processing"


class NAFNetDenoiseNode(BaseImageProcessingNode):
    """
    NAFNet Denoising Node
    
    State-of-the-art baseline-free network for image restoration.
    Simple architecture with excellent performance.
    
    Features:
    - SOTA performance with fewer parameters
    - Baseline-free design (no complex components)
    - Pre-trained models for denoising and deblurring
    - Efficient inference
    
    Perfect for:
    - High-quality denoising
    - Film grain removal
    - Motion deblurring
    - Professional restoration
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": ([
                    "denoising_sigma15",
                    "denoising_sigma25",
                    "deblurring"
                ], {
                    "default": "denoising_sigma15",
                    "tooltip": "Model type:\n* denoising_sigma15: Light noise (width=32)\n* denoising_sigma25: Medium noise (width=64)\n* deblurring: Motion/defocus blur removal"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("restored_image", "processing_report")
    FUNCTION = "process_nafnet"
    CATEGORY = "Eric's Image Processing/Pre-trained Denoisers"
    
    def __init__(self):
        super().__init__()
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def process_nafnet(self, image: torch.Tensor, model_type: str = "denoising_sigma15") -> Tuple[torch.Tensor, str]:
        """Process image with NAFNet"""
        
        if not BASE_NODE_AVAILABLE:
            return image, "❌ NAFNet components not available"
        
        try:
            # Initialize processor if needed
            if self.processor is None:
                self.processor = NAFNetProcessor(device=self.device)
            
            # Handle batch
            if len(image.shape) == 4:
                batch_size = image.shape[0]
                process_batch = batch_size > 1
            else:
                batch_size = 1
                process_batch = False
                image = image.unsqueeze(0)
            
            # Build report
            report_parts = []
            report_parts.append("⚡ NAFNet Image Restoration (Pre-trained)")
            report_parts.append("=" * 60)
            
            # Model info
            if self.processor.model is None:
                report_parts.append(f"📦 Initializing NAFNet: {model_type}")
                report_parts.append(f"   Device: {self.device}")
                
                self.processor.initialize_model(model_type=model_type)
                
                model_info = self.processor.get_model_info()
                report_parts.append(f"📊 Model Statistics:")
                report_parts.append(f"   * Parameters: {model_info['total_parameters']:,}")
                report_parts.append(f"   * Size: {model_info['model_size_mb']:.2f} MB")
                report_parts.append(f"   * Architecture: Baseline-free U-Net")
                report_parts.append("")
            
            # Task description
            if 'deblur' in model_type:
                task = "deblurring"
            else:
                task = "denoising"
            
            report_parts.append(f"🎯 Task: {task.capitalize()}")
            report_parts.append("")
            
            # Process images
            if process_batch:
                report_parts.append(f"🔄 Processing batch of {batch_size} images")
                report_parts.append("")
                
                denoised_batch = []
                total_psnr = 0.0
                
                for idx in range(batch_size):
                    np_image = image[idx].cpu().numpy()
                    
                    if np_image.max() > 1.0:
                        np_image = np_image / 255.0
                    
                    # Ensure 3 channels for NAFNet
                    if len(np_image.shape) == 2:
                        np_image = np.stack([np_image] * 3, axis=2)
                    elif np_image.shape[2] == 1:
                        np_image = np.repeat(np_image, 3, axis=2)
                    
                    denoised, info = self.processor.process_image(np_image)
                    
                    denoised_tensor = torch.from_numpy(denoised).float()
                    denoised_batch.append(denoised_tensor)
                    
                    total_psnr += info['psnr']
                    
                    if idx == 0 or idx == batch_size - 1:
                        report_parts.append(f"Image {idx + 1}/{batch_size}:")
                        report_parts.append(f"   * PSNR: {info['psnr']:.2f} dB")
                        report_parts.append("")
                
                result_tensor = torch.stack(denoised_batch, dim=0)
                
                avg_psnr = total_psnr / batch_size
                report_parts.append(f"📈 Batch Average PSNR: {avg_psnr:.2f} dB")
                
            else:
                # Single image
                np_image = image[0].cpu().numpy()
                
                if np_image.max() > 1.0:
                    np_image = np_image / 255.0
                
                # Ensure 3 channels
                if len(np_image.shape) == 2:
                    np_image = np.stack([np_image] * 3, axis=2)
                elif np_image.shape[2] == 1:
                    np_image = np.repeat(np_image, 3, axis=2)
                
                report_parts.append(f"📸 Processing single image")
                report_parts.append(f"   * Shape: {np_image.shape}")
                report_parts.append(f"   * Input range: [{np_image.min():.3f}, {np_image.max():.3f}]")
                report_parts.append("")
                
                denoised, info = self.processor.process_image(np_image)
                
                result_tensor = torch.from_numpy(denoised).float().unsqueeze(0)
                
                report_parts.append(f"✅ {task.capitalize()} completed")
                report_parts.append(f"📊 Results:")
                report_parts.append(f"   * PSNR: {info['psnr']:.2f} dB")
                report_parts.append(f"   * Output range: [{info['output_range'][0]:.3f}, {info['output_range'][1]:.3f}]")
                
                # Quality rating
                if info['psnr'] > 35:
                    quality = "Excellent"
                    icon = "🌟"
                elif info['psnr'] > 30:
                    quality = "Very Good"
                    icon = "✨"
                elif info['psnr'] > 25:
                    quality = "Good"
                    icon = "👍"
                else:
                    quality = "Fair"
                    icon = "👌"
                
                report_parts.append(f"   * Quality: {icon} {quality}")
            
            report_parts.append("")
            report_parts.append("ℹ️  About NAFNet:")
            report_parts.append("   * Simple Baselines for Image Restoration (ECCV 2022)")
            report_parts.append("   * Baseline-free design without complex components")
            report_parts.append("   * Achieves SOTA performance with simplicity")
            
            if 'deblur' in model_type:
                report_parts.append("   * Trained for motion/defocus blur removal")
            else:
                noise_level = model_type.split('sigma')[-1] if 'sigma' in model_type else "unknown"
                report_parts.append(f"   * Optimized for noise level σ={noise_level}")
            
            processing_report = "\n".join(report_parts)
            
            return (result_tensor, processing_report)
            
        except Exception as e:
            error_msg = f"❌ NAFNet error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return (image, error_msg)


# Node mappings
NAFNET_NODE_CLASS_MAPPINGS = {
    "NAFNetDenoiseNode": NAFNetDenoiseNode,
}

NAFNET_NODE_DISPLAY_NAME_MAPPINGS = {
    "NAFNetDenoiseNode": "NAFNet Restore (Pre-trained)",
}
