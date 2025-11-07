"""
DnCNN Denoising Node for ComfyUI
Classic CNN denoiser with pre-trained weights

Author: Eric Hiss (GitHub: EricRollei)
License: See LICENSE file in repository root

Original DnCNN Algorithm:
    Paper: "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
    Authors: Zhang et al., IEEE TIP 2017
    Source: https://github.com/cszn/DnCNN
    Pretrained weights auto-download from official repository

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
    from ..models.dncnn_architecture import DnCNNProcessor
    BASE_NODE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Base node or DnCNN model not available: {e}")
    BASE_NODE_AVAILABLE = False
    
    class BaseImageProcessingNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}}
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "process"
        CATEGORY = "Eric's Image Processing"


class DnCNNDenoiseNode(BaseImageProcessingNode):
    """
    DnCNN Denoising Node
    
    Classic and effective CNN denoiser with pre-trained weights.
    Automatically downloads weights on first use.
    
    Features:
    - Pre-trained models for various noise levels
    - Blind denoising (works for unknown noise levels)
    - Both grayscale and color support
    - Lightweight and fast inference
    
    Perfect for:
    - General image denoising
    - Film grain removal
    - Quick iterative editing
    - Production workflows
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": ([
                    "color_blind",
                    "grayscale_blind", 
                    "grayscale_sigma15",
                    "grayscale_sigma25",
                    "grayscale_sigma50"
                ], {
                    "default": "color_blind",
                    "tooltip": "Model type:\n* color_blind: Works for any noise level (color)\n* grayscale_blind: Works for any noise level (grayscale)\n* grayscale_sigmaXX: Optimized for specific noise level"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("denoised_image", "processing_report")
    FUNCTION = "process_dncnn"
    CATEGORY = "Eric's Image Processing/Pre-trained Denoisers"
    
    def __init__(self):
        super().__init__()
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def process_dncnn(self, image: torch.Tensor, model_type: str = "color_blind") -> Tuple[torch.Tensor, str]:
        """Process image with DnCNN"""
        
        if not BASE_NODE_AVAILABLE:
            return image, "❌ DnCNN components not available"
        
        try:
            # Initialize processor if needed
            if self.processor is None:
                self.processor = DnCNNProcessor(device=self.device)
            
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
            report_parts.append("🎯 DnCNN Denoising (Pre-trained)")
            report_parts.append("=" * 60)
            
            # Model info
            if self.processor.model is None:
                in_channels = image.shape[-1]
                
                # Override model type for grayscale
                if in_channels == 1 and 'color' in model_type:
                    model_type = 'grayscale_blind'
                    report_parts.append("ℹ️  Auto-switched to grayscale model")
                
                report_parts.append(f"📦 Initializing model: {model_type}")
                report_parts.append(f"   Device: {self.device}")
                
                self.processor.initialize_model(model_type=model_type, in_channels=in_channels)
                
                model_info = self.processor.get_model_info()
                report_parts.append(f"📊 Model Statistics:")
                report_parts.append(f"   * Parameters: {model_info['total_parameters']:,}")
                report_parts.append(f"   * Size: {model_info['model_size_mb']:.2f} MB")
                report_parts.append(f"   * Depth: {model_info['depth']} layers")
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
                    
                    denoised, info = self.processor.process_image(np_image)
                    
                    if len(denoised.shape) == 2:
                        denoised = np.expand_dims(denoised, axis=2)
                    
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
                
                report_parts.append(f"📸 Processing single image")
                report_parts.append(f"   * Shape: {np_image.shape}")
                report_parts.append(f"   * Input range: [{np_image.min():.3f}, {np_image.max():.3f}]")
                report_parts.append("")
                
                denoised, info = self.processor.process_image(np_image)
                
                if len(denoised.shape) == 2:
                    denoised = np.expand_dims(denoised, axis=2)
                
                result_tensor = torch.from_numpy(denoised).float().unsqueeze(0)
                
                report_parts.append("✅ Denoising completed")
                report_parts.append(f"📊 Results:")
                report_parts.append(f"   * PSNR: {info['psnr']:.2f} dB")
                report_parts.append(f"   * Output range: [{info['output_range'][0]:.3f}, {info['output_range'][1]:.3f}]")
                
                # Quality rating
                if info['psnr'] > 35:
                    quality = "Excellent"
                elif info['psnr'] > 30:
                    quality = "Very Good"
                elif info['psnr'] > 25:
                    quality = "Good"
                else:
                    quality = "Fair"
                
                report_parts.append(f"   * Quality: {quality}")
            
            report_parts.append("")
            report_parts.append("ℹ️  Model Info:")
            available = self.processor.get_available_models()
            report_parts.append(f"   Current: {model_type}")
            report_parts.append(f"   Description: {available.get(model_type, 'N/A')}")
            
            processing_report = "\n".join(report_parts)
            
            return (result_tensor, processing_report)
            
        except Exception as e:
            error_msg = f"❌ DnCNN error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return (image, error_msg)


# Node mappings
DNCNN_NODE_CLASS_MAPPINGS = {
    "DnCNNDenoiseNode": DnCNNDenoiseNode,
}

DNCNN_NODE_DISPLAY_NAME_MAPPINGS = {
    "DnCNNDenoiseNode": "DnCNN Denoise (Pre-trained)",
}
