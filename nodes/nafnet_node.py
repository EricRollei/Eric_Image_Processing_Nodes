"""
NAFNet ComfyUI Node
State-of-the-art pre-trained denoising with automatic weight downloading
"""

import torch
import numpy as np
from typing import Tuple

try:
    from ..base_node import BaseImageProcessingNode
    from ..models.nafnet_architecture import NAFNetProcessor
    BASE_NODE_AVAILABLE = True
except ImportError:
    BASE_NODE_AVAILABLE = False
    class BaseImageProcessingNode:
        CATEGORY = "Eric's Image Processing"


class NAFNetDenoiseNode(BaseImageProcessingNode):
    """
    NAFNet Denoising Node
    
    State-of-the-art denoiser using Nonlinear Activation Free Network.
    Automatically downloads pre-trained weights on first use.
    
    Features:
    - No activation functions (more efficient)
    - Multiple model sizes for speed/quality tradeoff
    - Real-world noise handling (SIDD model)
    - Works immediately - no training required!
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ([
                    "nafnet-width32",
                    "nafnet-width64",
                    "nafnet-sidd"
                ], {
                    "default": "nafnet-width32",
                    "tooltip": "Pre-trained models:\n"
                              "* nafnet-width32: Lighter, faster (~2M params)\n"
                              "* nafnet-width64: Balanced quality/speed (~8M params)\n"
                              "* nafnet-sidd: Real-world noise (trained on SIDD)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("denoised_image", "info_report")
    FUNCTION = "process_nafnet"
    CATEGORY = "Eric's Image Processing/Pre-trained Denoisers"
    
    def __init__(self):
        super().__init__()
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def process_nafnet(self, image: torch.Tensor, model: str = "nafnet-width32") -> Tuple[torch.Tensor, str]:
        if not BASE_NODE_AVAILABLE:
            return image, "❌ NAFNet components not available"
        
        try:
            # Initialize processor
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
            
            # Start report
            report_parts = []
            report_parts.append("🚀 NAFNet Denoising (State-of-the-Art)")
            report_parts.append("=" * 60)
            report_parts.append(f"📦 Model: {model}")
            report_parts.append(f"💾 Weights: Auto-downloaded and cached")
            report_parts.append(f"🖥️  Device: {self.device}")
            report_parts.append("")
            
            if process_batch:
                report_parts.append(f"Processing {batch_size} images...")
                denoised_batch = []
                
                for idx in range(batch_size):
                    np_image = image[idx].cpu().numpy()
                    if np_image.max() > 1.0:
                        np_image = np_image / 255.0
                    
                    denoised, info = self.processor.process_image(np_image, model)
                    
                    # Ensure 3 channels for ComfyUI
                    if len(denoised.shape) == 2:
                        denoised = np.stack([denoised] * 3, axis=2)
                    
                    denoised_tensor = torch.from_numpy(denoised).float()
                    denoised_batch.append(denoised_tensor)
                    
                    if idx == 0:
                        report_parts.append(f"✓ Using: {info['description']}")
                        report_parts.append(f"  PSNR: {info['psnr']:.2f} dB")
                
                result_tensor = torch.stack(denoised_batch, dim=0)
                report_parts.append(f"✅ Batch processing complete")
            else:
                np_image = image[0].cpu().numpy()
                if np_image.max() > 1.0:
                    np_image = np_image / 255.0
                
                denoised, info = self.processor.process_image(np_image, model)
                
                # Ensure 3 channels for ComfyUI
                if len(denoised.shape) == 2:
                    denoised = np.stack([denoised] * 3, axis=2)
                
                result_tensor = torch.from_numpy(denoised).float().unsqueeze(0)
                
                # Get model info
                model_info = self.processor.get_model_info()
                
                report_parts.append("✅ Denoising Complete")
                report_parts.append(f"")
                report_parts.append(f"📊 Results:")
                report_parts.append(f"   * Model: {info['model']}")
                report_parts.append(f"   * Description: {info['description']}")
                report_parts.append(f"   * PSNR: {info['psnr']:.2f} dB")
                report_parts.append(f"   * Output range: [{info['output_range'][0]:.3f}, {info['output_range'][1]:.3f}]")
                report_parts.append(f"")
                report_parts.append(f"⚙️  Model Info:")
                report_parts.append(f"   * Parameters: {model_info['total_parameters']:,}")
                report_parts.append(f"   * Size: {model_info['model_size_mb']:.1f} MB")
                report_parts.append(f"")
                report_parts.append(f"💡 Tip: NAFNet uses no activation functions!")
                report_parts.append(f"   This makes it faster and more efficient.")
            
            report = "\n".join(report_parts)
            return (result_tensor, report)
            
        except Exception as e:
            error_msg = f"❌ NAFNet error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return (image, error_msg)


# Node registration
NAFNET_NODE_CLASS_MAPPINGS = {
    "NAFNetDenoiseNode": NAFNetDenoiseNode,
}

NAFNET_NODE_DISPLAY_NAME_MAPPINGS = {
    "NAFNetDenoiseNode": "NAFNet Denoise (Pre-trained)",
}
