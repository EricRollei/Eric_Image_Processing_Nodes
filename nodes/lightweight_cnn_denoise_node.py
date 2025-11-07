"""
Lightweight Progressive CNN Denoising Node for ComfyUI
Efficient film grain removal with progressive residual fusion
"""

import torch
import numpy as np
from typing import Tuple, Optional
import os

try:
    from ..base_node import BaseImageProcessingNode
    from ..models.progressive_cnn_architecture import ProgressiveCNNProcessor
    BASE_NODE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Base node or Progressive CNN model not available: {e}")
    BASE_NODE_AVAILABLE = False
    
    # Fallback base class
    class BaseImageProcessingNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}}
        
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "process"
        CATEGORY = "Eric's Image Processing"


class LightweightCNNDenoiseNode(BaseImageProcessingNode):
    """
    Lightweight Progressive CNN Denoising Node
    
    Efficient neural network approach for film grain removal using:
    - Dense blocks for comprehensive feature extraction
    - Progressive residual fusion (shallow → deep features)
    - Lightweight attention mechanisms
    - Minimal parameters (~500KB) for fast inference
    
    Perfect for:
    - Real-time denoising applications
    - Resource-constrained environments
    - Batch processing workflows
    - Quick iterative editing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "show_comparison": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show before/after comparison metrics"
                }),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to pre-trained weights (leave empty for random init)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("denoised_image", "processing_report")
    FUNCTION = "process_lightweight_cnn"
    CATEGORY = "Eric's Image Processing/Film Grain"
    
    def __init__(self):
        super().__init__()
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def process_lightweight_cnn(self, image: torch.Tensor, show_comparison: bool = True,
                               model_path: str = "") -> Tuple[torch.Tensor, str]:
        """Process image with Lightweight Progressive CNN"""
        
        if not BASE_NODE_AVAILABLE:
            return image, "❌ Progressive CNN components not available"
        
        try:
            # Initialize processor if needed
            if self.processor is None:
                self.processor = ProgressiveCNNProcessor(device=self.device)
            
            # Handle batch dimension
            if len(image.shape) == 4:
                batch_size = image.shape[0]
                process_batch = batch_size > 1
            else:
                batch_size = 1
                process_batch = False
                image = image.unsqueeze(0)
            
            # Build report
            report_parts = []
            report_parts.append("⚡ Lightweight Progressive CNN Denoising")
            report_parts.append("=" * 60)
            
            # Model info
            if self.processor.model is None:
                # Determine channels from input
                in_channels = image.shape[-1]
                
                # Check for model weights
                model_weights_path = None
                if model_path and os.path.exists(model_path):
                    model_weights_path = model_path
                    report_parts.append(f"📦 Loading weights from: {model_path}")
                else:
                    report_parts.append("🔧 Using random initialization (untrained network)")
                    report_parts.append("💡 For best results, provide pre-trained weights")
                
                self.processor.initialize_model(in_channels=in_channels, pretrained_path=model_weights_path)
                
                # Get model info
                model_info = self.processor.get_model_info()
                report_parts.append(f"📊 Model Statistics:")
                report_parts.append(f"   • Total parameters: {model_info['total_parameters']:,}")
                report_parts.append(f"   • Model size: {model_info['model_size_mb']:.2f} MB")
                report_parts.append(f"   • Device: {model_info['device']}")
                report_parts.append(f"   • Architecture: Dense blocks + Progressive residual fusion")
                report_parts.append("")
            
            # Process images
            if process_batch:
                report_parts.append(f"🔄 Processing batch of {batch_size} images")
                report_parts.append("")
                
                denoised_batch = []
                total_psnr = 0.0
                
                for idx in range(batch_size):
                    # Convert single image to numpy
                    np_image = image[idx].cpu().numpy()
                    
                    # Ensure correct range
                    if np_image.max() > 1.0:
                        np_image = np_image / 255.0
                    
                    # Process with Progressive CNN
                    denoised, info = self.processor.process_image(np_image)
                    
                    # Convert back to tensor
                    if len(denoised.shape) == 2:
                        denoised = np.expand_dims(denoised, axis=2)
                    
                    denoised_tensor = torch.from_numpy(denoised).float()
                    denoised_batch.append(denoised_tensor)
                    
                    total_psnr += info['psnr']
                    
                    # Add info for first and last images
                    if idx == 0 or idx == batch_size - 1:
                        report_parts.append(f"Image {idx + 1}/{batch_size}:")
                        report_parts.append(f"   • PSNR: {info['psnr']:.2f} dB")
                        report_parts.append(f"   • Output range: [{info['output_range'][0]:.3f}, {info['output_range'][1]:.3f}]")
                        report_parts.append("")
                
                # Stack batch
                result_tensor = torch.stack(denoised_batch, dim=0)
                
                # Batch statistics
                avg_psnr = total_psnr / batch_size
                report_parts.append(f"📈 Batch Statistics:")
                report_parts.append(f"   • Average PSNR: {avg_psnr:.2f} dB")
                report_parts.append(f"   • Total images processed: {batch_size}")
                
            else:
                # Single image processing
                np_image = image[0].cpu().numpy()
                
                if np_image.max() > 1.0:
                    np_image = np_image / 255.0
                
                report_parts.append(f"📸 Processing single image")
                report_parts.append(f"   • Shape: {np_image.shape}")
                report_parts.append(f"   • Input range: [{np_image.min():.3f}, {np_image.max():.3f}]")
                
                # Calculate input noise estimation (std of Laplacian)
                if len(np_image.shape) == 3:
                    gray = np.dot(np_image[..., :3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = np_image
                
                laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                from scipy.ndimage import convolve
                laplacian_img = convolve(gray, laplacian)
                noise_estimate = np.std(laplacian_img)
                
                report_parts.append(f"   • Estimated noise level: {noise_estimate:.4f}")
                report_parts.append("")
                
                # Process with Progressive CNN
                denoised, info = self.processor.process_image(np_image)
                
                # Convert back to tensor
                if len(denoised.shape) == 2:
                    denoised = np.expand_dims(denoised, axis=2)
                
                result_tensor = torch.from_numpy(denoised).float().unsqueeze(0)
                
                # Add results to report
                report_parts.append("✅ Denoising completed")
                report_parts.append(f"📊 Results:")
                report_parts.append(f"   • PSNR: {info['psnr']:.2f} dB")
                report_parts.append(f"   • Output range: [{info['output_range'][0]:.3f}, {info['output_range'][1]:.3f}]")
                
                if show_comparison:
                    report_parts.append("")
                    report_parts.append("📉 Noise Reduction:")
                    
                    # Estimate noise in output
                    if len(denoised.shape) == 3:
                        gray_out = np.dot(denoised[..., :3], [0.2989, 0.5870, 0.1140])
                    else:
                        gray_out = denoised
                    
                    laplacian_out = convolve(gray_out, laplacian)
                    noise_estimate_out = np.std(laplacian_out)
                    
                    noise_reduction = ((noise_estimate - noise_estimate_out) / noise_estimate) * 100
                    
                    report_parts.append(f"   • Input noise: {noise_estimate:.4f}")
                    report_parts.append(f"   • Output noise: {noise_estimate_out:.4f}")
                    report_parts.append(f"   • Noise reduction: {noise_reduction:.1f}%")
                    
                    # Quality interpretation
                    if info['psnr'] > 35:
                        quality = "Excellent"
                    elif info['psnr'] > 30:
                        quality = "Very Good"
                    elif info['psnr'] > 25:
                        quality = "Good"
                    elif info['psnr'] > 20:
                        quality = "Fair"
                    else:
                        quality = "Needs Improvement"
                    
                    report_parts.append(f"   • Quality rating: {quality}")
                    
                    # Recommendations
                    if noise_reduction < 30:
                        report_parts.append("")
                        report_parts.append("💡 Recommendations:")
                        report_parts.append("   • Low noise reduction detected")
                        report_parts.append("   • Consider using pre-trained weights for better results")
                        report_parts.append("   • Or try the FGA-NN node for heavy grain removal")
            
            processing_report = "\n".join(report_parts)
            
            return (result_tensor, processing_report)
            
        except Exception as e:
            error_msg = f"❌ Progressive CNN error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return (image, error_msg)


# Node class mappings for ComfyUI registration
PROGRESSIVE_CNN_NODE_CLASS_MAPPINGS = {
    "LightweightCNNDenoiseNode": LightweightCNNDenoiseNode,
}

PROGRESSIVE_CNN_NODE_DISPLAY_NAME_MAPPINGS = {
    "LightweightCNNDenoiseNode": "Lightweight Progressive CNN Denoise",
}