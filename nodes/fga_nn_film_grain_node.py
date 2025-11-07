"""
FGA-NN Film Grain Denoising Node for ComfyUI
State-of-the-art film grain removal with parameter analysis
"""

import torch
import numpy as np
from typing import Tuple, Optional
import os

try:
    from ..base_node import BaseImageProcessingNode
    from ..models.fga_nn_architecture import FGANNProcessor
    BASE_NODE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Base node or FGA-NN model not available: {e}")
    BASE_NODE_AVAILABLE = False
    
    # Fallback base class
    class BaseImageProcessingNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}}
        
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "process"
        CATEGORY = "Eric's Image Processing"


class FGANNFilmGrainDenoiseNode(BaseImageProcessingNode):
    """
    FGA-NN Film Grain Denoising Node
    
    Modern neural network approach for film grain removal using:
    - Multi-scale feature extraction
    - Auto-regressive grain parameter estimation
    - CBAM attention mechanisms
    - Residual learning for noise prediction
    
    Perfect for:
    - Scanned film footage restoration
    - Digital film grain removal
    - Grain characteristic analysis
    - Professional post-production workflows
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "analyze_grain": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Analyze and report grain characteristics"
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
    RETURN_NAMES = ("denoised_image", "analysis_report")
    FUNCTION = "process_fgann"
    CATEGORY = "Eric's Image Processing/Film Grain"
    
    def __init__(self):
        super().__init__()
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def process_fgann(self, image: torch.Tensor, analyze_grain: bool = True, 
                     model_path: str = "") -> Tuple[torch.Tensor, str]:
        """Process image with FGA-NN denoising"""
        
        if not BASE_NODE_AVAILABLE:
            return image, "❌ FGA-NN components not available"
        
        try:
            # Initialize processor if needed
            if self.processor is None:
                self.processor = FGANNProcessor(device=self.device)
            
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
            report_parts.append("🎬 FGA-NN Film Grain Denoising")
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
                report_parts.append("")
            
            # Process images
            if process_batch:
                report_parts.append(f"🔄 Processing batch of {batch_size} images")
                report_parts.append("")
                
                denoised_batch = []
                
                for idx in range(batch_size):
                    # Convert single image to numpy
                    np_image = image[idx].cpu().numpy()
                    
                    # Ensure correct range
                    if np_image.max() > 1.0:
                        np_image = np_image / 255.0
                    
                    # Process with FGA-NN
                    denoised, info = self.processor.process_image(np_image, return_params=analyze_grain)
                    
                    # Convert back to tensor
                    if len(denoised.shape) == 2:
                        denoised = np.expand_dims(denoised, axis=2)
                    
                    denoised_tensor = torch.from_numpy(denoised).float()
                    denoised_batch.append(denoised_tensor)
                    
                    # Add info for first and last images
                    if idx == 0 or idx == batch_size - 1:
                        report_parts.append(f"Image {idx + 1}/{batch_size}:")
                        report_parts.append(f"   • PSNR: {info['psnr']:.2f} dB")
                        report_parts.append(f"   • Output range: [{info['output_range'][0]:.3f}, {info['output_range'][1]:.3f}]")
                        
                        if analyze_grain and 'grain_params' in info:
                            report_parts.append(f"   • Grain Analysis:")
                            ar_coef = info['ar_coefficients']
                            report_parts.append(f"     - AR coefficients: [{ar_coef[0]:.3f}, {ar_coef[1]:.3f}, {ar_coef[2]:.3f}]")
                            
                            freq_bands = info['frequency_bands']
                            report_parts.append(f"     - Frequency bands: [{freq_bands[0]:.3f}, {freq_bands[1]:.3f}, {freq_bands[2]:.3f}]")
                            
                            intensity = info['intensity_scaling']
                            report_parts.append(f"     - Intensity scaling: [{intensity[0]:.3f}, {intensity[1]:.3f}]")
                        
                        report_parts.append("")
                
                # Stack batch
                result_tensor = torch.stack(denoised_batch, dim=0)
                
            else:
                # Single image processing
                np_image = image[0].cpu().numpy()
                
                if np_image.max() > 1.0:
                    np_image = np_image / 255.0
                
                report_parts.append(f"📸 Processing single image")
                report_parts.append(f"   • Shape: {np_image.shape}")
                report_parts.append(f"   • Input range: [{np_image.min():.3f}, {np_image.max():.3f}]")
                report_parts.append("")
                
                # Process with FGA-NN
                denoised, info = self.processor.process_image(np_image, return_params=analyze_grain)
                
                # Convert back to tensor
                if len(denoised.shape) == 2:
                    denoised = np.expand_dims(denoised, axis=2)
                
                result_tensor = torch.from_numpy(denoised).float().unsqueeze(0)
                
                # Add results to report
                report_parts.append("✅ Denoising completed")
                report_parts.append(f"📊 Results:")
                report_parts.append(f"   • PSNR: {info['psnr']:.2f} dB")
                report_parts.append(f"   • Output range: [{info['output_range'][0]:.3f}, {info['output_range'][1]:.3f}]")
                
                if analyze_grain and 'grain_params' in info:
                    report_parts.append("")
                    report_parts.append("🔬 Film Grain Analysis:")
                    
                    ar_coef = info['ar_coefficients']
                    report_parts.append(f"   • Auto-regressive coefficients:")
                    report_parts.append(f"     - AR[0]: {ar_coef[0]:.4f} (temporal correlation)")
                    report_parts.append(f"     - AR[1]: {ar_coef[1]:.4f} (horizontal correlation)")
                    report_parts.append(f"     - AR[2]: {ar_coef[2]:.4f} (vertical correlation)")
                    
                    freq_bands = info['frequency_bands']
                    report_parts.append(f"   • Frequency band strengths:")
                    report_parts.append(f"     - Low freq:  {freq_bands[0]:.4f}")
                    report_parts.append(f"     - Mid freq:  {freq_bands[1]:.4f}")
                    report_parts.append(f"     - High freq: {freq_bands[2]:.4f}")
                    
                    intensity = info['intensity_scaling']
                    report_parts.append(f"   • Intensity-dependent scaling:")
                    report_parts.append(f"     - Dark regions:  {intensity[0]:.4f}")
                    report_parts.append(f"     - Bright regions: {intensity[1]:.4f}")
                    
                    # Grain strength interpretation
                    avg_grain = np.mean(info['grain_params'][:6])
                    if avg_grain < 0.2:
                        grain_level = "Very Light"
                    elif avg_grain < 0.4:
                        grain_level = "Light"
                    elif avg_grain < 0.6:
                        grain_level = "Moderate"
                    elif avg_grain < 0.8:
                        grain_level = "Heavy"
                    else:
                        grain_level = "Very Heavy"
                    
                    report_parts.append(f"   • Overall grain level: {grain_level} ({avg_grain:.3f})")
            
            analysis_report = "\n".join(report_parts)
            
            return (result_tensor, analysis_report)
            
        except Exception as e:
            error_msg = f"❌ FGA-NN error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return (image, error_msg)


# Node class mappings for ComfyUI registration
FGANN_NODE_CLASS_MAPPINGS = {
    "FGANNFilmGrainDenoiseNode": FGANNFilmGrainDenoiseNode,
}

FGANN_NODE_DISPLAY_NAME_MAPPINGS = {
    "FGANNFilmGrainDenoiseNode": "FGA-NN Film Grain Denoise",
}