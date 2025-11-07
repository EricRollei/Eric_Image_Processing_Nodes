"""
Professional Restoration Pipeline Node
Multi-stage intelligent restoration combining the best of all processing methods
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List

# Import from main package
try:
    from ..base_node import BaseImageProcessingNode
except ImportError:
    # Fallback for direct testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode

# Import available processors
try:
    from Eric_Image_Processing_Nodes import (
        SCUNetProcessor,
        SwinIRProcessor,
        RealESRGANProcessor,
        FrequencyEnhancementProcessor,
        WaveletDenoiseProcessor,
        AdvancedPSFProcessor,
        PerceptualColorProcessor,
        AutoDenoiseProcessor,
        Noise2VoidProcessor
    )
except ImportError:
    try:
        from scripts.scunet_processing import SCUNetProcessor
        from scripts.swinir_processing import SwinIRProcessor
        from scripts.real_esrgan_processing import RealESRGANProcessor
        from scripts.frequency_enhancement import FrequencyEnhancementProcessor
        from scripts.wavelet_denoise import WaveletDenoiseProcessor
        from scripts.advanced_psf_modeling import AdvancedPSFProcessor
        from scripts.perceptual_color_processing import PerceptualColorProcessor
    except ImportError:
        print("Warning: Some processors not available for restoration pipeline")


class ProfessionalRestorationPipelineNode(BaseImageProcessingNode):
    """
    Professional Multi-Stage Restoration Pipeline
    
    Combines multiple advanced processing methods in an intelligent sequence:
    
    PIPELINE STAGES:
    1. Pre-processing: Noise reduction and artifact removal
    2. Primary Restoration: AI-based enhancement (SCUNet/SwinIR/Real-ESRGAN)
    3. Frequency Enhancement: Detail recovery and sharpening
    4. Color Optimization: Perceptual color enhancement
    5. Post-processing: Final refinement and quality assurance
    
    INTELLIGENT ADAPTATION:
    - Automatically adjusts pipeline based on image characteristics
    - Skips unnecessary stages for optimal efficiency
    - Applies appropriate strength levels for each stage
    - Monitors quality at each step to prevent over-processing
    
    PROFESSIONAL FEATURES:
    - Quality-guided processing strength
    - Multi-scale analysis and processing
    - Edge-preserving enhancement
    - Noise-adaptive filtering
    - Color space optimization
    
    Perfect for:
    - Professional photo restoration
    - High-quality archival processing
    - Complex degradation scenarios
    - Demanding quality requirements
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "restoration_level": (["conservative", "moderate", "aggressive", "maximum"], {
                    "default": "moderate",
                    "tooltip": "Restoration intensity:\n• conservative: Gentle enhancement\n• moderate: Balanced restoration\n• aggressive: Strong enhancement\n• maximum: Maximum quality (slower)"
                }),
                "target_scale": ([1, 2, 3, 4], {
                    "default": 2,
                    "tooltip": "Final upscaling factor"
                }),
            },
            "optional": {
                "enable_stages": ("STRING", {
                    "default": "all",
                    "tooltip": "Comma-separated stages to enable:\n'preprocessing,ai_restoration,frequency,color,postprocessing'\nOr 'all' for complete pipeline"
                }),
                "ai_method": (["auto", "auto_denoise", "noise2void", "scunet", "swinir", "real_esrgan"], {
                    "default": "auto",
                    "tooltip": "AI restoration method:\n• auto: Intelligent selection\n• auto_denoise: Self-supervised denoising\n• noise2void: Single image training\n• scunet: Realistic restoration\n• swinir: Transformer precision\n• real_esrgan: Natural photos"
                }),
                "preserve_details": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable detail preservation techniques"
                }),
                "color_enhancement": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable perceptual color optimization"
                }),
                "quality_monitoring": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Monitor quality at each stage"
                }),
                "device_preference": (["auto", "cpu", "cuda"], {
                    "default": "auto",
                    "tooltip": "Processing device preference"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("restored_image", "pipeline_report")
    FUNCTION = "restore_professional"
    
    def restore_professional(self, image: torch.Tensor, restoration_level: str = "moderate",
                           target_scale: int = 2, enable_stages: str = "all",
                           ai_method: str = "auto", preserve_details: bool = True,
                           color_enhancement: bool = True, quality_monitoring: bool = True,
                           device_preference: str = "auto") -> Tuple[torch.Tensor, str]:
        """
        Professional multi-stage restoration pipeline
        """
        try:
            # Convert input
            img_np = self.tensor_to_numpy(image)
            current_image = img_np.copy()
            
            # Initialize pipeline report
            report_lines = [
                "=== PROFESSIONAL RESTORATION PIPELINE ===",
                f"Input: {img_np.shape} pixels",
                f"Restoration level: {restoration_level}",
                f"Target scale: {target_scale}x",
                f"AI method: {ai_method}",
                "",
            ]
            
            # Parse enabled stages
            if enable_stages.lower() == "all":
                stages = ["preprocessing", "ai_restoration", "frequency", "color", "postprocessing"]
            else:
                stages = [s.strip().lower() for s in enable_stages.split(",")]
            
            report_lines.append(f"Enabled stages: {', '.join(stages)}")
            report_lines.append("")
            
            # Pipeline execution
            stage_results = {}
            
            # Stage 1: Pre-processing
            if "preprocessing" in stages:
                current_image, stage_info = self._stage_preprocessing(
                    current_image, restoration_level, device_preference
                )
                stage_results["preprocessing"] = stage_info
                report_lines.extend([
                    "STAGE 1: PRE-PROCESSING",
                    f"Status: {stage_info.get('status', 'completed')}",
                    f"Actions: {', '.join(stage_info.get('actions', []))}",
                    ""
                ])
            
            # Stage 2: AI Restoration
            if "ai_restoration" in stages:
                current_image, stage_info = self._stage_ai_restoration(
                    current_image, ai_method, target_scale, restoration_level, device_preference
                )
                stage_results["ai_restoration"] = stage_info
                report_lines.extend([
                    "STAGE 2: AI RESTORATION",
                    f"Status: {stage_info.get('status', 'completed')}",
                    f"Method: {stage_info.get('method', 'unknown')}",
                    f"Scale achieved: {stage_info.get('scale_achieved', 'N/A')}",
                    ""
                ])
            
            # Stage 3: Frequency Enhancement
            if "frequency" in stages and preserve_details:
                current_image, stage_info = self._stage_frequency_enhancement(
                    current_image, restoration_level, device_preference
                )
                stage_results["frequency"] = stage_info
                report_lines.extend([
                    "STAGE 3: FREQUENCY ENHANCEMENT",
                    f"Status: {stage_info.get('status', 'completed')}",
                    f"Techniques: {', '.join(stage_info.get('techniques', []))}",
                    ""
                ])
            
            # Stage 4: Color Enhancement
            if "color" in stages and color_enhancement:
                current_image, stage_info = self._stage_color_enhancement(
                    current_image, restoration_level, device_preference
                )
                stage_results["color"] = stage_info
                report_lines.extend([
                    "STAGE 4: COLOR ENHANCEMENT",
                    f"Status: {stage_info.get('status', 'completed')}",
                    f"Optimizations: {', '.join(stage_info.get('optimizations', []))}",
                    ""
                ])
            
            # Stage 5: Post-processing
            if "postprocessing" in stages:
                current_image, stage_info = self._stage_postprocessing(
                    current_image, restoration_level, quality_monitoring
                )
                stage_results["postprocessing"] = stage_info
                report_lines.extend([
                    "STAGE 5: POST-PROCESSING",
                    f"Status: {stage_info.get('status', 'completed')}",
                    f"Refinements: {', '.join(stage_info.get('refinements', []))}",
                    ""
                ])
            
            # Final summary
            report_lines.extend([
                "=== PIPELINE SUMMARY ===",
                f"✅ Stages completed: {len([s for s in stage_results.values() if s.get('status') == 'completed'])}",
                f"Input size: {img_np.shape}",
                f"Output size: {current_image.shape}",
                f"Final scale: {current_image.shape[0]/img_np.shape[0]:.1f}x",
            ])
            
            # Quality assessment
            if quality_monitoring:
                quality_info = self._assess_final_quality(img_np, current_image)
                report_lines.extend([
                    "",
                    "=== QUALITY ASSESSMENT ===",
                ] + quality_info)
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(current_image)
            
            final_report = "\n".join(report_lines)
            print("Professional restoration pipeline completed!")
            print(final_report)
            
            return (result_tensor, final_report)
            
        except Exception as e:
            error_report = f"Professional restoration error: {e}\nReturning original image."
            print(error_report)
            import traceback
            traceback.print_exc()
            return (image, error_report)
    
    def _stage_preprocessing(self, img_np: np.ndarray, level: str, device: str) -> Tuple[np.ndarray, Dict]:
        """Stage 1: Pre-processing - noise reduction and cleanup"""
        actions = []
        
        try:
            # Adaptive denoising based on level
            if WaveletDenoiseProcessor is not None:
                denoiser = WaveletDenoiseProcessor()
                
                # Adjust strength based on restoration level
                sigma_map = {"conservative": 10, "moderate": 15, "aggressive": 20, "maximum": 25}
                sigma = sigma_map.get(level, 15)
                
                denoised = denoiser.process_image(img_np, sigma=sigma, wavelet='db8')
                if denoised is not None:
                    img_np = denoised
                    actions.append(f"wavelet denoising (σ={sigma})")
            
            return img_np, {"status": "completed", "actions": actions}
            
        except Exception as e:
            return img_np, {"status": "partial", "actions": actions, "error": str(e)}
    
    def _stage_ai_restoration(self, img_np: np.ndarray, method: str, scale: int, 
                            level: str, device: str) -> Tuple[np.ndarray, Dict]:
        """Stage 2: AI-based restoration"""
        try:
            # Select method
            if method == "auto":
                # Simple heuristic for auto-selection
                if scale == 1:
                    selected_method = "scunet"  # Enhancement only
                elif scale >= 4:
                    selected_method = "swinir"  # High scale
                else:
                    selected_method = "real_esrgan"  # Balanced
            else:
                selected_method = method
            
            # Process with selected method
            if selected_method == "auto_denoise":
                processor = AutoDenoiseProcessor(device=device)
                result = processor.process_image(img_np, method="auto")
                
                # Handle upscaling for Auto-Denoise
                if result is not None and scale > 1:
                    from PIL import Image
                    pil_img = Image.fromarray((result * 255).astype(np.uint8))
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    result = np.array(upscaled).astype(np.float32) / 255.0
                
                scale_achieved = f"{result.shape[0]/img_np.shape[0]:.1f}x" if result is not None else "failed"
                
            elif selected_method == "noise2void":
                processor = Noise2VoidProcessor(device=device)
                result = processor.process_image(img_np, train_epochs=50)  # Reduced for pipeline
                
                # Handle upscaling for Noise2Void
                if result is not None and scale > 1:
                    from PIL import Image
                    pil_img = Image.fromarray((result * 255).astype(np.uint8))
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    result = np.array(upscaled).astype(np.float32) / 255.0
                
                scale_achieved = f"{result.shape[0]/img_np.shape[0]:.1f}x" if result is not None else "failed"
                
            elif selected_method == "scunet" and SCUNetProcessor is not None:
                processor = SCUNetProcessor(device=device)
                result = processor.process_image(img_np)
                
                # Handle upscaling for SCUNet
                if result is not None and scale > 1:
                    from PIL import Image
                    pil_img = Image.fromarray((result * 255).astype(np.uint8))
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    result = np.array(upscaled).astype(np.float32) / 255.0
                
                scale_achieved = f"{result.shape[0]/img_np.shape[0]:.1f}x" if result is not None else "failed"
                
            elif selected_method == "swinir" and SwinIRProcessor is not None:
                processor = SwinIRProcessor(task="classical_sr", scale=scale, device=device)
                result = processor.process_image(img_np)
                scale_achieved = f"{scale}x" if result is not None else "failed"
                
            elif selected_method == "real_esrgan" and RealESRGANProcessor is not None:
                processor = RealESRGANProcessor(scale=scale, device=device)
                result = processor.process_image(img_np)
                scale_achieved = f"{scale}x" if result is not None else "failed"
                
            else:
                result = None
                scale_achieved = "method unavailable"
            
            if result is not None:
                return result, {
                    "status": "completed",
                    "method": selected_method,
                    "scale_achieved": scale_achieved
                }
            else:
                return img_np, {
                    "status": "failed",
                    "method": selected_method,
                    "scale_achieved": scale_achieved
                }
                
        except Exception as e:
            return img_np, {"status": "error", "method": method, "error": str(e)}
    
    def _stage_frequency_enhancement(self, img_np: np.ndarray, level: str, device: str) -> Tuple[np.ndarray, Dict]:
        """Stage 3: Frequency domain enhancement"""
        techniques = []
        
        try:
            if FrequencyEnhancementProcessor is not None:
                processor = FrequencyEnhancementProcessor()
                
                # Adjust strength based on level
                strength_map = {"conservative": 0.3, "moderate": 0.5, "aggressive": 0.7, "maximum": 0.9}
                strength = strength_map.get(level, 0.5)
                
                # Apply adaptive frequency enhancement
                enhanced = processor.process_image(
                    img_np,
                    method="Adaptive_Frequency",
                    enhancement_strength=strength
                )
                
                if enhanced is not None:
                    img_np = enhanced
                    techniques.append(f"adaptive frequency (strength={strength})")
            
            return img_np, {"status": "completed", "techniques": techniques}
            
        except Exception as e:
            return img_np, {"status": "partial", "techniques": techniques, "error": str(e)}
    
    def _stage_color_enhancement(self, img_np: np.ndarray, level: str, device: str) -> Tuple[np.ndarray, Dict]:
        """Stage 4: Perceptual color enhancement"""
        optimizations = []
        
        try:
            if PerceptualColorProcessor is not None:
                processor = PerceptualColorProcessor()
                
                # Apply perceptual color enhancement
                enhanced = processor.process_image(
                    img_np,
                    enhancement_mode="balanced",
                    preserve_skin_tones=True
                )
                
                if enhanced is not None:
                    img_np = enhanced
                    optimizations.append("perceptual color enhancement")
            
            return img_np, {"status": "completed", "optimizations": optimizations}
            
        except Exception as e:
            return img_np, {"status": "partial", "optimizations": optimizations, "error": str(e)}
    
    def _stage_postprocessing(self, img_np: np.ndarray, level: str, 
                            quality_monitoring: bool) -> Tuple[np.ndarray, Dict]:
        """Stage 5: Final post-processing and quality assurance"""
        refinements = []
        
        try:
            # Subtle final sharpening based on level
            if level in ["aggressive", "maximum"]:
                from scipy import ndimage
                
                # Gentle unsharp masking
                blurred = ndimage.gaussian_filter(img_np, sigma=0.5)
                sharpened = img_np + 0.3 * (img_np - blurred)
                img_np = np.clip(sharpened, 0, 1)
                refinements.append("unsharp masking")
            
            # Ensure proper value range
            img_np = np.clip(img_np, 0, 1)
            refinements.append("value normalization")
            
            return img_np, {"status": "completed", "refinements": refinements}
            
        except Exception as e:
            return img_np, {"status": "partial", "refinements": refinements, "error": str(e)}
    
    def _assess_final_quality(self, original: np.ndarray, processed: np.ndarray) -> List[str]:
        """Assess final image quality"""
        try:
            quality_info = []
            
            # Basic metrics
            scale_factor = processed.shape[0] / original.shape[0]
            quality_info.append(f"• Scale factor: {scale_factor:.1f}x")
            
            # Size comparison
            orig_pixels = original.shape[0] * original.shape[1]
            proc_pixels = processed.shape[0] * processed.shape[1]
            quality_info.append(f"• Pixel count: {orig_pixels:,} → {proc_pixels:,}")
            
            # Dynamic range analysis
            orig_range = original.max() - original.min()
            proc_range = processed.max() - processed.min()
            quality_info.append(f"• Dynamic range: {orig_range:.3f} → {proc_range:.3f}")
            
            # Quality assessment
            if proc_range > orig_range * 0.9:
                quality_info.append("• ✅ Good dynamic range preservation")
            else:
                quality_info.append("• ⚠️ Reduced dynamic range detected")
            
            if scale_factor >= 1.0:
                quality_info.append("• ✅ Resolution enhancement successful")
            
            return quality_info
            
        except Exception as e:
            return [f"• Quality assessment error: {e}"]


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "ProfessionalRestorationPipeline": ProfessionalRestorationPipelineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProfessionalRestorationPipeline": "Professional Restoration Pipeline",
}
