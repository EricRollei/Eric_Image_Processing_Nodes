"""
Smart Workflow Node - Intelligent Image Processing Selection
Automatically analyzes images and selects optimal processing methods
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List

# Import available processors
try:
    from ..base_node import BaseImageProcessingNode
except ImportError:
    # Fallback for direct testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode

# Import processors with fallback pattern
try:
    from Eric_Image_Processing_Nodes import (
        SCUNetProcessor,
        SwinIRProcessor,
        RealESRGANProcessor
    )
    # Note: ImageQualityAnalyzer not available (script doesn't exist)
    # from ..scripts.image_analysis import ImageQualityAnalyzer
except ImportError:
    try:
        from ..scripts.scunet_processing import SCUNetProcessor
        from ..scripts.swinir_processing import SwinIRProcessor
        from ..scripts.real_esrgan_processing import RealESRGANProcessor
        # from ..scripts.image_analysis import ImageQualityAnalyzer
    except ImportError:
        try:
            # Final fallback for direct execution
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from scripts.scunet_processing import SCUNetProcessor
            from scripts.swinir_processing import SwinIRProcessor
            from scripts.real_esrgan_processing import RealESRGANProcessor
        except ImportError:
            print("Warning: Some processors not available for smart workflow")
            SCUNetProcessor = None
            SwinIRProcessor = None
            RealESRGANProcessor = None
ImageQualityAnalyzer = None


class SmartWorkflowNode(BaseImageProcessingNode):
    """
    Smart Workflow - Intelligent Image Processing Selection
    
    Automatically analyzes your image and selects the optimal processing method:
    
    AI PROCESSORS:
    - SCUNet: Best for realistic image restoration and blind denoising
    - SwinIR: Excellent for classical super-resolution and clean images
    - Real-ESRGAN: Great for real-world images with complex degradation
    
    ANALYSIS FEATURES:
    - Noise level detection
    - Blur assessment
    - Compression artifact identification
    - Resolution analysis
    - Content type recognition
    
    SMART SELECTION:
    - High noise → SCUNet (blind denoising specialist)
    - Clean images → SwinIR (transformer precision)
    - Real-world photos → Real-ESRGAN (practical restoration)
    - Mixed issues → Multi-stage processing
    
    Perfect for:
    - Users unsure which method to use
    - Batch processing with varied image types
    - Professional workflows requiring consistent results
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_scale": ([1, 2, 3, 4], {
                    "default": 2,
                    "tooltip": "Desired upscaling factor:\n• 1: Enhancement only (no upscaling)\n• 2-4: Super-resolution levels"
                }),
                "priority": (["quality", "speed", "balanced"], {
                    "default": "balanced",
                    "tooltip": "Processing priority:\n• quality: Best possible results (slower)\n• speed: Fast processing (good quality)\n• balanced: Optimal speed/quality ratio"
                }),
            },
            "optional": {
                "analysis_detail": (["basic", "detailed", "comprehensive"], {
                    "default": "detailed",
                    "tooltip": "Image analysis depth:\n• basic: Quick assessment\n• detailed: Full quality analysis\n• comprehensive: Deep technical analysis"
                }),
                "allow_multi_stage": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow multi-stage processing for complex images"
                }),
                "manual_override": (["auto", "bm3d", "auto_denoise", "noise2void", "scunet", "swinir", "real_esrgan"], {
                    "default": "auto",
                    "tooltip": "Manual processor selection (overrides analysis)"
                }),
                "device_preference": (["auto", "cpu", "cuda"], {
                    "default": "auto",
                    "tooltip": "Processing device preference"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "analysis_report")
    FUNCTION = "smart_process"
    
    def smart_process(self, image: torch.Tensor, target_scale: int = 2, priority: str = "balanced",
                     analysis_detail: str = "detailed", allow_multi_stage: bool = True,
                     manual_override: str = "auto", device_preference: str = "auto") -> Tuple[torch.Tensor, str]:
        """
        Intelligently process image with optimal method selection
        """
        try:
            # Convert input - Use float32 conversion for better processing
            img_np = self._tensor_to_numpy_float(image)
            
            # Initialize analysis report
            report_lines = [
                "=== SMART WORKFLOW ANALYSIS ===",
                f"Input: {img_np.shape} pixels",
                f"Target scale: {target_scale}x",
                f"Priority: {priority}",
                "",
            ]
            
            # Analyze image quality
            analysis = self._analyze_image(img_np, analysis_detail)
            report_lines.extend(self._format_analysis(analysis))
            
            # Select optimal processor
            if manual_override != "auto":
                selected_processor = manual_override
                reason = f"Manual override: {manual_override}"
                use_multi_stage = False
            else:
                if allow_multi_stage:
                    pipeline, reason = self._select_multi_stage_pipeline(analysis, priority, target_scale)
                    selected_processor = " → ".join(pipeline)
                    use_multi_stage = True
                else:
                    selected_processor, reason = self._select_processor(analysis, priority, target_scale)
                    use_multi_stage = False
            
            report_lines.extend([
                "",
                "=== PROCESSOR SELECTION ===",
                f"Multi-stage: {'Enabled' if use_multi_stage else 'Disabled'}",
                f"Selected: {selected_processor.upper()}",
                f"Reason: {reason}",
                "",
            ])
            
            # Process with selected method
            device = self._get_device(device_preference)
            if use_multi_stage:
                result_np, processing_info = self._process_multi_stage(
                    img_np, pipeline, target_scale, analysis, device
                )
            else:
                result_np, processing_info = self._process_with_method(
                    img_np, selected_processor, target_scale, analysis, device
                )
            
            if result_np is not None:
                report_lines.extend([
                    "=== PROCESSING RESULTS ===",
                    f"Status: ✅ Success",
                    f"Method: {selected_processor.upper()}",
                    f"Input size: {img_np.shape}",
                    f"Output size: {result_np.shape}",
                    f"Scale achieved: {result_np.shape[0]/img_np.shape[0]:.1f}x",
                ])
                
                if processing_info:
                    report_lines.extend(["", "Processing details:"] + processing_info)
                
                # Convert back to tensor using float conversion
                result_tensor = self._numpy_to_tensor_float(result_np)
                
            else:
                report_lines.extend([
                    "=== PROCESSING RESULTS ===",
                    "Status: ❌ Failed",
                    "Fallback: Returning original image",
                ])
                result_tensor = image
            
            # Add recommendations
            if use_multi_stage:
                report_lines.extend(self._generate_multi_stage_recommendations(analysis, pipeline))
            else:
                report_lines.extend(self._generate_recommendations(analysis, selected_processor))
            
            final_report = "\n".join(report_lines)
            print("Smart workflow completed!")
            print(final_report)
            
            return (result_tensor, final_report)
            
        except Exception as e:
            error_report = f"Smart workflow error: {e}\nReturning original image."
            print(error_report)
            return (image, error_report)
    
    def _tensor_to_numpy_float(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert ComfyUI tensor to numpy float32 for processing
        
        Args:
            tensor: ComfyUI image tensor in format [N, H, W, C] with values 0-1
            
        Returns:
            numpy array in format [H, W, C] with values 0-1 (float32)
        """
        # Take first image if batch
        if len(tensor.shape) == 4:
            img = tensor[0]
        else:
            img = tensor
            
        # Convert to numpy and keep float32 0-1 range
        img_np = img.cpu().numpy().astype(np.float32)
        
        return img_np
    
    def _numpy_to_tensor_float(self, img_np: np.ndarray) -> torch.Tensor:
        """Convert numpy float32 array back to ComfyUI tensor format
        
        Args:
            img_np: numpy array in format [H, W, C] with values 0-1
            
        Returns:
            ComfyUI tensor in format [1, H, W, C] with values 0-1
        """
        # Ensure float32 and proper range
        if img_np.dtype != np.float32:
            img_np = img_np.astype(np.float32)
        
        # Clip to valid range
        img_np = np.clip(img_np, 0.0, 1.0)
            
        img_tensor = torch.from_numpy(img_np)
        
        # Add batch dimension if not present
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        return img_tensor

    def _get_device(self, device_preference: str) -> str:
        """Get the appropriate device for processing"""
        if device_preference == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        elif device_preference == "cuda":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("CUDA requested but not available, falling back to CPU")
                return "cpu"
        else:
            return "cpu"
    
    def _analyze_image(self, img_np: np.ndarray, detail_level: str) -> Dict:
        """Analyze image quality and characteristics"""
        try:
            if ImageQualityAnalyzer is None:
                # Basic fallback analysis
                return {
                    'noise_level': 'medium',
                    'blur_level': 'low',
                    'compression_artifacts': False,
                    'content_type': 'photo',
                    'resolution_category': 'standard'
                }
            
            analyzer = ImageQualityAnalyzer()
            
            if detail_level == "basic":
                return analyzer.quick_assessment(img_np)
            elif detail_level == "detailed":
                return analyzer.detailed_analysis(img_np)
            else:  # comprehensive
                return analyzer.comprehensive_analysis(img_np)
                
        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                'noise_level': 'medium',
                'blur_level': 'medium',
                'compression_artifacts': False,
                'content_type': 'photo',
                'resolution_category': 'standard'
            }
    
    def _format_analysis(self, analysis: Dict) -> List[str]:
        """Format analysis results for report"""
        lines = ["IMAGE ANALYSIS:"]
        
        for key, value in analysis.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    lines.append(f"• {key.replace('_', ' ').title()}: {value:.3f}")
                else:
                    lines.append(f"• {key.replace('_', ' ').title()}: {value}")
            else:
                lines.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        return lines
    
    def _select_processor(self, analysis: Dict, priority: str, target_scale: int) -> Tuple[str, str]:
        """Select optimal processor based on analysis"""
        
        # Extract key metrics
        noise_level = analysis.get('noise_level', 'medium')
        blur_level = analysis.get('blur_level', 'medium')
        compression_artifacts = analysis.get('compression_artifacts', False)
        content_type = analysis.get('content_type', 'photo')
        
        # Decision logic with Auto-Denoise and BM3D methods
        if noise_level in ['very_high'] or noise_level == 'unknown':
            return "auto_denoise", "Very high or unknown noise - Auto-Denoise with self-supervised learning"
        
        elif noise_level in ['high'] and target_scale == 1:
            return "bm3d", "High noise, no scaling - BM3D collaborative filtering for superior denoising"
        
        elif noise_level in ['medium'] and target_scale == 1 and priority == "quality":
            return "bm3d", "Medium noise, quality priority - BM3D provides excellent denoising"
        
        elif noise_level in ['high'] and target_scale > 1:
            return "scunet", "High noise with scaling - SCUNet excels at blind denoising + upscaling"
        
        elif compression_artifacts and target_scale <= 2:
            return "scunet", "JPEG artifacts detected - SCUNet handles compression well"
        
        elif content_type in ['graphics', 'text', 'line_art'] and target_scale > 1:
            return "swinir", "Clean/artificial content - SwinIR provides sharp upscaling"
        
        elif priority == "quality" and target_scale > 1:
            return "swinir", "Quality priority - SwinIR offers best transformer precision"
        
        elif priority == "speed":
            return "real_esrgan", "Speed priority - Real-ESRGAN is well-optimized"
        
        elif target_scale >= 4:
            return "swinir", "High upscaling factor - SwinIR handles large scales well"
        
        elif content_type == 'photo' and target_scale > 1:
            return "real_esrgan", "Real-world photo - Real-ESRGAN specializes in natural images"
        
        else:
            # Balanced default
            if target_scale == 1:
                return "scunet", "Enhancement only - SCUNet for restoration"
            else:
                return "swinir", "Balanced choice - SwinIR for general super-resolution"
    
    def _select_multi_stage_pipeline(self, analysis: Dict, priority: str, target_scale: int) -> Tuple[List[str], str]:
        """Select multi-stage processing pipeline based on analysis"""
        
        # Extract key metrics
        noise_level = analysis.get('noise_level', 'medium')
        blur_level = analysis.get('blur_level', 'medium')
        compression_artifacts = analysis.get('compression_artifacts', False)
        content_type = analysis.get('content_type', 'photo')
        
        pipeline = []
        reasons = []
        
        # Stage 1: Denoising (if needed)
        if noise_level in ['very_high']:
            pipeline.append("auto_denoise")
            reasons.append("Very high noise requires aggressive denoising")
        elif noise_level in ['high']:
            if priority == "quality":
                pipeline.append("bm3d")
                reasons.append("High noise, quality priority - BM3D denoising")
            else:
                pipeline.append("scunet")
                reasons.append("High noise - SCUNet restoration")
        elif noise_level in ['medium'] and priority == "quality":
            pipeline.append("bm3d")
            reasons.append("Medium noise, quality priority - BM3D denoising")
        
        # Stage 2: Compression artifact removal (if needed and not already handled)
        if compression_artifacts and "scunet" not in pipeline:
            pipeline.append("scunet")
            reasons.append("JPEG artifacts detected - SCUNet restoration")
        
        # Stage 3: Super-resolution (if needed)
        if target_scale > 1:
            if content_type in ['graphics', 'text', 'line_art']:
                pipeline.append("swinir")
                reasons.append("Clean/artificial content - SwinIR sharp upscaling")
            elif priority == "quality":
                pipeline.append("swinir")
                reasons.append("Quality priority - SwinIR transformer precision")
            elif priority == "speed":
                pipeline.append("real_esrgan")
                reasons.append("Speed priority - Real-ESRGAN optimization")
            elif target_scale >= 4:
                pipeline.append("swinir")
                reasons.append("High upscaling factor - SwinIR handles large scales")
            elif content_type == 'photo':
                pipeline.append("real_esrgan")
                reasons.append("Real-world photo - Real-ESRGAN specialization")
            else:
                pipeline.append("swinir")
                reasons.append("Balanced choice - SwinIR general super-resolution")
        
        # Stage 4: Sharpening (if image was noisy or blurry)
        if noise_level in ['medium', 'high', 'very_high'] or blur_level in ['medium', 'high', 'very_high']:
            pipeline.append("sharpen")
            reasons.append("Post-processing sharpening to restore detail")
        
        # If no pipeline was built, use single stage
        if not pipeline:
            if target_scale == 1:
                pipeline = ["scunet"]
                reasons = ["Enhancement only - SCUNet restoration"]
            else:
                pipeline = ["swinir"]
                reasons = ["Balanced choice - SwinIR general super-resolution"]
        
        reason = " → ".join(reasons)
        return pipeline, reason
    
    def _process_with_method(self, img_np: np.ndarray, method: str, scale: int, 
                           analysis: Dict, device: str) -> Tuple[Optional[np.ndarray], List[str]]:
        """Process image with selected method"""
        processing_info = []
        
        try:
            if method == "auto_denoise":
                try:
                    from Eric_Image_Processing_Nodes import AutoDenoiseProcessor
                except ImportError:
                    try:
                        from ..scripts.auto_denoise import AutoDenoiseProcessor
                    except ImportError:
                        processing_info.append("• Auto-Denoise processor not available")
                        return None, processing_info
                
                processor = AutoDenoiseProcessor(device=device)
                processing_info.append("• Using Auto-Denoise with intelligent method selection")
                
                result = processor.process_image(img_np, method="auto")
                if result is not None and scale > 1:
                    # Apply upscaling after denoising
                    from PIL import Image
                    pil_img = Image.fromarray((result * 255).astype(np.uint8))
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    result = np.array(upscaled).astype(np.float32) / 255.0
                    processing_info.append(f"• Applied {scale}x upscaling after auto-denoise")
                return result, processing_info
            
            elif method == "bm3d":
                try:
                    from Eric_Image_Processing_Nodes import BM3DProcessor
                except ImportError:
                    try:
                        from ..scripts.bm3d_denoise import BM3DProcessor
                    except ImportError:
                        processing_info.append("• BM3D processor not available")
                        return None, processing_info
                
                processor = BM3DProcessor()
                processing_info.append("• Using BM3D collaborative filtering denoising")
                
                # Select BM3D profile based on analysis
                noise_level = analysis.get('noise_level', 'medium')
                if noise_level == 'high':
                    profile = 'vn'  # Very noisy profile
                elif noise_level == 'low':
                    profile = 'high'  # High quality profile
                else:
                    profile = 'np'  # Normal profile
                
                result, info = processor.process_image(img_np, profile=profile)
                processing_info.append(f"• Applied BM3D profile: {profile}")
                
                if 'sigma_estimated' in info:
                    sigma = info['sigma_estimated']
                    if isinstance(sigma, list):
                        sigma_str = f"[{', '.join([f'{s:.3f}' for s in sigma])}]"
                    else:
                        sigma_str = f"{sigma:.3f}"
                    processing_info.append(f"• Estimated noise σ: {sigma_str}")
                
                if result is not None and scale > 1:
                    # Apply upscaling after denoising
                    from PIL import Image
                    pil_img = Image.fromarray((result * 255).astype(np.uint8))
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    result = np.array(upscaled).astype(np.float32) / 255.0
                    processing_info.append(f"• Applied {scale}x upscaling after BM3D")
                return result, processing_info
            
            elif method == "noise2void":
                try:
                    from Eric_Image_Processing_Nodes import Noise2VoidProcessor
                except ImportError:
                    try:
                        from ..scripts.noise2void import Noise2VoidProcessor
                    except ImportError:
                        processing_info.append("• Noise2Void processor not available")
                        return None, processing_info
                
                processor = Noise2VoidProcessor(device=device)
                processing_info.append("• Using Noise2Void self-supervised denoising")
                
                result = processor.process_image(img_np, train_epochs=50)  # Reduced for workflow
                if result is not None and scale > 1:
                    # Apply upscaling after denoising
                    from PIL import Image
                    pil_img = Image.fromarray((result * 255).astype(np.uint8))
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    result = np.array(upscaled).astype(np.float32) / 255.0
                    processing_info.append(f"• Applied {scale}x upscaling after Noise2Void")
                return result, processing_info
            
            elif method == "scunet" and SCUNetProcessor is not None:
                processor = SCUNetProcessor(device=device)
                processing_info.append("• Using SCUNet for realistic restoration")
                
                if scale > 1:
                    # SCUNet + upscaling
                    restored = processor.process_image(img_np)
                    if restored is not None and scale > 1:
                        # Simple upscaling after restoration
                        from PIL import Image
                        pil_img = Image.fromarray((restored * 255).astype(np.uint8))
                        new_size = (pil_img.width * scale, pil_img.height * scale)
                        upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                        result = np.array(upscaled).astype(np.float32) / 255.0
                        processing_info.append(f"• Applied {scale}x upscaling after restoration")
                        return result, processing_info
                    return restored, processing_info
                else:
                    return processor.process_image(img_np), processing_info
            
            elif method == "swinir" and SwinIRProcessor is not None:
                # Determine SwinIR task and parameters based on analysis
                noise_level_str = analysis.get('noise_level', 'low')
                
                if noise_level_str in ['high', 'very_high']:
                    task = "color_dn"
                    processing_info.append("• Using SwinIR color denoising mode")
                    # Map noise level to model variants
                    if noise_level_str == 'very_high':
                        model_variant = "heavy_noise"  # σ=50
                    else:
                        model_variant = "medium_noise"  # σ=25
                elif noise_level_str == 'medium':
                    task = "color_dn"
                    model_variant = "light_noise"  # σ=15
                    processing_info.append("• Using SwinIR color denoising mode (light)")
                else:
                    task = "classical_sr"
                    processing_info.append("• Using SwinIR classical super-resolution mode")
                    # Map scale to model variants
                    if scale == 2:
                        model_variant = "2x"
                    elif scale == 3:
                        model_variant = "3x"
                    elif scale == 4:
                        model_variant = "4x"
                    elif scale == 8:
                        model_variant = "8x"
                    else:
                        model_variant = "auto"
                
                # Create processor with new interface
                processor = SwinIRProcessor(task=task, device=device)
                
                # Parse model variant to get correct parameters
                scale_param, noise_param, jpeg_param = self._parse_swinir_variant(task, model_variant)
                processor.scale = scale_param
                processor.noise = noise_param
                processor.jpeg = jpeg_param
                
                result = processor.process_image(img_np)
                
                # Handle scaling for different tasks
                if result is not None:
                    # Validate result shape and fix if needed
                    if len(result.shape) == 4:
                        # Remove batch dimension if present
                        result = result[0]
                    
                    # Ensure result is valid numpy array with proper shape
                    if result.shape[0] == 1 and result.shape[1] == 1:
                        # Malformed result - use original image
                        processing_info.append("• Warning: SwinIR returned malformed result, using original")
                        result = img_np
                    
                    if task == "classical_sr":
                        # Classical SR models provide the exact scale requested
                        actual_scale = result.shape[0] / img_np.shape[0]
                        processing_info.append(f"• Achieved {actual_scale:.1f}x scaling with {model_variant}")
                    elif task == "color_dn":
                        # Denoising tasks don't upscale, apply scaling if needed
                        processing_info.append(f"• Applied {model_variant} denoising model")
                        if scale > 1:
                            from PIL import Image
                            # Convert to uint8 for PIL processing
                            result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
                            pil_img = Image.fromarray(result_uint8)
                            target_size = (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale))
                            upscaled = pil_img.resize(target_size, Image.Resampling.LANCZOS)
                            result = np.array(upscaled).astype(np.float32) / 255.0
                            processing_info.append(f"• Applied {scale}x upscaling after denoising")
                
                return result, processing_info
            
            elif method == "real_esrgan" and RealESRGANProcessor is not None:
                processor = RealESRGANProcessor(device=device)
                processing_info.append("• Using Real-ESRGAN for real-world restoration")
                return processor.process_image(img_np), processing_info
            
            else:
                processing_info.append(f"• {method} processor not available")
                return None, processing_info
            
        except Exception as e:
            import traceback
            processing_info.append(f"• Processing failed: {e}")
            processing_info.append(f"• Error details: {traceback.format_exc()}")
            print(f"Processing error in {method}: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            return None, processing_info
    
    def _generate_recommendations(self, analysis: Dict, selected_processor: str) -> List[str]:
        """Generate processing recommendations"""
        recommendations = [
            "",
            "=== RECOMMENDATIONS ===",
        ]
        
        noise_level = analysis.get('noise_level', 'medium')
        if noise_level in ['high', 'very_high']:
            recommendations.append("• Consider pre-processing with noise reduction")
        
        blur_level = analysis.get('blur_level', 'medium')
        if blur_level in ['high', 'very_high']:
            recommendations.append("• Image appears blurry - deconvolution might help")
        
        if analysis.get('compression_artifacts', False):
            recommendations.append("• JPEG artifacts detected - quality restoration recommended")
        
        content_type = analysis.get('content_type', 'photo')
        if content_type == 'graphics':
            recommendations.append("• Graphics/text content - consider specialized upscaling")
        
        # Method-specific tips
        if selected_processor == "scunet":
            recommendations.append("• SCUNet excels at realistic image restoration")
        elif selected_processor == "swinir":
            recommendations.append("• SwinIR provides excellent transformer-based processing")
        elif selected_processor == "real_esrgan":
            recommendations.append("• Real-ESRGAN optimized for natural photo enhancement")
        
        recommendations.append("• For best results, ensure adequate GPU memory")
        
        return recommendations
    
    def _generate_multi_stage_recommendations(self, analysis, pipeline):
        """Generate recommendations for multi-stage processing"""
        recommendations = [
            "",
            "=== MULTI-STAGE RECOMMENDATIONS ===",
        ]
        
        # Pipeline overview
        stage_names = {
            'auto_denoise': 'Auto-Denoise',
            'bm3d': 'BM3D Denoising',
            'noise2void': 'Noise2Void',
            'scunet': 'SCUNet Restoration',
            'swinir': 'SwinIR Processing',
            'real_esrgan': 'Real-ESRGAN',
            'sharpen': 'Sharpening'
        }
        
        pipeline_desc = " → ".join([stage_names.get(stage, stage.upper()) for stage in pipeline])
        recommendations.append(f"Pipeline: {pipeline_desc}")
        recommendations.append("")
        
        # Stage-specific recommendations
        for i, stage in enumerate(pipeline):
            if stage == 'auto_denoise':
                recommendations.append(f"• Stage {i+1}: Auto-Denoise will intelligently select denoising method")
            elif stage == 'bm3d':
                recommendations.append(f"• Stage {i+1}: BM3D provides superior collaborative filtering denoising")
            elif stage == 'noise2void':
                recommendations.append(f"• Stage {i+1}: Noise2Void uses self-supervised learning for denoising")
            elif stage == 'scunet':
                recommendations.append(f"• Stage {i+1}: SCUNet excels at realistic restoration and artifacts")
            elif stage == 'swinir':
                recommendations.append(f"• Stage {i+1}: SwinIR provides transformer-based super-resolution")
            elif stage == 'real_esrgan':
                recommendations.append(f"• Stage {i+1}: Real-ESRGAN optimized for natural photo enhancement")
            elif stage == 'sharpen':
                recommendations.append(f"• Stage {i+1}: Sharpening will enhance fine details and clarity")
        
        # Performance and quality notes
        recommendations.append("")
        if len(pipeline) > 2:
            recommendations.append("⚠️  Multi-stage processing takes longer but provides superior quality")
        
        # Analysis-based recommendations
        noise_level = analysis.get('noise_level', 'medium')
        if noise_level in ['high', 'very_high']:
            recommendations.append("• High noise detected - multi-stage approach recommended")
        
        if analysis.get('compression_artifacts', False):
            recommendations.append("• JPEG artifacts detected - restoration pipeline beneficial")
        
        recommendations.append("• GPU memory usage increases with pipeline complexity")
        
        return recommendations

    def _parse_swinir_variant(self, task: str, model_variant: str) -> Tuple[int, int, int]:
        """Parse SwinIR model variant to get scale, noise level, and jpeg quality parameters
        
        Args:
            task: The restoration task
            model_variant: Model variant selection string
            
        Returns:
            Tuple of (scale, noise_level, jpeg_quality)
        """
        # Default values
        scale = 2
        noise_level = 25
        jpeg_quality = 40
        
        if model_variant == "auto":
            # Auto selection based on task
            if task in ['classical_sr', 'lightweight_sr', 'real_sr']:
                scale = 4  # Default to 4x for super-resolution
            elif task in ['gray_dn', 'color_dn']:
                scale = 1  # No upscaling for denoising
                noise_level = 25  # Medium noise
            elif task == 'jpeg_car':
                scale = 1  # No upscaling for JPEG artifact removal
                jpeg_quality = 40  # Medium quality
        
        elif model_variant == "2x":
            scale = 2
        elif model_variant == "3x":
            scale = 3
        elif model_variant == "4x":
            scale = 4
        elif model_variant == "8x":
            scale = 8
        
        elif model_variant == "light_noise":
            scale = 1
            noise_level = 15  # σ=15 (light noise)
        elif model_variant == "medium_noise":
            scale = 1
            noise_level = 25  # σ=25 (medium noise)
        elif model_variant == "heavy_noise":
            scale = 1
            noise_level = 50  # σ=50 (heavy noise)
        
        elif model_variant == "jpeg_q10":
            scale = 1
            jpeg_quality = 10  # Heavy compression artifacts
        elif model_variant == "jpeg_q20":
            scale = 1
            jpeg_quality = 20
        elif model_variant == "jpeg_q30":
            scale = 1
            jpeg_quality = 30
        elif model_variant == "jpeg_q40":
            scale = 1
            jpeg_quality = 40  # Light compression artifacts
        
        return scale, noise_level, jpeg_quality
    
    def _process_multi_stage(self, img_np: np.ndarray, pipeline: List[str], target_scale: int, 
                           analysis: Dict, device: str) -> Tuple[Optional[np.ndarray], List[str]]:
        """Process image through multi-stage pipeline"""
        processing_info = []
        current_img = img_np.copy()
        
        processing_info.append(f"🔄 Multi-stage pipeline: {' → '.join(pipeline)}")
        
        for i, stage in enumerate(pipeline):
            stage_info = []
            processing_info.append(f"\n--- Stage {i+1}: {stage.upper()} ---")
            
            if stage == "sharpen":
                # Apply sharpening using unsharp mask
                current_img = self._apply_sharpening(current_img, stage_info)
                processing_info.extend(stage_info)
            else:
                # Use existing single-stage processing
                # For multi-stage, we only upscale in the final upscaling stage
                stage_scale = target_scale if (stage in ['swinir', 'real_esrgan'] and i == len(pipeline) - 1) else 1
                
                result, stage_info = self._process_with_method(
                    current_img, stage, stage_scale, analysis, device
                )
                
                if result is not None:
                    current_img = result
                    processing_info.extend(stage_info)
                else:
                    processing_info.extend(stage_info)
                    processing_info.append(f"⚠️  Stage {i+1} failed, continuing with previous result")
        
        # Final upscaling if not done in pipeline
        if target_scale > 1 and not any(stage in ['swinir', 'real_esrgan'] for stage in pipeline):
            processing_info.append(f"\n--- Final Stage: {target_scale}x UPSCALING ---")
            from PIL import Image
            result_uint8 = (np.clip(current_img, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(result_uint8)
            target_size = (int(img_np.shape[1] * target_scale), int(img_np.shape[0] * target_scale))
            upscaled = pil_img.resize(target_size, Image.Resampling.LANCZOS)
            current_img = np.array(upscaled).astype(np.float32) / 255.0
            processing_info.append(f"• Applied {target_scale}x Lanczos upscaling")
        
        return current_img, processing_info
    
    def _apply_sharpening(self, img_np: np.ndarray, info: List[str]) -> np.ndarray:
        """Apply unsharp mask sharpening to image"""
        try:
            from scipy import ndimage
            import cv2
            
            # Convert to proper format for processing
            if img_np.max() <= 1.0:
                img_work = (img_np * 255).astype(np.uint8)
            else:
                img_work = img_np.astype(np.uint8)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_work, (0, 0), 1.0)
            
            # Create unsharp mask
            unsharp_mask = cv2.addWeighted(img_work, 1.5, blurred, -0.5, 0)
            
            # Convert back to float32 [0,1] range
            result = unsharp_mask.astype(np.float32) / 255.0
            
            info.append("• Applied unsharp mask sharpening (radius=1.0, amount=1.5)")
            return result
            
        except ImportError:
            info.append("• Sharpening skipped (scipy/cv2 not available)")
            return img_np
        except Exception as e:
            info.append(f"• Sharpening failed: {e}")
            return img_np




# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SmartWorkflowNode": SmartWorkflowNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartWorkflowNode": "🧠 Smart Workflow",
}
