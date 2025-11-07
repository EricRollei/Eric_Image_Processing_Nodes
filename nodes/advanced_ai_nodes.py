"""
Advanced AI/ML Enhancement Nodes for ComfyUI
Based on 2024-2025 research findings

Implements:
- Real-ESRGAN super-resolution and enhancement
- Perceptual color space processing (Oklab, Jzazbz)
- Advanced PSF modeling and diffraction recovery
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Dict, Any

# Import our base class
try:
    from Eric_Image_Processing_Nodes import BaseImageProcessingNode
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base_node import BaseImageProcessingNode

# Import our processors
from scripts.real_esrgan_processing import RealESRGANProcessor, get_realesrgan_presets
from scripts.perceptual_color_processing import PerceptualColorProcessor, get_perceptual_color_presets
from scripts.advanced_psf_modeling import AdvancedPSFProcessor, get_psf_presets
from scripts.sfhformer_processing import SFHformerProcessor, get_sfhformer_presets

class RealESRGANNode(BaseImageProcessingNode):
    """Real-ESRGAN super-resolution node"""
    
    def __init__(self):
        super().__init__()
        self.processor = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "realesr-general-x4v3"], {
                    "default": "RealESRGAN_x4plus",
                    "tooltip": "Model variant selection:\n• RealESRGAN_x4plus: Best for general photographs\n• RealESRGAN_x4plus_anime_6B: Optimized for anime/illustrations\n• realesr-general-x4v3: Lightweight version"
                }),
                "scale_factor": ("INT", {
                    "default": 4, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Scale factor for super-resolution:\n• 1: No scaling (enhancement only)\n• 2: 2x super-resolution\n• 4: 4x super-resolution (default)\n• 8: 8x super-resolution (experimental)"
                }),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 512, "min": 128, "max": 2048, "step": 64,
                    "tooltip": "Tile size for memory-efficient processing:\n• 256: Low memory usage\n• 512: Balanced performance\n• 1024: High quality (requires more VRAM)"
                }),
                "blend_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Blend factor with original image:\n• 0.0: Original image only\n• 0.5: 50% blend\n• 1.0: Full Real-ESRGAN output"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("enhanced_image", "processing_info")
    FUNCTION = "process_with_real_esrgan"
    CATEGORY = "Eric's Nodes/AI Enhancement"
    
    def process_with_real_esrgan(self, image: torch.Tensor, model_variant: str, 
                                scale_factor: int, tile_size: int = 512,
                                blend_factor: float = 1.0) -> Tuple[torch.Tensor, str]:
        """Process image with Real-ESRGAN"""
        
        try:
            # Initialize processor if needed
            if self.processor is None or self.processor.model_name != model_variant:
                self.processor = RealESRGANProcessor(model_variant)
                self.processor.tile_size = tile_size
                self.processor.scale = scale_factor
            
            # Convert tensor to numpy
            np_image = self.tensor_to_numpy(image)
            
            # Process with Real-ESRGAN
            enhanced = self.processor.process_image(np_image)
            
            # Handle scaling if different from model default
            if scale_factor != 4:  # Default model scale
                import cv2
                h, w = enhanced.shape[:2]
                target_h = int(np_image.shape[0] * scale_factor)
                target_w = int(np_image.shape[1] * scale_factor)
                if (h, w) != (target_h, target_w):
                    enhanced = cv2.resize(enhanced, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            # Blend with original if needed
            if blend_factor < 1.0:
                # Resize original to match enhanced size
                import cv2
                original_resized = cv2.resize(np_image, (enhanced.shape[1], enhanced.shape[0]), 
                                            interpolation=cv2.INTER_CUBIC)
                enhanced = cv2.addWeighted(original_resized, 1 - blend_factor, 
                                         enhanced, blend_factor, 0)
            
            # Get processing info
            info = self.processor.get_model_info()
            
            info_lines = [
                f"Real-ESRGAN Enhancement: {model_variant}",
                f"Scale Factor: {scale_factor}x",
                f"Tile Size: {tile_size}px",
                f"Blend Factor: {blend_factor:.2f}",
                f"Input Size: {np_image.shape[1]}x{np_image.shape[0]}",
                f"Output Size: {enhanced.shape[1]}x{enhanced.shape[0]}",
                f"Model Description: {info.get('description', 'N/A')}",
                f"Memory Usage: {info.get('memory_usage', 'N/A')}"
            ]
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(enhanced)
            
            return (result_tensor, '\n'.join(info_lines))
            
        except Exception as e:
            error_msg = f"Real-ESRGAN processing failed: {str(e)}"
            return (image, error_msg)


class PerceptualColorNode(BaseImageProcessingNode):
    """Perceptual color space processing node"""
    
    def __init__(self):
        super().__init__()
        self.processor = PerceptualColorProcessor()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_space": (["oklab", "jzazbz", "lab"], {
                    "default": "oklab",
                    "tooltip": "Perceptual color space:\n• oklab: Better perceptual uniformity than CIELAB\n• jzazbz: Superior hue prediction and HDR compatibility\n• lab: Standard CIELAB with luminance-chrominance separation"
                }),
                "enhancement_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Enhancement strength:\n• 0.0: No enhancement\n• 0.5: Subtle enhancement\n• 1.0: Moderate enhancement\n• 1.5: Strong enhancement"
                }),
            },
            "optional": {
                "preset": (["none", "oklab_subtle", "oklab_moderate", "oklab_strong", 
                          "jzazbz_professional", "lab_luminance"], {
                    "default": "none",
                    "tooltip": "Processing presets:\n• none: Use manual settings\n• oklab_subtle: Subtle Oklab enhancement\n• jzazbz_professional: Professional color grading\n• lab_luminance: Luminance-only enhancement"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("enhanced_image", "processing_info")
    FUNCTION = "process_with_perceptual_color"
    CATEGORY = "Eric's Nodes/Advanced Enhancement"
    
    def process_with_perceptual_color(self, image: torch.Tensor, color_space: str,
                                    enhancement_strength: float, preset: str = "none") -> Tuple[torch.Tensor, str]:
        """Process image with perceptual color enhancement"""
        
        try:
            # Convert tensor to numpy
            np_image = self.tensor_to_numpy(image)
            
            # Convert uint8 [0,255] to float32 [0,1] for perceptual color processing
            np_image_float = np_image.astype(np.float32) / 255.0
            
            # Apply preset if selected
            if preset != "none":
                presets = get_perceptual_color_presets()
                if preset in presets:
                    preset_config = presets[preset]
                    color_space = preset_config["color_space"]
                    enhancement_strength = preset_config["enhancement_strength"]
            
            # Process with perceptual color enhancement
            enhanced = self.processor.perceptual_contrast_enhancement(
                np_image_float, color_space, enhancement_strength
            )
            
            # Get processing info
            info = self.processor.get_processing_info(color_space)
            
            info_lines = [
                f"Perceptual Color Enhancement: {info.get('name', color_space.upper())}",
                f"Enhancement Strength: {enhancement_strength:.2f}",
                f"Preset: {preset if preset != 'none' else 'Manual'}",
                f"Description: {info.get('description', 'N/A')}",
                f"Advantages: {', '.join(info.get('advantages', []))}",
                f"Input Size: {np_image_float.shape[1]}x{np_image_float.shape[0]}",
                f"Output Size: {enhanced.shape[1]}x{enhanced.shape[0]}"
            ]
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(enhanced)
            
            return (result_tensor, '\n'.join(info_lines))
            
        except Exception as e:
            error_msg = f"Perceptual color processing failed: {str(e)}"
            return (image, error_msg)


class AdvancedPSFNode(BaseImageProcessingNode):
    """Advanced PSF modeling and diffraction recovery node"""
    
    def __init__(self):
        super().__init__()
        self.processor = AdvancedPSFProcessor()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "psf_type": (["airy_disk", "gibson_lanni", "born_wolf", "blind_estimation"], {
                    "default": "airy_disk",
                    "tooltip": "PSF modeling type:\n• airy_disk: Theoretical diffraction limit\n• gibson_lanni: Fluorescence microscopy with aberrations\n• born_wolf: Vector diffraction theory\n• blind_estimation: Estimate PSF from image"
                }),
                "psf_size": ("INT", {
                    "default": 15, "min": 5, "max": 63, "step": 2,
                    "tooltip": "PSF size (odd numbers only):\n• 7-15: Small PSF for subtle effects\n• 15-31: Medium PSF for balanced processing\n• 31-63: Large PSF for strong aberrations"
                }),
            },
            "optional": {
                "wavelength": ("FLOAT", {
                    "default": 550e-9, "min": 400e-9, "max": 700e-9, "step": 10e-9,
                    "tooltip": "Light wavelength in meters:\n• 400-500nm: Blue light\n• 500-600nm: Green light (default 550nm)\n• 600-700nm: Red light"
                }),
                "numerical_aperture": ("FLOAT", {
                    "default": 1.4, "min": 0.1, "max": 1.5, "step": 0.1,
                    "tooltip": "Numerical aperture:\n• 0.1-0.5: Low resolution\n• 0.5-1.0: Medium resolution\n• 1.0-1.5: High resolution (oil immersion)"
                }),
                "noise_ratio": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001,
                    "tooltip": "Noise ratio for deconvolution:\n• 0.001: Low noise\n• 0.01: Medium noise\n• 0.1: High noise"
                }),
                "preset": (["none", "fluorescence_40x", "fluorescence_100x", "confocal_high_na", "airy_theoretical"], {
                    "default": "none",
                    "tooltip": "PSF presets:\n• none: Use manual settings\n• fluorescence_40x: 40x fluorescence objective\n• fluorescence_100x: 100x oil immersion\n• confocal_high_na: High-NA confocal"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "processing_info")
    FUNCTION = "process_with_psf_modeling"
    CATEGORY = "Eric's Nodes/Advanced Enhancement"
    
    def process_with_psf_modeling(self, image: torch.Tensor, psf_type: str, psf_size: int,
                                wavelength: float = 550e-9, numerical_aperture: float = 1.4,
                                noise_ratio: float = 0.01, preset: str = "none") -> Tuple[torch.Tensor, str]:
        """Process image with advanced PSF modeling"""
        
        try:
            # Convert tensor to numpy
            np_image = self.tensor_to_numpy(image)
            
            # Convert uint8 [0,255] to float32 [0,1] for processing
            np_image_float = np_image.astype(np.float32) / 255.0
            
            # Convert to grayscale for PSF processing
            if len(np_image_float.shape) == 3:
                gray_image = cv2.cvtColor(np_image_float, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = np_image_float
            
            # Apply preset if selected
            params = {
                'wavelength': wavelength,
                'numerical_aperture': numerical_aperture,
                'noise_ratio': noise_ratio
            }
            
            if preset != "none":
                presets = get_psf_presets()
                if preset in presets:
                    preset_config = presets[preset]
                    psf_type = preset_config.get("type", psf_type)
                    params.update({k: v for k, v in preset_config.items() if k != "type" and k != "description"})
            
            # Generate PSF and process
            if psf_type == "airy_disk":
                psf = self.processor.airy_disk_psf(psf_size, **params)
                processed = self.processor._wiener_deconvolution(gray_image, psf, params['noise_ratio'])
                
            elif psf_type == "gibson_lanni":
                psf = self.processor.gibson_lanni_psf(psf_size, **params)
                processed = self.processor._wiener_deconvolution(gray_image, psf, params['noise_ratio'])
                
            elif psf_type == "born_wolf":
                psf = self.processor.born_wolf_psf(psf_size, **params)
                processed = self.processor._wiener_deconvolution(gray_image, psf, params['noise_ratio'])
                
            elif psf_type == "blind_estimation":
                psf, processed = self.processor.blind_psf_estimation(
                    gray_image, psf_size, 
                    num_iterations=10, 
                    regularization=params['noise_ratio']
                )
            
            # Convert back to RGB if needed
            if len(np_image_float.shape) == 3:
                processed_rgb = np.stack([processed] * 3, axis=2)
            else:
                processed_rgb = processed
            
            # Get processing info
            info = self.processor.get_psf_info(psf_type)
            
            info_lines = [
                f"PSF Processing: {info.get('name', psf_type.upper())}",
                f"PSF Size: {psf_size}x{psf_size}",
                f"Wavelength: {wavelength*1e9:.0f}nm",
                f"Numerical Aperture: {numerical_aperture:.2f}",
                f"Noise Ratio: {noise_ratio:.4f}",
                f"Preset: {preset if preset != 'none' else 'Manual'}",
                f"Description: {info.get('description', 'N/A')}",
                f"Formula: {info.get('formula', 'N/A')}",
                f"Input Size: {np_image_float.shape[1]}x{np_image_float.shape[0]}",
                f"Output Size: {processed_rgb.shape[1]}x{processed_rgb.shape[0]}"
            ]
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(processed_rgb)
            
            return (result_tensor, '\n'.join(info_lines))
            
        except Exception as e:
            error_msg = f"PSF processing failed: {str(e)}"
            return (image, error_msg)


class SFHformerNode(BaseImageProcessingNode):
    """SFHformer dual-domain enhancement node"""
    
    def __init__(self):
        super().__init__()
        self.processor = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["restoration", "denoising", "sharpening", "high_quality"], {
                    "default": "restoration",
                    "tooltip": "Processing preset:\n• restoration: Balanced restoration for general images\n• denoising: Strong denoising with detail preservation\n• sharpening: Detail enhancement and sharpening\n• high_quality: High-quality processing (requires more VRAM)"
                }),
                "domain_blend": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Blend between spatial and frequency domain processing:\n• 0.0: Frequency domain only\n• 0.5: Balanced blend\n• 1.0: Spatial domain only"
                }),
            },
            "optional": {
                "sharpening_strength": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Sharpening strength for detail enhancement"
                }),
                "bilateral_sigma": ("FLOAT", {
                    "default": 75.0, "min": 10.0, "max": 150.0, "step": 5.0,
                    "tooltip": "Bilateral filter sigma for noise reduction"
                }),
                "use_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use neural network model if available:\n• True: Use SFHformer model\n• False: Use fallback dual-domain processing"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("enhanced_image", "processing_info")
    FUNCTION = "process_with_sfhformer"
    CATEGORY = "Eric's Nodes/AI Enhancement"
    
    def process_with_sfhformer(self, image: torch.Tensor, preset: str, domain_blend: float,
                              sharpening_strength: float = 0.4, bilateral_sigma: float = 75.0,
                              use_model: bool = True) -> Tuple[torch.Tensor, str]:
        """Process image with SFHformer dual-domain enhancement"""
        
        try:
            # Initialize processor if needed
            if self.processor is None:
                self.processor = SFHformerProcessor()
            
            # Convert tensor to numpy
            np_image = self.tensor_to_numpy(image)
            
            # Get preset configuration
            presets = get_sfhformer_presets()
            config = presets.get(preset, {})
            
            # Override with manual settings
            config.update({
                'domain_blend': domain_blend,
                'sharpening_strength': sharpening_strength,
                'bilateral_sigma': bilateral_sigma
            })
            
            # Process with SFHformer
            if use_model:
                enhanced = self.processor.process_image(np_image, **config)
            else:
                enhanced = self.processor._fallback_dual_domain_processing(np_image, **config)
            
            # Get processing info
            info = self.processor.get_model_info()
            
            info_lines = [
                f"SFHformer Enhancement: {preset.upper()}",
                f"Domain Blend: {domain_blend:.2f} (0=Freq, 1=Spatial)",
                f"Sharpening Strength: {sharpening_strength:.2f}",
                f"Bilateral Sigma: {bilateral_sigma:.1f}",
                f"Use Model: {'Yes' if use_model else 'No (Fallback)'}",
                f"Model Loaded: {'Yes' if info['model_loaded'] else 'No'}",
                f"Architecture: {info['architecture']}",
                f"Input Size: {np_image.shape[1]}x{np_image.shape[0]}",
                f"Output Size: {enhanced.shape[1]}x{enhanced.shape[0]}",
                f"Features: {', '.join(info['features'][:2])}"
            ]
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(enhanced)
            
            return (result_tensor, '\n'.join(info_lines))
            
        except Exception as e:
            error_msg = f"SFHformer processing failed: {str(e)}"
            return (image, error_msg)


class AIEnhancementBatchNode(BaseImageProcessingNode):
    """Batch processing node for AI enhancement methods"""
    
    def __init__(self):
        super().__init__()
        self.processors = {}
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enhancement_method": (["real_esrgan", "perceptual_color", "combined"], {
                    "default": "real_esrgan",
                    "tooltip": "Enhancement method:\n• real_esrgan: AI super-resolution\n• perceptual_color: Color space enhancement\n• combined: Both methods in sequence"
                }),
                "batch_size": ("INT", {
                    "default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Batch processing size:\n• 1: Single image processing\n• 4: Small batch (recommended)\n• 8-16: Large batch (requires more VRAM)"
                }),
            },
            "optional": {
                "real_esrgan_model": (["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "realesr-general-x4v3"], {
                    "default": "RealESRGAN_x4plus"
                }),
                "color_space": (["oklab", "jzazbz", "lab"], {
                    "default": "oklab"
                }),
                "enhancement_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("enhanced_images", "processing_info")
    FUNCTION = "process_batch_ai_enhancement"
    CATEGORY = "Eric's Nodes/AI Enhancement"
    
    def process_batch_ai_enhancement(self, images: torch.Tensor, enhancement_method: str,
                                   batch_size: int, real_esrgan_model: str = "RealESRGAN_x4plus",
                                   color_space: str = "oklab", enhancement_strength: float = 0.7) -> Tuple[torch.Tensor, str]:
        """Process batch of images with AI enhancement"""
        
        try:
            # Initialize processors
            if enhancement_method in ["real_esrgan", "combined"]:
                if "real_esrgan" not in self.processors:
                    self.processors["real_esrgan"] = RealESRGANProcessor(real_esrgan_model)
            
            if enhancement_method in ["perceptual_color", "combined"]:
                if "perceptual_color" not in self.processors:
                    self.processors["perceptual_color"] = PerceptualColorProcessor()
            
            # Process images in batches
            num_images = images.shape[0]
            processed_images = []
            processing_stats = []
            
            for i in range(0, num_images, batch_size):
                batch_end = min(i + batch_size, num_images)
                batch_images = images[i:batch_end]
                
                for j in range(batch_images.shape[0]):
                    # Convert to numpy
                    np_image = self.tensor_to_numpy(batch_images[j:j+1])
                    np_image = np_image.squeeze(0)
                    
                    # Apply enhancement
                    if enhancement_method == "real_esrgan":
                        enhanced = self.processors["real_esrgan"].process_image(np_image)
                    elif enhancement_method == "perceptual_color":
                        enhanced = self.processors["perceptual_color"].perceptual_contrast_enhancement(
                            np_image, color_space, enhancement_strength
                        )
                    elif enhancement_method == "combined":
                        # Apply Real-ESRGAN first
                        enhanced = self.processors["real_esrgan"].process_image(np_image)
                        # Then apply perceptual color enhancement
                        enhanced = self.processors["perceptual_color"].perceptual_contrast_enhancement(
                            enhanced, color_space, enhancement_strength
                        )
                    
                    processed_images.append(enhanced)
                    processing_stats.append({
                        'index': i + j,
                        'input_size': f"{np_image.shape[1]}x{np_image.shape[0]}",
                        'output_size': f"{enhanced.shape[1]}x{enhanced.shape[0]}"
                    })
            
            # Convert back to tensor
            result_tensors = []
            for enhanced in processed_images:
                result_tensors.append(self.numpy_to_tensor(enhanced))
            
            result_tensor = torch.cat(result_tensors, dim=0)
            
            # Create processing info
            info_lines = [
                f"AI Enhancement Batch Processing: {enhancement_method.upper()}",
                f"Total Images: {num_images}",
                f"Batch Size: {batch_size}",
                f"Enhancement Method: {enhancement_method}",
            ]
            
            if enhancement_method in ["real_esrgan", "combined"]:
                info_lines.append(f"Real-ESRGAN Model: {real_esrgan_model}")
            
            if enhancement_method in ["perceptual_color", "combined"]:
                info_lines.append(f"Color Space: {color_space}")
                info_lines.append(f"Enhancement Strength: {enhancement_strength:.2f}")
            
            info_lines.append(f"Processing Stats:")
            for stat in processing_stats[:5]:  # Show first 5
                info_lines.append(f"  Image {stat['index']}: {stat['input_size']} → {stat['output_size']}")
            
            if len(processing_stats) > 5:
                info_lines.append(f"  ... and {len(processing_stats) - 5} more images")
            
            return (result_tensor, '\n'.join(info_lines))
            
        except Exception as e:
            error_msg = f"Batch AI enhancement failed: {str(e)}"
            return (images, error_msg)


# Node mappings for ComfyUI registration
ADVANCED_AI_MAPPINGS = {
    "RealESRGANNode": RealESRGANNode,
    "PerceptualColorNode": PerceptualColorNode,
    "AdvancedPSFNode": AdvancedPSFNode,
    "SFHformerNode": SFHformerNode,
    "AIEnhancementBatchNode": AIEnhancementBatchNode,
}

ADVANCED_AI_DISPLAY = {
    "RealESRGANNode": "Real-ESRGAN Super-Resolution",
    "PerceptualColorNode": "Perceptual Color Enhancement",
    "AdvancedPSFNode": "Advanced PSF Modeling",
    "SFHformerNode": "SFHformer Dual-Domain Enhancement",
    "AIEnhancementBatchNode": "AI Enhancement Batch Processing",
}
