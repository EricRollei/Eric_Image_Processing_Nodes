"""
Advanced Sharpening Nodes for ComfyUI
Implementing cutting-edge sharpening techniques from 2024-2025 research
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# Import base node with fallback for different import contexts
try:
    from ..base_node import BaseImageProcessingNode
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode

# Try to import the processor - use fallback for circular import issues
try:
    from ..scripts.advanced_sharpening import AdvancedSharpeningProcessor
except ImportError:
    try:
        # Direct import fallback
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from scripts.advanced_sharpening import AdvancedSharpeningProcessor
    except ImportError:
        print("Warning: AdvancedSharpeningProcessor not available")
        AdvancedSharpeningProcessor = None

class AdvancedSharpeningNode(BaseImageProcessingNode):
    """Advanced sharpening with multiple sophisticated techniques"""
    
    CATEGORY = "Eric's Image Processing/Advanced Sharpening"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["smart", "hiraloam", "directional", "multiscale", "guided", "auto"], {
                    "default": "auto",
                    "tooltip": "Sharpening method:\n• auto: Intelligent selection\n• smart: Overshoot protection + adaptive radius\n• hiraloam: High radius, low amount technique\n• directional: Orientation-specific enhancement\n• multiscale: Laplacian pyramid approach\n• guided: Edge-preserving with feedback"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Overall sharpening strength"
                }),
            },
            "optional": {
                "radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Base sharpening radius (method-dependent)"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Edge threshold for selective sharpening"
                }),
                "overshoot_protection": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable overshoot detection and control"
                }),
                "luminance_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process only luminance to avoid color artifacts"
                }),
            }
        }
    
    def process_image(self, image: torch.Tensor, method: str, strength: float,
                     radius: float = 1.0, threshold: float = 0.1,
                     overshoot_protection: bool = True, luminance_only: bool = True) -> Tuple[torch.Tensor, str]:
        """Process image with advanced sharpening"""
        
        if AdvancedSharpeningProcessor is None:
            return (image, "❌ AdvancedSharpeningProcessor not available")
        
        try:
            # Convert tensor to numpy - keep in float32 [0,1] range for sharpening
            # Do NOT use tensor_to_numpy() as it converts to uint8 [0,255]
            img_np = image.cpu().numpy()  # Keep as float32 [0,1]
            
            # DEBUG: Print input data info
            print(f"\n=== Advanced Sharpening Debug ===")
            print(f"Input tensor shape: {image.shape}")
            print(f"Input tensor dtype: {image.dtype}")
            print(f"Numpy array shape: {img_np.shape}")
            print(f"Numpy array dtype: {img_np.dtype}")
            print(f"Numpy array range: [{img_np.min():.4f}, {img_np.max():.4f}]")
            print(f"Numpy is contiguous: {img_np.flags['C_CONTIGUOUS']}")
            print(f"Method: {method}")
            print(f"=================================\n")
            
            # Initialize processor
            processor = AdvancedSharpeningProcessor()
            
            # Process each image in batch
            processed_images = []
            processing_info = []
            
            for i in range(img_np.shape[0]):
                single_img = np.ascontiguousarray(img_np[i])
                
                # Apply sharpening based on method
                if method == 'smart':
                    result, info = processor.smart_sharpening(
                        single_img, strength=strength, radius=radius, 
                        threshold=threshold, overshoot_protection=overshoot_protection
                    )
                elif method == 'hiraloam':
                    result, info = processor.hiraloam_sharpening(
                        single_img, radius_ratio=radius*2, amount_ratio=strength*0.25
                    )
                elif method == 'directional':
                    result, info = processor.edge_directional_sharpening(
                        single_img, strength=strength, luminance_only=luminance_only
                    )
                elif method == 'multiscale':
                    result, info = processor.multiscale_laplacian_sharpening(
                        single_img, strength=strength
                    )
                elif method == 'guided':
                    result, info = processor.guided_filter_sharpening(
                        single_img, strength=strength, radius=int(radius*8)
                    )
                else:  # auto
                    result, info = processor.process_image(
                        single_img, method='auto', strength=strength, radius=radius
                    )
                
                if result is not None:
                    processed_images.append(result)
                    processing_info.append(info)
                else:
                    # Return original on failure
                    processed_images.append(single_img)
                    processing_info.append({'error': 'Processing failed'})
            
            # Convert back to tensor
            processed_tensor = self.numpy_to_tensor(np.stack(processed_images))
            
            # Create info string
            if processing_info and 'error' not in processing_info[0]:
                method_used = processing_info[0].get('method', method)
                if 'auto_selected' in processing_info[0]:
                    method_used += f" (auto-selected)"
                
                info_lines = [
                    f"🔧 Advanced Sharpening: {method_used}",
                    f"💪 Strength: {strength:.1f}",
                ]
                
                # Add method-specific info
                if 'edge_pixels_detected' in processing_info[0]:
                    edges = processing_info[0]['edge_pixels_detected']
                    info_lines.append(f"🔍 Edge pixels detected: {edges:,}")
                
                if 'frequency_bands' in processing_info[0]:
                    bands = processing_info[0]['frequency_bands']
                    info_lines.append(f"📊 Frequency bands: {bands}")
                
                if 'num_directions' in processing_info[0]:
                    dirs = processing_info[0]['num_directions']
                    info_lines.append(f"🧭 Directional filters: {dirs}")
                
                if 'psnr' in processing_info[0]:
                    psnr = processing_info[0]['psnr']
                    info_lines.append(f"📈 PSNR: {psnr:.2f} dB")
                
                info_str = "\n".join(info_lines)
            else:
                info_str = f"❌ Advanced sharpening failed"
            
            return (processed_tensor, info_str)
            
        except Exception as e:
            import traceback
            print("\n" + "="*80)
            print("FULL TRACEBACK FROM ADVANCED SHARPENING NODE:")
            print("="*80)
            traceback.print_exc()
            print("="*80 + "\n")
            return (image, f"❌ Advanced sharpening error: {str(e)}")

class SmartSharpeningNode(BaseImageProcessingNode):
    """Smart sharpening with overshoot detection and adaptive radius control"""
    
    CATEGORY = "Eric's Image Processing/Advanced Sharpening"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Sharpening strength"
                }),
                "radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Base sharpening radius"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Edge threshold for selective sharpening"
                }),
                "overshoot_protection": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable overshoot detection and control"
                }),
            }
        }
    
    def process_image(self, image: torch.Tensor, strength: float, radius: float,
                     threshold: float, overshoot_protection: bool) -> Tuple[torch.Tensor, str]:
        """Process image with smart sharpening"""
        
        if AdvancedSharpeningProcessor is None:
            return (image, "❌ AdvancedSharpeningProcessor not available")
        
        try:
            # Keep in float32 [0,1] range - do not convert to uint8
            img_np = image.cpu().numpy()
            processor = AdvancedSharpeningProcessor()
            
            processed_images = []
            processing_info = []
            
            for i in range(img_np.shape[0]):
                # CRITICAL: Ensure array slice is contiguous
                single_img = np.ascontiguousarray(img_np[i])
                result, info = processor.smart_sharpening(
                    single_img, strength=strength, radius=radius,
                    threshold=threshold, overshoot_protection=overshoot_protection
                )
                
                if result is not None:
                    processed_images.append(result)
                    processing_info.append(info)
                else:
                    processed_images.append(single_img)
                    processing_info.append({'error': 'Processing failed'})
            
            processed_tensor = self.numpy_to_tensor(np.stack(processed_images))
            
            if processing_info and 'error' not in processing_info[0]:
                info = processing_info[0]
                edges = info.get('edge_pixels_detected', 0)
                sig_edges = info.get('significant_edges', 0)
                radius_range = info.get('adaptive_radius_range', [radius, radius])
                
                info_str = (f"🧠 Smart Sharpening Applied\n"
                           f"💪 Strength: {strength:.1f}\n"
                           f"🔍 Edge pixels: {edges:,}\n"
                           f"⚡ Significant edges: {sig_edges:,}\n"
                           f"📏 Adaptive radius: {radius_range[0]:.2f}-{radius_range[1]:.2f}\n"
                           f"🛡️ Overshoot protection: {'On' if overshoot_protection else 'Off'}")
                
                if 'psnr' in info:
                    info_str += f"\n📈 PSNR: {info['psnr']:.2f} dB"
            else:
                info_str = "❌ Smart sharpening failed"
            
            return (processed_tensor, info_str)
            
        except Exception as e:
            return (image, f"❌ Smart sharpening error: {str(e)}")

class HiRaLoAmSharpeningNode(BaseImageProcessingNode):
    """High Radius Low Amount sharpening technique"""
    
    CATEGORY = "Eric's Image Processing/Advanced Sharpening"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius_ratio": ("FLOAT", {
                    "default": 4.0,
                    "min": 2.0,
                    "max": 6.0,
                    "step": 0.1,
                    "tooltip": "High radius multiplier (higher = more subtle)"
                }),
                "amount_ratio": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.1,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "Low amount multiplier (lower = more subtle)"
                }),
                "blur_type": (["gaussian", "bilateral", "mixed"], {
                    "default": "mixed",
                    "tooltip": "Type of blur kernel"
                }),
                "frequency_bands": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 5,
                    "tooltip": "Number of frequency bands to process"
                }),
            }
        }
    
    def process_image(self, image: torch.Tensor, radius_ratio: float, amount_ratio: float,
                     blur_type: str, frequency_bands: int) -> Tuple[torch.Tensor, str]:
        """Process image with HiRaLoAm sharpening"""
        
        if AdvancedSharpeningProcessor is None:
            return (image, "❌ AdvancedSharpeningProcessor not available")
        
        try:
            # Keep in float32 [0,1] range - do not convert to uint8
            img_np = image.cpu().numpy()
            processor = AdvancedSharpeningProcessor()
            
            processed_images = []
            processing_info = []
            
            for i in range(img_np.shape[0]):
                # CRITICAL: Ensure array slice is contiguous
                single_img = np.ascontiguousarray(img_np[i])
                result, info = processor.hiraloam_sharpening(
                    single_img, radius_ratio=radius_ratio, amount_ratio=amount_ratio,
                    blur_type=blur_type, frequency_bands=frequency_bands
                )
                
                if result is not None:
                    processed_images.append(result)
                    processing_info.append(info)
                else:
                    processed_images.append(single_img)
                    processing_info.append({'error': 'Processing failed'})
            
            processed_tensor = self.numpy_to_tensor(np.stack(processed_images))
            
            if processing_info and 'error' not in processing_info[0]:
                info = processing_info[0]
                
                info_str = (f"🎛️ HiRaLoAm Sharpening Applied\n"
                           f"📏 Radius ratio: {radius_ratio:.1f}x\n"
                           f"💪 Amount ratio: {amount_ratio:.2f}x\n"
                           f"🔄 Blur type: {blur_type}\n"
                           f"📊 Frequency bands: {frequency_bands}")
                
                # Show band details
                if 'processing_details' in info:
                    details = info['processing_details']
                    info_str += f"\n🎵 Band details:"
                    for detail in details[:3]:  # Show first 3 bands
                        band = detail['band']
                        radius = detail['radius']
                        amount = detail['amount']
                        info_str += f"\n  Band {band}: r={radius:.1f}, a={amount:.3f}"
            else:
                info_str = "❌ HiRaLoAm sharpening failed"
            
            return (processed_tensor, info_str)
            
        except Exception as e:
            return (image, f"❌ HiRaLoAm sharpening error: {str(e)}")

class EdgeDirectionalSharpeningNode(BaseImageProcessingNode):
    """Edge-directional sharpening with orientation-specific enhancement"""
    
    CATEGORY = "Eric's Image Processing/Advanced Sharpening"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sharpening strength"
                }),
                "num_directions": ([4, 8, 16], {
                    "default": 8,
                    "tooltip": "Number of directional filters"
                }),
                "luminance_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process only luminance to avoid color artifacts"
                }),
            }
        }
    
    def process_image(self, image: torch.Tensor, strength: float, num_directions: int,
                     luminance_only: bool) -> Tuple[torch.Tensor, str]:
        """Process image with edge-directional sharpening"""
        
        if AdvancedSharpeningProcessor is None:
            return (image, "❌ AdvancedSharpeningProcessor not available")
        
        try:
            # Keep in float32 [0,1] range - do not convert to uint8
            img_np = image.cpu().numpy()
            processor = AdvancedSharpeningProcessor()
            
            processed_images = []
            processing_info = []
            
            for i in range(img_np.shape[0]):
                # CRITICAL: Ensure array slice is contiguous
                single_img = np.ascontiguousarray(img_np[i])
                result, info = processor.edge_directional_sharpening(
                    single_img, strength=strength, num_directions=num_directions,
                    luminance_only=luminance_only
                )
                
                if result is not None:
                    processed_images.append(result)
                    processing_info.append(info)
                else:
                    processed_images.append(single_img)
                    processing_info.append({'error': 'Processing failed'})
            
            processed_tensor = self.numpy_to_tensor(np.stack(processed_images))
            
            if processing_info and 'error' not in processing_info[0]:
                info = processing_info[0]
                response_range = info.get('max_response_range', [0, 0])
                dominant_dirs = info.get('dominant_directions', [])
                
                info_str = (f"🧭 Edge-Directional Sharpening Applied\n"
                           f"💪 Strength: {strength:.1f}\n"
                           f"🔄 Directions: {num_directions}\n"
                           f"📊 Response range: {response_range[0]:.3f} to {response_range[1]:.3f}\n"
                           f"🎯 Dominant directions: {len(dominant_dirs)}\n"
                           f"🎨 Luminance only: {'Yes' if luminance_only else 'No'}")
            else:
                info_str = "❌ Edge-directional sharpening failed"
            
            return (processed_tensor, info_str)
            
        except Exception as e:
            return (image, f"❌ Edge-directional sharpening error: {str(e)}")

class MultiScaleLaplacianSharpeningNode(BaseImageProcessingNode):
    """Multi-scale Laplacian pyramid sharpening"""
    
    CATEGORY = "Eric's Image Processing/Advanced Sharpening"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Overall sharpening strength"
                }),
                "num_scales": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 6,
                    "tooltip": "Number of pyramid scales"
                }),
                "scale_progression": (["linear", "exponential", "custom"], {
                    "default": "exponential",
                    "tooltip": "How scales progress"
                }),
            },
            "optional": {
                "custom_scales": ("STRING", {
                    "default": "0.5,1.0,2.0,4.0",
                    "tooltip": "Custom scales (comma-separated)"
                }),
                "adaptive_scaling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use adaptive scaling factors"
                }),
            }
        }
    
    def process_image(self, image: torch.Tensor, strength: float, num_scales: int,
                     scale_progression: str, custom_scales: str = "0.5,1.0,2.0,4.0",
                     adaptive_scaling: bool = True) -> Tuple[torch.Tensor, str]:
        """Process image with multi-scale Laplacian sharpening"""
        
        if AdvancedSharpeningProcessor is None:
            return (image, "❌ AdvancedSharpeningProcessor not available")
        
        try:
            # Keep in float32 [0,1] range - do not convert to uint8
            img_np = image.cpu().numpy()
            processor = AdvancedSharpeningProcessor()
            
            # Generate scales based on progression type
            if scale_progression == "linear":
                scales = [0.5 + i * 1.0 for i in range(num_scales)]
            elif scale_progression == "exponential":
                scales = [0.5 * (2 ** i) for i in range(num_scales)]
            else:  # custom
                try:
                    scales = [float(s.strip()) for s in custom_scales.split(',')]
                    scales = scales[:num_scales]  # Limit to num_scales
                except:
                    scales = [0.5, 1.0, 2.0, 4.0][:num_scales]
            
            # Generate kernel sizes and scaling factors
            kernel_sizes = [3 + 2*i for i in range(num_scales)]
            if adaptive_scaling:
                scaling_factors = [0.2 + 0.2*i for i in range(num_scales)]
            else:
                scaling_factors = [0.5] * num_scales
            
            processed_images = []
            processing_info = []
            
            for i in range(img_np.shape[0]):
                # CRITICAL: Ensure array slice is contiguous
                single_img = np.ascontiguousarray(img_np[i])
                result, info = processor.multiscale_laplacian_sharpening(
                    single_img, strength=strength, scales=scales,
                    kernel_sizes=kernel_sizes, scaling_factors=scaling_factors
                )
                
                if result is not None:
                    processed_images.append(result)
                    processing_info.append(info)
                else:
                    processed_images.append(single_img)
                    processing_info.append({'error': 'Processing failed'})
            
            processed_tensor = self.numpy_to_tensor(np.stack(processed_images))
            
            if processing_info and 'error' not in processing_info[0]:
                info = processing_info[0]
                
                info_str = (f"🏔️ Multi-Scale Laplacian Sharpening Applied\n"
                           f"💪 Strength: {strength:.1f}\n"
                           f"📊 Scales: {num_scales} ({scale_progression})\n"
                           f"🎛️ Scale values: {', '.join([f'{s:.1f}' for s in scales])}\n"
                           f"🔧 Adaptive scaling: {'Yes' if adaptive_scaling else 'No'}")
                
                # Show scale details
                if 'scale_details' in info:
                    details = info['scale_details']
                    info_str += f"\n📈 Scale details:"
                    for detail in details[:3]:  # Show first 3 scales
                        scale = detail['scale']
                        factor = detail['scaling_factor']
                        info_str += f"\n  Scale {scale:.1f}: factor={factor:.2f}"
            else:
                info_str = "❌ Multi-scale Laplacian sharpening failed"
            
            return (processed_tensor, info_str)
            
        except Exception as e:
            return (image, f"❌ Multi-scale Laplacian sharpening error: {str(e)}")

class GuidedFilterSharpeningNode(BaseImageProcessingNode):
    """Guided filtering with edge-preserving smoothing and sharpening feedback"""
    
    CATEGORY = "Eric's Image Processing/Advanced Sharpening"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sharpening strength"
                }),
                "radius": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 16,
                    "tooltip": "Guided filter radius"
                }),
                "epsilon": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Regularization parameter (lower = more edge-preserving)"
                }),
            }
        }
    
    def process_image(self, image: torch.Tensor, strength: float, radius: int,
                     epsilon: float) -> Tuple[torch.Tensor, str]:
        """Process image with guided filter sharpening"""
        
        if AdvancedSharpeningProcessor is None:
            return (image, "❌ AdvancedSharpeningProcessor not available")
        
        try:
            # Keep in float32 [0,1] range - do not convert to uint8
            img_np = image.cpu().numpy()
            processor = AdvancedSharpeningProcessor()
            
            processed_images = []
            processing_info = []
            
            for i in range(img_np.shape[0]):
                # CRITICAL: Ensure array slice is contiguous
                single_img = np.ascontiguousarray(img_np[i])
                result, info = processor.guided_filter_sharpening(
                    single_img, strength=strength, radius=radius, epsilon=epsilon
                )
                
                if result is not None:
                    processed_images.append(result)
                    processing_info.append(info)
                else:
                    processed_images.append(single_img)
                    processing_info.append({'error': 'Processing failed'})
            
            processed_tensor = self.numpy_to_tensor(np.stack(processed_images))
            
            if processing_info and 'error' not in processing_info[0]:
                info = processing_info[0]
                detail_range = info.get('detail_range', [0, 0])
                preserved_range = info.get('preserved_detail_range', [0, 0])
                
                info_str = (f"🎯 Guided Filter Sharpening Applied\n"
                           f"💪 Strength: {strength:.1f}\n"
                           f"📏 Radius: {radius}\n"
                           f"🔧 Epsilon: {epsilon:.3f}\n"
                           f"📊 Detail range: {detail_range[0]:.3f} to {detail_range[1]:.3f}\n"
                           f"🛡️ Preserved range: {preserved_range[0]:.3f} to {preserved_range[1]:.3f}")
            else:
                info_str = "❌ Guided filter sharpening failed"
            
            return (processed_tensor, info_str)
            
        except Exception as e:
            return (image, f"❌ Guided filter sharpening error: {str(e)}")

# Node mappings for ComfyUI registration
ADVANCED_SHARPENING_MAPPINGS = {
    "AdvancedSharpeningNode": AdvancedSharpeningNode,
    "SmartSharpeningNode": SmartSharpeningNode,
    "HiRaLoAmSharpeningNode": HiRaLoAmSharpeningNode,
    "EdgeDirectionalSharpeningNode": EdgeDirectionalSharpeningNode,
    "MultiScaleLaplacianSharpeningNode": MultiScaleLaplacianSharpeningNode,
    "GuidedFilterSharpeningNode": GuidedFilterSharpeningNode,
}

ADVANCED_SHARPENING_DISPLAY = {
    "AdvancedSharpeningNode": "🔧 Advanced Sharpening (Auto)",
    "SmartSharpeningNode": "🧠 Smart Sharpening",
    "HiRaLoAmSharpeningNode": "🎛️ HiRaLoAm Sharpening",
    "EdgeDirectionalSharpeningNode": "🧭 Edge-Directional Sharpening",
    "MultiScaleLaplacianSharpeningNode": "🏔️ Multi-Scale Laplacian",
    "GuidedFilterSharpeningNode": "🎯 Guided Filter Sharpening",
}
