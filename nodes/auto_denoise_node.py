"""
Auto-Denoise ComfyUI Nodes
Self-supervised and unsupervised CNN denoising methods for ComfyUI

Provides nodes for:
- Noise2Void: Single image self-supervised denoising
- Deep Image Prior: Unsupervised restoration
- Auto-Denoise: Intelligent method selection
- Method comparison
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import warnings

try:
    from ..base_node import BaseImageProcessingNode
    from ..scripts.auto_denoise import (
        AutoDenoiseProcessor,
        Noise2VoidProcessor,
        DeepImagePriorProcessor
    )
    BASE_NODE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Base node or auto-denoise scripts not available: {e}")
    try:
        # Fallback for direct execution
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from base_node import BaseImageProcessingNode
        from scripts.auto_denoise import (
            AutoDenoiseProcessor,
            Noise2VoidProcessor,
            DeepImagePriorProcessor
        )
        BASE_NODE_AVAILABLE = True
    except ImportError as e2:
        print(f"Fallback import also failed: {e2}")
        BASE_NODE_AVAILABLE = False
        # Fallback base class
        class BaseImageProcessingNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",)}}
            
            RETURN_TYPES = ("IMAGE",)
            FUNCTION = "process"
            CATEGORY = "Eric's Image Processing"


class AutoDenoiseNode(BaseImageProcessingNode):
    """
    Auto-Denoise Node: Intelligent denoising method selection and processing
    
    Automatically analyzes the image and selects the optimal denoising method:
    - Noise2Void for high-noise images
    - Deep Image Prior for low-complexity images
    
    Features:
    - No clean reference images required
    - Self-supervised learning
    - Automatic method selection
    - GPU acceleration support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["auto", "noise2void", "deep_image_prior"], {"default": "auto"}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
            "optional": {
                "train_epochs": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),
                "iterations": ("INT", {"default": 3000, "min": 500, "max": 10000, "step": 100}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
                "analyze_image": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("denoised_image", "processing_info")
    FUNCTION = "process_auto_denoise"
    CATEGORY = "Eric's Image Processing/Auto-Denoise"
    
    def __init__(self):
        super().__init__()
        self.processor = None
    
    def process_auto_denoise(self, image: torch.Tensor, method: str = "auto", 
                           device: str = "auto", train_epochs: int = 100,
                           iterations: int = 3000, learning_rate: float = 0.01,
                           analyze_image: bool = True) -> Tuple[torch.Tensor, str]:
        """Process image with Auto-Denoise"""
        
        if not BASE_NODE_AVAILABLE:
            return image, "❌ Auto-denoise components not available"
        
        try:
            # Convert ComfyUI tensor to numpy
            if len(image.shape) == 4:
                np_image = image[0].cpu().numpy()  # Take first image from batch
            else:
                np_image = image.cpu().numpy()
            
            # Ensure values are in [0, 1]
            if np_image.max() > 1.0:
                np_image = np_image / 255.0
            
            # Initialize processor
            if self.processor is None:
                self.processor = AutoDenoiseProcessor(device=device)
            
            info_parts = []
            
            # Analyze image if requested
            if analyze_image:
                analysis = self.processor.analyze_image(np_image)
                info_parts.append(f"📊 Image Analysis:")
                info_parts.append(f"  • Noise level: {analysis.get('noise_level', 0):.4f}")
                info_parts.append(f"  • Complexity: {analysis.get('complexity', 0):.4f}")
                info_parts.append(f"  • Recommended: {analysis.get('recommended_method', 'unknown')}")
                info_parts.append(f"  • Reason: {analysis.get('reason', 'N/A')}")
                info_parts.append("")
            
            # Process image
            info_parts.append(f"🔧 Processing with method: {method}")
            
            kwargs = {
                'train_epochs': train_epochs,
                'iterations': iterations,
                'learning_rate': learning_rate
            }
            
            result = self.processor.process_image(np_image, method=method, **kwargs)
            
            if result is not None:
                # Convert back to ComfyUI tensor
                if len(result.shape) == 2:
                    result = np.expand_dims(result, axis=2)
                
                result_tensor = torch.from_numpy(result).float()
                
                # Add batch dimension if needed
                if len(result_tensor.shape) == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                
                info_parts.append("✅ Auto-denoise processing completed successfully")
                
                # Add performance info
                if hasattr(self.processor, 'n2v_processor') and self.processor.n2v_processor.trained:
                    info_parts.append("🧠 Noise2Void model trained on input image")
                
                processing_info = "\n".join(info_parts)
                return (result_tensor, processing_info)
            else:
                error_info = "\n".join(info_parts + ["❌ Auto-denoise processing failed"])
                return (image, error_info)
                
        except Exception as e:
            error_msg = f"❌ Auto-denoise error: {str(e)}"
            print(error_msg)
            return (image, error_msg)


class Noise2VoidNode(BaseImageProcessingNode):
    """
    Noise2Void Node: Self-supervised single image denoising
    
    Revolutionary approach that doesn't require clean reference images.
    Works by training on blind spots in the same image.
    
    Perfect for:
    - Single noisy images
    - Unknown noise characteristics
    - Real-world noise patterns
    - Film grain and digital noise
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "train_epochs": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),
                "model_size": (["small", "medium", "large"], {"default": "medium"}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
            "optional": {
                "learning_rate": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.01, "step": 0.0001}),
                "show_training_progress": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("denoised_image", "training_info")
    FUNCTION = "process_noise2void"
    CATEGORY = "Eric's Image Processing/Auto-Denoise"
    
    def __init__(self):
        super().__init__()
        self.processor = None
    
    def process_noise2void(self, image: torch.Tensor, train_epochs: int = 100,
                          model_size: str = "medium", device: str = "auto",
                          learning_rate: float = 0.001, 
                          show_training_progress: bool = True) -> Tuple[torch.Tensor, str]:
        """Process image with Noise2Void"""
        
        if not BASE_NODE_AVAILABLE:
            return image, "❌ Noise2Void components not available"
        
        try:
            # Convert ComfyUI tensor to numpy
            if len(image.shape) == 4:
                np_image = image[0].cpu().numpy()
            else:
                np_image = image.cpu().numpy()
            
            # Ensure values are in [0, 1]
            if np_image.max() > 1.0:
                np_image = np_image / 255.0
            
            # Initialize processor
            self.processor = Noise2VoidProcessor(device=device, model_size=model_size)
            
            info_parts = []
            info_parts.append(f"🔬 Noise2Void Processing:")
            info_parts.append(f"  • Model size: {model_size}")
            info_parts.append(f"  • Training epochs: {train_epochs}")
            info_parts.append(f"  • Learning rate: {learning_rate}")
            info_parts.append(f"  • Device: {device}")
            info_parts.append("")
            
            # Train and process
            training_stats = self.processor.train_model(
                np_image, 
                epochs=train_epochs, 
                learning_rate=learning_rate
            )
            
            # Get training info
            info_parts.append(f"📈 Training Results:")
            info_parts.append(f"  • Final loss: {training_stats['final_loss']:.6f}")
            info_parts.append(f"  • Model architecture: {training_stats['model_size']}")
            info_parts.append("")
            
            # Process image (inference)
            result = self.processor.process_image(np_image, train_epochs=0)  # Skip retraining
            
            if result is not None:
                # Convert back to ComfyUI tensor
                if len(result.shape) == 2:
                    result = np.expand_dims(result, axis=2)
                
                result_tensor = torch.from_numpy(result).float()
                
                if len(result_tensor.shape) == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                
                info_parts.append("✅ Noise2Void processing completed successfully")
                info_parts.append("🧠 Self-supervised model trained on input image")
                
                training_info = "\n".join(info_parts)
                return (result_tensor, training_info)
            else:
                error_info = "\n".join(info_parts + ["❌ Noise2Void processing failed"])
                return (image, error_info)
                
        except Exception as e:
            error_msg = f"❌ Noise2Void error: {str(e)}"
            print(error_msg)
            return (image, error_msg)


class DeepImagePriorNode(BaseImageProcessingNode):
    """
    Deep Image Prior Node: Unsupervised restoration using CNN architecture
    
    Uses the implicit bias of CNN architectures for image restoration.
    No training data required - the network structure acts as a prior.
    
    Excellent for:
    - Image restoration without training data
    - Denoising with unknown noise models
    - Super-resolution and inpainting
    - Preserving image structure
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "iterations": ("INT", {"default": 3000, "min": 500, "max": 10000, "step": 100}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
            "optional": {
                "show_progress": ("BOOLEAN", {"default": True}),
                "early_stopping": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("restored_image", "optimization_info")
    FUNCTION = "process_deep_image_prior"
    CATEGORY = "Eric's Image Processing/Auto-Denoise"
    
    def __init__(self):
        super().__init__()
        self.processor = None
    
    def process_deep_image_prior(self, image: torch.Tensor, iterations: int = 3000,
                               learning_rate: float = 0.01, device: str = "auto",
                               show_progress: bool = True, 
                               early_stopping: bool = True) -> Tuple[torch.Tensor, str]:
        """Process image with Deep Image Prior"""
        
        if not BASE_NODE_AVAILABLE:
            return image, "❌ Deep Image Prior components not available"
        
        try:
            # Convert ComfyUI tensor to numpy
            if len(image.shape) == 4:
                np_image = image[0].cpu().numpy()
            else:
                np_image = image.cpu().numpy()
            
            # Ensure values are in [0, 1]
            if np_image.max() > 1.0:
                np_image = np_image / 255.0
            
            # Initialize processor
            self.processor = DeepImagePriorProcessor(device=device)
            
            info_parts = []
            info_parts.append(f"🎯 Deep Image Prior Processing:")
            info_parts.append(f"  • Optimization iterations: {iterations}")
            info_parts.append(f"  • Learning rate: {learning_rate}")
            info_parts.append(f"  • Device: {device}")
            info_parts.append(f"  • Early stopping: {early_stopping}")
            info_parts.append("")
            info_parts.append("🔧 Using CNN architecture as implicit image prior...")
            
            # Process image
            result = self.processor.process_image(
                np_image, 
                iterations=iterations, 
                learning_rate=learning_rate
            )
            
            if result is not None:
                # Convert back to ComfyUI tensor
                if len(result.shape) == 2:
                    result = np.expand_dims(result, axis=2)
                
                result_tensor = torch.from_numpy(result).float()
                
                if len(result_tensor.shape) == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                
                info_parts.append("✅ Deep Image Prior processing completed")
                info_parts.append("🏗️ Network architecture used as image prior")
                info_parts.append("🔄 Unsupervised optimization completed")
                
                optimization_info = "\n".join(info_parts)
                return (result_tensor, optimization_info)
            else:
                error_info = "\n".join(info_parts + ["❌ Deep Image Prior processing failed"])
                return (image, error_info)
                
        except Exception as e:
            error_msg = f"❌ Deep Image Prior error: {str(e)}"
            print(error_msg)
            return (image, error_msg)


class AutoDenoiseComparisonNode(BaseImageProcessingNode):
    """
    Auto-Denoise Comparison Node: Compare multiple state-of-the-art methods
    
    Processes the same image with different Auto-Denoise methods and provides
    visual comparison with quality metrics.
    
    Methods compared:
    - Noise2Void (self-supervised)
    - Deep Image Prior (unsupervised)
    - Original image for reference
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
            "optional": {
                "n2v_epochs": ("INT", {"default": 50, "min": 10, "max": 200, "step": 10}),
                "dip_iterations": ("INT", {"default": 1000, "min": 300, "max": 3000, "step": 100}),
                "calculate_metrics": ("BOOLEAN", {"default": True}),
                "show_analysis": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("original", "noise2void_result", "deep_image_prior_result", "comparison_report")
    FUNCTION = "compare_methods"
    CATEGORY = "Eric's Image Processing/Auto-Denoise"
    
    def __init__(self):
        super().__init__()
        self.processor = None
    
    def compare_methods(self, image: torch.Tensor, device: str = "auto",
                       n2v_epochs: int = 50, dip_iterations: int = 1000,
                       calculate_metrics: bool = True, 
                       show_analysis: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Compare Auto-Denoise methods"""
        
        if not BASE_NODE_AVAILABLE:
            return image, image, image, "❌ Auto-denoise comparison components not available"
        
        try:
            # Convert ComfyUI tensor to numpy
            if len(image.shape) == 4:
                np_image = image[0].cpu().numpy()
            else:
                np_image = image.cpu().numpy()
            
            # Ensure values are in [0, 1]
            if np_image.max() > 1.0:
                np_image = np_image / 255.0
            
            # Initialize processor
            self.processor = AutoDenoiseProcessor(device=device)
            
            report_parts = []
            report_parts.append("🔬 Auto-Denoise Method Comparison")
            report_parts.append("=" * 50)
            
            # Image analysis
            if show_analysis:
                analysis = self.processor.analyze_image(np_image)
                report_parts.append("📊 Image Analysis:")
                report_parts.append(f"  • Noise level: {analysis.get('noise_level', 0):.4f}")
                report_parts.append(f"  • Complexity: {analysis.get('complexity', 0):.4f}")
                report_parts.append(f"  • Recommended method: {analysis.get('recommended_method', 'unknown')}")
                report_parts.append("")
            
            # Process with Noise2Void
            report_parts.append("🧠 Processing with Noise2Void...")
            n2v_result = self.processor.n2v_processor.process_image(np_image, train_epochs=n2v_epochs)
            
            # Process with Deep Image Prior
            report_parts.append("🎯 Processing with Deep Image Prior...")
            dip_result = self.processor.dip_processor.process_image(np_image, iterations=dip_iterations)
            
            # Prepare outputs
            def to_tensor(img_array):
                if img_array is None:
                    return image  # Return original if processing failed
                
                if len(img_array.shape) == 2:
                    img_array = np.expand_dims(img_array, axis=2)
                
                tensor = torch.from_numpy(img_array).float()
                
                if len(tensor.shape) == 3:
                    tensor = tensor.unsqueeze(0)
                
                return tensor
            
            n2v_tensor = to_tensor(n2v_result)
            dip_tensor = to_tensor(dip_result)
            
            # Calculate metrics if requested
            if calculate_metrics:
                report_parts.append("")
                report_parts.append("📈 Quality Metrics:")
                
                try:
                    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
                    
                    if n2v_result is not None:
                        n2v_psnr = peak_signal_noise_ratio(np_image, n2v_result, data_range=1.0)
                        n2v_ssim = structural_similarity(np_image, n2v_result, multichannel=True, channel_axis=-1, data_range=1.0)
                        report_parts.append(f"  • Noise2Void: PSNR={n2v_psnr:.2f}dB, SSIM={n2v_ssim:.4f}")
                    else:
                        report_parts.append("  • Noise2Void: Processing failed")
                    
                    if dip_result is not None:
                        dip_psnr = peak_signal_noise_ratio(np_image, dip_result, data_range=1.0)
                        dip_ssim = structural_similarity(np_image, dip_result, multichannel=True, channel_axis=-1, data_range=1.0)
                        report_parts.append(f"  • Deep Image Prior: PSNR={dip_psnr:.2f}dB, SSIM={dip_ssim:.4f}")
                    else:
                        report_parts.append("  • Deep Image Prior: Processing failed")
                        
                except ImportError:
                    report_parts.append("  • Metrics calculation requires scikit-image")
                except Exception as e:
                    report_parts.append(f"  • Metrics calculation failed: {e}")
            
            # Summary
            report_parts.append("")
            report_parts.append("📋 Processing Summary:")
            report_parts.append(f"  • Noise2Void: {'✅ Success' if n2v_result is not None else '❌ Failed'}")
            report_parts.append(f"  • Deep Image Prior: {'✅ Success' if dip_result is not None else '❌ Failed'}")
            report_parts.append("")
            report_parts.append("🎯 Method Characteristics:")
            report_parts.append("  • Noise2Void: Self-supervised, single image training")
            report_parts.append("  • Deep Image Prior: Unsupervised, architecture as prior")
            
            comparison_report = "\n".join(report_parts)
            
            return (image, n2v_tensor, dip_tensor, comparison_report)
            
        except Exception as e:
            error_msg = f"❌ Auto-denoise comparison error: {str(e)}"
            print(error_msg)
            return (image, image, image, error_msg)


# Node class mappings for ComfyUI registration
AUTO_DENOISE_NODE_CLASS_MAPPINGS = {
    "AutoDenoiseNode": AutoDenoiseNode,
    "Noise2VoidNode": Noise2VoidNode,
    "DeepImagePriorNode": DeepImagePriorNode,
    "AutoDenoiseComparisonNode": AutoDenoiseComparisonNode,
}

AUTO_DENOISE_NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoDenoiseNode": "Auto-Denoise (Smart Selection)",
    "Noise2VoidNode": "Noise2Void (Self-Supervised)",
    "DeepImagePriorNode": "Deep Image Prior (Unsupervised)",
    "AutoDenoiseComparisonNode": "Auto-Denoise Comparison",
}
