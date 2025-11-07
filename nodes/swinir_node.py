"""
ComfyUI node for SwinIR (Swin Transformer for Image Restoration)
State-of-the-art transformer-based restoration with 67% parameter reduction
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union

# Import from main package
try:
    from ..base_node import BaseImageProcessingNode
except ImportError:
    # Fallback for direct testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode

# Import SwinIR processor
try:
    from Eric_Image_Processing_Nodes import SwinIRProcessor
except ImportError:
    try:
        from scripts.swinir_processing import SwinIRProcessor
    except ImportError:
        print("Warning: SwinIR processor not found. Node will not function.")
        SwinIRProcessor = None


class SwinIRRestorationNode(BaseImageProcessingNode):
    """
    SwinIR Transformer-based Image Restoration
    
    State-of-the-art restoration using Swin Transformer architecture:
    - 67% parameter reduction compared to CNN methods
    - Up to 0.45dB PSNR improvement
    - Hierarchical feature extraction with local and global context
    - Residual Swin Transformer Blocks with shifted window attention
    
    Excellent for:
    - Classical image super-resolution
    - Image denoising (Gaussian, real noise)
    - JPEG compression artifact reduction
    - Professional photo restoration
    
    Performance: Transformer-based, excellent quality with efficient parameters
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "task": (["classical_sr", "lightweight_sr", "real_sr", "color_dn", "gray_dn", "jpeg_car"], {
                    "default": "classical_sr",
                    "tooltip": "Restoration task:\n• classical_sr: Classical super-resolution (clean images)\n• lightweight_sr: Lightweight super-resolution (faster)\n• real_sr: Real-world super-resolution (GAN-based)\n• color_dn: Color image denoising\n• gray_dn: Grayscale image denoising\n• jpeg_car: JPEG compression artifact removal"
                }),
            },
            "optional": {
                "model_variant": (["auto", "2x", "3x", "4x", "8x", "light_noise", "medium_noise", "heavy_noise", "jpeg_q10", "jpeg_q20", "jpeg_q30", "jpeg_q40"], {
                    "default": "auto",
                    "tooltip": "Model variant selection:\n• auto: Automatically select best model for task\n• 2x-8x: Super-resolution scales\n• light_noise: σ=15 (for denoising)\n• medium_noise: σ=25 (for denoising)\n• heavy_noise: σ=50 (for denoising)\n• jpeg_q10-40: JPEG quality levels for artifact removal"
                }),
                "window_size": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 16,
                    "step": 2,
                    "tooltip": "Transformer window size:\n• 4-6: Faster processing, less context\n• 8: Balanced (recommended)\n• 10-16: Better quality, slower"
                }),
                "device_preference": (["auto", "cpu", "cuda"], {
                    "default": "auto",
                    "tooltip": "Processing device preference"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore_image"
    
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

    def restore_image(self, image: torch.Tensor, task: str = "classical_sr", 
                     model_variant: str = "auto", window_size: int = 8,
                     device_preference: str = "auto") -> Tuple[torch.Tensor]:
        """
        Restore image using SwinIR transformer architecture
        
        Args:
            image: Input image tensor
            task: Type of restoration task
            model_variant: Model variant selection (auto, 2x, 3x, 4x, 8x, light_noise, medium_noise, heavy_noise, jpeg_q10-40)
            window_size: Transformer window size
            device_preference: Preferred processing device
            
        Returns:
            Tuple containing restored image tensor
        """
        try:
            if SwinIRProcessor is None:
                print("Error: SwinIR processor not available")
                return (image,)
            
            # Convert ComfyUI tensor to numpy float32
            img_np = self._tensor_to_numpy_float(image)
            
            # Parse model variant to get scale, noise, and jpeg parameters
            scale, noise_level, jpeg_quality = self._parse_model_variant(task, model_variant)
            
            # Create processor
            print(f"Initializing SwinIR for task: {task}")
            print(f"Model variant: {model_variant} → scale={scale}, noise={noise_level}, jpeg={jpeg_quality}")
            processor = SwinIRProcessor(
                task=task,
                scale=scale,
                device=device_preference,
                noise=noise_level,
                jpeg=jpeg_quality
            )
            
            # Process the image
            print(f"Processing with SwinIR (model_variant={model_variant}, task={task})...")
            result_np = processor.process_image(img_np, strength=1.0)
            
            # Validate result
            if result_np is None:
                print("Error: SwinIR processing failed")
                return (image,)
            
            # Convert back to ComfyUI tensor
            result_tensor = self._numpy_to_tensor_float(result_np)
            
            print(f"SwinIR restoration completed successfully!")
            print(f"Input shape: {img_np.shape}")
            print(f"Output shape: {result_np.shape}")
            if task in ['classical_sr', 'lightweight_sr', 'real_sr']:
                print(f"Scale factor achieved: {result_np.shape[0]/img_np.shape[0]:.1f}x")
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"SwinIR restoration error: {e}")
            import traceback
            traceback.print_exc()
            return (image,)  # Return original on error

    def _parse_model_variant(self, task: str, model_variant: str) -> Tuple[int, int, int]:
        """Parse model variant to get scale, noise level, and jpeg quality parameters
        
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


class SwinIRBatchNode(BaseImageProcessingNode):
    """
    SwinIR Batch Processing for Multiple Images
    
    Efficiently process multiple images with SwinIR transformer
    Ideal for:
    - Batch photo restoration
    - Consistent processing across image sets
    - Professional workflow integration
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of images
                "task": (["classical_sr", "lightweight_sr", "real_sr", "color_dn", "gray_dn"], {
                    "default": "classical_sr",
                    "tooltip": "Restoration task for all images"
                }),
                "scale": ([1, 2, 3, 4], {
                    "default": 2,
                    "tooltip": "Upscaling factor for batch"
                }),
            },
            "optional": {
                "batch_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of images to process simultaneously"
                }),
                "noise_level": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 75,
                    "step": 5,
                    "tooltip": "Noise level for denoising tasks"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_images",)
    FUNCTION = "restore_batch"
    
    def restore_batch(self, images: torch.Tensor, task: str = "classical_sr", scale: int = 2,
                     batch_size: int = 2, noise_level: int = 15) -> Tuple[torch.Tensor]:
        """
        Restore a batch of images using SwinIR
        """
        try:
            if SwinIRProcessor is None:
                print("Error: SwinIR processor not available")
                return (images,)
            
            # Create processor once for all images
            processor = SwinIRProcessor(task=task, scale=scale, device="auto")
            
            # Process images in batches
            num_images = images.shape[0]
            restored_images = []
            
            print(f"Batch processing {num_images} images with SwinIR...")
            
            for i in range(0, num_images, batch_size):
                end_idx = min(i + batch_size, num_images)
                batch = images[i:end_idx]
                
                print(f"Processing batch {i//batch_size + 1}/{(num_images-1)//batch_size + 1}")
                
                # Process each image in the batch
                batch_results = []
                for j in range(batch.shape[0]):
                    img_tensor = batch[j:j+1]  # Keep batch dimension
                    img_np = self.tensor_to_numpy(img_tensor)
                    
                    # Process with SwinIR
                    if task in ["color_dn", "gray_dn"]:
                        result_np = processor.process_image(img_np, noise_level=noise_level)
                    else:
                        result_np = processor.process_image(img_np, scale=scale)
                    
                    result_tensor = self.numpy_to_tensor(result_np)
                    batch_results.append(result_tensor)
                
                # Combine batch results
                if batch_results:
                    batch_tensor = torch.cat(batch_results, dim=0)
                    restored_images.append(batch_tensor)
            
            # Combine all results
            if restored_images:
                final_result = torch.cat(restored_images, dim=0)
                print(f"SwinIR batch processing completed! Processed {final_result.shape[0]} images.")
                return (final_result,)
            else:
                return (images,)
                
        except Exception as e:
            print(f"SwinIR batch processing error: {e}")
            return (images,)


class MemoryOptimizationNode(BaseImageProcessingNode):
    """
    Memory Management and Optimization Node
    
    Provides intelligent memory management for image processing workflows:
    - Automatic tile size optimization
    - Memory usage monitoring
    - GPU memory cleanup
    - Processing optimization recommendations
    
    Essential for:
    - Large image processing (>4MP)
    - GPU memory management
    - Workflow optimization
    - Preventing out-of-memory errors
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_memory_gb": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 24.0,
                    "step": 0.5,
                    "tooltip": "Target memory usage in GB:\n• 1-2GB: Conservative (slower)\n• 4-8GB: Balanced\n• 8-16GB: Aggressive (faster)\n• 16GB+: Maximum performance"
                }),
            },
            "optional": {
                "optimization_mode": (["memory_efficient", "balanced", "performance"], {
                    "default": "balanced",
                    "tooltip": "Optimization strategy:\n• memory_efficient: Minimize memory usage\n• balanced: Balance speed and memory\n• performance: Maximize processing speed"
                }),
                "enable_cleanup": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable automatic memory cleanup after processing"
                }),
                "force_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force CPU processing to avoid GPU memory issues"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("optimized_image", "memory_report")
    FUNCTION = "optimize_memory"
    
    def optimize_memory(self, image: torch.Tensor, target_memory_gb: float = 4.0,
                       optimization_mode: str = "balanced", enable_cleanup: bool = True,
                       force_cpu: bool = False) -> Tuple[torch.Tensor, str]:
        """
        Optimize memory usage for image processing
        """
        try:
            # Import memory utilities
            try:
                from scripts.memory_utils import MemoryManager
            except ImportError:
                print("Warning: Memory utilities not available")
                return (image, "Memory utilities not available")
            
            img_np = self.tensor_to_numpy(image)
            
            # Get memory information
            report_lines = [
                "=== MEMORY OPTIMIZATION REPORT ===",
                f"Input image: {img_np.shape} ({img_np.nbytes / 1024**2:.1f} MB)",
                f"Target memory: {target_memory_gb:.1f} GB",
                f"Optimization mode: {optimization_mode}",
            ]
            
            # Determine optimal device
            if force_cpu:
                device = 'cpu'
                report_lines.append("Device: CPU (forced)")
            else:
                device = MemoryManager.safe_device_selection('auto', min_gpu_memory_gb=target_memory_gb)
                report_lines.append(f"Recommended device: {device}")
            
            # Calculate optimal tile size
            tile_size = MemoryManager.optimize_tile_size(
                img_np.shape,
                max_memory_gb=target_memory_gb,
                min_tile_size=64 if optimization_mode == "memory_efficient" else 128,
                max_tile_size=512 if optimization_mode == "performance" else 256
            )
            report_lines.append(f"Optimal tile size: {tile_size}x{tile_size}")
            
            # Memory usage recommendations
            estimated_memory = (img_np.nbytes * 4) / 1024**3  # Rough estimate for processing
            if estimated_memory > target_memory_gb:
                report_lines.append(f"⚠️ Warning: Estimated memory usage ({estimated_memory:.1f}GB) exceeds target")
                report_lines.append("Recommendation: Use smaller tile size or enable tiling")
            else:
                report_lines.append(f"✅ Memory usage within target ({estimated_memory:.1f}GB)")
            
            # Processing recommendations
            if img_np.shape[0] * img_np.shape[1] > 4000000:  # >4MP
                report_lines.append("Recommendation: Enable tiling for large image processing")
            
            if device == 'cuda':
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
                        report_lines.append(f"GPU memory: {gpu_free:.1f}GB free / {gpu_memory:.1f}GB total")
                except:
                    pass
            
            # Cleanup if requested
            if enable_cleanup:
                MemoryManager.cleanup_memory()
                report_lines.append("✅ Memory cleanup completed")
            
            # Add processing tips
            report_lines.extend([
                "",
                "=== PROCESSING TIPS ===",
                f"• Use tile_size={tile_size} for optimal performance",
                f"• Process on {device} for best results",
                "• Enable tiling for images >4MP",
                "• Monitor memory usage during processing"
            ])
            
            memory_report = "\n".join(report_lines)
            
            print("Memory optimization completed!")
            print(memory_report)
            
            return (image, memory_report)
            
        except Exception as e:
            error_report = f"Memory optimization error: {e}"
            print(error_report)
            return (image, error_report)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SwinIRRestoration": SwinIRRestorationNode,
    "SwinIRBatch": SwinIRBatchNode,
    "MemoryOptimization": MemoryOptimizationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwinIRRestoration": "SwinIR Image Restoration",
    "SwinIRBatch": "SwinIR Batch Processing",
    "MemoryOptimization": "Memory Optimization",
}
