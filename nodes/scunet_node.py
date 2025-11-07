"""
ComfyUI nodes for SCUNet (Swin-Conv-UNet) image restoration
State-of-the-art deep learning restoration for mixed degradations
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import os
from pathlib import Path

# Import from main package
try:
    from ..base_node import BaseImageProcessingNode
except ImportError:
    # Fallback for direct testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode

# Import SCUNet processor
try:
    from ..scripts.scunet_processing import SCUNetProcessor
except ImportError:
    try:
        from scripts.scunet_processing import SCUNetProcessor
    except ImportError:
        print("Warning: SCUNet processor not found. Node will not function.")
        SCUNetProcessor = None


class SCUNetRestorationNode(BaseImageProcessingNode):
    """
    SCUNet (Swin-Conv-UNet) Deep Learning Image Restoration
    
    State-of-the-art restoration for:
    - Film grain and digital noise removal
    - JPEG compression artifact reduction
    - Motion blur and defocus blur correction
    - Mixed degradation restoration
    - Old photo restoration
    - Low-light image enhancement
    
    Performance: GPU-accelerated, excellent quality on complex degradations
    Memory: Optimized with tiling for large images
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available SCUNet models from the models directory
        models_dir = Path(__file__).parent.parent / "models"
        available_models = []
        
        if models_dir.exists():
            for model_file in models_dir.glob("scunet_*.pth"):
                model_name = model_file.stem
                available_models.append(model_name)
        
        # If no models found, provide fallback list
        if not available_models:
            available_models = [
                "scunet_color_15", "scunet_color_25", "scunet_color_50",
                "scunet_color_real_psnr", "scunet_color_real_gan",
                "scunet_gray_15", "scunet_gray_25", "scunet_gray_50"
            ]
        
        # Sort models for better organization
        available_models.sort()
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (available_models, {
                    "default": available_models[0] if available_models else "scunet_color_25",
                    "tooltip": "SCUNet model selection:\n"
                              "• Color models: For color image denoising\n"
                              "• Gray models: For grayscale image denoising\n"
                              "• Real models: For real-world degradation (PSNR/GAN trained)\n"
                              "• Numbers (15/25/50): Gaussian noise levels"
                }),
            },
            "optional": {
                "device_preference": (["auto", "cpu", "cuda"], {
                    "default": "auto",
                    "tooltip": "Processing device:\n• Auto: Automatic selection\n• CPU: Force CPU processing\n• CUDA: Force GPU processing"
                }),
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Processing tile size:\n• 64-128: Safe for low memory\n• 256: Balanced (recommended)\n• 512+: Faster but needs more memory"
                }),
                "overlap": ("INT", {
                    "default": 16,
                    "min": 8,
                    "max": 64,
                    "step": 8,
                    "tooltip": "Overlap between tiles to reduce seam artifacts"
                }),
                "enable_tiling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable tile-based processing for memory safety"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore_image"
    
    def restore_image(self, image: torch.Tensor, model_name: str, 
                     device_preference: str = "auto", tile_size: int = 256, 
                     overlap: int = 16, enable_tiling: bool = True) -> Tuple[torch.Tensor]:
        """
        Restore image using SCUNet deep learning model
        
        Args:
            image: Input image tensor
            model_name: SCUNet model to use
            device_preference: Preferred processing device
            tile_size: Size of processing tiles
            overlap: Overlap between tiles
            enable_tiling: Whether to use tile-based processing
            
        Returns:
            Tuple containing restored image tensor
        """
        try:
            if SCUNetProcessor is None:
                print("Error: SCUNet processor not available")
                return (image,)
            
            # Convert ComfyUI tensor to numpy
            img_np = self.tensor_to_numpy(image)
            
            # Determine input channels from model name and image
            if "gray" in model_name.lower():
                # Convert to grayscale if using gray model
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_np = np.mean(img_np, axis=2, keepdims=True)
                    print(f"Converting to grayscale for {model_name}")
            elif "color" in model_name.lower():
                # Ensure color image for color model
                if len(img_np.shape) == 3 and img_np.shape[2] == 1:
                    img_np = np.repeat(img_np, 3, axis=2)
                    print(f"Converting to color for {model_name}")
            
            # Get device
            device = self._get_device(device_preference)
            
            # Create processor with specific model
            processor = SCUNetProcessor(
                model_name=model_name,
                device=device,
                lightweight=True  # Always use lightweight for stability
            )
            
            # Process the image
            print(f"Processing with {model_name} on {device}...")
            
            if enable_tiling:
                result_np = processor.process_image(img_np, tile_size=tile_size, overlap=overlap)
            else:
                result_np = processor.process_image(img_np)
            
            # Validate result
            if result_np is None:
                print("Error: Processing returned None")
                return (image,)
            
            # Ensure result has correct shape
            if result_np.shape != img_np.shape:
                print(f"Warning: Shape mismatch. Input: {img_np.shape}, Output: {result_np.shape}")
                # Try to fix common shape issues
                if len(result_np.shape) == 2:
                    result_np = np.expand_dims(result_np, axis=2)
                elif len(result_np.shape) == 3 and result_np.shape[2] == 1 and img_np.shape[2] == 3:
                    result_np = np.repeat(result_np, 3, axis=2)
            
            # Convert back to ComfyUI tensor
            result_tensor = self.numpy_to_tensor(result_np)
            
            print(f"SCUNet restoration completed successfully!")
            print(f"Model: {model_name}")
            print(f"Device: {device}")
            print(f"Input shape: {img_np.shape}")
            print(f"Output shape: {result_np.shape}")
            print(f"Input range: [{img_np.min():.3f}, {img_np.max():.3f}]")
            print(f"Output range: [{result_np.min():.3f}, {result_np.max():.3f}]")
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"SCUNet restoration error: {e}")
            import traceback
            traceback.print_exc()
            return (image,)  # Return original on error
    
    def _get_device(self, device_preference: str) -> str:
        """Get the appropriate device for processing"""
        if device_preference == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device_preference == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("CUDA requested but not available, falling back to CPU")
                return "cpu"
        else:
            return "cpu"


class SCUNetBatchRestorationNode(BaseImageProcessingNode):
    """
    SCUNet Batch Processing for Multiple Images
    
    Efficiently process multiple images with shared model loading
    Ideal for:
    - Processing entire photo collections
    - Batch restoration workflows
    - Video frame processing
    
    Performance: Optimized for batch processing with model reuse
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available SCUNet models
        models_dir = Path(__file__).parent.parent / "models"
        available_models = []
        
        if models_dir.exists():
            for model_file in models_dir.glob("scunet_*.pth"):
                model_name = model_file.stem
                available_models.append(model_name)
        
        if not available_models:
            available_models = ["scunet_color_real_psnr", "scunet_color_real_gan"]
        
        available_models.sort()
        
        return {
            "required": {
                "images": ("IMAGE",),  # Expects batch of images
                "model_name": (available_models, {
                    "default": available_models[0],
                    "tooltip": "SCUNet model for batch processing"
                }),
            },
            "optional": {
                "batch_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Number of images to process simultaneously"
                }),
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 512,
                    "step": 64,
                    "tooltip": "Processing tile size for memory management"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_images",)
    FUNCTION = "restore_batch"
    
    def restore_batch(self, images: torch.Tensor, model_name: str,
                     batch_size: int = 4, tile_size: int = 256) -> Tuple[torch.Tensor]:
        """
        Restore a batch of images using SCUNet
        
        Args:
            images: Batch of input image tensors
            model_name: SCUNet model to use
            batch_size: Number of images to process at once
            tile_size: Size of processing tiles
            
        Returns:
            Tuple containing batch of restored image tensors
        """
        try:
            if SCUNetProcessor is None:
                return (images,)
            
            # Get processor once for all images
            processor = SCUNetProcessor(model_name=model_name, device="auto", lightweight=True)
            
            # Process images in batches
            num_images = images.shape[0]
            restored_images = []
            
            print(f"Batch processing {num_images} images with {model_name}...")
            
            for i in range(0, num_images, batch_size):
                end_idx = min(i + batch_size, num_images)
                batch = images[i:end_idx]
                
                print(f"Processing batch {i//batch_size + 1}/{(num_images-1)//batch_size + 1}")
                
                # Process each image in the batch
                batch_results = []
                for j in range(batch.shape[0]):
                    # FIXED: Remove batch dimension by using [j] instead of [j:j+1]
                    img_tensor = batch[j]  # Shape: [H,W,C]
                    img_np = self.tensor_to_numpy(img_tensor)
                    
                    # Process with SCUNet
                    result_np = processor.process_image(img_np, tile_size=tile_size)
                    result_tensor = self.numpy_to_tensor(result_np)
                    batch_results.append(result_tensor)
                
                # Combine batch results
                if batch_results:
                    batch_tensor = torch.cat(batch_results, dim=0)
                    restored_images.append(batch_tensor)
            
            # Combine all results
            if restored_images:
                final_result = torch.cat(restored_images, dim=0)
                print(f"Batch processing completed! Processed {final_result.shape[0]} images.")
                return (final_result,)
            else:
                return (images,)
                
        except Exception as e:
            print(f"Batch SCUNet restoration error: {e}")
            return (images,)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SCUNetRestoration": SCUNetRestorationNode,
    "SCUNetBatchRestoration": SCUNetBatchRestorationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCUNetRestoration": "SCUNet Image Restoration",
    "SCUNetBatchRestoration": "SCUNet Batch Processing",
}
