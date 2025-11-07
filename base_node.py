"""
Base node class for Eric's Image Processing Nodes
Handles common ComfyUI integration, tensor conversions, and device management
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import gc


class BaseImageProcessingNode:
    """Base class for all image processing nodes with ComfyUI integration"""
    
    CATEGORY = "Eric's Image Processing"
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert ComfyUI image tensor (NHWC) to numpy array (NHWC or HWC)
        
        Args:
            tensor: ComfyUI image tensor in format [N, H, W, C] or [H, W, C] with values 0-1
            
        Returns:
            numpy array in format [N, H, W, C] or [H, W, C] with values 0-255 (uint8)
            GUARANTEED to be contiguous for OpenCV 4.11+ compatibility
        """
        # CRITICAL: Ensure tensor is contiguous before numpy conversion
        # PyTorch operations like permute() can create non-contiguous tensors
        # OpenCV 4.11+ strictly requires contiguous arrays
        tensor = tensor.contiguous()
        
        # Keep batch dimension if present
        img_np = tensor.cpu().numpy()
        
        # Convert to numpy and scale to 0-255
        img_np = (img_np * 255).astype(np.uint8)
        
        # CRITICAL: Ensure final array is contiguous
        # Arithmetic operations and astype() can create non-contiguous arrays
        img_np = np.ascontiguousarray(img_np)
        
        return img_np
    
    @staticmethod
    def numpy_to_tensor(img_np: np.ndarray) -> torch.Tensor:
        """Convert numpy array back to ComfyUI tensor format
        
        Args:
            img_np: numpy array in format [N, H, W, C] or [H, W, C] with values 0-255 (uint8) or 0-1 (float32)
            
        Returns:
            ComfyUI tensor in format [N, H, W, C] or [1, H, W, C] with values 0-1
        """
        # CRITICAL: Ensure input array is contiguous before ANY operations
        # Processing functions may return non-contiguous arrays
        img_np = np.ascontiguousarray(img_np)
        
        # Handle different input formats
        if img_np.dtype == np.uint8:
            # Convert uint8 [0,255] to float32 [0,1]
            img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
        elif img_np.dtype in [np.float32, np.float64]:
            # Handle float input - check if it's in [0,1] or [0,255] range
            if img_np.max() <= 1.0:
                # Already in [0,1] range
                img_tensor = torch.from_numpy(img_np.astype(np.float32))
            else:
                # Assume it's in [0,255] range
                img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
        else:
            # Convert to uint8 first, then to float32
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_np = np.ascontiguousarray(img_np)  # Ensure contiguous after clip
            img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
        
        # Add batch dimension if not present (single image)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        return img_tensor
    
    @staticmethod
    def get_device_info() -> Tuple[str, bool]:
        """Get optimal device for processing
        
        Returns:
            Tuple of (device_name, gpu_available)
        """
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            gpu_available = True
        else:
            device = "cpu"
            gpu_available = False
            
        return device, gpu_available
    
    @staticmethod
    def cleanup_memory():
        """Clean up GPU memory after processing"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def validate_image_tensor(tensor: torch.Tensor) -> bool:
        """Validate that tensor is a proper ComfyUI image tensor
        
        Args:
            tensor: Input tensor to validate
            
        Returns:
            True if valid, raises ValueError if not
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
            
        if len(tensor.shape) not in [3, 4]:
            raise ValueError(f"Image tensor must have 3 or 4 dimensions, got {len(tensor.shape)}")
            
        if len(tensor.shape) == 4:
            n, h, w, c = tensor.shape
            if c not in [1, 3, 4]:
                raise ValueError(f"Image must have 1, 3, or 4 channels, got {c}")
        else:
            h, w, c = tensor.shape
            if c not in [1, 3, 4]:
                raise ValueError(f"Image must have 1, 3, or 4 channels, got {c}")
                
        if tensor.min() < 0 or tensor.max() > 1:
            raise ValueError("Image tensor values must be in range [0, 1]")
            
        return True
    
    def process_image_safe(self, image: torch.Tensor, processing_func, **kwargs) -> torch.Tensor:
        """Safely process an image with error handling and memory cleanup
        
        Args:
            image: Input image tensor [N, H, W, C] or [H, W, C]
            processing_func: Function to apply to the image (expects [H, W, C])
            **kwargs: Additional arguments for processing_func
            
        Returns:
            Processed image tensor [N, H, W, C] or [H, W, C]
        """
        try:
            # Validate input
            self.validate_image_tensor(image)
            
            # Handle batch dimension properly
            is_batch = len(image.shape) == 4
            
            if is_batch:
                # Process each image in the batch
                batch_size = image.shape[0]
                results = []
                
                for i in range(batch_size):
                    # Get single image [H, W, C]
                    single_img = image[i]
                    
                    # Convert to numpy
                    img_np = self.tensor_to_numpy(single_img)
                    
                    # Apply processing (expects 2D or 3D array)
                    processed_np = processing_func(img_np, **kwargs)
                    
                    # Convert back to tensor
                    result_tensor = self.numpy_to_tensor(processed_np)
                    
                    # Remove batch dimension if added by numpy_to_tensor
                    if len(result_tensor.shape) == 4:
                        result_tensor = result_tensor[0]
                    
                    results.append(result_tensor)
                
                # Stack results back into batch
                result = torch.stack(results, dim=0)
            else:
                # Single image [H, W, C]
                # Convert to numpy
                img_np = self.tensor_to_numpy(image)
                
                # Apply processing
                processed_np = processing_func(img_np, **kwargs)
                
                # Convert back to tensor
                result = self.numpy_to_tensor(processed_np)
                
                # Remove batch dimension if added by numpy_to_tensor
                if len(result.shape) == 4 and result.shape[0] == 1:
                    result = result[0]
            
            return result
            
        except Exception as e:
            print(f"Error in image processing: {str(e)}")
            raise
        finally:
            # Clean up memory
            self.cleanup_memory()
