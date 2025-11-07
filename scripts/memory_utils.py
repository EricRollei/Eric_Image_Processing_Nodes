"""
Memory management utilities for advanced image processing
Helps prevent memory issues when working with large models and images
"""

import torch
import gc
import psutil
import numpy as np
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Utility class for managing memory during image processing"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get current memory usage information"""
        info = {
            'system_ram_total': psutil.virtual_memory().total / (1024**3),  # GB
            'system_ram_available': psutil.virtual_memory().available / (1024**3),  # GB
            'system_ram_percent': psutil.virtual_memory().percent,
        }
        
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                
                info.update({
                    'gpu_memory_total': gpu_memory,
                    'gpu_memory_allocated': gpu_allocated,
                    'gpu_memory_cached': gpu_cached,
                    'gpu_memory_free': gpu_memory - gpu_cached,
                })
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
                
        return info
    
    @staticmethod
    def cleanup_memory():
        """Clean up memory (both CPU and GPU)"""
        # Python garbage collection
        gc.collect()
        
        # PyTorch CUDA cache cleanup
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.warning(f"GPU cleanup failed: {e}")
    
    @staticmethod
    def safe_device_selection(preferred_device: str = 'auto', 
                            min_gpu_memory_gb: float = 4.0) -> str:
        """
        Safely select device based on available memory
        
        Args:
            preferred_device: 'auto', 'cuda', or 'cpu'
            min_gpu_memory_gb: Minimum GPU memory required for CUDA
        """
        if preferred_device == 'cpu':
            return 'cpu'
            
        if not torch.cuda.is_available():
            return 'cpu'
            
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_gb = gpu_props.total_memory / (1024**3)
            
            # Check if we have enough free memory
            torch.cuda.empty_cache()
            allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
            free_memory_gb = total_memory_gb - allocated_gb
            
            if free_memory_gb < min_gpu_memory_gb:
                logger.info(f"GPU has {free_memory_gb:.1f}GB free, need {min_gpu_memory_gb}GB. Using CPU.")
                return 'cpu'
            else:
                logger.info(f"GPU has {free_memory_gb:.1f}GB free. Using CUDA.")
                return 'cuda'
                
        except Exception as e:
            logger.warning(f"Error checking GPU memory: {e}. Using CPU.")
            return 'cpu'
    
    @staticmethod
    def estimate_memory_requirements(image_shape: tuple, 
                                   model_params: int,
                                   processing_factor: float = 4.0) -> Dict[str, float]:
        """
        Estimate memory requirements for processing
        
        Args:
            image_shape: (H, W, C) shape of input image
            model_params: Number of model parameters
            processing_factor: Memory multiplier for intermediate tensors
        """
        h, w, c = image_shape
        
        # Image memory (input + output + intermediate)
        image_memory_mb = (h * w * c * 4 * processing_factor) / (1024**2)  # float32
        
        # Model memory
        model_memory_mb = (model_params * 4) / (1024**2)  # float32 parameters
        
        # Total estimate
        total_mb = image_memory_mb + model_memory_mb
        
        return {
            'image_memory_mb': image_memory_mb,
            'model_memory_mb': model_memory_mb,
            'total_memory_mb': total_mb,
            'total_memory_gb': total_mb / 1024
        }
    
    @staticmethod
    def optimize_tile_size(image_shape: tuple,
                          max_memory_gb: float = 2.0,
                          min_tile_size: int = 64,
                          max_tile_size: int = 1024) -> int:
        """
        Calculate optimal tile size based on available memory
        
        Args:
            image_shape: (H, W, C) shape of input image
            max_memory_gb: Maximum memory to use for processing
            min_tile_size: Minimum allowed tile size
            max_tile_size: Maximum allowed tile size
        """
        h, w, c = image_shape
        max_memory_mb = max_memory_gb * 1024
        
        # Start with max tile size and reduce if needed
        for tile_size in range(max_tile_size, min_tile_size - 1, -64):
            # Estimate memory for this tile size
            tile_memory_mb = (tile_size * tile_size * c * 4 * 8) / (1024**2)  # Processing factor
            
            if tile_memory_mb <= max_memory_mb:
                return tile_size
                
        return min_tile_size

class SafeModelLoader:
    """Context manager for safe model loading and cleanup"""
    
    def __init__(self, device: str = 'auto', cleanup_on_exit: bool = True):
        self.device = MemoryManager.safe_device_selection(device)
        self.cleanup_on_exit = cleanup_on_exit
        self.initial_memory = None
        
    def __enter__(self):
        self.initial_memory = MemoryManager.get_memory_info()
        logger.info(f"Starting model loading on {self.device}")
        logger.info(f"Initial memory: {self.initial_memory.get('gpu_memory_free', 'N/A')} GB GPU free")
        return self.device
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            MemoryManager.cleanup_memory()
            
        final_memory = MemoryManager.get_memory_info()
        logger.info(f"Final memory: {final_memory.get('gpu_memory_free', 'N/A')} GB GPU free")
        
        if exc_type is not None:
            logger.error(f"Model loading failed: {exc_val}")

def safe_torch_operation(func):
    """Decorator for safe PyTorch operations with memory cleanup"""
    def wrapper(*args, **kwargs):
        try:
            # Clean up before operation
            MemoryManager.cleanup_memory()
            
            # Run operation
            result = func(*args, **kwargs)
            
            # Clean up after operation
            MemoryManager.cleanup_memory()
            
            return result
            
        except Exception as e:
            # Clean up on error
            MemoryManager.cleanup_memory()
            logger.error(f"Operation {func.__name__} failed: {e}")
            raise
            
    return wrapper

# Convenience functions
def get_safe_batch_size(image_size: int, base_batch_size: int = 4) -> int:
    """Get safe batch size based on image dimensions"""
    memory_info = MemoryManager.get_memory_info()
    
    if 'gpu_memory_free' in memory_info:
        free_gb = memory_info['gpu_memory_free']
        if free_gb < 2:
            return 1
        elif free_gb < 4:
            return max(1, base_batch_size // 2)
    
    # Adjust based on image size
    if image_size > 1024:
        return 1
    elif image_size > 512:
        return max(1, base_batch_size // 2)
    else:
        return base_batch_size

def check_memory_requirements(required_gb: float) -> bool:
    """Check if we have enough memory for an operation"""
    memory_info = MemoryManager.get_memory_info()
    
    # Check system RAM
    available_ram = memory_info.get('system_ram_available', 0)
    if available_ram < required_gb:
        return False
    
    # Check GPU memory if using CUDA
    if torch.cuda.is_available():
        gpu_free = memory_info.get('gpu_memory_free', 0)
        if gpu_free < required_gb:
            return False
            
    return True

# Example usage
if __name__ == "__main__":
    # Test memory utilities
    print("Memory Info:")
    info = MemoryManager.get_memory_info()
    for key, value in info.items():
        if 'memory' in key and isinstance(value, float):
            print(f"  {key}: {value:.2f} GB")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nSafe device: {MemoryManager.safe_device_selection()}")
    
    # Test tile size optimization
    tile_size = MemoryManager.optimize_tile_size((2048, 2048, 3))
    print(f"Optimal tile size for 2048x2048 image: {tile_size}")
    
    # Test memory requirements estimation
    reqs = MemoryManager.estimate_memory_requirements((1024, 1024, 3), 1000000)
    print(f"Memory requirements: {reqs['total_memory_gb']:.2f} GB")
