"""
GPU-accelerated image processing utilities
Uses CuPy for CUDA acceleration when available
"""

import numpy as np
import cv2
from typing import Optional, Union, Tuple, Any
import warnings

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    import cupyx.scipy.signal as cp_signal
    CUPY_AVAILABLE = True
    print("✓ CuPy available - GPU acceleration enabled")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠ CuPy not available - using CPU fallback")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_gpu_info():
    """Get GPU information and memory status"""
    if not CUPY_AVAILABLE:
        return {"available": False, "device": None, "memory": None}
    
    try:
        device = cp.cuda.Device()
        memory = cp.cuda.MemoryPool().used_bytes()
        return {
            "available": True,
            "device": device.id,
            "memory": memory,
            "name": device.compute_capability
        }
    except:
        return {"available": False, "device": None, "memory": None}


def to_gpu(array: np.ndarray, force_cpu: bool = False) -> Union[np.ndarray, Any]:
    """Transfer array to GPU if available"""
    if force_cpu or not CUPY_AVAILABLE:
        return array
    
    try:
        return cp.asarray(array)
    except:
        warnings.warn("GPU transfer failed, using CPU")
        return array


def to_cpu(array: Union[np.ndarray, Any]) -> np.ndarray:
    """Transfer array back to CPU"""
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array


def gpu_gaussian_blur(image: np.ndarray, sigma: float, force_cpu: bool = False) -> np.ndarray:
    """GPU-accelerated Gaussian blur"""
    if force_cpu or not CUPY_AVAILABLE:
        return cv2.GaussianBlur(image, (0, 0), sigma)
    
    try:
        gpu_image = to_gpu(image.astype(np.float32))
        
        # Use CuPy's Gaussian filter
        if len(gpu_image.shape) == 3:
            # Process each channel
            result_channels = []
            for c in range(gpu_image.shape[2]):
                blurred = cp_ndimage.gaussian_filter(gpu_image[:, :, c], sigma=sigma)
                result_channels.append(blurred)
            gpu_result = cp.stack(result_channels, axis=2)
        else:
            gpu_result = cp_ndimage.gaussian_filter(gpu_image, sigma=sigma)
        
        return to_cpu(gpu_result).astype(image.dtype)
    
    except Exception as e:
        warnings.warn(f"GPU blur failed: {e}, using CPU")
        return cv2.GaussianBlur(image, (0, 0), sigma)


def gpu_bilateral_filter(image: np.ndarray, d: int, sigma_color: float, sigma_space: float, force_cpu: bool = False) -> np.ndarray:
    """GPU-accelerated bilateral filter (falls back to CPU for now)"""
    # Note: CuPy doesn't have bilateral filter, so we use CPU
    # FIXED: Ensure contiguous array before OpenCV operation
    image = np.ascontiguousarray(image)
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def gpu_non_local_means(image: np.ndarray, h: float, patch_size: int, patch_distance: int, force_cpu: bool = False) -> np.ndarray:
    """GPU-accelerated Non-Local Means (uses OpenCV's implementation)"""
    # OpenCV's implementation can use GPU if compiled with CUDA
    # FIXED: Ensure contiguous array before OpenCV operation
    image = np.ascontiguousarray(image)
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, patch_size, patch_distance)
    else:
        return cv2.fastNlMeansDenoising(image, None, h, patch_size, patch_distance)


def gpu_frequency_filter(image: np.ndarray, filter_func, force_cpu: bool = False) -> np.ndarray:
    """GPU-accelerated frequency domain filtering"""
    if force_cpu or not CUPY_AVAILABLE:
        # CPU fallback using numpy
        if len(image.shape) == 3:
            result_channels = []
            for c in range(image.shape[2]):
                channel = image[:, :, c].astype(np.float32) / 255.0
                
                # FFT
                f_transform = np.fft.fft2(channel)
                f_shifted = np.fft.fftshift(f_transform)
                
                # Apply filter
                filtered = filter_func(f_shifted)
                
                # IFFT
                f_ishifted = np.fft.ifftshift(filtered)
                result = np.fft.ifft2(f_ishifted)
                result = np.real(result)
                
                result_channels.append(result)
            
            result = np.stack(result_channels, axis=2)
        else:
            channel = image.astype(np.float32) / 255.0
            f_transform = np.fft.fft2(channel)
            f_shifted = np.fft.fftshift(f_transform)
            filtered = filter_func(f_shifted)
            f_ishifted = np.fft.ifftshift(filtered)
            result = np.fft.ifft2(f_ishifted)
            result = np.real(result)
        
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    try:
        # GPU implementation using CuPy
        gpu_image = to_gpu(image.astype(np.float32) / 255.0)
        
        if len(gpu_image.shape) == 3:
            result_channels = []
            for c in range(gpu_image.shape[2]):
                channel = gpu_image[:, :, c]
                
                # FFT
                f_transform = cp.fft.fft2(channel)
                f_shifted = cp.fft.fftshift(f_transform)
                
                # Apply filter
                filtered = filter_func(f_shifted)
                
                # IFFT
                f_ishifted = cp.fft.ifftshift(filtered)
                result = cp.fft.ifft2(f_ishifted)
                result = cp.real(result)
                
                result_channels.append(result)
            
            gpu_result = cp.stack(result_channels, axis=2)
        else:
            f_transform = cp.fft.fft2(gpu_image)
            f_shifted = cp.fft.fftshift(f_transform)
            filtered = filter_func(f_shifted)
            f_ishifted = cp.fft.ifftshift(filtered)
            gpu_result = cp.fft.ifft2(f_ishifted)
            gpu_result = cp.real(gpu_result)
        
        result = to_cpu(gpu_result)
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    except Exception as e:
        warnings.warn(f"GPU frequency filtering failed: {e}, using CPU")
        return gpu_frequency_filter(image, filter_func, force_cpu=True)


def gpu_memory_info():
    """Get GPU memory information"""
    if not CUPY_AVAILABLE:
        return {"available": False}
    
    try:
        mempool = cp.get_default_memory_pool()
        return {
            "available": True,
            "used_bytes": mempool.used_bytes(),
            "total_bytes": mempool.total_bytes(),
            "free_bytes": mempool.free_bytes()
        }
    except:
        return {"available": False}


def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if CUPY_AVAILABLE:
        try:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        except:
            pass


def estimate_gpu_memory_usage(image_shape: Tuple[int, ...], num_copies: int = 4) -> int:
    """Estimate GPU memory usage for image processing"""
    # Estimate memory needed for intermediate arrays
    pixels = np.prod(image_shape)
    bytes_per_pixel = 4  # float32
    return pixels * bytes_per_pixel * num_copies


def can_use_gpu(image_shape: Tuple[int, ...], safety_margin: float = 0.8) -> bool:
    """Check if GPU has enough memory for processing"""
    if not CUPY_AVAILABLE:
        return False
    
    try:
        estimated_usage = estimate_gpu_memory_usage(image_shape)
        gpu_info = gpu_memory_info()
        
        if gpu_info["available"]:
            # Check if we have enough free memory with safety margin
            available_memory = gpu_info["free_bytes"]
            return estimated_usage < (available_memory * safety_margin)
        
        return False
    except:
        return False
