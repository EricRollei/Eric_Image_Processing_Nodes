"""
GPU-Accelerated BM3D Denoising using pytorch-bm3d

This module provides GPU-accelerated BM3D denoising using the pytorch-bm3d library.
Requires CUDA-capable GPU and compiled CUDA extension.

Performance: ~15-30x faster than CPU BM3D
- 256x256: ~0.08s (vs ~2.5s CPU)
- 1080p: ~0.14s (vs ~5-10s CPU)
"""

import numpy as np
import torch

# Try to import pytorch-bm3d
try:
    from pytorch_bm3d import BM3D
    PYTORCH_BM3D_AVAILABLE = True
except ImportError:
    PYTORCH_BM3D_AVAILABLE = False
    BM3D = None

# Try to import CUDA extension
try:
    import bm3d_cuda
    BM3D_CUDA_AVAILABLE = True
except ImportError:
    BM3D_CUDA_AVAILABLE = False


def is_available():
    """
    Check if GPU BM3D is available.
    
    Returns:
        tuple: (available, reason) where available is bool and reason is str
    """
    if not PYTORCH_BM3D_AVAILABLE:
        return False, "pytorch-bm3d not installed"
    
    if not BM3D_CUDA_AVAILABLE:
        return False, "CUDA extension not compiled"
    
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    return True, "GPU BM3D available"


def get_preset_parameters():
    """
    Get preset parameter combinations for different noise types.
    
    Returns:
        dict: Preset name to parameters mapping
    """
    return {
        "light_noise": {
            "sigma": 10,
            "two_step": True,
            "description": "Light noise (sigma=10), two-step refinement"
        },
        "medium_noise": {
            "sigma": 25,
            "two_step": True,
            "description": "Medium noise (sigma=25), two-step refinement"
        },
        "heavy_noise": {
            "sigma": 40,
            "two_step": True,
            "description": "Heavy noise (sigma=40), two-step refinement"
        },
        "fast_light": {
            "sigma": 10,
            "two_step": False,
            "description": "Fast mode, light noise (single-step)"
        },
        "fast_medium": {
            "sigma": 25,
            "two_step": False,
            "description": "Fast mode, medium noise (single-step)"
        }
    }


def bm3d_gpu_denoise(image, sigma=25, two_step=True, device=None):
    """
    GPU-accelerated BM3D denoising.
    
    Args:
        image: Input image as numpy array [H, W, C] or [H, W], float [0-1] or uint8 [0-255]
        sigma: Noise standard deviation (0-100). Represents noise level in 0-255 scale.
               sigma=25 means "25/255 = 9.8% noise"
        two_step: Use two-step refinement (higher quality, ~1.5x slower)
                  True = equivalent to 'refilter' profile
                  False = single-step (faster)
        device: CUDA device (e.g., 'cuda:0' or torch.device). If None, uses cuda:0
        
    Returns:
        tuple: (denoised_image, info_dict)
            - denoised_image: numpy array same shape as input, float [0-1]
            - info_dict: Processing information
            
    Raises:
        RuntimeError: If GPU BM3D is not available
        ValueError: If input parameters are invalid
    """
    # Check availability
    available, reason = is_available()
    if not available:
        raise RuntimeError(f"GPU BM3D not available: {reason}")
    
    # Validate input
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if image.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")
    
    if sigma < 0 or sigma > 100:
        raise ValueError(f"Sigma must be in [0, 100], got {sigma}")
    
    # Store original properties
    original_shape = image.shape
    original_dtype = image.dtype
    is_grayscale = (image.ndim == 2)
    
    # Convert to float [0-1] if needed
    if image.dtype == np.uint8:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.astype(np.float32)
        # Ensure [0-1] range
        if image_float.max() > 1.0 or image_float.min() < 0.0:
            image_float = np.clip(image_float, 0.0, 1.0)
    
    # Handle grayscale
    if is_grayscale:
        image_float = image_float[:, :, np.newaxis]  # [H, W] -> [H, W, 1]
    
    h, w, c = image_float.shape
    
    # Prepare info dictionary
    info = {
        "method": "BM3D-GPU",
        "sigma": sigma,
        "two_step": two_step,
        "input_shape": original_shape,
        "channels": c,
        "device": str(device) if device else "cuda:0"
    }
    
    try:
        # Set device
        if device is None:
            device = torch.device("cuda:0")
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Convert to pytorch-bm3d format: [1, C, H, W], int32 [0-255]
        # NumPy: [H, W, C] float [0-1] -> PyTorch: [1, C, H, W] int [0-255]
        image_uint8 = (image_float * 255.0).clip(0, 255).astype(np.uint8)
        
        # Convert to torch tensor: [H, W, C] -> [C, H, W] -> [1, C, H, W]
        image_torch = torch.from_numpy(image_uint8).permute(2, 0, 1).unsqueeze(0)
        image_torch = image_torch.to(torch.int32).to(device)
        
        # CRITICAL: Ensure contiguous memory layout
        image_torch = image_torch.contiguous()
        
        info["torch_shape"] = list(image_torch.shape)
        info["torch_dtype"] = str(image_torch.dtype)
        
        # Calculate variance from sigma
        # variance = (sigma_in_255_scale)^2
        variance = float(sigma ** 2)
        info["variance"] = variance
        
        # Initialize BM3D model
        bm3d_model = BM3D(two_step=two_step)
        
        # Run denoising with timing
        torch.cuda.synchronize()
        import time
        start_time = time.time()
        
        with torch.no_grad():
            output_torch = bm3d_model(image_torch, variance=variance)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        info["processing_time"] = f"{elapsed:.3f}s"
        
        # Convert back to numpy: [1, C, H, W] int [0-255] -> [H, W, C] float [0-1]
        output_np = output_torch.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = output_np.astype(np.float32) / 255.0
        output_np = np.clip(output_np, 0.0, 1.0)
        
        # Restore original shape if grayscale
        if is_grayscale:
            output_np = output_np[:, :, 0]  # [H, W, 1] -> [H, W]
        
        # Calculate quality metrics if input wasn't too noisy
        if sigma <= 50:  # Only for reasonable noise levels
            input_flat = image_float.flatten()
            output_flat = output_np.flatten() if output_np.ndim == 3 else output_np[:, :, np.newaxis].flatten()
            
            # Mean Squared Error
            mse = np.mean((input_flat - output_flat[:len(input_flat)]) ** 2)
            if mse > 0:
                psnr = 10 * np.log10(1.0 / mse)
                info["psnr"] = f"{psnr:.2f} dB"
        
        # Clean up GPU memory
        del image_torch, output_torch, bm3d_model
        torch.cuda.empty_cache()
        
        info["status"] = "success"
        
        return output_np, info
        
    except Exception as e:
        info["status"] = "failed"
        info["error"] = str(e)
        
        # Clean up on error
        torch.cuda.empty_cache()
        
        raise RuntimeError(f"GPU BM3D denoising failed: {e}") from e


def get_optimal_sigma(noise_level_description):
    """
    Get optimal sigma value based on noise description.
    
    Args:
        noise_level_description: String describing noise level
        
    Returns:
        int: Recommended sigma value
    """
    noise_map = {
        "very_light": 5,
        "light": 10,
        "medium": 25,
        "heavy": 40,
        "very_heavy": 60
    }
    
    return noise_map.get(noise_level_description.lower(), 25)


def estimate_performance(width, height):
    """
    Estimate processing time for given image dimensions.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        dict: Performance estimates
    """
    # Based on benchmark: 1080p (1920x1080 = 2,073,600 pixels) takes ~0.14s
    pixels = width * height
    base_pixels = 1920 * 1080
    base_time = 0.14
    
    # Roughly linear scaling with pixel count
    estimated_time = (pixels / base_pixels) * base_time
    
    # CPU equivalent (rough estimate: 20-30x slower)
    estimated_cpu_time = estimated_time * 25
    
    return {
        "estimated_gpu_time": f"{estimated_time:.3f}s",
        "estimated_cpu_time": f"{estimated_cpu_time:.1f}s",
        "speedup": "~20-30x vs CPU",
        "pixels": f"{pixels:,}"
    }


# Convenience function for common use case
def denoise_image(image, noise_level="medium", quality="high"):
    """
    Simplified interface for common denoising scenarios.
    
    Args:
        image: Input image numpy array
        noise_level: "light", "medium", or "heavy"
        quality: "high" (two-step) or "fast" (single-step)
        
    Returns:
        tuple: (denoised_image, info_dict)
    """
    sigma_map = {
        "light": 10,
        "medium": 25,
        "heavy": 40
    }
    
    sigma = sigma_map.get(noise_level.lower(), 25)
    two_step = (quality.lower() == "high")
    
    return bm3d_gpu_denoise(image, sigma=sigma, two_step=two_step)
