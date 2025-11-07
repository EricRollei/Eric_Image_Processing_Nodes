"""
Wavelet denoising implementation using PyWavelets
Supports multiple wavelet types and thresholding methods
Enhanced with GPU acceleration for CUDA-capable systems
"""

import numpy as np
import pywt
import cv2
from typing import Optional, Union, Tuple
from skimage.restoration import estimate_sigma

# GPU acceleration support
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import local GPU utilities
from .gpu_utils import (
    to_gpu, to_cpu, can_use_gpu, 
    gpu_memory_info, cleanup_gpu_memory, 
    estimate_gpu_memory_usage
)


def wavelet_denoise(
    image: np.ndarray,
    wavelet: str = 'db8',
    sigma: Optional[float] = None,
    mode: str = 'soft',
    method: str = 'BayesShrink',
    levels: Optional[int] = None,
    multichannel: bool = True,
    rescale_sigma: bool = True
) -> np.ndarray:
    """
    Advanced wavelet denoising with multiple threshold selection methods
    
    Args:
        image: Input image as numpy array [H, W, C] with values 0-255
        wavelet: Wavelet type ('db8', 'db4', 'bior2.2', 'coif2', 'haar')
                 - 'db8': Daubechies 8 - Good all-around choice, smooth reconstruction
                 - 'db4': Daubechies 4 - Faster, good for natural images  
                 - 'bior2.2': Biorthogonal - Good for images with edges
                 - 'coif2': Coiflets - Good reconstruction properties
                 - 'haar': Haar - Fastest, good for cartoon-like images
        sigma: Noise standard deviation (auto-estimated if None)
               - Typical range: 5-25 for moderate noise, 25-50 for heavy noise
        mode: Thresholding mode ('soft', 'hard')
              - 'soft': Gradual transition, smoother results (recommended)
              - 'hard': Sharp cutoff, preserves edges better but may create artifacts
        method: Threshold selection method ('BayesShrink', 'VisuShrink', 'manual')
                - 'BayesShrink': Adaptive, optimal for most natural images
                - 'VisuShrink': Conservative, good for preserving details
                - 'manual': Use sigma parameter directly as threshold
        levels: Decomposition levels (auto-calculated if None)
                - Typical range: 3-6 levels, more levels = more denoising
        multichannel: Process color channels independently
        rescale_sigma: Whether to rescale sigma estimate (recommended True)
    
    Returns:
        Denoised image as numpy array [H, W, C] with values 0-255
    """
    
    # Validate inputs
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")
    
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Handle grayscale vs color
    if img_float.ndim == 2:
        return _denoise_single_channel(
            img_float, wavelet, sigma, mode, method, levels, rescale_sigma
        ) * 255.0
    
    # Process color image
    if multichannel:
        # Process each channel independently
        denoised_channels = []
        for c in range(img_float.shape[2]):
            denoised_channel = _denoise_single_channel(
                img_float[:, :, c], wavelet, sigma, mode, method, levels, rescale_sigma
            )
            denoised_channels.append(denoised_channel)
        
        result = np.stack(denoised_channels, axis=2)
    else:
        # Convert to grayscale and denoise
        gray = np.mean(img_float, axis=2)
        denoised_gray = _denoise_single_channel(
            gray, wavelet, sigma, mode, method, levels, rescale_sigma
        )
        # Broadcast back to color
        result = np.stack([denoised_gray] * img_float.shape[2], axis=2)
    
    # Convert back to uint8 range
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    return result


def _denoise_single_channel(
    image: np.ndarray,
    wavelet: str,
    sigma: Optional[float],
    mode: str,
    method: str,
    levels: Optional[int],
    rescale_sigma: bool
) -> np.ndarray:
    """Denoise a single channel image using wavelets"""
    
    # Auto-calculate decomposition levels if not specified
    if levels is None:
        # Calculate max levels manually (compatible with all PyWavelets versions)
        min_size = min(image.shape)
        # More conservative level calculation for better performance
        levels = min(5, max(2, int(np.log2(min_size)) - 2))
    else:
        # Validate user-provided levels
        min_size = min(image.shape)
        max_levels = int(np.log2(min_size)) - 1
        if levels > max_levels:
            print(f"Warning: Requested levels ({levels}) exceeds maximum for image size ({max_levels}). Using {max_levels} instead.")
            levels = max_levels
        elif levels < 1:
            print(f"Warning: Levels must be >= 1. Using 1 instead of {levels}.")
            levels = 1
    
    # Estimate noise if sigma not provided
    if sigma is None:
        sigma = estimate_sigma(image, channel_axis=None)
        if rescale_sigma:
            # Improved rescaling factor based on wavelet type
            if wavelet.startswith('db'):
                sigma *= 1.1  # Daubechies wavelets
            elif wavelet.startswith('bior'):
                sigma *= 0.9  # Biorthogonal wavelets
            else:
                sigma *= 1.0  # Default
    
    # Perform wavelet decomposition - use correct parameter name
    coeffs = pywt.wavedec2(image, wavelet, level=levels, mode='symmetric')
    
    # Apply thresholding based on method
    if method == 'BayesShrink':
        coeffs_thresh = _bayes_shrink_threshold(coeffs, sigma)
    elif method == 'VisuShrink':
        threshold = sigma * np.sqrt(2 * np.log(image.size))
        coeffs_thresh = _apply_threshold(coeffs, threshold, mode)
    elif method == 'SureShrink':
        coeffs_thresh = _sure_shrink_threshold(coeffs, sigma, mode)
    elif method == 'manual':
        # For manual method, use sigma as noise level but calculate appropriate threshold
        # This prevents using sigma directly as threshold which is often too aggressive
        manual_threshold = sigma * 0.1  # More conservative threshold
        coeffs_thresh = _apply_threshold(coeffs, manual_threshold, mode)
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
    
    # Reconstruct image
    denoised = pywt.waverec2(coeffs_thresh, wavelet, mode='symmetric')
    
    # Ensure output is same size as input (handle boundary effects)
    if denoised.shape != image.shape:
        # Crop to original size
        h, w = image.shape
        denoised = denoised[:h, :w]
    
    return np.clip(denoised, 0, 1)


def _bayes_shrink_threshold(coeffs, sigma):
    """Apply BayesShrink thresholding to wavelet coefficients"""
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients unchanged
    
    for i in range(1, len(coeffs)):
        # For each detail level
        detail_coeffs = []
        for detail in coeffs[i]:  # (cH, cV, cD)
            # Estimate signal variance
            var_y = np.var(detail)
            var_x = max(0, var_y - sigma**2)
            
            if var_x > 0:
                # Calculate Bayes threshold
                threshold = sigma**2 / np.sqrt(var_x)
                # Apply soft thresholding
                detail_thresh = pywt.threshold(detail, threshold, mode='soft')
            else:
                # If no signal estimated, use conservative threshold instead of zeroing
                # This prevents over-denoising when noise estimation is poor
                conservative_threshold = sigma * 0.5  # Much more conservative
                detail_thresh = pywt.threshold(detail, conservative_threshold, mode='soft')
            
            detail_coeffs.append(detail_thresh)
        
        coeffs_thresh.append(tuple(detail_coeffs))
    
    return coeffs_thresh


def _apply_threshold(coeffs, threshold, mode):
    """Apply uniform threshold to all detail coefficients"""
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
    
    for i in range(1, len(coeffs)):
        detail_coeffs = []
        for detail in coeffs[i]:
            detail_thresh = pywt.threshold(detail, threshold, mode=mode)
            detail_coeffs.append(detail_thresh)
        coeffs_thresh.append(tuple(detail_coeffs))
    
    return coeffs_thresh


def _sure_shrink_threshold(coeffs, sigma, mode):
    """Apply SureShrink thresholding to wavelet coefficients
    
    SureShrink is a hybrid approach that selects between soft and hard thresholding
    based on Stein's Unbiased Risk Estimator (SURE).
    """
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients unchanged
    
    for i in range(1, len(coeffs)):
        # For each detail level
        detail_coeffs = []
        for detail in coeffs[i]:  # (cH, cV, cD)
            # Calculate threshold using SURE
            n = detail.size
            sorted_coeffs = np.sort(np.abs(detail.flatten()))
            
            # Calculate risk for each threshold
            risks = []
            thresholds = sorted_coeffs[::max(1, n//100)]  # Sample thresholds
            
            for t in thresholds:
                # Number of coefficients below threshold
                k = np.sum(np.abs(detail) <= t)
                
                # SURE formula
                risk = n - 2*k + np.sum(np.minimum(np.abs(detail), t)**2)
                risks.append(risk)
            
            # Find threshold with minimum risk
            if risks:
                min_idx = np.argmin(risks)
                threshold = thresholds[min_idx]
            else:
                threshold = sigma * np.sqrt(2 * np.log(n))  # Fallback to VisuShrink
            
            # Apply thresholding
            detail_thresh = pywt.threshold(detail, threshold, mode=mode)
            detail_coeffs.append(detail_thresh)
        
        coeffs_thresh.append(tuple(detail_coeffs))
    
    return coeffs_thresh


def wavelet_denoise_stationary(
    image: np.ndarray,
    wavelet: str = 'db8',
    sigma: Optional[float] = None,
    mode: str = 'soft',
    method: str = 'BayesShrink',
    levels: Optional[int] = None,
    multichannel: bool = True,
    rescale_sigma: bool = True
) -> np.ndarray:
    """
    Stationary (translation-invariant) wavelet denoising
    
    This method uses the Stationary Wavelet Transform (SWT) which is shift-invariant,
    producing better results for images with important structures at different positions.
    
    Args:
        Same as wavelet_denoise() but uses SWT instead of DWT
        
    Returns:
        Denoised image with reduced shift-sensitivity artifacts
    """
    
    # Validate inputs
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")
    
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Handle grayscale vs color
    if img_float.ndim == 2:
        result = _denoise_single_channel_stationary(
            img_float, wavelet, sigma, mode, method, levels, rescale_sigma
        )
    else:
        if multichannel:
            # Process each channel independently
            result_channels = []
            for c in range(img_float.shape[2]):
                denoised_channel = _denoise_single_channel_stationary(
                    img_float[:, :, c], wavelet, sigma, mode, method, levels, rescale_sigma
                )
                result_channels.append(denoised_channel)
            
            result = np.stack(result_channels, axis=2)
        else:
            # Convert to grayscale and denoise
            gray = np.mean(img_float, axis=2)
            denoised_gray = _denoise_single_channel_stationary(
                gray, wavelet, sigma, mode, method, levels, rescale_sigma
            )
            # Broadcast back to color
            result = np.stack([denoised_gray] * img_float.shape[2], axis=2)
    
    # Convert back to uint8 range
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    return result


def _denoise_single_channel_stationary(
    image: np.ndarray,
    wavelet: str,
    sigma: Optional[float],
    mode: str,
    method: str,
    levels: Optional[int],
    rescale_sigma: bool
) -> np.ndarray:
    """Denoise a single channel using Stationary Wavelet Transform"""
    
    # Auto-calculate decomposition levels if not specified
    if levels is None:
        min_size = min(image.shape)
        levels = min(4, max(2, int(np.log2(min_size)) - 3))  # Conservative for SWT
    else:
        # Validate user-provided levels for stationary wavelet
        min_size = min(image.shape)
        max_levels = int(np.log2(min_size)) - 2  # Even more conservative for SWT
        if levels > max_levels:
            print(f"Warning: Requested SWT levels ({levels}) exceeds maximum for image size ({max_levels}). Using {max_levels} instead.")
            levels = max_levels
        elif levels < 1:
            print(f"Warning: SWT levels must be >= 1. Using 1 instead of {levels}.")
            levels = 1
    
    # Estimate noise if sigma not provided
    if sigma is None:
        sigma = estimate_sigma(image, channel_axis=None)
        if rescale_sigma:
            sigma *= 1.0
    
    # Perform stationary wavelet decomposition
    coeffs = pywt.swt2(image, wavelet, level=levels, norm=True)
    
    # Apply thresholding based on method
    if method == 'BayesShrink':
        coeffs_thresh = _bayes_shrink_threshold_swt(coeffs, sigma)
    elif method == 'VisuShrink':
        threshold = sigma * np.sqrt(2 * np.log(image.size))
        coeffs_thresh = _apply_threshold_swt(coeffs, threshold, mode)
    elif method == 'SureShrink':
        coeffs_thresh = _sure_shrink_threshold_swt(coeffs, sigma, mode)
    elif method == 'manual':
        # For manual method, use sigma as noise level but calculate appropriate threshold
        manual_threshold = sigma * 0.1  # More conservative threshold
        coeffs_thresh = _apply_threshold_swt(coeffs, manual_threshold, mode)
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
    
    # Reconstruct image
    denoised = pywt.iswt2(coeffs_thresh, wavelet, norm=True)
    
    # Ensure output is same size as input
    if denoised.shape != image.shape:
        h, w = image.shape
        denoised = denoised[:h, :w]
    
    return np.clip(denoised, 0, 1)


def _bayes_shrink_threshold_swt(coeffs, sigma):
    """Apply BayesShrink thresholding to SWT coefficients"""
    coeffs_thresh = []
    
    for i, (approx, (cH, cV, cD)) in enumerate(coeffs):
        if i == 0:
            # Keep first level approximation coefficients unchanged
            coeffs_thresh.append((approx, (cH, cV, cD)))
        else:
            # Process detail coefficients
            detail_coeffs = []
            for detail in [cH, cV, cD]:
                # Estimate signal variance
                var_y = np.var(detail)
                var_x = max(0, var_y - sigma**2)
                
                if var_x > 0:
                    # Calculate Bayes threshold
                    threshold = sigma**2 / np.sqrt(var_x)
                    # Apply soft thresholding
                    detail_thresh = pywt.threshold(detail, threshold, mode='soft')
                else:
                    # If no signal estimated, use conservative threshold instead of zeroing
                    conservative_threshold = sigma * 0.5  # Much more conservative
                    detail_thresh = pywt.threshold(detail, conservative_threshold, mode='soft')
                
                detail_coeffs.append(detail_thresh)
            
            coeffs_thresh.append((approx, tuple(detail_coeffs)))
    
    return coeffs_thresh


def _apply_threshold_swt(coeffs, threshold, mode):
    """Apply uniform threshold to SWT coefficients"""
    coeffs_thresh = []
    
    for i, (approx, (cH, cV, cD)) in enumerate(coeffs):
        if i == 0:
            # Keep first level approximation coefficients unchanged
            coeffs_thresh.append((approx, (cH, cV, cD)))
        else:
            detail_coeffs = []
            for detail in [cH, cV, cD]:
                detail_thresh = pywt.threshold(detail, threshold, mode=mode)
                detail_coeffs.append(detail_thresh)
            coeffs_thresh.append((approx, tuple(detail_coeffs)))
    
    return coeffs_thresh


def _sure_shrink_threshold_swt(coeffs, sigma, mode):
    """Apply SureShrink thresholding to SWT coefficients
    
    SureShrink is a hybrid approach that selects between soft and hard thresholding
    based on Stein's Unbiased Risk Estimator (SURE).
    """
    coeffs_thresh = []
    
    for i, (approx, (cH, cV, cD)) in enumerate(coeffs):
        if i == 0:
            # Keep first level approximation coefficients unchanged
            coeffs_thresh.append((approx, (cH, cV, cD)))
        else:
            detail_coeffs = []
            for detail in [cH, cV, cD]:
                # Calculate threshold using SURE
                n = detail.size
                sorted_coeffs = np.sort(np.abs(detail.flatten()))
                
                # Calculate risk for each threshold
                risks = []
                thresholds = sorted_coeffs[::max(1, n//50)]  # Sample thresholds
                
                for t in thresholds:
                    # Number of coefficients below threshold
                    k = np.sum(np.abs(detail) <= t)
                    
                    # SURE formula
                    risk = n - 2*k + np.sum(np.minimum(np.abs(detail), t)**2)
                    risks.append(risk)
                
                # Find threshold with minimum risk
                if risks:
                    min_idx = np.argmin(risks)
                    threshold = thresholds[min_idx]
                else:
                    threshold = sigma * np.sqrt(2 * np.log(n))  # Fallback to VisuShrink
                
                # Apply thresholding
                detail_thresh = pywt.threshold(detail, threshold, mode=mode)
                detail_coeffs.append(detail_thresh)
        
        coeffs_thresh.append((approx, tuple(detail_coeffs)))
    
    return coeffs_thresh


def gpu_wavelet_denoise(
    image: np.ndarray,
    wavelet: str = 'db8',
    sigma: Optional[float] = None,
    mode: str = 'soft',
    method: str = 'BayesShrink',
    levels: Optional[int] = None,
    multichannel: bool = True,
    rescale_sigma: bool = True,
    force_cpu: bool = False
) -> np.ndarray:
    """
    GPU-accelerated wavelet denoising with automatic fallback to CPU
    
    Args:
        image: Input image as numpy array [H, W, C] with values 0-255
        wavelet: Wavelet type ('db8', 'db4', 'bior2.2', 'coif2', 'haar')
        sigma: Noise standard deviation (auto-estimated if None)
        mode: Thresholding mode ('soft', 'hard')
        method: Threshold selection method ('BayesShrink', 'VisuShrink', 'SureShrink', 'manual')
        levels: Decomposition levels (auto-calculated if None)
        multichannel: Process color channels independently
        rescale_sigma: Whether to rescale sigma estimate
        force_cpu: Force CPU processing even if GPU is available
    
    Returns:
        Denoised image as numpy array [H, W, C] with values 0-255
    """
    
    # Check if GPU acceleration is beneficial
    use_gpu = (
        not force_cpu and 
        CUPY_AVAILABLE and 
        can_use_gpu(image.shape) and
        np.prod(image.shape) > 512*512  # Only use GPU for larger images
    )
    
    if use_gpu:
        try:
            return _gpu_wavelet_denoise_impl(
                image, wavelet, sigma, mode, method, levels, multichannel, rescale_sigma
            )
        except Exception as e:
            print(f"GPU wavelet denoising failed: {e}, falling back to CPU")
            cleanup_gpu_memory()
    
    # CPU fallback
    return wavelet_denoise(
        image, wavelet, sigma, mode, method, levels, multichannel, rescale_sigma
    )


def _gpu_wavelet_denoise_impl(
    image: np.ndarray,
    wavelet: str,
    sigma: Optional[float],
    mode: str,
    method: str,
    levels: Optional[int],
    multichannel: bool,
    rescale_sigma: bool
) -> np.ndarray:
    """GPU implementation of wavelet denoising"""
    
    # Validate inputs
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")
    
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    gpu_img = to_gpu(img_float)
    
    # Handle grayscale vs color
    if gpu_img.ndim == 2:
        result = _gpu_denoise_single_channel(
            gpu_img, wavelet, sigma, mode, method, levels, rescale_sigma
        )
    else:
        if multichannel:
            # Process each channel independently on GPU
            result_channels = []
            for c in range(gpu_img.shape[2]):
                denoised_channel = _gpu_denoise_single_channel(
                    gpu_img[:, :, c], wavelet, sigma, mode, method, levels, rescale_sigma
                )
                result_channels.append(denoised_channel)
            
            result = cp.stack(result_channels, axis=2)
        else:
            # Convert to grayscale and denoise
            gray = cp.mean(gpu_img, axis=2)
            denoised_gray = _gpu_denoise_single_channel(
                gray, wavelet, sigma, mode, method, levels, rescale_sigma
            )
            # Broadcast back to color
            result = cp.stack([denoised_gray] * gpu_img.shape[2], axis=2)
    
    # Convert back to CPU and uint8 range
    result_cpu = to_cpu(result)
    result_cpu = np.clip(result_cpu * 255.0, 0, 255).astype(np.uint8)
    
    return result_cpu


def _gpu_denoise_single_channel(
    gpu_image,
    wavelet: str,
    sigma: Optional[float],
    mode: str,
    method: str,
    levels: Optional[int],
    rescale_sigma: bool
):
    """GPU-accelerated single channel denoising"""
    
    # Convert to CPU for wavelet transform (PyWavelets doesn't support GPU directly)
    cpu_image = to_cpu(gpu_image)
    
    # Auto-calculate decomposition levels if not specified
    if levels is None:
        min_size = min(cpu_image.shape)
        levels = min(5, max(2, int(np.log2(min_size)) - 2))
    
    # Estimate noise if sigma not provided
    if sigma is None:
        sigma = estimate_sigma(cpu_image, channel_axis=None)
        if rescale_sigma:
            if wavelet.startswith('db'):
                sigma *= 1.1
            elif wavelet.startswith('bior'):
                sigma *= 0.9
            else:
                sigma *= 1.0
    
    # Perform wavelet decomposition on CPU
    coeffs = pywt.wavedec2(cpu_image, wavelet, level=levels, mode='symmetric')
    
    # Apply thresholding using GPU when possible
    if method == 'BayesShrink':
        coeffs_thresh = _gpu_bayes_shrink_threshold(coeffs, sigma)
    elif method == 'VisuShrink':
        threshold = sigma * np.sqrt(2 * np.log(cpu_image.size))
        coeffs_thresh = _gpu_apply_threshold(coeffs, threshold, mode)
    elif method == 'SureShrink':
        coeffs_thresh = _gpu_sure_shrink_threshold(coeffs, sigma, mode)
    elif method == 'manual':
        coeffs_thresh = _gpu_apply_threshold(coeffs, sigma, mode)
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
    
    # Reconstruct image on CPU
    denoised = pywt.waverec2(coeffs_thresh, wavelet, mode='symmetric')
    
    # Ensure output is same size as input
    if denoised.shape != cpu_image.shape:
        h, w = cpu_image.shape
        denoised = denoised[:h, :w]
    
    # Convert back to GPU for final processing
    gpu_result = to_gpu(np.clip(denoised, 0, 1))
    
    return gpu_result


def _gpu_bayes_shrink_threshold(coeffs, sigma):
    """GPU-accelerated BayesShrink thresholding"""
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients unchanged
    
    for i in range(1, len(coeffs)):
        detail_coeffs = []
        for detail in coeffs[i]:
            # Move to GPU for variance calculation
            gpu_detail = to_gpu(detail)
            
            # Calculate variance on GPU
            var_y = cp.var(gpu_detail)
            var_x = cp.maximum(0, var_y - sigma**2)
            
            if float(var_x) > 0:
                # Calculate Bayes threshold
                threshold = sigma**2 / cp.sqrt(var_x)
                # Apply soft thresholding on GPU
                gpu_detail_thresh = cp.where(
                    cp.abs(gpu_detail) > threshold,
                    cp.sign(gpu_detail) * (cp.abs(gpu_detail) - threshold),
                    0
                )
            else:
                gpu_detail_thresh = cp.zeros_like(gpu_detail)
            
            # Convert back to CPU for wavelet reconstruction
            detail_thresh = to_cpu(gpu_detail_thresh)
            detail_coeffs.append(detail_thresh)
        
        coeffs_thresh.append(tuple(detail_coeffs))
    
    return coeffs_thresh


def _gpu_apply_threshold(coeffs, threshold, mode):
    """GPU-accelerated uniform thresholding"""
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
    
    for i in range(1, len(coeffs)):
        detail_coeffs = []
        for detail in coeffs[i]:
            gpu_detail = to_gpu(detail)
            
            if mode == 'soft':
                # Soft thresholding on GPU
                gpu_detail_thresh = cp.where(
                    cp.abs(gpu_detail) > threshold,
                    cp.sign(gpu_detail) * (cp.abs(gpu_detail) - threshold),
                    0
                )
            else:  # hard thresholding
                gpu_detail_thresh = cp.where(
                    cp.abs(gpu_detail) > threshold,
                    gpu_detail,
                    0
                )
            
            detail_thresh = to_cpu(gpu_detail_thresh)
            detail_coeffs.append(detail_thresh)
        
        coeffs_thresh.append(tuple(detail_coeffs))
    
    return coeffs_thresh


def _gpu_sure_shrink_threshold(coeffs, sigma, mode):
    """GPU-accelerated SureShrink thresholding"""
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients unchanged
    
    for i in range(1, len(coeffs)):
        detail_coeffs = []
        for detail in coeffs[i]:
            gpu_detail = to_gpu(detail)
            n = gpu_detail.size
            
            # Calculate SURE on GPU
            sorted_coeffs = cp.sort(cp.abs(gpu_detail.flatten()))
            
            # Sample thresholds for efficiency
            sample_size = min(100, n)
            indices = cp.linspace(0, n-1, sample_size, dtype=int)
            thresholds = sorted_coeffs[indices]
            
            # Calculate risk for each threshold
            risks = []
            for t in thresholds:
                k = cp.sum(cp.abs(gpu_detail) <= t)
                risk = n - 2*k + cp.sum(cp.minimum(cp.abs(gpu_detail), t)**2)
                risks.append(float(risk))
            
            # Find threshold with minimum risk
            if risks:
                min_idx = np.argmin(risks)
                threshold = float(thresholds[min_idx])
            else:
                threshold = sigma * np.sqrt(2 * np.log(n))  # Fallback
            
            # Apply thresholding on GPU
            if mode == 'soft':
                gpu_detail_thresh = cp.where(
                    cp.abs(gpu_detail) > threshold,
                    cp.sign(gpu_detail) * (cp.abs(gpu_detail) - threshold),
                    0
                )
            else:  # hard thresholding
                gpu_detail_thresh = cp.where(
                    cp.abs(gpu_detail) > threshold,
                    gpu_detail,
                    0
                )
            
            detail_thresh = to_cpu(gpu_detail_thresh)
            detail_coeffs.append(detail_thresh)
        
        coeffs_thresh.append(tuple(detail_coeffs))
    
    return coeffs_thresh


def gpu_wavelet_denoise_stationary(
    image: np.ndarray,
    wavelet: str = 'db8',
    sigma: Optional[float] = None,
    mode: str = 'soft',
    method: str = 'BayesShrink',
    levels: Optional[int] = None,
    multichannel: bool = True,
    rescale_sigma: bool = True,
    force_cpu: bool = False
) -> np.ndarray:
    """
    GPU-accelerated stationary wavelet denoising
    
    Uses the Stationary Wavelet Transform (SWT) which is shift-invariant,
    producing better results for images with important structures at different positions.
    
    Args:
        Same as gpu_wavelet_denoise() but uses SWT instead of DWT
        
    Returns:
        Denoised image with reduced shift-sensitivity artifacts
    """
    
    # Check if GPU acceleration is beneficial
    use_gpu = (
        not force_cpu and 
        CUPY_AVAILABLE and 
        can_use_gpu(image.shape) and
        np.prod(image.shape) > 512*512
    )
    
    if use_gpu:
        try:
            return _gpu_wavelet_denoise_stationary_impl(
                image, wavelet, sigma, mode, method, levels, multichannel, rescale_sigma
            )
        except Exception as e:
            print(f"GPU stationary wavelet denoising failed: {e}, falling back to CPU")
            cleanup_gpu_memory()
    
    # CPU fallback
    return wavelet_denoise_stationary(
        image, wavelet, sigma, mode, method, levels, multichannel, rescale_sigma
    )


def _gpu_wavelet_denoise_stationary_impl(
    image: np.ndarray,
    wavelet: str,
    sigma: Optional[float],
    mode: str,
    method: str,
    levels: Optional[int],
    multichannel: bool,
    rescale_sigma: bool
) -> np.ndarray:
    """GPU implementation of stationary wavelet denoising"""
    
    # Validate inputs
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")
    
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    gpu_img = to_gpu(img_float)
    
    # Handle grayscale vs color
    if gpu_img.ndim == 2:
        result = _gpu_denoise_single_channel_stationary(
            gpu_img, wavelet, sigma, mode, method, levels, rescale_sigma
        )
    else:
        if multichannel:
            # Process each channel independently on GPU
            result_channels = []
            for c in range(gpu_img.shape[2]):
                denoised_channel = _gpu_denoise_single_channel_stationary(
                    gpu_img[:, :, c], wavelet, sigma, mode, method, levels, rescale_sigma
                )
                result_channels.append(denoised_channel)
            
            result = cp.stack(result_channels, axis=2)
        else:
            # Convert to grayscale and denoise
            gray = cp.mean(gpu_img, axis=2)
            denoised_gray = _gpu_denoise_single_channel_stationary(
                gray, wavelet, sigma, mode, method, levels, rescale_sigma
            )
            # Broadcast back to color
            result = cp.stack([denoised_gray] * gpu_img.shape[2], axis=2)
    
    # Convert back to CPU and uint8 range
    result_cpu = to_cpu(result)
    result_cpu = np.clip(result_cpu * 255.0, 0, 255).astype(np.uint8)
    
    return result_cpu


def _gpu_denoise_single_channel_stationary(
    gpu_image,
    wavelet: str,
    sigma: Optional[float],
    mode: str,
    method: str,
    levels: Optional[int],
    rescale_sigma: bool
):
    """GPU-accelerated single channel stationary denoising"""
    
    # Convert to CPU for stationary wavelet transform
    cpu_image = to_cpu(gpu_image)
    
    # Auto-calculate decomposition levels if not specified
    if levels is None:
        min_size = min(cpu_image.shape)
        levels = min(4, max(2, int(np.log2(min_size)) - 3))  # More conservative for SWT
    
    # Estimate noise if sigma not provided
    if sigma is None:
        sigma = estimate_sigma(cpu_image, channel_axis=None)
        if rescale_sigma:
            if wavelet.startswith('db'):
                sigma *= 1.1
            elif wavelet.startswith('bior'):
                sigma *= 0.9
            else:
                sigma *= 1.0
    
    # Perform stationary wavelet decomposition on CPU
    coeffs = pywt.swt2(cpu_image, wavelet, level=levels, trim_approx=True)
    
    # Apply thresholding using GPU when possible
    if method == 'BayesShrink':
        coeffs_thresh = _gpu_bayes_shrink_threshold_swt(coeffs, sigma)
    elif method == 'VisuShrink':
        threshold = sigma * np.sqrt(2 * np.log(cpu_image.size))
        coeffs_thresh = _gpu_apply_threshold_swt(coeffs, threshold, mode)
    elif method == 'SureShrink':
        coeffs_thresh = _gpu_sure_shrink_threshold_swt(coeffs, sigma, mode)
    elif method == 'manual':
        coeffs_thresh = _gpu_apply_threshold_swt(coeffs, sigma, mode)
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
    
    # Reconstruct image on CPU
    denoised = pywt.iswt2(coeffs_thresh, wavelet)
    
    # Ensure output is same size as input
    if denoised.shape != cpu_image.shape:
        h, w = cpu_image.shape
        denoised = denoised[:h, :w]
    
    # Convert back to GPU for final processing
    gpu_result = to_gpu(np.clip(denoised, 0, 1))
    
    return gpu_result


def _gpu_bayes_shrink_threshold_swt(coeffs, sigma):
    """GPU-accelerated BayesShrink thresholding for SWT coefficients"""
    coeffs_thresh = []
    
    for i, (cA, (cH, cV, cD)) in enumerate(coeffs):
        # For SWT, we threshold all coefficients including approximation at deeper levels
        if i == len(coeffs) - 1:
            # Keep the final approximation coefficients
            cA_thresh = cA
        else:
            # Threshold approximation coefficients at deeper levels
            gpu_cA = to_gpu(cA)
            var_y = cp.var(gpu_cA)
            var_x = cp.maximum(0, var_y - sigma**2)
            
            if float(var_x) > 0:
                threshold = sigma**2 / cp.sqrt(var_x)
                cA_thresh = to_cpu(cp.where(
                    cp.abs(gpu_cA) > threshold,
                    cp.sign(gpu_cA) * (cp.abs(gpu_cA) - threshold),
                    0
                ))
            else:
                cA_thresh = np.zeros_like(cA)
        
        # Threshold detail coefficients
        detail_coeffs = []
        for detail in [cH, cV, cD]:
            gpu_detail = to_gpu(detail)
            var_y = cp.var(gpu_detail)
            var_x = cp.maximum(0, var_y - sigma**2)
            
            if float(var_x) > 0:
                threshold = sigma**2 / cp.sqrt(var_x)
                detail_thresh = to_cpu(cp.where(
                    cp.abs(gpu_detail) > threshold,
                    cp.sign(gpu_detail) * (cp.abs(gpu_detail) - threshold),
                    0
                ))
            else:
                detail_thresh = np.zeros_like(detail)
            
            detail_coeffs.append(detail_thresh)
        
        coeffs_thresh.append((cA_thresh, tuple(detail_coeffs)))
    
    return coeffs_thresh


def _gpu_apply_threshold_swt(coeffs, threshold, mode):
    """GPU-accelerated uniform thresholding for SWT coefficients"""
    coeffs_thresh = []
    
    for i, (cA, (cH, cV, cD)) in enumerate(coeffs):
        # For SWT, we can threshold all coefficients
        if i == len(coeffs) - 1:
            # Keep the final approximation coefficients
            cA_thresh = cA
        else:
            # Threshold approximation coefficients at deeper levels
            gpu_cA = to_gpu(cA)
            if mode == 'soft':
                cA_thresh = to_cpu(cp.where(
                    cp.abs(gpu_cA) > threshold,
                    cp.sign(gpu_cA) * (cp.abs(gpu_cA) - threshold),
                    0
                ))
            else:  # hard thresholding
                cA_thresh = to_cpu(cp.where(
                    cp.abs(gpu_cA) > threshold,
                    gpu_cA,
                    0
                ))
        
        # Threshold detail coefficients
        detail_coeffs = []
        for detail in [cH, cV, cD]:
            gpu_detail = to_gpu(detail)
            
            if mode == 'soft':
                detail_thresh = to_cpu(cp.where(
                    cp.abs(gpu_detail) > threshold,
                    cp.sign(gpu_detail) * (cp.abs(gpu_detail) - threshold),
                    0
                ))
            else:  # hard thresholding
                detail_thresh = to_cpu(cp.where(
                    cp.abs(gpu_detail) > threshold,
                    gpu_detail,
                    0
                ))
            
            detail_coeffs.append(detail_thresh)
        
        coeffs_thresh.append((cA_thresh, tuple(detail_coeffs)))
    
    return coeffs_thresh


def _gpu_sure_shrink_threshold_swt(coeffs, sigma, mode):
    """GPU-accelerated SureShrink thresholding for SWT coefficients"""
    coeffs_thresh = []
    
    for i, (cA, (cH, cV, cD)) in enumerate(coeffs):
        # For SWT, we can threshold all coefficients
        if i == len(coeffs) - 1:
            # Keep the final approximation coefficients
            cA_thresh = cA
        else:
            # Apply SURE to approximation coefficients at deeper levels
            cA_thresh = _gpu_sure_threshold_single(cA, sigma, mode)
        
        # Apply SURE to detail coefficients
        detail_coeffs = []
        for detail in [cH, cV, cD]:
            detail_thresh = _gpu_sure_threshold_single(detail, sigma, mode)
            detail_coeffs.append(detail_thresh)
        
        coeffs_thresh.append((cA_thresh, tuple(detail_coeffs)))
    
    return coeffs_thresh


def _gpu_sure_threshold_single(detail, sigma, mode):
    """Apply SURE thresholding to a single coefficient array"""
    gpu_detail = to_gpu(detail)
    n = gpu_detail.size
    
    # Calculate SURE on GPU
    sorted_coeffs = cp.sort(cp.abs(gpu_detail.flatten()))
    
    # Sample thresholds for efficiency
    sample_size = min(100, n)
    indices = cp.linspace(0, n-1, sample_size, dtype=int)
    thresholds = sorted_coeffs[indices]
    
    # Calculate risk for each threshold
    risks = []
    for t in thresholds:
        k = cp.sum(cp.abs(gpu_detail) <= t)
        risk = n - 2*k + cp.sum(cp.minimum(cp.abs(gpu_detail), t)**2)
        risks.append(float(risk))
    
    # Find threshold with minimum risk
    if risks:
        min_idx = np.argmin(risks)
        threshold = float(thresholds[min_idx])
    else:
        threshold = sigma * np.sqrt(2 * np.log(n))  # Fallback
    
    # Apply thresholding on GPU
    if mode == 'soft':
        gpu_detail_thresh = cp.where(
            cp.abs(gpu_detail) > threshold,
            cp.sign(gpu_detail) * (cp.abs(gpu_detail) - threshold),
            0
        )
    else:  # hard thresholding
        gpu_detail_thresh = cp.where(
            cp.abs(gpu_detail) > threshold,
            gpu_detail,
            0
        )
    
    return to_cpu(gpu_detail_thresh)


def get_available_wavelets():
    """Get list of available wavelet families suitable for denoising"""
    return {
        'db8': 'Daubechies 8 (recommended for natural images)',
        'db4': 'Daubechies 4 (faster, good balance)',
        'bior2.2': 'Biorthogonal 2.2 (good for edges)',
        'coif2': 'Coiflets 2 (symmetric, good reconstruction)',
        'haar': 'Haar (fastest, simple images)',
        'db1': 'Daubechies 1 (same as Haar)',
        'db6': 'Daubechies 6 (smoother than db4)',
        'bior4.4': 'Biorthogonal 4.4 (higher order)',
        'sym8': 'Symlets 8 (nearly symmetric)'
    }


def estimate_noise_level(image: np.ndarray) -> float:
    """Estimate noise level in image using robust methods
    
    Args:
        image: Input image with values 0-255 (uint8) or 0-1 (float)
        
    Returns:
        Noise level in same scale as input (0-255 if input is uint8, 0-1 if input is float)
    """
    # Store original data type
    original_dtype = image.dtype
    is_uint8 = (original_dtype == np.uint8)
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        if is_uint8:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            # Manual conversion for float images
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            # FIXED: Ensure contiguous array after arithmetic operation
            gray = np.ascontiguousarray(gray)
    else:
        gray = image.copy()
    
    # Method 1: Try skimage estimate_sigma (robust but sometimes fails)
    try:
        if is_uint8:
            gray_float = gray.astype(np.float32) / 255.0
        else:
            gray_float = gray.astype(np.float32)
            gray_float = np.clip(gray_float, 0, 1)
        
        sigma_normalized = estimate_sigma(gray_float, channel_axis=None)
        
        # Convert back to original scale
        if is_uint8:
            return sigma_normalized * 255.0
        else:
            return sigma_normalized
            
    except Exception:
        pass  # Fall through to backup method
    
    # Method 2: Laplacian-based noise estimation (more reliable fallback)
    if is_uint8:
        gray_for_laplacian = gray.astype(np.float32)
    else:
        gray_for_laplacian = gray * 255.0  # Scale to 0-255 for Laplacian
        # FIXED: Ensure contiguous array after arithmetic operation
        gray_for_laplacian = np.ascontiguousarray(gray_for_laplacian)
    
    # Calculate Laplacian
    laplacian = cv2.Laplacian(gray_for_laplacian, cv2.CV_64F)
    
    # Estimate noise using median absolute deviation of Laplacian
    sigma_laplacian = np.median(np.abs(laplacian)) / 0.6745
    
    # Convert back to appropriate scale
    if is_uint8:
        return sigma_laplacian
    else:
        return sigma_laplacian / 255.0
