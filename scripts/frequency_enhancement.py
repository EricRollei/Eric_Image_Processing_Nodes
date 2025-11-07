"""
Advanced frequency domain enhancement techniques
Phase-preserving enhancement, homomorphic filtering, and multi-scale FFT enhancement
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple, Union
import warnings


def homomorphic_filter(
    image: np.ndarray,
    d0: float = 30.0,
    gamma_h: float = 2.0,
    gamma_l: float = 0.5,
    c: float = 1.0,
    filter_type: str = 'gaussian'
) -> np.ndarray:
    """
    Homomorphic filtering for illumination and reflectance separation
    
    Excellent for:
    - Uneven lighting correction in scanned photos
    - Shadow removal and highlight recovery
    - Normalizing illumination across image regions
    - Scientific imaging with lighting variations
    
    Args:
        image: Input image [H, W, C] with values 0-255
        d0: Cutoff frequency for high-pass characteristic (10-100)
           - 10-30: Strong illumination correction
           - 30-60: Moderate correction (recommended)
           - 60-100: Subtle correction
        gamma_h: High frequency gain (0.5-5.0)
                - 1.5-2.5: Enhance details (recommended)
                - 2.5-5.0: Strong detail enhancement
                - 0.5-1.0: Reduce high frequencies
        gamma_l: Low frequency gain (0.1-1.0)
                - 0.1-0.5: Strong illumination suppression
                - 0.5-0.8: Moderate suppression (recommended)
                - 0.8-1.0: Gentle suppression
        c: Sharpness of transition (0.5-2.0)
        filter_type: Filter shape ('gaussian', 'butterworth')
    
    Returns:
        Enhanced image [H, W, C] with values 0-255
    """
    
    # Validate inputs
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")
    
    # Convert to float for processing
    img_float = image.astype(np.float64)
    
    # Handle grayscale vs color
    if img_float.ndim == 2:
        result = _homomorphic_single_channel(img_float, d0, gamma_h, gamma_l, c, filter_type)
    else:
        # Process each channel independently
        result_channels = []
        for ch in range(img_float.shape[2]):
            enhanced_channel = _homomorphic_single_channel(
                img_float[:, :, ch], d0, gamma_h, gamma_l, c, filter_type
            )
            result_channels.append(enhanced_channel)
        result = np.stack(result_channels, axis=2)
    
    # Convert back to uint8 range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def _homomorphic_single_channel(
    image: np.ndarray,
    d0: float,
    gamma_h: float,
    gamma_l: float,
    c: float,
    filter_type: str
) -> np.ndarray:
    """Apply homomorphic filtering to single channel"""
    
    h, w = image.shape
    
    # Normalize to 0-1 range and add small epsilon to avoid log(0)
    img_normalized = image / 255.0
    img_safe = img_normalized + 1e-6
    
    # Take logarithm (image = illumination × reflectance → log(image) = log(illum) + log(refl))
    log_image = np.log(img_safe)
    
    # Transform to frequency domain
    fft_image = fft2(log_image)
    fft_shifted = fftshift(fft_image)
    
    # Create frequency filter
    filter_func = _create_homomorphic_filter(h, w, d0, gamma_h, gamma_l, c, filter_type)
    
    # Apply filter
    filtered_fft = fft_shifted * filter_func
    
    # Transform back to spatial domain
    filtered_fft_unshifted = ifftshift(filtered_fft)
    filtered_log = np.real(ifft2(filtered_fft_unshifted))
    
    # Take exponential to get back to original domain
    result_normalized = np.exp(filtered_log)
    
    # Remove the epsilon we added
    result_normalized = np.maximum(result_normalized - 1e-6, 0)
    
    # Scale back to 0-255 range and ensure reasonable bounds
    result = result_normalized * 255.0
    
    # Ensure result is in reasonable range
    result = np.clip(result, 0, 255)
    
    # Check for any NaN or inf values
    if not np.all(np.isfinite(result)):
        print("Warning: Non-finite values detected in homomorphic filter result, clipping...")
        result = np.nan_to_num(result, nan=128.0, posinf=255.0, neginf=0.0)
    
    return result


def _create_homomorphic_filter(
    h: int, w: int, d0: float, gamma_h: float, gamma_l: float, c: float, filter_type: str
) -> np.ndarray:
    """Create homomorphic filter in frequency domain"""
    
    # Create frequency coordinates
    u = np.arange(h) - h // 2
    v = np.arange(w) - w // 2
    U, V = np.meshgrid(v, u)
    
    # Distance from center
    D = np.sqrt(U**2 + V**2)
    
    if filter_type == 'gaussian':
        # Gaussian high-pass characteristic
        H = 1 - np.exp(-(D**2) / (2 * d0**2))
    elif filter_type == 'butterworth':
        # Butterworth high-pass characteristic
        H = 1 / (1 + (d0 / (D + 1e-6))**(2 * c))
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Scale to desired frequency response
    homomorphic_filter = (gamma_h - gamma_l) * H + gamma_l
    
    return homomorphic_filter


def phase_preserving_enhancement(
    image: np.ndarray,
    enhancement_factor: float = 1.5,
    frequency_range: Tuple[float, float] = (0.1, 0.8),
    method: str = 'magnitude_scaling',
    preserve_dc: bool = True
) -> np.ndarray:
    """
    Phase-preserving frequency domain enhancement
    
    Enhances image details while preserving phase information for natural results
    
    Args:
        image: Input image [H, W, C] with values 0-255
        enhancement_factor: Magnitude enhancement strength (1.0-3.0)
                          - 1.0-1.5: Subtle enhancement (recommended)
                          - 1.5-2.0: Moderate enhancement
                          - 2.0-3.0: Strong enhancement
        frequency_range: (low, high) frequency range to enhance (0.0-1.0)
                        - (0.1, 0.8): Standard detail enhancement
                        - (0.05, 0.5): Low-frequency emphasis
                        - (0.2, 0.9): High-frequency emphasis
        method: Enhancement method ('magnitude_scaling', 'adaptive_scaling')
        preserve_dc: Keep DC component unchanged (recommended True)
    
    Returns:
        Enhanced image [H, W, C] with values 0-255
    """
    
    # Validate inputs
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")
    
    # Convert to float for processing
    img_float = image.astype(np.float64) / 255.0
    
    # Handle grayscale vs color
    if img_float.ndim == 2:
        result = _phase_preserving_single_channel(
            img_float, enhancement_factor, frequency_range, method, preserve_dc
        )
    else:
        # Process each channel independently
        result_channels = []
        for ch in range(img_float.shape[2]):
            enhanced_channel = _phase_preserving_single_channel(
                img_float[:, :, ch], enhancement_factor, frequency_range, method, preserve_dc
            )
            result_channels.append(enhanced_channel)
        result = np.stack(result_channels, axis=2)
    
    # Convert back to uint8 range
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    return result


def _phase_preserving_single_channel(
    image: np.ndarray,
    enhancement_factor: float,
    frequency_range: Tuple[float, float],
    method: str,
    preserve_dc: bool
) -> np.ndarray:
    """Apply phase-preserving enhancement to single channel"""
    
    h, w = image.shape
    
    # Transform to frequency domain
    fft_image = fft2(image)
    fft_shifted = fftshift(fft_image)
    
    # Separate magnitude and phase
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    
    # Create frequency mask for enhancement region
    enhancement_mask = _create_frequency_mask(h, w, frequency_range)
    
    if method == 'magnitude_scaling':
        # Simple magnitude scaling
        enhanced_magnitude = magnitude * (1 + (enhancement_factor - 1) * enhancement_mask)
        
    elif method == 'adaptive_scaling':
        # Adaptive scaling based on local magnitude
        local_mean = gaussian_filter(magnitude, sigma=5)
        adaptive_factor = magnitude / (local_mean + 1e-6)
        enhanced_magnitude = magnitude * (1 + (enhancement_factor - 1) * enhancement_mask * adaptive_factor)
        
    else:
        raise ValueError(f"Unknown enhancement method: {method}")
    
    # Preserve DC component if requested
    if preserve_dc:
        center_h, center_w = h // 2, w // 2
        enhanced_magnitude[center_h, center_w] = magnitude[center_h, center_w]
    
    # Reconstruct complex spectrum with original phase
    enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
    
    # Transform back to spatial domain
    enhanced_fft_unshifted = ifftshift(enhanced_fft)
    result = np.real(ifft2(enhanced_fft_unshifted))
    
    return result


def _create_frequency_mask(h: int, w: int, frequency_range: Tuple[float, float]) -> np.ndarray:
    """Create frequency mask for selective enhancement"""
    
    # Create frequency coordinates
    u = np.arange(h) - h // 2
    v = np.arange(w) - w // 2
    U, V = np.meshgrid(v, u)
    
    # Normalized distance from center
    max_freq = min(h, w) // 2
    D_norm = np.sqrt(U**2 + V**2) / max_freq
    
    # Create mask for frequency range
    low_freq, high_freq = frequency_range
    mask = np.logical_and(D_norm >= low_freq, D_norm <= high_freq).astype(np.float64)
    
    # Smooth transitions
    mask = gaussian_filter(mask, sigma=2.0)
    
    return mask


def multiscale_fft_enhancement(
    image: np.ndarray,
    scales: int = 4,
    enhancement_factors: Optional[list] = None,
    blur_sigmas: Optional[list] = None
) -> np.ndarray:
    """
    Multi-scale frequency domain enhancement using Laplacian pyramid
    
    Args:
        image: Input image [H, W, C] with values 0-255
        scales: Number of scales to process (2-6)
        enhancement_factors: Enhancement factor for each scale
        blur_sigmas: Gaussian blur sigma for each scale
    
    Returns:
        Enhanced image [H, W, C] with values 0-255
    """
    
    # Default parameters
    if enhancement_factors is None:
        enhancement_factors = [1.0, 1.2, 1.5, 1.3, 1.1][:scales]
    
    if blur_sigmas is None:
        blur_sigmas = [1.0, 2.0, 4.0, 8.0, 16.0][:scales]
    
    # Ensure we have enough parameters
    while len(enhancement_factors) < scales:
        enhancement_factors.append(1.0)
    while len(blur_sigmas) < scales:
        blur_sigmas.append(blur_sigmas[-1] * 2)
    
    # Convert to float for processing
    img_float = image.astype(np.float64) / 255.0
    
    # Handle grayscale vs color
    if img_float.ndim == 2:
        result = _multiscale_fft_single_channel(img_float, scales, enhancement_factors, blur_sigmas)
    else:
        # Process each channel independently
        result_channels = []
        for ch in range(img_float.shape[2]):
            enhanced_channel = _multiscale_fft_single_channel(
                img_float[:, :, ch], scales, enhancement_factors, blur_sigmas
            )
            result_channels.append(enhanced_channel)
        result = np.stack(result_channels, axis=2)
    
    # Convert back to uint8 range
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    return result


def _multiscale_fft_single_channel(
    image: np.ndarray,
    scales: int,
    enhancement_factors: list,
    blur_sigmas: list
) -> np.ndarray:
    """Apply multi-scale FFT enhancement to single channel"""
    
    # Create Laplacian pyramid
    pyramid = []
    current_image = image.copy()
    
    for i in range(scales - 1):
        # Blur current level
        blurred = gaussian_filter(current_image, sigma=blur_sigmas[i])
        
        # Compute Laplacian (difference)
        laplacian = current_image - blurred
        pyramid.append(laplacian)
        
        # Move to next level
        current_image = blurred
    
    # Add the final blurred image
    pyramid.append(current_image)
    
    # Enhance each level in frequency domain
    enhanced_pyramid = []
    
    for i, level in enumerate(pyramid[:-1]):  # Skip the base level
        if enhancement_factors[i] != 1.0:
            # Transform to frequency domain
            fft_level = fft2(level)
            magnitude = np.abs(fft_level)
            phase = np.angle(fft_level)
            
            # Enhance magnitude
            enhanced_magnitude = magnitude * enhancement_factors[i]
            
            # Reconstruct and transform back
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_level = np.real(ifft2(enhanced_fft))
            
            enhanced_pyramid.append(enhanced_level)
        else:
            enhanced_pyramid.append(level)
    
    # Add the unchanged base level
    enhanced_pyramid.append(pyramid[-1])
    
    # Reconstruct image
    result = enhanced_pyramid[-1]  # Start with base
    
    for i in range(len(enhanced_pyramid) - 2, -1, -1):
        result = result + enhanced_pyramid[i]
    
    return result


def adaptive_frequency_filter(
    image: np.ndarray,
    local_variance_threshold: float = 0.01,
    smooth_enhancement: float = 1.2,
    detail_enhancement: float = 1.8,
    noise_reduction: float = 0.8
) -> np.ndarray:
    """
    Adaptive frequency filtering based on local image statistics
    
    Args:
        image: Input image [H, W, C] with values 0-255
        local_variance_threshold: Threshold for smooth vs detailed regions
        smooth_enhancement: Enhancement factor for smooth regions
        detail_enhancement: Enhancement factor for detailed regions  
        noise_reduction: Reduction factor for likely noise regions
    
    Returns:
        Enhanced image [H, W, C] with values 0-255
    """
    
    # Convert to float for processing
    img_float = image.astype(np.float64) / 255.0
    
    # Handle grayscale vs color
    if img_float.ndim == 2:
        result = _adaptive_frequency_single_channel(
            img_float, local_variance_threshold, smooth_enhancement, 
            detail_enhancement, noise_reduction
        )
    else:
        # Process each channel independently
        result_channels = []
        for ch in range(img_float.shape[2]):
            enhanced_channel = _adaptive_frequency_single_channel(
                img_float[:, :, ch], local_variance_threshold, 
                smooth_enhancement, detail_enhancement, noise_reduction
            )
            result_channels.append(enhanced_channel)
        result = np.stack(result_channels, axis=2)
    
    # Convert back to uint8 range
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    return result


def _adaptive_frequency_single_channel(
    image: np.ndarray,
    variance_threshold: float,
    smooth_enhancement: float,
    detail_enhancement: float,
    noise_reduction: float
) -> np.ndarray:
    """Apply adaptive frequency filtering to single channel"""
    
    h, w = image.shape
    
    # Compute local variance
    local_mean = gaussian_filter(image, sigma=3)
    local_variance = gaussian_filter(image**2, sigma=3) - local_mean**2
    
    # Transform to frequency domain
    fft_image = fft2(image)
    fft_shifted = fftshift(fft_image)
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    
    # Create adaptive enhancement map based on spatial variance
    # High variance = details, low variance = smooth regions
    
    # Transform variance map to frequency domain for adaptive filtering
    fft_variance = fft2(local_variance)
    fft_var_shifted = fftshift(fft_variance)
    var_magnitude = np.abs(fft_var_shifted)
    
    # Normalize variance magnitude
    var_magnitude_norm = var_magnitude / (np.max(var_magnitude) + 1e-6)
    
    # Create adaptive enhancement factor
    enhancement_factor = np.ones_like(magnitude)
    
    # Smooth regions (low frequency, low variance) - gentle enhancement
    smooth_mask = var_magnitude_norm < variance_threshold
    enhancement_factor[smooth_mask] = smooth_enhancement
    
    # Detail regions (medium frequency, high variance) - strong enhancement
    detail_mask = var_magnitude_norm >= variance_threshold
    enhancement_factor[detail_mask] = detail_enhancement
    
    # High frequency regions (likely noise) - reduction
    u = np.arange(h) - h // 2
    v = np.arange(w) - w // 2
    U, V = np.meshgrid(v, u)
    D_norm = np.sqrt(U**2 + V**2) / (min(h, w) // 2)
    noise_mask = D_norm > 0.7
    enhancement_factor[noise_mask] = noise_reduction
    
    # Apply adaptive enhancement
    enhanced_magnitude = magnitude * enhancement_factor
    
    # Reconstruct complex spectrum
    enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
    
    # Transform back to spatial domain
    enhanced_fft_unshifted = ifftshift(enhanced_fft)
    result = np.real(ifft2(enhanced_fft_unshifted))
    
    return result


def debug_homomorphic_filter(image: np.ndarray, d0: float = 40.0) -> dict:
    """
    Debug version of homomorphic filter that returns intermediate results
    Use this to diagnose issues with the main filter
    """
    # Convert to single channel for debugging
    if image.ndim == 3:
        image = np.ascontiguousarray(image)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # FIXED: Ensure contiguous array after cv2.cvtColor
        gray = np.ascontiguousarray(gray)
    else:
        gray = image.copy()
    
    # Step 1: Normalize and add epsilon
    img_normalized = gray.astype(np.float64) / 255.0
    img_safe = img_normalized + 1e-6
    
    print(f"Step 1 - Input range: {gray.min()}-{gray.max()}")
    print(f"Step 1 - Normalized range: {img_normalized.min():.6f}-{img_normalized.max():.6f}")
    print(f"Step 1 - Safe range: {img_safe.min():.6f}-{img_safe.max():.6f}")
    
    # Step 2: Take logarithm
    log_image = np.log(img_safe)
    print(f"Step 2 - Log range: {log_image.min():.6f}-{log_image.max():.6f}")
    
    # Step 3: FFT
    fft_image = fft2(log_image)
    fft_shifted = fftshift(fft_image)
    print(f"Step 3 - FFT magnitude range: {np.abs(fft_shifted).min():.6f}-{np.abs(fft_shifted).max():.6f}")
    
    # Step 4: Create and apply filter
    h, w = gray.shape
    filter_func = _create_homomorphic_filter(h, w, d0, 1.8, 0.6, 1.0, 'gaussian')
    print(f"Step 4 - Filter range: {filter_func.min():.6f}-{filter_func.max():.6f}")
    
    filtered_fft = fft_shifted * filter_func
    print(f"Step 4 - Filtered FFT magnitude range: {np.abs(filtered_fft).min():.6f}-{np.abs(filtered_fft).max():.6f}")
    
    # Step 5: IFFT
    filtered_fft_unshifted = ifftshift(filtered_fft)
    filtered_log = np.real(ifft2(filtered_fft_unshifted))
    print(f"Step 5 - Filtered log range: {filtered_log.min():.6f}-{filtered_log.max():.6f}")
    
    # Step 6: Exponential
    result_normalized = np.exp(filtered_log)
    print(f"Step 6 - Exp result range: {result_normalized.min():.6f}-{result_normalized.max():.6f}")
    
    # Step 7: Remove epsilon and scale
    result_normalized = np.maximum(result_normalized - 1e-6, 0)
    result = result_normalized * 255.0
    result = np.clip(result, 0, 255)
    print(f"Step 7 - Final range: {result.min():.6f}-{result.max():.6f}")
    
    return {
        'original': gray,
        'normalized': img_normalized,
        'log': log_image,
        'filter': filter_func,
        'filtered_log': filtered_log,
        'exp_result': result_normalized,
        'final': result.astype(np.uint8)
    }


def get_frequency_enhancement_presets():
    """Get dictionary of frequency enhancement presets for different scenarios"""
    return {
        'illumination_correction': {
            'method': 'homomorphic',
            'params': {'d0': 40, 'gamma_h': 1.8, 'gamma_l': 0.6},
            'description': 'Correct uneven lighting in scanned photos'
        },
        'detail_enhancement': {
            'method': 'phase_preserving',
            'params': {'enhancement_factor': 1.4, 'frequency_range': (0.1, 0.7)},
            'description': 'Enhance image details naturally'
        },
        'shadow_highlight': {
            'method': 'homomorphic',
            'params': {'d0': 25, 'gamma_h': 2.2, 'gamma_l': 0.4},
            'description': 'Strong shadow/highlight recovery'
        },
        'fine_detail_boost': {
            'method': 'phase_preserving',
            'params': {'enhancement_factor': 1.8, 'frequency_range': (0.3, 0.9)},
            'description': 'Boost fine details and textures'
        },
        'multiscale_sharpening': {
            'method': 'multiscale_fft',
            'params': {'scales': 4, 'enhancement_factors': [1.0, 1.3, 1.6, 1.2]},
            'description': 'Multi-scale detail enhancement'
        },
        'adaptive_enhancement': {
            'method': 'adaptive',
            'params': {'detail_enhancement': 1.6, 'smooth_enhancement': 1.1},
            'description': 'Adaptive enhancement based on image content'
        }
    }
