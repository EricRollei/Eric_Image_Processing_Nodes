"""
Wiener filter implementation for frequency domain image restoration
Optimal restoration when noise characteristics are known
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Optional, Tuple, Union
import warnings


def wiener_filter_restoration(
    image: np.ndarray,
    psf: Optional[np.ndarray] = None,
    noise_variance: Optional[float] = None,
    signal_variance: Optional[float] = None,
    K: Optional[float] = None,
    blur_type: str = 'gaussian',
    blur_size: float = 2.0,
    motion_angle: float = 0.0,
    motion_length: float = 10.0,
    estimate_noise: bool = True,
    clip: bool = True
) -> np.ndarray:
    """
    Wiener filter for optimal image restoration in frequency domain
    
    The Wiener filter minimizes mean square error between the original and 
    restored image, providing optimal restoration when noise characteristics
    are known.
    
    Args:
        image: Input degraded image [H, W, C] with values 0-255
        psf: Point Spread Function (None = auto-generate)
        noise_variance: Noise power (auto-estimated if None)
        signal_variance: Signal power (auto-estimated if None)  
        K: Regularization parameter (auto-calculated if None)
           - 0.001-0.01: Light regularization, sharp results
           - 0.01-0.1: Moderate regularization (recommended)
           - 0.1-1.0: Heavy regularization, smooth results
        blur_type: Type of degradation ('gaussian', 'motion')
                  - 'gaussian': Lens defocus, atmospheric blur
                  - 'motion': Camera shake, object motion
        blur_size: Gaussian blur standard deviation (1.0-10.0)
        motion_angle: Motion direction in degrees (0-180)
        motion_length: Motion distance in pixels (5-50)
        estimate_noise: Auto-estimate noise parameters
        clip: Clip output to valid range [0, 255]
    
    Returns:
        Restored image [H, W, C] with values 0-255
    """
    
    # Validate inputs
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")
    
    # Convert to float for processing
    img_float = image.astype(np.float64) / 255.0
    
    # Generate PSF if not provided
    if psf is None:
        psf = _generate_psf_for_wiener(blur_type, blur_size, motion_angle, motion_length, img_float.shape[:2])
    else:
        psf = psf.astype(np.float64)
        psf = psf / np.sum(psf)  # Normalize PSF
    
    # Handle grayscale vs color
    if img_float.ndim == 2:
        result = _wiener_filter_single_channel(
            img_float, psf, noise_variance, signal_variance, K, estimate_noise
        )
    else:
        # Process each channel independently
        result_channels = []
        for c in range(img_float.shape[2]):
            filtered_channel = _wiener_filter_single_channel(
                img_float[:, :, c], psf, noise_variance, signal_variance, K, estimate_noise
            )
            result_channels.append(filtered_channel)
        result = np.stack(result_channels, axis=2)
    
    # Convert back to uint8
    if clip:
        result = np.clip(result, 0, 1)
    
    return (result * 255).astype(np.uint8)


def _wiener_filter_single_channel(
    image: np.ndarray,
    psf: np.ndarray,
    noise_variance: Optional[float],
    signal_variance: Optional[float], 
    K: Optional[float],
    estimate_noise: bool
) -> np.ndarray:
    """Apply Wiener filter to single channel image"""
    
    h, w = image.shape
    
    # Pad PSF to image size
    psf_padded = np.zeros((h, w))
    ph, pw = psf.shape
    
    # Center the PSF
    start_h = (h - ph) // 2
    start_w = (w - pw) // 2
    psf_padded[start_h:start_h + ph, start_w:start_w + pw] = psf
    
    # Shift PSF so that center is at (0,0) for FFT
    psf_padded = ifftshift(psf_padded)
    
    # Transform to frequency domain
    image_fft = fft2(image)
    psf_fft = fft2(psf_padded)
    
    # Estimate noise and signal parameters if needed
    if estimate_noise or K is None:
        if noise_variance is None:
            # Estimate noise variance using high-frequency components
            noise_variance = _estimate_noise_variance_fft(image_fft)
            
        if signal_variance is None:
            # Estimate signal variance from image
            try:
                signal_variance = np.var(image)
                if signal_variance <= 0 or not np.isfinite(signal_variance):
                    signal_variance = 0.1  # Default fallback
            except Exception as e:
                print(f"Warning: Signal variance estimation failed ({e}), using default")
                signal_variance = 0.1
            
        if K is None:
            # Calculate regularization parameter
            if noise_variance is not None and signal_variance is not None and signal_variance > 0:
                K = noise_variance / signal_variance
                K = np.clip(K, 0.001, 1.0)  # Reasonable bounds
            else:
                K = 0.05  # Default fallback
    
    # Ensure all parameters are valid before printing
    if K is not None and noise_variance is not None and signal_variance is not None:
        print(f"Wiener filter parameters: K={K:.4f}, noise_var={noise_variance:.6f}, signal_var={signal_variance:.6f}")
    else:
        print(f"Wiener filter: Using K={K if K is not None else 'None'} (some parameters could not be estimated)")
    
    # Ensure K has a valid value
    if K is None:
        K = 0.05  # Reasonable default
        print("Warning: Could not estimate K parameter, using default K=0.05")
    
    # Compute Wiener filter in frequency domain
    # W(u,v) = H*(u,v) / (|H(u,v)|² + K)
    psf_conj = np.conj(psf_fft)
    psf_magnitude_squared = np.abs(psf_fft) ** 2
    
    # Avoid division by zero
    epsilon = 1e-10
    wiener_filter = psf_conj / (psf_magnitude_squared + K + epsilon)
    
    # Apply filter
    restored_fft = image_fft * wiener_filter
    
    # Transform back to spatial domain
    restored = np.real(ifft2(restored_fft))
    
    return restored


def _generate_psf_for_wiener(
    blur_type: str,
    blur_size: float,
    motion_angle: float,
    motion_length: float,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """Generate PSF for Wiener filtering"""
    
    if blur_type == 'gaussian':
        # Gaussian PSF
        size = int(2 * np.ceil(3 * blur_size) + 1)
        center = size // 2
        
        y, x = np.ogrid[:size, :size]
        psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * blur_size**2))
        psf = psf / np.sum(psf)
        
    elif blur_type == 'motion':
        # Motion blur PSF (same as Richardson-Lucy implementation)
        length = int(motion_length)
        if length < 1:
            length = 1
            
        angle_rad = np.deg2rad(motion_angle)
        
        dx = int(np.abs(length * np.cos(angle_rad)))
        dy = int(np.abs(length * np.sin(angle_rad)))
        size = max(dx, dy, 3)
        
        if size % 2 == 0:
            size += 1
            
        psf = np.zeros((size, size))
        center = size // 2
        
        x0, y0 = center, center
        x1 = center + int(length * np.cos(angle_rad) / 2)
        y1 = center + int(length * np.sin(angle_rad) / 2)
        x2 = center - int(length * np.cos(angle_rad) / 2)
        y2 = center - int(length * np.sin(angle_rad) / 2)
        
        points = _bresenham_line(x2, y2, x1, y1)
        for x, y in points:
            if 0 <= x < size and 0 <= y < size:
                psf[y, x] = 1
        
        if np.sum(psf) > 0:
            psf = psf / np.sum(psf)
        else:
            psf[center, center] = 1
            
    else:
        raise ValueError(f"Unknown blur type: {blur_type}")
    
    return psf


def _bresenham_line(x0: int, y0: int, x1: int, y1: int):
    """Bresenham's line algorithm (same as Richardson-Lucy)"""
    points = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        
        if x0 == x1 and y0 == y1:
            break
            
        e2 = 2 * err
        
        if e2 > -dy:
            err -= dy
            x0 += sx
            
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points


def _estimate_noise_variance_fft(image_fft: np.ndarray) -> float:
    """Estimate noise variance from FFT high-frequency components"""
    
    try:
        h, w = image_fft.shape
        
        # Create mask for high-frequency components
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # Distance from center
        distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_distance = min(center_h, center_w)
        
        # High-frequency mask (outer 25% of spectrum)
        high_freq_mask = distance > (0.75 * max_distance)
        
        # Ensure we have some high-frequency components
        if np.sum(high_freq_mask) == 0:
            # Fallback: use outer 10% if 25% gives no pixels
            high_freq_mask = distance > (0.9 * max_distance)
        
        if np.sum(high_freq_mask) == 0:
            # Ultimate fallback: use a default value
            return 0.01
        
        # Estimate noise from high-frequency power
        high_freq_power = np.abs(image_fft[high_freq_mask]) ** 2
        noise_variance = np.mean(high_freq_power) / (h * w)
        
        # Ensure we return a reasonable value
        if noise_variance <= 0 or not np.isfinite(noise_variance):
            return 0.01
            
        return float(noise_variance)
        
    except Exception as e:
        print(f"Warning: FFT noise estimation failed ({e}), using default")
        return 0.01


def adaptive_wiener_filter(
    image: np.ndarray,
    blur_strength: str = 'medium',
    noise_level: str = 'auto'
) -> np.ndarray:
    """
    Simplified Wiener filter with automatic parameter selection
    
    Args:
        image: Input image [H, W, C] with values 0-255
        blur_strength: 'light', 'medium', 'heavy'
        noise_level: 'low', 'medium', 'high', 'auto'
    
    Returns:
        Filtered image
    """
    
    # Parameter presets
    blur_params = {
        'light': {'blur_size': 1.5, 'K_factor': 0.01},
        'medium': {'blur_size': 3.0, 'K_factor': 0.05}, 
        'heavy': {'blur_size': 5.0, 'K_factor': 0.1}
    }
    
    noise_params = {
        'low': 0.01,
        'medium': 0.05,
        'high': 0.15,
        'auto': None  # Will be estimated
    }
    
    params = blur_params[blur_strength]
    K_base = noise_params[noise_level] if noise_level != 'auto' else None
    
    if K_base is not None:
        K = K_base * params['K_factor']
    else:
        K = None  # Will be estimated
    
    return wiener_filter_restoration(
        image,
        blur_type='gaussian',
        blur_size=params['blur_size'],
        K=K,
        estimate_noise=(noise_level == 'auto')
    )


def parametric_wiener_filter(
    image: np.ndarray,
    psf_sigma: float = 2.0,
    snr_db: float = 30.0
) -> np.ndarray:
    """
    Wiener filter with SNR specified in dB
    
    Args:
        image: Input image
        psf_sigma: Gaussian PSF standard deviation
        snr_db: Signal-to-noise ratio in dB (10-60)
               - 10-20 dB: Very noisy images
               - 20-30 dB: Moderately noisy (typical)
               - 30-60 dB: Clean images
    
    Returns:
        Filtered image
    """
    
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    K = 1.0 / snr_linear
    
    print(f"Parametric Wiener: SNR={snr_db}dB, K={K:.6f}")
    
    return wiener_filter_restoration(
        image,
        blur_type='gaussian',
        blur_size=psf_sigma,
        K=K,
        estimate_noise=False
    )


def compare_restoration_methods(image: np.ndarray, blur_size: float = 3.0):
    """Compare different restoration approaches on the same image"""
    
    results = {}
    
    # Wiener filter with different K values
    for k_val in [0.01, 0.05, 0.1]:
        result = wiener_filter_restoration(
            image, blur_type='gaussian', blur_size=blur_size, K=k_val
        )
        results[f'Wiener_K={k_val}'] = result
    
    # Adaptive Wiener
    result = adaptive_wiener_filter(image, 'medium', 'auto')
    results['Adaptive_Wiener'] = result
    
    return results


def get_wiener_presets():
    """Get dictionary of Wiener filter presets for different scenarios"""
    return {
        'light_blur_clean': {
            'blur_size': 1.5,
            'K': 0.005,
            'description': 'Light blur, clean image'
        },
        'light_blur_noisy': {
            'blur_size': 1.5,
            'K': 0.05,
            'description': 'Light blur, moderate noise'
        },
        'medium_blur_clean': {
            'blur_size': 3.0,
            'K': 0.01,
            'description': 'Medium blur, clean image'
        },
        'medium_blur_noisy': {
            'blur_size': 3.0,
            'K': 0.1,
            'description': 'Medium blur, noisy image'
        },
        'heavy_blur_clean': {
            'blur_size': 5.0,
            'K': 0.02,
            'description': 'Heavy blur, clean image'
        },
        'heavy_blur_noisy': {
            'blur_size': 5.0,
            'K': 0.2,
            'description': 'Heavy blur, very noisy'
        },
        'motion_horizontal': {
            'blur_type': 'motion',
            'motion_angle': 0.0,
            'motion_length': 15.0,
            'K': 0.05,
            'description': 'Horizontal motion blur'
        },
        'motion_vertical': {
            'blur_type': 'motion',
            'motion_angle': 90.0,
            'motion_length': 15.0,
            'K': 0.05,
            'description': 'Vertical motion blur'
        }
    }
