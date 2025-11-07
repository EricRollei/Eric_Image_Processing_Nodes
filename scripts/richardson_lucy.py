"""
Richardson-Lucy deconvolution implementation
Advanced deconvolution for lens blur, motion blur, and PSF correction
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import convolve2d
from typing import Optional, Tuple, Union
import warnings


def richardson_lucy_deconvolution(
    image: np.ndarray,
    psf: Optional[np.ndarray] = None,
    iterations: int = 10,
    clip: bool = True,
    filter_epsilon: float = None,
    blur_type: str = 'gaussian',
    blur_size: float = 2.0,
    motion_angle: float = 0.0,
    motion_length: float = 10.0,
    regularization: float = 0.0
) -> np.ndarray:
    """
    Richardson-Lucy deconvolution for image restoration
    
    This algorithm iteratively estimates the original image from a blurred observation
    using maximum likelihood estimation under Poisson noise assumptions.
    
    Args:
        image: Input blurred image [H, W, C] with values 0-255
        psf: Point Spread Function kernel (None = auto-generate)
        iterations: Number of RL iterations (5-50)
                   - 5-10: Fast, light restoration
                   - 10-20: Balanced quality/speed (recommended)
                   - 20-50: High quality, may amplify noise
        clip: Clip output to valid range [0, 255]
        filter_epsilon: Small value to prevent division by zero (auto if None)
        blur_type: Type of blur to correct ('gaussian', 'motion', 'custom')
                  - 'gaussian': Lens defocus, camera shake
                  - 'motion': Linear motion blur  
                  - 'custom': User-provided PSF
        blur_size: Size parameter for Gaussian blur (standard deviation)
                  - 1.0-3.0: Light blur correction
                  - 3.0-5.0: Moderate blur (recommended)
                  - 5.0-10.0: Heavy blur correction
        motion_angle: Motion blur angle in degrees (0-180)
        motion_length: Motion blur length in pixels (5-50)
        regularization: TV regularization strength (0-0.1)
                       - 0.0: No regularization (may amplify noise)
                       - 0.001-0.01: Light smoothing (recommended)
                       - 0.01-0.1: Heavy smoothing, reduces noise
    
    Returns:
        Restored image [H, W, C] with values 0-255
    """
    
    # Validate inputs
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")
    
    if iterations < 1:
        raise ValueError("Iterations must be >= 1")
    
    # Convert to float for processing
    img_float = image.astype(np.float64) / 255.0
    
    # Generate PSF if not provided
    if psf is None:
        psf = _generate_psf(blur_type, blur_size, motion_angle, motion_length, img_float.shape[:2])
    else:
        psf = psf.astype(np.float64)
        psf = psf / np.sum(psf)  # Normalize PSF
    
    # Set epsilon for numerical stability
    if filter_epsilon is None:
        filter_epsilon = np.finfo(img_float.dtype).eps
    
    # Handle grayscale vs color
    if img_float.ndim == 2:
        result = _richardson_lucy_single_channel(
            img_float, psf, iterations, filter_epsilon, regularization
        )
    else:
        # Process each channel independently
        result_channels = []
        for c in range(img_float.shape[2]):
            deconv_channel = _richardson_lucy_single_channel(
                img_float[:, :, c], psf, iterations, filter_epsilon, regularization
            )
            result_channels.append(deconv_channel)
        result = np.stack(result_channels, axis=2)
    
    # Convert back to uint8
    if clip:
        result = np.clip(result, 0, 1)
    
    return (result * 255).astype(np.uint8)


def _richardson_lucy_single_channel(
    image: np.ndarray,
    psf: np.ndarray, 
    iterations: int,
    epsilon: float,
    regularization: float
) -> np.ndarray:
    """Richardson-Lucy algorithm for single channel"""
    
    # Initialize estimate with input image
    estimate = image.copy()
    
    # Flip PSF for correlation
    psf_flipped = np.flipud(np.fliplr(psf))
    
    for i in range(iterations):
        # Forward convolution: estimate * psf
        convolved = convolve2d(estimate, psf, mode='same', boundary='symm')
        
        # Avoid division by zero
        convolved = np.maximum(convolved, epsilon)
        
        # Ratio of observed to estimated
        ratio = image / convolved
        
        # Backward convolution: ratio * psf_flipped  
        correlation = convolve2d(ratio, psf_flipped, mode='same', boundary='symm')
        
        # Update estimate
        estimate = estimate * correlation
        
        # Apply total variation regularization if requested
        if regularization > 0:
            estimate = _apply_tv_regularization(estimate, regularization)
        
        # Ensure positivity
        estimate = np.maximum(estimate, epsilon)
    
    return estimate


def _generate_psf(
    blur_type: str,
    blur_size: float,
    motion_angle: float,
    motion_length: float,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """Generate Point Spread Function based on blur type"""
    
    if blur_type == 'gaussian':
        # Gaussian PSF for lens blur
        size = int(2 * np.ceil(3 * blur_size) + 1)  # 6-sigma kernel
        center = size // 2
        
        y, x = np.ogrid[:size, :size]
        psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * blur_size**2))
        psf = psf / np.sum(psf)
        
    elif blur_type == 'motion':
        # Linear motion blur PSF
        length = int(motion_length)
        if length < 1:
            length = 1
            
        # Create motion kernel
        angle_rad = np.deg2rad(motion_angle)
        
        # Calculate kernel size to fit motion
        dx = int(np.abs(length * np.cos(angle_rad)))
        dy = int(np.abs(length * np.sin(angle_rad)))
        size = max(dx, dy, 3)
        
        # Make size odd
        if size % 2 == 0:
            size += 1
            
        psf = np.zeros((size, size))
        center = size // 2
        
        # Draw line for motion blur
        x0, y0 = center, center
        x1 = center + int(length * np.cos(angle_rad) / 2)
        y1 = center + int(length * np.sin(angle_rad) / 2)
        x2 = center - int(length * np.cos(angle_rad) / 2)
        y2 = center - int(length * np.sin(angle_rad) / 2)
        
        # Use Bresenham's line algorithm
        points = _bresenham_line(x2, y2, x1, y1)
        for x, y in points:
            if 0 <= x < size and 0 <= y < size:
                psf[y, x] = 1
        
        # Normalize
        if np.sum(psf) > 0:
            psf = psf / np.sum(psf)
        else:
            # Fallback to single point
            psf[center, center] = 1
            
    else:
        raise ValueError(f"Unknown blur type: {blur_type}")
    
    return psf


def _bresenham_line(x0: int, y0: int, x1: int, y1: int):
    """Bresenham's line algorithm for motion blur PSF"""
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


def _apply_tv_regularization(image: np.ndarray, strength: float) -> np.ndarray:
    """Apply total variation regularization to reduce noise amplification"""
    
    # Compute gradients
    grad_x = np.diff(image, axis=1, prepend=image[:, [0]])
    grad_y = np.diff(image, axis=0, prepend=image[[0], :])
    
    # Total variation magnitude
    tv_magnitude = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    # Compute divergence of normalized gradients
    div_x = np.diff(grad_x / tv_magnitude, axis=1, append=0)
    div_y = np.diff(grad_y / tv_magnitude, axis=0, append=0)
    
    # Update image
    regularized = image + strength * (div_x + div_y)
    
    return np.maximum(regularized, 0)  # Ensure positivity


def estimate_motion_blur(image: np.ndarray, method: str = 'fft') -> Tuple[float, float]:
    """
    Estimate motion blur parameters from an image
    
    Args:
        image: Input blurred image
        method: Estimation method ('fft', 'gradient')
        
    Returns:
        Tuple of (angle_degrees, length_pixels)
    """
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        image = np.ascontiguousarray(image)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # FIXED: Ensure contiguous array after cv2.cvtColor
        gray = np.ascontiguousarray(gray)
    else:
        gray = image.copy()
    
    if method == 'fft':
        # FFT-based motion blur estimation
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        
        # Find dominant direction in frequency domain
        # Look for dark lines in the magnitude spectrum
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Compute radial average to find blur direction
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Estimate angle by finding minimum in radial profile
        angles = np.linspace(0, 180, 180)
        min_variance = float('inf')
        best_angle = 0
        
        for angle in angles:
            angle_rad = np.deg2rad(angle)
            
            # Sample along perpendicular to motion direction
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Get profile perpendicular to motion
            profile = []
            for offset in range(-20, 21):
                sample_x = center_x + offset * cos_a
                sample_y = center_y + offset * sin_a
                
                if 0 <= sample_x < w and 0 <= sample_y < h:
                    profile.append(magnitude[int(sample_y), int(sample_x)])
            
            if len(profile) > 10:
                variance = np.var(profile)
                if variance < min_variance:
                    min_variance = variance
                    best_angle = angle
        
        # Estimate length (very rough)
        estimated_length = max(5, min(30, int(w * 0.05)))
        
        return best_angle, estimated_length
        
    else:
        # Fallback method
        return 0.0, 10.0


def create_motion_psf(angle: float, length: float, shape: Tuple[int, int] = (31, 31)) -> np.ndarray:
    """Create a motion blur PSF with specified parameters"""
    return _generate_psf('motion', 0, angle, length, shape)


def create_gaussian_psf(sigma: float, shape: Tuple[int, int] = (31, 31)) -> np.ndarray:
    """Create a Gaussian PSF with specified standard deviation"""
    return _generate_psf('gaussian', sigma, 0, 0, shape)


def get_blur_presets():
    """Get dictionary of common blur correction presets"""
    return {
        'light_camera_shake': {
            'blur_type': 'gaussian',
            'blur_size': 1.5,
            'iterations': 8,
            'regularization': 0.002
        },
        'moderate_defocus': {
            'blur_type': 'gaussian', 
            'blur_size': 3.0,
            'iterations': 15,
            'regularization': 0.005
        },
        'heavy_defocus': {
            'blur_type': 'gaussian',
            'blur_size': 5.0,
            'iterations': 20,
            'regularization': 0.01
        },
        'horizontal_motion': {
            'blur_type': 'motion',
            'motion_angle': 0.0,
            'motion_length': 15.0,
            'iterations': 12,
            'regularization': 0.003
        },
        'vertical_motion': {
            'blur_type': 'motion',
            'motion_angle': 90.0, 
            'motion_length': 15.0,
            'iterations': 12,
            'regularization': 0.003
        },
        'diagonal_motion': {
            'blur_type': 'motion',
            'motion_angle': 45.0,
            'motion_length': 15.0,
            'iterations': 12,
            'regularization': 0.003
        }
    }
