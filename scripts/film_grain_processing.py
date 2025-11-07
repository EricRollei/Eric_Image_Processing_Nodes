"""
Specialized film grain processing algorithms
Handles authentic film grain vs. simulated/digital grain with different approaches
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from skimage.restoration import estimate_sigma, denoise_nl_means
from skimage.filters import gaussian, median
from skimage.morphology import disk, opening, closing
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
import warnings

try:
    from .gpu_utils import (
        gpu_gaussian_blur, gpu_bilateral_filter, gpu_non_local_means,
        can_use_gpu, cleanup_gpu_memory, to_gpu, to_cpu
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    warnings.warn("GPU utilities not available")


def analyze_grain_type(image: np.ndarray, show_analysis: bool = False) -> Dict[str, Any]:
    """
    Analyze image to determine grain characteristics and type
    
    Returns:
        Dictionary with grain analysis results
    """
    # Ensure input is contiguous before any OpenCV operations
    image = np.ascontiguousarray(image)
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # FIXED: Ensure contiguous array after cv2.cvtColor
        gray = np.ascontiguousarray(gray)
    else:
        gray = np.ascontiguousarray(image.copy())
    
    gray_float = gray.astype(np.float32) / 255.0
    # FIXED: Ensure contiguous array after arithmetic operation
    gray_float = np.ascontiguousarray(gray_float)
    
    analysis = {}
    
    # 1. Noise level estimation
    analysis['noise_level'] = estimate_sigma(gray_float, channel_axis=None)
    
    # 2. Grain structure analysis
    # Use different kernel sizes to analyze grain structure
    grain_responses = []
    kernel_sizes = [3, 5, 7, 9]
    
    for kernel_size in kernel_sizes:
        # Local standard deviation (texture measure)
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        local_mean = cv2.filter2D(np.ascontiguousarray(gray_float), -1, kernel)
        gray_sq = np.ascontiguousarray(gray_float ** 2)
        local_variance = cv2.filter2D(gray_sq, -1, kernel) - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_variance, 0))
        grain_responses.append(np.mean(local_std))
    
    analysis['grain_responses'] = grain_responses
    
    # 3. Frequency domain analysis
    f_transform = np.fft.fft2(gray_float)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    
    # Analyze different frequency bands
    # High frequency (potential digital noise)
    high_freq_mask = np.zeros_like(magnitude)
    high_freq_radius = min(h, w) // 8
    y, x = np.ogrid[:h, :w]
    high_freq_mask[(y - center_h)**2 + (x - center_w)**2 > high_freq_radius**2] = 1
    analysis['high_freq_energy'] = np.sum(magnitude * high_freq_mask) / np.sum(magnitude)
    
    # Mid frequency (film grain typically here)
    mid_freq_mask = np.zeros_like(magnitude)
    mid_freq_inner = min(h, w) // 16
    mid_freq_outer = min(h, w) // 8
    mid_freq_mask[
        ((y - center_h)**2 + (x - center_w)**2 > mid_freq_inner**2) &
        ((y - center_h)**2 + (x - center_w)**2 <= mid_freq_outer**2)
    ] = 1
    analysis['mid_freq_energy'] = np.sum(magnitude * mid_freq_mask) / np.sum(magnitude)
    
    # 4. Grain pattern regularity
    # Autocorrelation analysis to detect repetitive patterns
    autocorr = cv2.matchTemplate(
        np.ascontiguousarray(gray_float),
        np.ascontiguousarray(gray_float),
        cv2.TM_CCOEFF_NORMED
    )
    # Remove center peak
    center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    autocorr[center_y-5:center_y+5, center_x-5:center_x+5] = 0
    analysis['pattern_regularity'] = np.max(autocorr)
    
    # 5. Grain size estimation
    # Use morphological operations to estimate grain size
    grain_sizes = []
    for size in range(1, 6):
        kernel = disk(size)
        opened = opening(gray_float, kernel)
        difference = np.mean(np.abs(gray_float - opened))
        grain_sizes.append(difference)
    
    analysis['grain_size_profile'] = grain_sizes
    analysis['estimated_grain_size'] = np.argmax(grain_sizes) + 1
    
    # 6. Determine grain type
    analysis['grain_type'] = _classify_grain_type(analysis)
    
    if show_analysis:
        _print_grain_analysis(analysis)
    
    return analysis


def _classify_grain_type(analysis: Dict[str, Any]) -> str:
    """Classify grain type based on analysis results"""
    
    noise_level = analysis['noise_level']
    high_freq_energy = analysis['high_freq_energy']
    mid_freq_energy = analysis['mid_freq_energy']
    pattern_regularity = analysis['pattern_regularity']
    grain_size = analysis['estimated_grain_size']
    
    # Decision thresholds
    if pattern_regularity > 0.3:
        return "simulated_regular"  # Artificially added grain with regular pattern
    elif high_freq_energy > 0.4:
        return "digital_noise"  # High frequency digital noise
    elif mid_freq_energy > 0.3 and grain_size <= 2:
        return "fine_film_grain"  # Fine film grain (35mm, high quality)
    elif mid_freq_energy > 0.2 and grain_size <= 4:
        return "medium_film_grain"  # Medium film grain (16mm, standard)
    elif noise_level > 0.05:
        return "coarse_film_grain"  # Coarse film grain (8mm, low light)
    else:
        return "minimal_grain"  # Very little grain present


def _print_grain_analysis(analysis: Dict[str, Any]):
    """Print grain analysis results"""
    print("\n=== GRAIN ANALYSIS ===")
    print(f"Grain Type: {analysis['grain_type']}")
    print(f"Noise Level: {analysis['noise_level']:.4f}")
    print(f"High Freq Energy: {analysis['high_freq_energy']:.3f}")
    print(f"Mid Freq Energy: {analysis['mid_freq_energy']:.3f}")
    print(f"Pattern Regularity: {analysis['pattern_regularity']:.3f}")
    print(f"Estimated Grain Size: {analysis['estimated_grain_size']}")
    print("======================\n")


def denoise_film_grain(
    image: np.ndarray,
    grain_type: Optional[str] = None,
    preserve_texture: bool = True,
    use_gpu: bool = True,
    show_analysis: bool = False
) -> np.ndarray:
    """
    Specialized film grain denoising based on grain type
    
    Args:
        image: Input image with film grain
        grain_type: Type of grain (auto-detected if None)
        preserve_texture: Whether to preserve film texture characteristics
        use_gpu: Use GPU acceleration if available
        show_analysis: Show grain analysis results
    
    Returns:
        Denoised image with preserved film characteristics
    """
    # Ensure input is contiguous before processing
    image = np.ascontiguousarray(image)
    
    # Analyze grain if type not specified
    if grain_type is None:
        analysis = analyze_grain_type(image, show_analysis)
        grain_type = analysis['grain_type']
    else:
        analysis = analyze_grain_type(image, show_analysis)
    
    if show_analysis:
        print(f"Processing as: {grain_type}")
    
    # Choose processing method based on grain type
    if grain_type == "fine_film_grain":
        return _process_fine_film_grain(image, analysis, preserve_texture, use_gpu)
    elif grain_type == "medium_film_grain":
        return _process_medium_film_grain(image, analysis, preserve_texture, use_gpu)
    elif grain_type == "coarse_film_grain":
        return _process_coarse_film_grain(image, analysis, preserve_texture, use_gpu)
    elif grain_type == "simulated_regular":
        return _process_simulated_grain(image, analysis, preserve_texture, use_gpu)
    elif grain_type == "digital_noise":
        return _process_digital_noise(image, analysis, preserve_texture, use_gpu)
    else:  # minimal_grain
        return _process_minimal_grain(image, analysis, preserve_texture, use_gpu)


def _process_fine_film_grain(
    image: np.ndarray,
    analysis: Dict[str, Any],
    preserve_texture: bool,
    use_gpu: bool
) -> np.ndarray:
    """Process fine film grain (35mm, high quality film)"""
    
    # Fine film grain requires gentle processing to preserve film character
    image = np.ascontiguousarray(image)
    
    # Stage 1: Gentle bilateral filtering to reduce noise while preserving edges
    sigma_color = 15 if preserve_texture else 25
    sigma_space = 15 if preserve_texture else 25
    
    if use_gpu and GPU_UTILS_AVAILABLE:
        result = gpu_bilateral_filter(image, 5, sigma_color, sigma_space)
    else:
        result = cv2.bilateralFilter(np.ascontiguousarray(image), 5, sigma_color, sigma_space)
    # FIXED: Ensure contiguous array after filtering
    result = np.ascontiguousarray(result)
    
    # Stage 2: Selective smoothing in flat regions only
    if preserve_texture:
        # Create edge mask
        gray = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        gray = np.ascontiguousarray(gray)
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = cv2.dilate(np.ascontiguousarray(edges), np.ones((3, 3)), iterations=1)
        
        # Apply additional smoothing only in non-edge regions
        smooth_strength = 0.3  # Gentle smoothing
        
        if use_gpu and GPU_UTILS_AVAILABLE:
            smoothed = gpu_gaussian_blur(result, 0.8)
        else:
            smoothed = cv2.GaussianBlur(np.ascontiguousarray(result), (0, 0), 0.8)
        
        # Blend based on edge mask
        if len(image.shape) == 3:
            edge_mask_3d = np.stack([edge_mask] * 3, axis=2) / 255.0
        else:
            edge_mask_3d = edge_mask / 255.0
        
        # CRITICAL FIX: Array arithmetic creates non-contiguous array
        result = (result * edge_mask_3d + smoothed * (1 - edge_mask_3d) * smooth_strength + 
                 result * (1 - edge_mask_3d) * (1 - smooth_strength)).astype(np.uint8)
        result = np.ascontiguousarray(result)
    
    return result


def _process_medium_film_grain(
    image: np.ndarray,
    analysis: Dict[str, Any],
    preserve_texture: bool,
    use_gpu: bool
) -> np.ndarray:
    """Process medium film grain (16mm, standard film)"""
    
    # Medium film grain can handle slightly more aggressive processing
    image = np.ascontiguousarray(image)
    
    # Stage 1: Non-local means for texture preservation
    h = 8 if preserve_texture else 12
    
    if use_gpu and GPU_UTILS_AVAILABLE:
        result = gpu_non_local_means(image, h, 5, 11)
    else:
        if len(image.shape) == 3:
            result = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(image), None, h, h, 5, 11)
        else:
            result = cv2.fastNlMeansDenoising(np.ascontiguousarray(image), None, h, 5, 11)
    # FIXED: Ensure contiguous array after denoising
    result = np.ascontiguousarray(result)
    
    # Stage 2: Gentle additional smoothing if needed
    if analysis['noise_level'] > 0.03:
        if use_gpu and GPU_UTILS_AVAILABLE:
            result = gpu_gaussian_blur(result, 0.5)
        else:
            result = cv2.GaussianBlur(np.ascontiguousarray(result), (0, 0), 0.5)
    
    return result


def _process_coarse_film_grain(
    image: np.ndarray,
    analysis: Dict[str, Any],
    preserve_texture: bool,
    use_gpu: bool
) -> np.ndarray:
    """Process coarse film grain (8mm, low light, push processed)"""
    
    # Coarse film grain requires more aggressive processing
    image = np.ascontiguousarray(image)
    
    # Stage 1: Bilateral filtering for noise reduction
    sigma_color = 25 if preserve_texture else 35
    sigma_space = 25 if preserve_texture else 35
    
    if use_gpu and GPU_UTILS_AVAILABLE:
        result = gpu_bilateral_filter(image, 7, sigma_color, sigma_space)
    else:
        result = cv2.bilateralFilter(np.ascontiguousarray(image), 7, sigma_color, sigma_space)
    # FIXED: Ensure contiguous array after bilateral filtering
    result = np.ascontiguousarray(result)
    
    # Stage 2: Non-local means for texture preservation
    h = 12 if preserve_texture else 18
    
    if use_gpu and GPU_UTILS_AVAILABLE:
        result = gpu_non_local_means(result, h, 7, 15)
    else:
        if len(image.shape) == 3:
            result = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(result), None, h, h, 7, 15)
        else:
            result = cv2.fastNlMeansDenoising(np.ascontiguousarray(result), None, h, 7, 15)
    # FIXED: Ensure contiguous array after non-local means
    result = np.ascontiguousarray(result)
    
    # Stage 3: Gentle sharpening to restore detail
    if preserve_texture:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.1
        # CRITICAL FIX: Ensure contiguous before cv2.filter2D
        result = np.ascontiguousarray(result)
        result = cv2.filter2D(result, -1, kernel)
        # CRITICAL FIX: Array arithmetic (np.clip) creates non-contiguous array
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = np.ascontiguousarray(result)
    
    return result


def _process_simulated_grain(
    image: np.ndarray,
    analysis: Dict[str, Any],
    preserve_texture: bool,
    use_gpu: bool
) -> np.ndarray:
    """Process simulated/artificial film grain"""
    
    # Simulated grain often has regular patterns and can be removed more aggressively
    image = np.ascontiguousarray(image)
    
    # Stage 1: Detect and remove regular patterns
    # Use morphological operations to remove regular grain patterns
    kernel_size = analysis['estimated_grain_size'] * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if len(image.shape) == 3:
        result_channels = []
        for c in range(image.shape[2]):
            # CRITICAL FIX: Channel slice creates non-contiguous array
            channel = np.ascontiguousarray(image[:, :, c])
            # Opening to remove small bright spots
            opened = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
            # Closing to remove small dark spots
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            result_channels.append(closed)
        result = np.stack(result_channels, axis=2)
        result = np.ascontiguousarray(result)
    else:
        contiguous_image = np.ascontiguousarray(image)
        result = cv2.morphologyEx(contiguous_image, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        result = np.ascontiguousarray(result)
    
    # Stage 2: Smoothing to blend the morphological operations
    if use_gpu and GPU_UTILS_AVAILABLE:
        result = gpu_gaussian_blur(result, 1.0)
    else:
        result = cv2.GaussianBlur(np.ascontiguousarray(result), (0, 0), 1.0)
    # FIXED: Ensure contiguous array after smoothing
    result = np.ascontiguousarray(result)
    
    # Stage 3: Blend with original to preserve some texture if requested
    if preserve_texture:
        alpha = 0.7  # Blend factor
        # CRITICAL FIX: Array arithmetic creates non-contiguous array
        result = (alpha * result + (1 - alpha) * image).astype(np.uint8)
        result = np.ascontiguousarray(result)
    
    return result


def _process_digital_noise(
    image: np.ndarray,
    analysis: Dict[str, Any],
    preserve_texture: bool,
    use_gpu: bool
) -> np.ndarray:
    """Process digital noise (high ISO, sensor noise)"""
    
    # Digital noise requires frequency domain processing
    image = np.ascontiguousarray(image)
    
    # Stage 1: Frequency domain filtering
    def high_freq_filter(f_shifted):
        h, w = f_shifted.shape
        center_h, center_w = h // 2, w // 2
        
        # Create high-frequency suppression filter
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((y - center_h)**2 + (x - center_w)**2)
        
        # Gaussian high-frequency suppression
        cutoff = min(h, w) // 8
        filter_response = np.exp(-(distance / cutoff)**2)
        
        return f_shifted * filter_response
    
    if use_gpu and GPU_UTILS_AVAILABLE:
        from .gpu_utils import gpu_frequency_filter
        result = gpu_frequency_filter(image, high_freq_filter)
    else:
        # CPU implementation
        if len(image.shape) == 3:
            result_channels = []
            for c in range(image.shape[2]):
                # CRITICAL FIX: Channel slice + arithmetic creates non-contiguous array
                channel = np.ascontiguousarray(image[:, :, c].astype(np.float32) / 255.0)
                f_transform = np.fft.fft2(channel)
                f_shifted = np.fft.fftshift(f_transform)
                filtered = high_freq_filter(f_shifted)
                f_ishifted = np.fft.ifftshift(filtered)
                result_channel = np.fft.ifft2(f_ishifted)
                result_channel = np.real(result_channel)
                result_channels.append(np.ascontiguousarray(result_channel))
            result = np.stack(result_channels, axis=2)
        else:
            # CRITICAL FIX: Arithmetic creates non-contiguous array
            channel = np.ascontiguousarray(image.astype(np.float32) / 255.0)
            f_transform = np.fft.fft2(channel)
            f_shifted = np.fft.fftshift(f_transform)
            filtered = high_freq_filter(f_shifted)
            f_ishifted = np.fft.ifftshift(filtered)
            result = np.fft.ifft2(f_ishifted)
            result = np.real(result)
        
        # CRITICAL FIX: Array arithmetic creates non-contiguous array
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        result = np.ascontiguousarray(result)
    
    # Stage 2: Additional bilateral filtering for remaining noise
    if preserve_texture:
        # CRITICAL FIX: Ensure contiguous before cv2.bilateralFilter
        result = np.ascontiguousarray(result)
        result = cv2.bilateralFilter(result, 5, 20, 20)
    
    return result


def _process_minimal_grain(
    image: np.ndarray,
    analysis: Dict[str, Any],
    preserve_texture: bool,
    use_gpu: bool
) -> np.ndarray:
    """Process images with minimal grain"""
    
    # Very gentle processing to avoid over-smoothing
    image = np.ascontiguousarray(image)
    
    # Light bilateral filtering
    if use_gpu and GPU_UTILS_AVAILABLE:
        result = gpu_bilateral_filter(image, 3, 10, 10)
    else:
        result = cv2.bilateralFilter(np.ascontiguousarray(image), 3, 10, 10)
    
    # Blend with original to maintain natural look
    alpha = 0.5  # Very conservative blending
    # CRITICAL FIX: Array arithmetic creates non-contiguous array
    result = (alpha * result + (1 - alpha) * image).astype(np.uint8)
    result = np.ascontiguousarray(result)
    
    return result


def get_grain_processing_recommendations(grain_type: str) -> Dict[str, Any]:
    """Get processing recommendations for different grain types"""
    
    recommendations = {
        "fine_film_grain": {
            "description": "High-quality 35mm film grain",
            "processing": "Gentle bilateral filtering with edge preservation",
            "preserve_texture": True,
            "strength": "Light",
            "notes": "Maintains film character while reducing noise"
        },
        "medium_film_grain": {
            "description": "Standard 16mm film grain",
            "processing": "Non-local means with moderate parameters",
            "preserve_texture": True,
            "strength": "Moderate",
            "notes": "Good balance of noise reduction and texture preservation"
        },
        "coarse_film_grain": {
            "description": "Heavy 8mm or push-processed film grain",
            "processing": "Multi-stage bilateral + NLM with optional sharpening",
            "preserve_texture": True,
            "strength": "Strong",
            "notes": "Aggressive denoising with detail recovery"
        },
        "simulated_regular": {
            "description": "Artificially added regular grain patterns",
            "processing": "Morphological operations + smoothing",
            "preserve_texture": False,
            "strength": "Targeted",
            "notes": "Removes artificial patterns while preserving natural texture"
        },
        "digital_noise": {
            "description": "High ISO sensor noise",
            "processing": "Frequency domain filtering + bilateral",
            "preserve_texture": True,
            "strength": "Adaptive",
            "notes": "Targets high-frequency noise while preserving detail"
        },
        "minimal_grain": {
            "description": "Very low noise levels",
            "processing": "Light bilateral filtering",
            "preserve_texture": True,
            "strength": "Minimal",
            "notes": "Conservative processing to avoid over-smoothing"
        }
    }
    
    return recommendations.get(grain_type, {
        "description": "Unknown grain type",
        "processing": "Default processing",
        "preserve_texture": True,
        "strength": "Moderate",
        "notes": "Use general-purpose denoising"
    })
