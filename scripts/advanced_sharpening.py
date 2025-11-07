"""
Advanced Sharpening Processor
Implementation of cutting-edge sharpening techniques from 2024-2025 research:
- Smart Sharpening with overshoot detection
- High Radius Low Amount (HiRaLoAm) technique
- Edge-directional kernels with orientation-specific enhancement
- Multi-scale Laplacian pyramids
- Guided filtering with edge-preserving smoothing

FIXED: NumPy 2.0 / OpenCV 4.11 compatibility - ensures all arrays are contiguous
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import filters, feature, color
from skimage.restoration import denoise_bilateral

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

class AdvancedSharpeningProcessor:
    """Advanced sharpening processor with multiple sophisticated techniques"""
    
    def __init__(self):
        self.name = "Advanced Sharpening Processor"
        self.version = "1.0.1"  # Updated version for NumPy 2.0 compatibility
        
    def smart_sharpening(self, image: np.ndarray, strength: float = 1.0, 
                        radius: float = 1.0, threshold: float = 0.1,
                        overshoot_protection: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Smart sharpening with overshoot detection and adaptive radius control
        FIXED: Scale-aware processing for real-world megapixel images
        
        Args:
            image: Input image (H, W) or (H, W, C) in range [0, 1]
            strength: Sharpening strength (0.1 to 3.0)
            radius: Base sharpening radius (0.5 to 5.0)
            threshold: Edge threshold for selective sharpening (0.01 to 0.5)
            overshoot_protection: Enable overshoot detection and control
            
        Returns:
            Tuple of (sharpened_image, processing_info)
        """
        try:
            # FIXED: Calculate scale factor based on image size
            image_area = image.shape[0] * image.shape[1]
            scale_factor = np.sqrt(image_area / (512 * 512))  # Base size 512x512
            scale_factor = np.clip(scale_factor, 0.5, 4.0)
            
            # FIXED: Scale-aware parameter adjustment
            adjusted_radius = radius * scale_factor
            adjusted_threshold = threshold / scale_factor  # Lower threshold for larger images
            # MODIFIED: Use full strength instead of reducing it (was * 0.7)
            # This gives more visible results as requested by users
            adjusted_strength = strength
            
            # Convert to working format
            if len(image.shape) == 3:
                is_color = True
                # Work in luminance for edge detection, preserve color
                lab_image = color.rgb2lab(image)
                # CRITICAL: Ensure lab_image is contiguous immediately after conversion
                lab_image = np.ascontiguousarray(lab_image)
                # CRITICAL: Ensure lab_image is contiguous immediately after conversion
                lab_image = np.ascontiguousarray(lab_image)
                luminance = lab_image[:, :, 0] / 100.0  # Normalize L channel
                # FIXED: Ensure contiguous array for OpenCV operations
                work_image = np.ascontiguousarray(luminance)
            else:
                is_color = False
                work_image = np.ascontiguousarray(image.copy())
            
            # Adaptive radius control based on local image content
            edge_map = feature.canny(work_image, sigma=adjusted_radius * 0.5, 
                                   low_threshold=adjusted_threshold * 0.5, 
                                   high_threshold=adjusted_threshold * 2.0)
            
            # Calculate local edge density for adaptive radius
            kernel_size = max(3, int(adjusted_radius * 4) // 2 * 2 + 1)
            edge_density = ndimage.uniform_filter(edge_map.astype(np.float32), size=kernel_size)
            
            # Adaptive radius: smaller radius for high edge density areas
            adaptive_radius = adjusted_radius * (1.0 - 0.5 * edge_density)
            adaptive_radius = np.clip(adaptive_radius, adjusted_radius * 0.3, adjusted_radius * 1.5)
            # FIXED: Ensure contiguous array after arithmetic operations
            adaptive_radius = np.ascontiguousarray(adaptive_radius)
            
            # Create adaptive blur using variable radius
            blurred = self._adaptive_gaussian_blur(work_image, adaptive_radius)
            
            # Calculate unsharp mask
            detail = work_image - blurred
            # FIXED: Ensure contiguous array after arithmetic operation
            detail = np.ascontiguousarray(detail)
            
            # Threshold-based selective sharpening with soft falloff
            # MODIFIED: Instead of hard cutoff, use soft weighting for more visible results
            significant_edges = np.abs(detail) > adjusted_threshold
            edge_weight = np.clip(np.abs(detail) / (adjusted_threshold + 1e-6), 0, 1)
            # Blend between full detail (edges) and partial detail (smooth areas)
            # This gives more visible sharpening across the entire image
            selective_detail = detail * (0.3 + 0.7 * edge_weight)  # Min 30% effect everywhere
            # FIXED: Ensure contiguous array after arithmetic operation
            selective_detail = np.ascontiguousarray(selective_detail)
            
            # Overshoot detection and control
            if overshoot_protection:
                selective_detail = self._control_overshoot(work_image, selective_detail, adjusted_strength)
                # FIXED: Ensure contiguous array after overshoot control
                selective_detail = np.ascontiguousarray(selective_detail)
            
            # Apply sharpening
            sharpened_luminance = work_image + adjusted_strength * selective_detail
            sharpened_luminance = np.clip(sharpened_luminance, 0, 1)
            
            # Reconstruct color image if needed
            if is_color:
                lab_result = lab_image.copy()
                lab_result[:, :, 0] = sharpened_luminance * 100.0
                # FIXED: Ensure contiguous array before color conversion
                lab_result = np.ascontiguousarray(lab_result)
                result = color.lab2rgb(lab_result)
                # CRITICAL FIX: Ensure result from color.lab2rgb is contiguous
                result = np.ascontiguousarray(result)
            else:
                result = sharpened_luminance
                # Ensure result is contiguous
                result = np.ascontiguousarray(result)
            
            # Calculate processing info
            info = {
                'method': 'Smart Sharpening',
                'strength': strength,
                'adjusted_strength': adjusted_strength,
                'base_radius': radius,
                'adjusted_radius': adjusted_radius,
                'threshold': threshold,
                'adjusted_threshold': adjusted_threshold,
                'scale_factor': scale_factor,
                'image_size': f"{image.shape[0]}x{image.shape[1]}",
                'overshoot_protection': overshoot_protection,
                'edge_pixels_detected': int(np.sum(edge_map)),
                'significant_edges': int(np.sum(significant_edges)),
                'adaptive_radius_range': [float(adaptive_radius.min()), float(adaptive_radius.max())],
                'luminance_only_processing': is_color
            }
            
            # Calculate quality metrics if possible
            if SKIMAGE_AVAILABLE:
                try:
                    if is_color:
                        psnr = peak_signal_noise_ratio(image, result, data_range=1.0)
                        ssim = structural_similarity(image, result, multichannel=True, 
                                                   channel_axis=-1, data_range=1.0)
                    else:
                        psnr = peak_signal_noise_ratio(image, result, data_range=1.0)
                        ssim = structural_similarity(image, result, data_range=1.0)
                    info['psnr'] = psnr
                    info['ssim'] = ssim
                except:
                    pass
            
            return result, info
            
        except Exception as e:
            import traceback
            print(f"\nSmart sharpening error: {str(e)}")
            traceback.print_exc()
            return None, {'error': f"Smart sharpening failed: {str(e)}"}
    
    def hiraloam_sharpening(self, image: np.ndarray, radius_ratio: float = 4.0,
                           amount_ratio: float = 0.25, blur_type: str = 'gaussian',
                           frequency_bands: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        High Radius Low Amount (HiRaLoAm) technique with multiple blur kernels
        
        Args:
            image: Input image (H, W) or (H, W, C) in range [0, 1]
            radius_ratio: High radius multiplier (2.0 to 6.0)
            amount_ratio: Low amount multiplier (0.1 to 0.5)
            blur_type: Type of blur ('gaussian', 'bilateral', 'mixed')
            frequency_bands: Number of frequency bands (2 to 5)
            
        Returns:
            Tuple of (sharpened_image, processing_info)
        """
        try:
            # Work in luminance for color images
            if len(image.shape) == 3:
                is_color = True
                lab_image = color.rgb2lab(image)
                # CRITICAL: Ensure lab_image is contiguous immediately after conversion
                lab_image = np.ascontiguousarray(lab_image)
                luminance = lab_image[:, :, 0] / 100.0
                # FIXED: Ensure contiguous array for OpenCV operations
                work_image = np.ascontiguousarray(luminance)
            else:
                is_color = False
                work_image = np.ascontiguousarray(image.copy())
            
            result = work_image.copy()
            processing_details = []
            
            # Multi-frequency band processing
            for band in range(frequency_bands):
                # Calculate radius for this frequency band
                base_radius = 1.0 + (band * radius_ratio / frequency_bands)
                
                # Create appropriate blur for this band
                if blur_type == 'gaussian':
                    blurred = gaussian_filter(work_image, sigma=base_radius)
                elif blur_type == 'bilateral':
                    # Convert to 8-bit for bilateral filter
                    img_8bit = (work_image * 255).astype(np.uint8)
                    # FIXED: Ensure contiguous array for OpenCV
                    img_8bit = np.ascontiguousarray(img_8bit)
                    try:
                        blurred_8bit = cv2.bilateralFilter(img_8bit, 
                                                          d=int(base_radius * 2) + 1,
                                                          sigmaColor=50, sigmaSpace=50)
                        blurred = blurred_8bit.astype(np.float32) / 255.0
                    except:
                        # Fallback to Gaussian if bilateral filter fails
                        blurred = gaussian_filter(work_image, sigma=base_radius)
                elif blur_type == 'mixed':
                    # Alternate between gaussian and bilateral
                    if band % 2 == 0:
                        blurred = gaussian_filter(work_image, sigma=base_radius)
                    else:
                        img_8bit = (work_image * 255).astype(np.uint8)
                        # FIXED: Ensure contiguous array for OpenCV
                        img_8bit = np.ascontiguousarray(img_8bit)
                        try:
                            blurred_8bit = cv2.bilateralFilter(img_8bit, 
                                                              d=int(base_radius * 2) + 1,
                                                              sigmaColor=50, sigmaSpace=50)
                            blurred = blurred_8bit.astype(np.float32) / 255.0
                        except:
                            # Fallback to Gaussian if bilateral filter fails
                            blurred = gaussian_filter(work_image, sigma=base_radius)
                
                # FIXED: Ensure blurred is contiguous before arithmetic
                blurred = np.ascontiguousarray(blurred)
                
                # Calculate detail for this frequency band
                detail = work_image - blurred
                # FIXED: Ensure contiguous array after arithmetic operation
                detail = np.ascontiguousarray(detail)
                
                # Apply low amount enhancement
                band_amount = amount_ratio / (band + 1)  # Reduce amount for higher frequencies
                result = result + band_amount * detail
                # FIXED: Ensure contiguous after arithmetic
                result = np.ascontiguousarray(result)
                
                processing_details.append({
                    'band': band,
                    'radius': base_radius,
                    'amount': band_amount,
                    'blur_type': blur_type if blur_type != 'mixed' else ('gaussian' if band % 2 == 0 else 'bilateral')
                })
            
            result = np.clip(result, 0, 1)
            
            # Reconstruct color image if needed
            if is_color:
                lab_result = lab_image.copy()
                lab_result[:, :, 0] = result * 100.0
                # FIXED: Ensure contiguous array before color conversion
                lab_result = np.ascontiguousarray(lab_result)
                result = color.lab2rgb(lab_result)
                # CRITICAL FIX: Ensure result from color.lab2rgb is contiguous
                result = np.ascontiguousarray(result)
            else:
                # Ensure result is contiguous
                result = np.ascontiguousarray(result)
            
            info = {
                'method': 'HiRaLoAm Sharpening',
                'radius_ratio': radius_ratio,
                'amount_ratio': amount_ratio,
                'blur_type': blur_type,
                'frequency_bands': frequency_bands,
                'processing_details': processing_details,
                'luminance_only_processing': is_color
            }
            
            return result, info
            
        except Exception as e:
            import traceback
            print(f"\nHiRaLoAm sharpening error: {str(e)}")
            traceback.print_exc()
            return None, {'error': f"HiRaLoAm sharpening failed: {str(e)}"}
    
    def edge_directional_sharpening(self, image: np.ndarray, strength: float = 1.0,
                                   num_directions: int = 8, 
                                   luminance_only: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Edge-directional sharpening with orientation-specific enhancement
        
        Args:
            image: Input image (H, W) or (H, W, C) in range [0, 1]
            strength: Sharpening strength (0.1 to 2.0)
            num_directions: Number of directional filters (4, 8, or 16)
            luminance_only: Process only luminance to avoid color artifacts
            
        Returns:
            Tuple of (sharpened_image, processing_info)
        """
        try:
            # Work in luminance for color images if specified
            if len(image.shape) == 3 and luminance_only:
                is_color = True
                lab_image = color.rgb2lab(image)
                # CRITICAL: Ensure lab_image is contiguous immediately after conversion
                lab_image = np.ascontiguousarray(lab_image)
                luminance = lab_image[:, :, 0] / 100.0
                # FIXED: Ensure contiguous array for OpenCV operations
                work_image = np.ascontiguousarray(luminance)
            else:
                is_color = False
                work_image = np.ascontiguousarray(image.copy())
            
            # Create directional filters
            directional_responses = []
            angles = np.linspace(0, np.pi, num_directions, endpoint=False)
            
            for i, angle in enumerate(angles):
                # Create oriented edge detection kernel
                kernel = self._create_directional_kernel(angle, size=5)
                
                # Apply directional filter
                response = ndimage.convolve(work_image, kernel, mode='reflect')
                directional_responses.append(response)
            
            # Combine directional responses with maximum response selection
            max_response = np.zeros_like(work_image)
            direction_map = np.zeros_like(work_image, dtype=np.int32)
            
            for i, response in enumerate(directional_responses):
                mask = np.abs(response) > np.abs(max_response)
                max_response[mask] = response[mask]
                direction_map[mask] = i
            
            # Apply orientation-specific enhancement
            enhanced = work_image + strength * max_response
            enhanced = np.clip(enhanced, 0, 1)
            
            # Reconstruct color image if needed
            if is_color:
                lab_result = lab_image.copy()
                lab_result[:, :, 0] = enhanced * 100.0
                # FIXED: Ensure contiguous array before color conversion
                lab_result = np.ascontiguousarray(lab_result)
                result = color.lab2rgb(lab_result)
                # CRITICAL FIX: Ensure result from color.lab2rgb is contiguous
                result = np.ascontiguousarray(result)
            else:
                result = enhanced
                # Ensure result is contiguous
                result = np.ascontiguousarray(result)
            
            info = {
                'method': 'Edge-Directional Sharpening',
                'strength': strength,
                'num_directions': num_directions,
                'luminance_only': luminance_only,
                'direction_angles': angles.tolist(),
                'max_response_range': [float(max_response.min()), float(max_response.max())],
                'dominant_directions': [int(d) for d in np.unique(direction_map)]
            }
            
            return result, info
            
        except Exception as e:
            import traceback
            print(f"\nEdge-directional sharpening error: {str(e)}")
            traceback.print_exc()
            return None, {'error': f"Edge-directional sharpening failed: {str(e)}"}
    
    def multiscale_laplacian_sharpening(self, image: np.ndarray, strength: float = 1.0,
                                       scales: List[float] = [0.5, 1.0, 2.0, 4.0],
                                       kernel_sizes: List[int] = [3, 5, 7, 9],
                                       scaling_factors: List[float] = [0.2, 0.4, 0.6, 0.8]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Multi-scale Laplacian pyramid sharpening with adaptive scaling
        
        Args:
            image: Input image (H, W) or (H, W, C) in range [0, 1]
            strength: Overall sharpening strength (0.1 to 2.0)
            scales: Gaussian pyramid scales
            kernel_sizes: Laplacian kernel sizes (must match scales length)
            scaling_factors: Adaptive scaling factors (0.2-0.8 range)
            
        Returns:
            Tuple of (sharpened_image, processing_info)
        """
        try:
            # Ensure parameter lists have same length
            num_scales = len(scales)
            if len(kernel_sizes) != num_scales:
                kernel_sizes = kernel_sizes[:num_scales] + [kernel_sizes[-1]] * (num_scales - len(kernel_sizes))
            if len(scaling_factors) != num_scales:
                scaling_factors = scaling_factors[:num_scales] + [scaling_factors[-1]] * (num_scales - len(scaling_factors))
            
            # Work in luminance for color images
            if len(image.shape) == 3:
                is_color = True
                lab_image = color.rgb2lab(image)
                # CRITICAL: Ensure lab_image is contiguous immediately after conversion
                lab_image = np.ascontiguousarray(lab_image)
                luminance = lab_image[:, :, 0] / 100.0
                # FIXED: Ensure contiguous array for OpenCV operations
                work_image = np.ascontiguousarray(luminance)
            else:
                is_color = False
                work_image = np.ascontiguousarray(image.copy())
            
            # Build Gaussian pyramid
            pyramid = [work_image]
            for scale in scales[1:]:
                sigma = scale
                blurred = gaussian_filter(work_image, sigma=sigma)
                pyramid.append(blurred)
            
            # Calculate Laplacian pyramid
            laplacian_pyramid = []
            scale_details = []
            
            for i in range(len(pyramid) - 1):
                # Calculate Laplacian
                laplacian = pyramid[i] - pyramid[i + 1]
                
                # Apply kernel-specific enhancement
                kernel_size = kernel_sizes[i]
                if kernel_size > 3:
                    # Create custom Laplacian kernel
                    kernel = self._create_laplacian_kernel(kernel_size)
                    enhanced_laplacian = ndimage.convolve(laplacian, kernel, mode='reflect')
                else:
                    enhanced_laplacian = laplacian
                
                # Apply adaptive scaling
                scaling_factor = scaling_factors[i]
                scaled_laplacian = enhanced_laplacian * scaling_factor
                laplacian_pyramid.append(scaled_laplacian)
                
                scale_details.append({
                    'scale': scales[i],
                    'kernel_size': kernel_size,
                    'scaling_factor': scaling_factor,
                    'laplacian_range': [float(scaled_laplacian.min()), float(scaled_laplacian.max())]
                })
            
            # Reconstruct enhanced image
            result = pyramid[-1]  # Start with finest blur
            for laplacian in reversed(laplacian_pyramid):
                result = result + strength * laplacian
                # FIXED: Ensure contiguous after each addition
                result = np.ascontiguousarray(result)
            
            result = np.clip(result, 0, 1)
            
            # Reconstruct color image if needed
            if is_color:
                lab_result = lab_image.copy()
                lab_result[:, :, 0] = result * 100.0
                # FIXED: Ensure contiguous array before color conversion
                lab_result = np.ascontiguousarray(lab_result)
                result = color.lab2rgb(lab_result)
                # CRITICAL FIX: Ensure result from color.lab2rgb is contiguous
                result = np.ascontiguousarray(result)
            else:
                # Ensure result is contiguous
                result = np.ascontiguousarray(result)
            
            info = {
                'method': 'Multi-scale Laplacian Sharpening',
                'strength': strength,
                'num_scales': num_scales,
                'scales': scales,
                'kernel_sizes': kernel_sizes,
                'scaling_factors': scaling_factors,
                'scale_details': scale_details,
                'luminance_only_processing': is_color
            }
            
            return result, info
            
        except Exception as e:
            import traceback
            print(f"\nMulti-scale Laplacian sharpening error: {str(e)}")
            traceback.print_exc()
            return None, {'error': f"Multi-scale Laplacian sharpening failed: {str(e)}"}
    
    def guided_filter_sharpening(self, image: np.ndarray, strength: float = 1.0,
                                radius: int = 8, epsilon: float = 0.1) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Guided filtering with edge-preserving smoothing and sharpening feedback
        
        Args:
            image: Input image (H, W) or (H, W, C) in range [0, 1]
            strength: Sharpening strength (0.1 to 2.0)
            radius: Guided filter radius (4 to 16)
            epsilon: Regularization parameter (0.01 to 1.0)
            
        Returns:
            Tuple of (sharpened_image, processing_info)
        """
        try:
            # Work in luminance for color images
            if len(image.shape) == 3:
                is_color = True
                lab_image = color.rgb2lab(image)
                # CRITICAL: Ensure lab_image is contiguous immediately after conversion
                lab_image = np.ascontiguousarray(lab_image)
                luminance = lab_image[:, :, 0] / 100.0
                # FIXED: Ensure contiguous array for OpenCV operations
                work_image = np.ascontiguousarray(luminance)
            else:
                is_color = False
                work_image = np.ascontiguousarray(image.copy())
            
            # Apply guided filter (edge-preserving smoothing)
            guided_filtered = self._guided_filter(work_image, work_image, radius, epsilon)
            
            # Calculate detail layer
            detail = work_image - guided_filtered
            # FIXED: Ensure contiguous array after arithmetic operation
            detail = np.ascontiguousarray(detail)
            
            # Apply edge-preserving enhancement with feedback
            # Use bilateral filter to preserve edge structure in detail layer
            detail_preserved = denoise_bilateral(detail, sigma_color=0.1, sigma_spatial=radius//2)
            # Ensure contiguous after denoise_bilateral
            detail_preserved = np.ascontiguousarray(detail_preserved)
            
            # Combine with sharpening feedback
            enhanced = work_image + strength * detail_preserved
            enhanced = np.clip(enhanced, 0, 1)
            
            # Reconstruct color image if needed
            if is_color:
                lab_result = lab_image.copy()
                lab_result[:, :, 0] = enhanced * 100.0
                # FIXED: Ensure contiguous array before color conversion
                lab_result = np.ascontiguousarray(lab_result)
                result = color.lab2rgb(lab_result)
                # CRITICAL FIX: Ensure result from color.lab2rgb is contiguous
                result = np.ascontiguousarray(result)
            else:
                result = enhanced
                # Ensure result is contiguous
                result = np.ascontiguousarray(result)
            
            info = {
                'method': 'Guided Filter Sharpening',
                'strength': strength,
                'radius': radius,
                'epsilon': epsilon,
                'detail_range': [float(detail.min()), float(detail.max())],
                'preserved_detail_range': [float(detail_preserved.min()), float(detail_preserved.max())],
                'luminance_only_processing': is_color
            }
            
            return result, info
            
        except Exception as e:
            import traceback
            print(f"\nGuided filter sharpening error: {str(e)}")
            traceback.print_exc()
            return None, {'error': f"Guided filter sharpening failed: {str(e)}"}
    
    def process_image(self, image: np.ndarray, method: str = 'smart', **kwargs) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Main processing method with automatic method selection
        
        Args:
            image: Input image (H, W) or (H, W, C) in range [0, 1]
            method: Sharpening method ('smart', 'hiraloam', 'directional', 'multiscale', 'guided', 'auto')
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        try:
            if method == 'auto':
                # Automatic method selection based on image characteristics
                method = self._select_optimal_method(image)
                kwargs['auto_selected'] = True
            
            if method == 'smart':
                return self.smart_sharpening(image, **kwargs)
            elif method == 'hiraloam':
                return self.hiraloam_sharpening(image, **kwargs)
            elif method == 'directional':
                return self.edge_directional_sharpening(image, **kwargs)
            elif method == 'multiscale':
                return self.multiscale_laplacian_sharpening(image, **kwargs)
            elif method == 'guided':
                return self.guided_filter_sharpening(image, **kwargs)
            else:
                return None, {'error': f"Unknown sharpening method: {method}"}
                
        except Exception as e:
            import traceback
            print(f"\nAdvanced sharpening error: {str(e)}")
            traceback.print_exc()
            return None, {'error': f"Advanced sharpening processing failed: {str(e)}"}
    
    # Helper methods
    def _adaptive_gaussian_blur(self, image: np.ndarray, radius_map: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur with spatially varying radius"""
        # Approximate adaptive blur with multiple fixed-radius blurs
        # This is a simplified implementation - full adaptive blur would require more complex algorithms
        
        # Sample a few different radii
        radii = [radius_map.min(), np.median(radius_map), radius_map.max()]
        
        # Create blurred versions
        blurred_versions = []
        for r in radii:
            blurred = gaussian_filter(image, sigma=r)
            blurred_versions.append(blurred)
        
        # Interpolate based on local radius
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                local_radius = radius_map[i, j]
                
                # Find interpolation weights
                if local_radius <= radii[0]:
                    result[i, j] = blurred_versions[0][i, j]
                elif local_radius >= radii[-1]:
                    result[i, j] = blurred_versions[-1][i, j]
                else:
                    # Linear interpolation between nearest radii
                    for k in range(len(radii) - 1):
                        if radii[k] <= local_radius <= radii[k + 1]:
                            t = (local_radius - radii[k]) / (radii[k + 1] - radii[k])
                            result[i, j] = (1 - t) * blurred_versions[k][i, j] + t * blurred_versions[k + 1][i, j]
                            break
        
        # FIXED: Ensure contiguous array after pixel-wise construction
        return np.ascontiguousarray(result)
    
    def _control_overshoot(self, original: np.ndarray, detail: np.ndarray, strength: float) -> np.ndarray:
        """Control overshoot artifacts in sharpening"""
        # Calculate potential result
        potential_result = original + strength * detail
        
        # Detect overshoot (values outside [0, 1] range)
        overshoot_mask = (potential_result < 0) | (potential_result > 1)
        
        # Reduce detail strength in overshoot areas
        controlled_detail = detail.copy()
        
        # For negative overshoot
        negative_mask = potential_result < 0
        if np.any(negative_mask):
            # Reduce detail to prevent going below 0
            max_negative_detail = -original[negative_mask]
            controlled_detail[negative_mask] = np.maximum(detail[negative_mask], max_negative_detail / strength)
        
        # For positive overshoot
        positive_mask = potential_result > 1
        if np.any(positive_mask):
            # Reduce detail to prevent going above 1
            max_positive_detail = (1 - original[positive_mask])
            controlled_detail[positive_mask] = np.minimum(detail[positive_mask], max_positive_detail / strength)
        
        return controlled_detail
    
    def _create_directional_kernel(self, angle: float, size: int = 5) -> np.ndarray:
        """Create edge detection kernel for specific direction"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Create oriented edge kernel using Sobel-like approach
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        for i in range(size):
            for j in range(size):
                x = j - center
                y = i - center
                
                # Project coordinates onto direction
                parallel = x * cos_angle + y * sin_angle
                perpendicular = -x * sin_angle + y * cos_angle
                
                # Create edge response (derivative in perpendicular direction)
                if abs(perpendicular) <= 1:
                    kernel[i, j] = np.sign(perpendicular) * np.exp(-parallel**2 / 2)
        
        # Normalize kernel
        kernel = kernel / (np.sum(np.abs(kernel)) + 1e-10)
        
        return kernel
    
    def _create_laplacian_kernel(self, size: int) -> np.ndarray:
        """Create Laplacian kernel of specified size"""
        if size == 3:
            return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        elif size == 5:
            return np.array([
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0]
            ]) / 8
        else:
            # Generate larger Laplacian kernel
            kernel = np.zeros((size, size))
            center = size // 2
            
            # Simple approach: negative weight at distance 1, positive at center
            for i in range(size):
                for j in range(size):
                    if i == center and j == center:
                        kernel[i, j] = (size - 1) * 4  # Positive center
                    elif abs(i - center) + abs(j - center) == 1:
                        kernel[i, j] = -1  # Negative neighbors
            
            return kernel
    
    def _guided_filter(self, guide: np.ndarray, input_img: np.ndarray, 
                      radius: int, epsilon: float) -> np.ndarray:
        """Guided filter implementation"""
        # Box filter
        def box_filter(img, r):
            return ndimage.uniform_filter(img.astype(np.float64), size=2*r+1, mode='reflect')
        
        # Mean and variance of guide
        mean_guide = box_filter(guide, radius)
        mean_input = box_filter(input_img, radius)
        mean_guide_input = box_filter(guide * input_img, radius)
        
        # Covariance and variance
        cov_guide_input = mean_guide_input - mean_guide * mean_input
        var_guide = box_filter(guide * guide, radius) - mean_guide * mean_guide
        
        # Coefficients
        a = cov_guide_input / (var_guide + epsilon)
        b = mean_input - a * mean_guide
        
        # Filter coefficients
        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)
        
        # Output
        output = mean_a * guide + mean_b
        # FIXED: Ensure contiguous after arithmetic
        output = np.ascontiguousarray(output)
        
        return output
    
    def _select_optimal_method(self, image: np.ndarray) -> str:
        """Automatically select optimal sharpening method based on image characteristics"""
        # Analyze image characteristics
        if len(image.shape) == 3:
            try:
                gray = color.rgb2gray(image)
            except:
                # Fallback if color conversion fails
                gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calculate image metrics with error handling
        try:
            edges = feature.canny(gray, sigma=1.0)
            edge_density = np.mean(edges)
        except:
            edge_density = 0.0
        
        # Calculate local variance (texture measure)
        try:
            local_variance = ndimage.generic_filter(gray, np.var, size=5)
            texture_level = np.mean(local_variance)
        except:
            texture_level = 0.0
        
        # Calculate gradient magnitude
        try:
            grad_x = ndimage.sobel(gray, axis=0)
            grad_y = ndimage.sobel(gray, axis=1)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_level = np.mean(gradient_magnitude)
        except:
            gradient_level = 0.0
        
        # Decision logic with fallbacks
        if edge_density > 0.1 and gradient_level > 0.2:
            return 'directional'  # High edge content - use directional
        elif texture_level > 0.01:
            return 'multiscale'   # High texture - use multi-scale
        elif edge_density > 0.05:
            return 'guided'       # Moderate edges - use guided filter
        elif gradient_level < 0.1:
            return 'hiraloam'     # Low contrast - use HiRaLoAm
        else:
            return 'smart'        # Default to smart sharpening
    
    def get_method_info(self) -> Dict[str, str]:
        """Get information about available sharpening methods"""
        return {
            'smart': 'Smart Sharpening with overshoot detection and adaptive radius control',
            'hiraloam': 'High Radius Low Amount technique with multiple blur kernels',
            'directional': 'Edge-directional sharpening with orientation-specific enhancement',
            'multiscale': 'Multi-scale Laplacian pyramid sharpening with adaptive scaling',
            'guided': 'Guided filtering with edge-preserving smoothing and feedback',
            'auto': 'Automatic method selection based on image characteristics'
        }
