"""
Adaptive Frequency Decomposition (AFD) Processor
Implementation of cutting-edge frequency domain enhancement from 2024-2025 research:
- Multi-scale frequency band decomposition with adaptive thresholds
- Color-aware frequency processing in perceptual spaces
- Selective frequency enhancement based on image content analysis
- Advanced wavelet and Fourier hybrid decomposition
- Perceptual frequency weighting and reconstruction
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List, Union
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import color, restoration, feature
from skimage.util import img_as_float
import pywt

try:
    from scipy.signal import butter, filtfilt
    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False

class AdaptiveFrequencyDecompositionProcessor:
    """Adaptive Frequency Decomposition processor with color-aware multi-scale analysis"""
    
    def __init__(self):
        self.name = "Adaptive Frequency Decomposition Processor"
        self.version = "1.0"
        self.wavelet_cache = {}
        
    def adaptive_frequency_decomposition(self, image: np.ndarray,
                                       color_space: str = 'lab',
                                       decomposition_method: str = 'hybrid',
                                       frequency_bands: int = 6,
                                       adaptive_thresholds: bool = True,
                                       color_aware_processing: bool = True,
                                       enhancement_strength: float = 1.0,
                                       selective_enhancement: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adaptive Frequency Decomposition with color-aware multi-scale processing
        FIXED: Scale-aware processing for real-world megapixel images
        
        Args:
            image: Input image (H, W) or (H, W, C) in range [0, 1]
            color_space: Color space ('lab', 'oklab', 'jzazbz', 'hsv', 'rgb')
            decomposition_method: Method ('wavelet', 'fourier', 'hybrid', 'adaptive')
            frequency_bands: Number of frequency bands (3 to 8)
            adaptive_thresholds: Use adaptive thresholds for each band
            color_aware_processing: Process frequency bands differently per color channel
            enhancement_strength: Overall enhancement strength (0.1 to 3.0)
            selective_enhancement: Apply selective enhancement based on content analysis
            
        Returns:
            Tuple of (enhanced_image, processing_info)
        """
        try:
            # FIXED: Calculate scale factor based on image size
            image_area = image.shape[0] * image.shape[1]
            scale_factor = np.sqrt(image_area / (512 * 512))  # Normalize to 512x512 base
            scale_factor = np.clip(scale_factor, 0.5, 4.0)    # Reasonable bounds
            
            # Convert to working color space
            work_image, color_info = self._convert_to_colorspace(image, color_space)
            
            # Analyze image characteristics for adaptive processing
            image_analysis = self._analyze_image_characteristics(work_image, color_space)
            
            # FIXED: Add scale information to analysis
            image_analysis['scale_factor'] = scale_factor
            image_analysis['image_size'] = f"{image.shape[0]}x{image.shape[1]}"
            
            # Select optimal decomposition method if adaptive
            if decomposition_method == 'adaptive':
                decomposition_method = self._select_optimal_decomposition(image_analysis)
            
            # Perform frequency decomposition with scale-aware parameters
            frequency_bands_data = self._decompose_frequencies(
                work_image, decomposition_method, frequency_bands, color_space, scale_factor
            )
            
            # Calculate adaptive thresholds if enabled
            if adaptive_thresholds:
                thresholds = self._calculate_adaptive_thresholds(
                    frequency_bands_data, image_analysis
                )
            else:
                # FIXED: Scale-aware default thresholds
                base_threshold = 0.05 / scale_factor  # Smaller threshold for larger images
                thresholds = [base_threshold * (i + 1) for i in range(frequency_bands)]
            
            # Apply color-aware processing if enabled
            if color_aware_processing and len(work_image.shape) == 3:
                enhanced_bands = self._color_aware_frequency_processing(
                    frequency_bands_data, thresholds, enhancement_strength, color_space, scale_factor
                )
            else:
                enhanced_bands = self._standard_frequency_processing(
                    frequency_bands_data, thresholds, enhancement_strength, scale_factor, color_space
                )
            
            # Apply selective enhancement if enabled
            if selective_enhancement:
                enhanced_bands = self._selective_frequency_enhancement(
                    enhanced_bands, image_analysis, work_image
                )
            
            # Reconstruct enhanced image
            result = self._reconstruct_from_frequency_bands(
                enhanced_bands, decomposition_method, work_image.shape
            )
            
            # Convert back to original color space
            final_result = self._convert_from_colorspace(result, color_space, color_info)
            
            # Calculate processing metrics
            processing_info = self._calculate_processing_metrics(
                image, final_result, frequency_bands_data, enhanced_bands, image_analysis
            )
            
            info = {
                'method': 'Adaptive Frequency Decomposition',
                'color_space': color_space,
                'decomposition_method': decomposition_method,
                'frequency_bands': frequency_bands,
                'adaptive_thresholds': adaptive_thresholds,
                'color_aware_processing': color_aware_processing,
                'enhancement_strength': enhancement_strength,
                'selective_enhancement': selective_enhancement,
                'image_analysis': image_analysis,
                'thresholds': thresholds,
                'processing_metrics': processing_info
            }
            
            return final_result, info
            
        except Exception as e:
            return None, {'error': f"Adaptive frequency decomposition failed: {str(e)}"}
    
    def _convert_to_colorspace(self, image: np.ndarray, color_space: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convert image to specified color space"""
        if len(image.shape) == 2:
            return image, {'grayscale': True}
        
        color_info = {'grayscale': False, 'original_space': 'rgb'}
        
        if color_space == 'lab':
            work_image = color.rgb2lab(image)
            work_image[:, :, 0] = work_image[:, :, 0] / 100.0  # Normalize L channel
            color_info['lab_conversion'] = True
            
        elif color_space == 'oklab':
            # Approximate with LAB for now
            work_image = color.rgb2lab(image)
            work_image[:, :, 0] = work_image[:, :, 0] / 100.0
            color_info['oklab_approximation'] = True
            
        elif color_space == 'jzazbz':
            # Approximate with LAB for now
            work_image = color.rgb2lab(image)
            work_image[:, :, 0] = work_image[:, :, 0] / 100.0
            color_info['jzazbz_approximation'] = True
            
        elif color_space == 'hsv':
            work_image = color.rgb2hsv(image)
            color_info['hsv_conversion'] = True
            
        else:  # RGB
            work_image = image.copy()
            color_info['rgb_passthrough'] = True
            
        return work_image, color_info
    
    def _convert_from_colorspace(self, image: np.ndarray, color_space: str, 
                                color_info: Dict[str, Any]) -> np.ndarray:
        """Convert image back from specified color space"""
        if color_info.get('grayscale', False):
            return image
        
        if color_space == 'lab' or color_info.get('oklab_approximation') or color_info.get('jzazbz_approximation'):
            lab_image = image.copy()
            lab_image[:, :, 0] = lab_image[:, :, 0] * 100.0  # Denormalize L channel
            # FIXED: Ensure contiguous array after slice assignment before color conversion
            lab_image = np.ascontiguousarray(lab_image)
            result = color.lab2rgb(lab_image)
            
        elif color_space == 'hsv':
            result = color.hsv2rgb(image)
            
        else:  # RGB
            result = image.copy()
            
        return np.clip(result, 0, 1)
    
    def _analyze_image_characteristics(self, image: np.ndarray, color_space: str) -> Dict[str, Any]:
        """Analyze image characteristics for adaptive processing"""
        # Use luminance for analysis
        if len(image.shape) == 3:
            if color_space in ['lab', 'oklab', 'jzazbz']:
                luminance = image[:, :, 0]
                # FIXED: Ensure contiguous array after channel extraction
                luminance = np.ascontiguousarray(luminance)
            else:
                luminance = color.rgb2gray(image)
        else:
            luminance = image
        
        analysis = {}
        
        try:
            # Basic statistics
            analysis['mean_luminance'] = float(np.mean(luminance))
            analysis['std_luminance'] = float(np.std(luminance))
            analysis['dynamic_range'] = float(np.max(luminance) - np.min(luminance))
            
            # Frequency content analysis
            fft_lum = np.abs(fft2(luminance))
            fft_centered = fftshift(fft_lum)
            
            # Calculate frequency energy distribution
            h, w = fft_centered.shape
            center_y, center_x = h // 2, w // 2
            
            # Define frequency regions
            low_freq_mask = self._create_frequency_mask(fft_centered.shape, center_y, center_x, 0.1)
            mid_freq_mask = self._create_frequency_mask(fft_centered.shape, center_y, center_x, 0.4) & ~low_freq_mask
            high_freq_mask = ~(low_freq_mask | mid_freq_mask)
            
            total_energy = np.sum(fft_centered**2)
            analysis['low_freq_energy'] = float(np.sum(fft_centered[low_freq_mask]**2) / total_energy)
            analysis['mid_freq_energy'] = float(np.sum(fft_centered[mid_freq_mask]**2) / total_energy)
            analysis['high_freq_energy'] = float(np.sum(fft_centered[high_freq_mask]**2) / total_energy)
            
            # Texture analysis
            try:
                edges = feature.canny(luminance, sigma=1.0)
                analysis['edge_density'] = float(np.mean(edges))
                
                # Local variance (texture measure)
                local_variance = ndimage.generic_filter(luminance, np.var, size=5)
                analysis['texture_level'] = float(np.mean(local_variance))
                analysis['texture_variation'] = float(np.std(local_variance))
            except:
                analysis['edge_density'] = 0.0
                analysis['texture_level'] = 0.0
                analysis['texture_variation'] = 0.0
            
            # Color distribution analysis (if color image)
            if len(image.shape) == 3:
                try:
                    if color_space in ['lab', 'oklab', 'jzazbz']:
                        a_channel = image[:, :, 1]
                        b_channel = image[:, :, 2]
                        analysis['color_variance_a'] = float(np.var(a_channel))
                        analysis['color_variance_b'] = float(np.var(b_channel))
                        analysis['color_saturation'] = float(np.sqrt(a_channel**2 + b_channel**2).mean())
                    else:
                        # For RGB/HSV, calculate general color variance
                        analysis['color_variance_1'] = float(np.var(image[:, :, 1]))
                        analysis['color_variance_2'] = float(np.var(image[:, :, 2]))
                        analysis['color_range'] = float(np.std(image, axis=2).mean())
                except:
                    analysis['color_variance_a'] = 0.0
                    analysis['color_variance_b'] = 0.0
                    analysis['color_saturation'] = 0.0
            
        except Exception as e:
            analysis['analysis_error'] = str(e)
            # Fallback values
            analysis.update({
                'mean_luminance': 0.5,
                'std_luminance': 0.2,
                'dynamic_range': 1.0,
                'low_freq_energy': 0.5,
                'mid_freq_energy': 0.3,
                'high_freq_energy': 0.2,
                'edge_density': 0.1,
                'texture_level': 0.1
            })
        
        return analysis
    
    def _create_frequency_mask(self, shape: Tuple[int, int], center_y: int, center_x: int, 
                              radius_fraction: float) -> np.ndarray:
        """Create circular frequency mask"""
        h, w = shape
        y, x = np.ogrid[:h, :w]
        radius = min(h, w) * radius_fraction / 2
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        return mask
    
    def _select_optimal_decomposition(self, analysis: Dict[str, Any]) -> str:
        """Select optimal decomposition method based on image characteristics"""
        high_freq_energy = analysis.get('high_freq_energy', 0.2)
        texture_level = analysis.get('texture_level', 0.1)
        edge_density = analysis.get('edge_density', 0.1)
        
        if high_freq_energy > 0.3 and texture_level > 0.15:
            return 'wavelet'  # Good for textured images
        elif edge_density > 0.15:
            return 'hybrid'   # Good for images with strong edges
        else:
            return 'fourier'  # Good for smooth images
    
    def _decompose_frequencies(self, image: np.ndarray, method: str, 
                             num_bands: int, color_space: str, scale_factor: float = 1.0) -> List[np.ndarray]:
        """Decompose image into frequency bands with scale-aware parameters"""
        if method == 'wavelet':
            return self._wavelet_decomposition(image, num_bands, scale_factor)
        elif method == 'fourier':
            return self._fourier_decomposition(image, num_bands, scale_factor)
        elif method == 'hybrid':
            return self._hybrid_decomposition(image, num_bands, scale_factor)
        else:
            # Default to wavelet
            return self._wavelet_decomposition(image, num_bands, scale_factor)
    
    def _wavelet_decomposition(self, image: np.ndarray, num_bands: int, scale_factor: float = 1.0) -> List[np.ndarray]:
        """Wavelet-based frequency decomposition"""
        bands = []
        
        # Process each channel separately if color image
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                channel_bands = self._wavelet_decompose_channel(image[:, :, c], num_bands)
                if c == 0:
                    bands = [np.zeros((*band.shape, image.shape[2])) for band in channel_bands]
                for i, band in enumerate(channel_bands):
                    bands[i][:, :, c] = band
        else:
            bands = self._wavelet_decompose_channel(image, num_bands)
        
        return bands
    
    def _wavelet_decompose_channel(self, channel: np.ndarray, num_bands: int) -> List[np.ndarray]:
        """Decompose single channel using wavelets"""
        # Use Daubechies wavelet
        wavelet = 'db8'
        
        # Perform multi-level decomposition
        coeffs = pywt.wavedec2(channel, wavelet, level=min(num_bands-1, 6))
        
        # Convert coefficients to frequency bands
        bands = []
        
        # Approximation coefficients (lowest frequency)
        approx = coeffs[0]
        # Resize to original size
        bands.append(cv2.resize(approx, (channel.shape[1], channel.shape[0]), interpolation=cv2.INTER_CUBIC))
        
        # Detail coefficients (higher frequencies)
        for i, (cH, cV, cD) in enumerate(coeffs[1:]):
            # Combine horizontal, vertical, and diagonal details
            detail_combined = np.sqrt(cH**2 + cV**2 + cD**2)
            # Resize to original size
            detail_resized = cv2.resize(detail_combined, (channel.shape[1], channel.shape[0]), 
                                      interpolation=cv2.INTER_CUBIC)
            bands.append(detail_resized)
        
        # If we need more bands, create additional bands through Gaussian pyramid
        while len(bands) < num_bands:
            # Create additional band by further decomposing the highest frequency band
            last_band = bands[-1]
            gaussian_blur = ndimage.gaussian_filter(last_band, sigma=0.5)
            high_freq = last_band - gaussian_blur
            bands.append(high_freq)
        
        return bands[:num_bands]
    
    def _fourier_decomposition(self, image: np.ndarray, num_bands: int, scale_factor: float = 1.0) -> List[np.ndarray]:
        """Fourier-based frequency decomposition"""
        bands = []
        
        # Process each channel separately if color image
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                channel_bands = self._fourier_decompose_channel(image[:, :, c], num_bands)
                if c == 0:
                    bands = [np.zeros((*band.shape, image.shape[2])) for band in channel_bands]
                for i, band in enumerate(channel_bands):
                    bands[i][:, :, c] = band
        else:
            bands = self._fourier_decompose_channel(image, num_bands)
        
        return bands
    
    def _fourier_decompose_channel(self, channel: np.ndarray, num_bands: int) -> List[np.ndarray]:
        """Decompose single channel using Fourier transform"""
        # Take FFT
        fft_channel = fft2(channel)
        fft_shifted = fftshift(fft_channel)
        
        h, w = channel.shape
        center_y, center_x = h // 2, w // 2
        
        bands = []
        
        # Create frequency bands with increasing radii
        for i in range(num_bands):
            if i == 0:
                # Lowest frequencies
                radius_fraction = 0.1
                mask = self._create_frequency_mask(channel.shape, center_y, center_x, radius_fraction)
            elif i == num_bands - 1:
                # Highest frequencies (everything else)
                if num_bands > 2:
                    prev_radius = 0.1 + (i-1) * (0.5 - 0.1) / (num_bands - 2)
                else:
                    prev_radius = 0.1  # For num_bands = 2, just use the base radius
                mask = ~self._create_frequency_mask(channel.shape, center_y, center_x, prev_radius)
            else:
                # Intermediate frequencies - avoid division by zero
                if num_bands > 1:
                    radius_fraction = 0.1 + i * (0.5 - 0.1) / (num_bands - 1)
                    prev_radius = 0.1 + (i-1) * (0.5 - 0.1) / (num_bands - 1)
                else:
                    radius_fraction = 0.3  # Default for single band
                    prev_radius = 0.1
                
                outer_mask = self._create_frequency_mask(channel.shape, center_y, center_x, radius_fraction)
                inner_mask = self._create_frequency_mask(channel.shape, center_y, center_x, prev_radius)
                mask = outer_mask & ~inner_mask
            
            # Apply mask and inverse FFT
            filtered_fft = fft_shifted * mask
            filtered_channel = np.real(ifft2(ifftshift(filtered_fft)))
            bands.append(filtered_channel)
        
        return bands
    
    def _hybrid_decomposition(self, image: np.ndarray, num_bands: int, scale_factor: float = 1.0) -> List[np.ndarray]:
        """Hybrid wavelet-Fourier decomposition"""
        # Use wavelet for first few bands (good for texture)
        wavelet_bands = min(3, num_bands // 2)
        fourier_bands = num_bands - wavelet_bands
        
        # Get wavelet bands
        wav_bands = self._wavelet_decomposition(image, wavelet_bands, scale_factor)
        
        # Get Fourier bands from residual
        if fourier_bands > 0:
            # Use the highest frequency wavelet band as input for Fourier decomposition
            if len(image.shape) == 3:
                residual = wav_bands[-1]
            else:
                residual = wav_bands[-1]
            
            four_bands = self._fourier_decomposition(residual, fourier_bands, scale_factor)
            
            # Combine bands (replace last wavelet band with Fourier bands)
            combined_bands = wav_bands[:-1] + four_bands
        else:
            combined_bands = wav_bands
        
        return combined_bands[:num_bands]
    
    def _calculate_adaptive_thresholds(self, frequency_bands: List[np.ndarray], 
                                     analysis: Dict[str, Any]) -> List[float]:
        """Calculate adaptive thresholds for each frequency band"""
        thresholds = []
        
        # Base threshold depends on image characteristics
        base_threshold = 0.1
        
        # Adjust based on noise level (estimated from high frequency content)
        high_freq_energy = analysis.get('high_freq_energy', 0.2)
        if high_freq_energy > 0.3:
            base_threshold *= 1.5  # More aggressive for noisy images
        elif high_freq_energy < 0.1:
            base_threshold *= 0.7  # Gentler for clean images
        
        for i, band in enumerate(frequency_bands):
            # Calculate band-specific statistics
            band_std = np.std(band)
            band_mean = np.abs(np.mean(band))
            
            # Adaptive threshold based on band characteristics
            if i == 0:  # Lowest frequency band
                threshold = base_threshold * 0.5  # More conservative
            elif i == len(frequency_bands) - 1:  # Highest frequency band
                threshold = base_threshold * 2.0  # More aggressive
            else:  # Intermediate bands
                threshold = base_threshold * (1.0 + 0.3 * i / len(frequency_bands))
            
            # Adjust based on band statistics
            if band_std > 0.2:  # High variation band
                threshold *= 1.2
            elif band_std < 0.05:  # Low variation band
                threshold *= 0.8
            
            thresholds.append(float(np.clip(threshold, 0.01, 0.5)))
        
        return thresholds
    
    def _color_aware_frequency_processing(self, frequency_bands: List[np.ndarray], 
                                        thresholds: List[float], strength: float,
                                        color_space: str, scale_factor: float = 1.0) -> List[np.ndarray]:
        """Apply color-aware frequency processing with scale awareness"""
        enhanced_bands = []
        
        # FIXED: Scale-aware strength adjustment
        base_strength = strength * scale_factor * 0.5  # More conservative for larger images
        
        for i, band in enumerate(frequency_bands):
            if len(band.shape) == 3:  # Color image
                enhanced_band = np.zeros_like(band)
                
                # Different processing for different channels based on color space
                for c in range(band.shape[2]):
                    if color_space in ['lab', 'oklab', 'jzazbz']:
                        if c == 0:  # Luminance channel
                            channel_strength = base_strength  # Full strength for luminance
                        else:  # Color channels - DON'T PROCESS AT ALL
                            channel_strength = 0.0  # No processing for color channels
                    elif color_space == 'hsv':
                        if c == 2:  # Value channel
                            channel_strength = base_strength
                        else:  # Hue and saturation
                            channel_strength = base_strength * 0.2  # More conservative
                    else:  # RGB
                        channel_strength = base_strength  # Equal for all channels
                    
                    enhanced_band[:, :, c] = self._enhance_frequency_band(
                        band[:, :, c], thresholds[i], channel_strength
                    )
                
                enhanced_bands.append(enhanced_band)
            else:  # Grayscale
                enhanced_band = self._enhance_frequency_band(band, thresholds[i], strength)
                enhanced_bands.append(enhanced_band)
        
        return enhanced_bands
    
    def _standard_frequency_processing(self, frequency_bands: List[np.ndarray], 
                                     thresholds: List[float], strength: float, 
                                     scale_factor: float = 1.0, color_space: str = 'rgb') -> List[np.ndarray]:
        """Apply standard frequency processing with scale-aware enhancement"""
        enhanced_bands = []
        
        # FIXED: Scale-aware strength adjustment
        adjusted_strength = strength * scale_factor * 0.5  # More conservative for larger images
        
        for i, band in enumerate(frequency_bands):
            if len(band.shape) == 3:  # Color image
                enhanced_band = np.zeros_like(band)
                for c in range(band.shape[2]):
                    # Apply different strength based on color space even in standard processing
                    if color_space in ['lab', 'oklab', 'jzazbz']:
                        if c == 0:  # Luminance channel
                            channel_strength = adjusted_strength
                        else:  # Color channels - DON'T PROCESS AT ALL
                            channel_strength = 0.0  # No processing for color channels
                    elif color_space == 'hsv':
                        if c == 2:  # Value channel
                            channel_strength = adjusted_strength
                        else:  # Hue and saturation
                            channel_strength = adjusted_strength * 0.2
                    else:  # RGB
                        channel_strength = adjusted_strength
                    
                    enhanced_band[:, :, c] = self._enhance_frequency_band(
                        band[:, :, c], thresholds[i], channel_strength
                    )
                enhanced_bands.append(enhanced_band)
            else:  # Grayscale
                enhanced_band = self._enhance_frequency_band(band, thresholds[i], adjusted_strength)
                enhanced_bands.append(enhanced_band)
        
        return enhanced_bands
    
    def _enhance_frequency_band(self, band: np.ndarray, threshold: float, strength: float) -> np.ndarray:
        """Enhance individual frequency band"""
        # If strength is 0, return original band unchanged
        if strength == 0.0:
            return band.copy()
        
        # Apply threshold
        mask = np.abs(band) > threshold
        
        # Selective enhancement
        enhanced = band.copy()
        enhanced[mask] = enhanced[mask] * (1.0 + strength)
        
        # Soft clipping to prevent artifacts
        enhanced = np.tanh(enhanced / 0.5) * 0.5
        
        return enhanced
    
    def _selective_frequency_enhancement(self, enhanced_bands: List[np.ndarray], 
                                       analysis: Dict[str, Any], 
                                       original_image: np.ndarray) -> List[np.ndarray]:
        """Apply selective enhancement based on image content"""
        # Create enhancement map based on local image characteristics
        if len(original_image.shape) == 3:
            luminance = original_image[:, :, 0]  # Assume first channel is luminance
        else:
            luminance = original_image
        
        # Calculate local enhancement factors
        local_contrast = ndimage.generic_filter(luminance, np.std, size=5)
        local_edges = ndimage.generic_filter(luminance, lambda x: np.sum(np.abs(np.diff(x))), size=3)
        
        # Normalize factors
        contrast_norm = (local_contrast - local_contrast.min()) / (local_contrast.max() - local_contrast.min() + 1e-10)
        edges_norm = (local_edges - local_edges.min()) / (local_edges.max() - local_edges.min() + 1e-10)
        
        # Combine into enhancement map
        enhancement_map = 0.6 * contrast_norm + 0.4 * edges_norm
        
        # Apply selective enhancement
        selective_bands = []
        for i, band in enumerate(enhanced_bands):
            # Different weights for different frequency bands
            if i == 0:  # Low frequency
                weight = 0.3
            elif i == len(enhanced_bands) - 1:  # High frequency
                weight = 1.0
            else:  # Mid frequency
                weight = 0.5 + 0.5 * i / len(enhanced_bands)
            
            if len(band.shape) == 3:
                selective_band = np.zeros_like(band)
                for c in range(band.shape[2]):
                    # Blend original and enhanced based on enhancement map
                    original_band = enhanced_bands[i][:, :, c]
                    enhanced_part = band[:, :, c]
                    blend_factor = enhancement_map * weight
                    selective_band[:, :, c] = original_band + blend_factor * (enhanced_part - original_band)
            else:
                # Blend original and enhanced based on enhancement map
                original_band = enhanced_bands[i]
                enhanced_part = band
                blend_factor = enhancement_map * weight
                selective_band = original_band + blend_factor * (enhanced_part - original_band)
            
            selective_bands.append(selective_band)
        
        return selective_bands
    
    def _reconstruct_from_frequency_bands(self, enhanced_bands: List[np.ndarray], 
                                        method: str, original_shape: Tuple) -> np.ndarray:
        """Reconstruct image from enhanced frequency bands"""
        # Simple reconstruction: sum all bands
        if enhanced_bands:
            result = enhanced_bands[0].copy()
            for band in enhanced_bands[1:]:
                result = result + band
            
            # FIXED: Proper clamping instead of destructive normalization
            # Only clamp values outside [0, 1] range, preserving brightness relationships
            result = np.clip(result, 0, 1)
            
            return result
        else:
            # Fallback: return zeros
            return np.zeros(original_shape)
    
    def _calculate_processing_metrics(self, original: np.ndarray, enhanced: np.ndarray,
                                    original_bands: List[np.ndarray], enhanced_bands: List[np.ndarray],
                                    analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate processing metrics"""
        try:
            # Convert to grayscale for metrics if needed
            if len(original.shape) == 3:
                orig_gray = color.rgb2gray(original)
                enh_gray = color.rgb2gray(enhanced)
            else:
                orig_gray = original
                enh_gray = enhanced
            
            # Frequency content changes
            orig_fft = np.abs(fft2(orig_gray))
            enh_fft = np.abs(fft2(enh_gray))
            
            freq_energy_change = np.sum(enh_fft**2) / (np.sum(orig_fft**2) + 1e-10)
            
            # Contrast improvement
            orig_contrast = np.std(orig_gray)
            enh_contrast = np.std(enh_gray)
            contrast_improvement = enh_contrast / (orig_contrast + 1e-10)
            
            # Detail enhancement
            orig_laplacian = np.abs(ndimage.laplace(orig_gray))
            enh_laplacian = np.abs(ndimage.laplace(enh_gray))
            detail_enhancement = np.mean(enh_laplacian) / (np.mean(orig_laplacian) + 1e-10)
            
            return {
                'frequency_energy_change': float(freq_energy_change),
                'contrast_improvement': float(contrast_improvement),
                'detail_enhancement': float(detail_enhancement),
                'num_bands_processed': len(enhanced_bands),
                'original_contrast': float(orig_contrast),
                'enhanced_contrast': float(enh_contrast)
            }
            
        except Exception as e:
            return {'metrics_error': str(e)}
    
    def process_image(self, image: np.ndarray, method: str = 'adaptive', **kwargs) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Main processing method
        
        Args:
            image: Input image (H, W) or (H, W, C) in range [0, 1]
            method: Processing method (only 'adaptive' for now)
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        try:
            return self.adaptive_frequency_decomposition(image, **kwargs)
        except Exception as e:
            return None, {'error': f"Adaptive frequency decomposition processing failed: {str(e)}"}
    
    def get_method_info(self) -> Dict[str, str]:
        """Get information about available methods"""
        return {
            'adaptive_frequency_decomposition': 'Adaptive Frequency Decomposition with color-aware multi-scale processing'
        }
