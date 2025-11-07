"""
Advanced Film Grain Processing and Denoising
Specialized algorithms for authentic film grain, digital grain, and simulated grain overlays
"""

import numpy as np
import cv2
from typing import Optional, Union, Tuple, Dict, List
from scipy import ndimage
from scipy.signal import medfilt2d
from skimage.restoration import estimate_sigma
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from skimage.filters import rank
from skimage.morphology import disk
import pywt

# Import local modules
from .wavelet_denoise import gpu_wavelet_denoise, wavelet_denoise
from .nonlocal_means import nonlocal_means_denoise
from .gpu_utils import to_gpu, to_cpu, can_use_gpu, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage


class FilmGrainAnalyzer:
    """Advanced analysis of film grain characteristics"""
    
    def __init__(self):
        self.grain_features = {}
        self.recommendations = {}
    
    def analyze_grain_type(self, image: np.ndarray) -> Dict:
        """
        Comprehensive analysis of grain type and characteristics
        
        Args:
            image: Input image as numpy array [H, W, C] with values 0-255
            
        Returns:
            Dictionary with grain analysis results and recommendations
        """
        
        # Convert to grayscale for analysis
        if image.ndim == 3:
            image = np.ascontiguousarray(image)

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # FIXED: Ensure contiguous array after cv2.cvtColor
            gray = np.ascontiguousarray(gray)
        else:
            gray = image.copy()
        
        # Basic noise statistics
        noise_stats = self._estimate_noise_statistics(gray)
        
        # Grain structure analysis
        grain_structure = self._analyze_grain_structure(gray)
        
        # Frequency domain analysis
        freq_analysis = self._analyze_frequency_characteristics(gray)
        
        # Local texture analysis
        texture_analysis = self._analyze_local_texture(gray)
        
        # Determine grain type
        grain_type = self._classify_grain_type(
            noise_stats, grain_structure, freq_analysis, texture_analysis
        )
        
        # Generate recommendations
        recommendations = self._generate_denoising_recommendations(
            grain_type, noise_stats, grain_structure, freq_analysis
        )
        
        return {
            'grain_type': grain_type,
            'noise_statistics': noise_stats,
            'grain_structure': grain_structure,
            'frequency_analysis': freq_analysis,
            'texture_analysis': texture_analysis,
            'recommendations': recommendations
        }
    
    def _estimate_noise_statistics(self, image: np.ndarray) -> Dict:
        """Estimate comprehensive noise statistics"""
        
        # Standard deviation estimate
        sigma = estimate_sigma(image, channel_axis=None)
        
        # Noise variance in different regions
        h, w = image.shape
        regions = [
            image[:h//2, :w//2],      # Top-left
            image[:h//2, w//2:],      # Top-right
            image[h//2:, :w//2],      # Bottom-left
            image[h//2:, w//2:],      # Bottom-right
            image[h//4:3*h//4, w//4:3*w//4]  # Center
        ]
        
        regional_sigmas = [estimate_sigma(region, channel_axis=None) for region in regions]
        
        # Estimate signal-to-noise ratio
        signal_power = np.var(image)
        noise_power = sigma**2
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
        
        return {
            'global_sigma': sigma,
            'regional_sigmas': regional_sigmas,
            'sigma_variation': np.std(regional_sigmas),
            'snr_db': snr,
            'noise_power': noise_power,
            'signal_power': signal_power
        }
    
    def _analyze_grain_structure(self, image: np.ndarray) -> Dict:
        """Analyze grain structure and patterns"""
        
        # High-pass filter to isolate grain
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_pass = cv2.filter2D(image.astype(np.float32), -1, kernel)
        
        # Grain size estimation using morphological operations
        grain_sizes = []
        for radius in range(1, 6):
            elem = disk(radius)
            opened = rank.gradient(image, elem)
            grain_sizes.append(np.mean(opened))
        
        # Dominant grain size
        dominant_size = np.argmax(grain_sizes) + 1
        
        # Grain regularity (uniformity of grain distribution)
        # Use local binary pattern for texture analysis
        lbp = local_binary_pattern(image, 8, 1, method='uniform')
        grain_regularity = shannon_entropy(lbp)
        
        # Grain directionality analysis
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        
        # Measure directionality
        hist, _ = np.histogram(gradient_direction, bins=36, range=(-np.pi, np.pi))
        directionality = 1 - (np.max(hist) - np.min(hist)) / np.sum(hist)
        
        return {
            'dominant_grain_size': dominant_size,
            'grain_sizes': grain_sizes,
            'grain_regularity': grain_regularity,
            'directionality': directionality,
            'gradient_magnitude_mean': np.mean(gradient_magnitude),
            'gradient_magnitude_std': np.std(gradient_magnitude)
        }
    
    def _analyze_frequency_characteristics(self, image: np.ndarray) -> Dict:
        """Analyze frequency domain characteristics"""
        
        # FFT analysis
        f_transform = np.fft.fft2(image)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shifted)
        
        # Power spectral density
        psd = magnitude_spectrum**2
        
        # Radial frequency analysis
        h, w = image.shape
        center_y, center_x = h//2, w//2
        
        # Create radial frequency grid
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Radial average of power spectrum
        max_radius = int(np.sqrt(center_x**2 + center_y**2))
        radial_profile = np.zeros(max_radius)
        
        for radius in range(max_radius):
            mask = (r >= radius) & (r < radius + 1)
            radial_profile[radius] = np.mean(psd[mask])
        
        # Find dominant frequencies
        peak_indices = np.argsort(radial_profile)[-5:]  # Top 5 peaks
        dominant_freqs = peak_indices / max_radius  # Normalized frequencies
        
        # High frequency energy ratio
        high_freq_threshold = max_radius * 0.7
        high_freq_mask = r >= high_freq_threshold
        high_freq_energy = np.sum(psd[high_freq_mask])
        total_energy = np.sum(psd)
        high_freq_ratio = high_freq_energy / total_energy
        
        return {
            'radial_profile': radial_profile,
            'dominant_frequencies': dominant_freqs,
            'high_freq_ratio': high_freq_ratio,
            'spectral_centroid': np.sum(np.arange(max_radius) * radial_profile) / np.sum(radial_profile),
            'spectral_spread': np.sqrt(np.sum(((np.arange(max_radius) - 
                                              np.sum(np.arange(max_radius) * radial_profile) / np.sum(radial_profile))**2) * 
                                              radial_profile) / np.sum(radial_profile))
        }
    
    def _analyze_local_texture(self, image: np.ndarray) -> Dict:
        """Analyze local texture characteristics"""
        
        # Local standard deviation
        local_std = ndimage.generic_filter(image.astype(np.float32), np.std, size=5)
        
        # Local entropy
        local_entropy = rank.entropy(image, disk(3))
        
        # Texture energy and contrast
        gray_levels = 256
        glcm = np.zeros((gray_levels, gray_levels))
        
        # Simple GLCM computation for texture analysis
        for i in range(image.shape[0]-1):
            for j in range(image.shape[1]-1):
                glcm[image[i, j], image[i, j+1]] += 1
        
        # Normalize GLCM
        glcm = glcm / np.sum(glcm)
        
        # Texture features
        contrast = np.sum(glcm * np.power(np.arange(gray_levels)[:, None] - np.arange(gray_levels), 2))
        energy = np.sum(glcm**2)
        homogeneity = np.sum(glcm / (1 + np.abs(np.arange(gray_levels)[:, None] - np.arange(gray_levels))))
        
        return {
            'local_std_mean': np.mean(local_std),
            'local_std_variation': np.std(local_std),
            'local_entropy_mean': np.mean(local_entropy),
            'local_entropy_variation': np.std(local_entropy),
            'glcm_contrast': contrast,
            'glcm_energy': energy,
            'glcm_homogeneity': homogeneity
        }
    
    def _classify_grain_type(self, noise_stats: Dict, grain_structure: Dict, 
                           freq_analysis: Dict, texture_analysis: Dict) -> str:
        """Classify the type of grain based on analysis"""
        
        # Decision tree for grain classification
        high_freq_ratio = freq_analysis['high_freq_ratio']
        grain_regularity = grain_structure['grain_regularity']
        sigma_variation = noise_stats['sigma_variation']
        snr = noise_stats['snr_db']
        
        # Authentic film grain characteristics
        if (high_freq_ratio > 0.15 and 
            grain_regularity > 3.0 and 
            sigma_variation < 2.0 and 
            snr > 10):
            return 'authentic_film'
        
        # Digital camera grain characteristics
        elif (high_freq_ratio > 0.2 and 
              grain_regularity < 2.5 and 
              sigma_variation > 3.0 and 
              snr < 15):
            return 'digital_noise'
        
        # Simulated grain overlay characteristics
        elif (high_freq_ratio > 0.25 and 
              grain_regularity > 2.0 and 
              sigma_variation > 5.0):
            return 'simulated_overlay'
        
        # Low noise / clean image
        elif snr > 25:
            return 'clean'
        
        # Mixed or unknown grain
        else:
            return 'mixed_grain'
    
    def _generate_denoising_recommendations(self, grain_type: str, noise_stats: Dict, 
                                          grain_structure: Dict, freq_analysis: Dict) -> Dict:
        """Generate tailored denoising recommendations"""
        
        recommendations = {
            'primary_method': '',
            'secondary_method': '',
            'parameters': {},
            'preserve_grain': False,
            'multi_stage': False,
            'post_processing': []
        }
        
        if grain_type == 'authentic_film':
            recommendations.update({
                'primary_method': 'wavelet_denoise',
                'secondary_method': 'nonlocal_means',
                'parameters': {
                    'wavelet': 'db8',
                    'method': 'BayesShrink',
                    'mode': 'soft',
                    'levels': 4,
                    'sigma_scale': 0.8  # Preserve more grain
                },
                'preserve_grain': True,
                'multi_stage': True,
                'post_processing': ['grain_enhancement', 'selective_sharpening']
            })
        
        elif grain_type == 'digital_noise':
            recommendations.update({
                'primary_method': 'nonlocal_means',
                'secondary_method': 'wavelet_denoise',
                'parameters': {
                    'h': noise_stats['global_sigma'] * 0.8,
                    'patch_size': 7,
                    'patch_distance': 11,
                    'wavelet': 'db4',
                    'method': 'SureShrink',
                    'mode': 'soft'
                },
                'preserve_grain': False,
                'multi_stage': True,
                'post_processing': ['edge_enhancement']
            })
        
        elif grain_type == 'simulated_overlay':
            recommendations.update({
                'primary_method': 'frequency_filter',
                'secondary_method': 'wavelet_denoise',
                'parameters': {
                    'filter_type': 'gaussian_high_pass',
                    'cutoff_freq': 0.1,
                    'wavelet': 'bior2.2',
                    'method': 'VisuShrink',
                    'mode': 'soft',
                    'levels': 5
                },
                'preserve_grain': False,
                'multi_stage': True,
                'post_processing': ['texture_preservation']
            })
        
        elif grain_type == 'clean':
            recommendations.update({
                'primary_method': 'gentle_enhancement',
                'secondary_method': 'none',
                'parameters': {
                    'enhancement_factor': 1.1,
                    'preserve_details': True
                },
                'preserve_grain': True,
                'multi_stage': False,
                'post_processing': ['contrast_enhancement']
            })
        
        else:  # mixed_grain
            recommendations.update({
                'primary_method': 'adaptive_denoise',
                'secondary_method': 'wavelet_denoise',
                'parameters': {
                    'adaptive_strength': 0.7,
                    'wavelet': 'db8',
                    'method': 'BayesShrink',
                    'mode': 'soft',
                    'levels': 4
                },
                'preserve_grain': True,
                'multi_stage': True,
                'post_processing': ['adaptive_enhancement']
            })
        
        return recommendations


class FilmGrainProcessor:
    """Advanced film grain processing with specialized denoising strategies"""
    
    def __init__(self):
        self.analyzer = FilmGrainAnalyzer()
    
    def process_grain_aware(self, image: np.ndarray, 
                          auto_analyze: bool = True,
                          grain_type: Optional[str] = None,
                          preservation_level: float = 0.7,
                          use_gpu: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process image with grain-aware denoising
        
        Args:
            image: Input image as numpy array [H, W, C] with values 0-255
            auto_analyze: Whether to automatically analyze grain type
            grain_type: Manual grain type specification
            preservation_level: How much grain to preserve (0.0 = remove all, 1.0 = preserve all)
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            Tuple of (processed_image, analysis_results)
        """
        
        # Analyze grain if needed
        if auto_analyze or grain_type is None:
            analysis = self.analyzer.analyze_grain_type(image)
            detected_grain_type = analysis['grain_type']
            recommendations = analysis['recommendations']
        else:
            detected_grain_type = grain_type
            analysis = {'grain_type': grain_type, 'recommendations': {}}
            recommendations = self._get_default_recommendations(grain_type)
        
        # Apply grain-specific processing
        processed_image = self._apply_grain_specific_processing(
            image, detected_grain_type, recommendations, preservation_level, use_gpu
        )
        
        # Update analysis with processing results
        analysis['processed_grain_type'] = detected_grain_type
        analysis['preservation_level'] = preservation_level
        analysis['gpu_used'] = use_gpu
        
        return processed_image, analysis
    
    def _apply_grain_specific_processing(self, image: np.ndarray, grain_type: str, 
                                       recommendations: Dict, preservation_level: float,
                                       use_gpu: bool) -> np.ndarray:
        """Apply grain-specific processing pipeline"""
        
        # Adjust parameters based on preservation level
        adjusted_params = self._adjust_parameters_for_preservation(
            recommendations.get('parameters', {}), preservation_level
        )
        
        if grain_type == 'authentic_film':
            return self._process_authentic_film_grain(image, adjusted_params, use_gpu)
        
        elif grain_type == 'digital_noise':
            return self._process_digital_noise(image, adjusted_params, use_gpu)
        
        elif grain_type == 'simulated_overlay':
            return self._process_simulated_overlay(image, adjusted_params, use_gpu)
        
        elif grain_type == 'clean':
            return self._process_clean_image(image, adjusted_params, use_gpu)
        
        else:  # mixed_grain
            return self._process_mixed_grain(image, adjusted_params, use_gpu)
    
    def _adjust_parameters_for_preservation(self, params: Dict, preservation_level: float) -> Dict:
        """Adjust denoising parameters based on preservation level"""
        
        adjusted = params.copy()
        
        # Adjust sigma scaling for wavelet methods
        if 'sigma_scale' in adjusted:
            adjusted['sigma_scale'] = adjusted['sigma_scale'] * (0.5 + preservation_level * 0.5)
        
        # Adjust NLM parameters
        if 'h' in adjusted:
            adjusted['h'] = adjusted['h'] * (1.0 - preservation_level * 0.3)
        
        # Adjust wavelet levels
        if 'levels' in adjusted:
            adjusted['levels'] = max(2, int(adjusted['levels'] * (0.7 + preservation_level * 0.3)))
        
        return adjusted
    
    def _process_authentic_film_grain(self, image: np.ndarray, params: Dict, use_gpu: bool) -> np.ndarray:
        """Process authentic film grain with preservation"""
        
        # Stage 1: Gentle wavelet denoising
        if use_gpu:
            denoised = gpu_wavelet_denoise(
                image,
                wavelet=params.get('wavelet', 'db8'),
                method=params.get('method', 'BayesShrink'),
                mode=params.get('mode', 'soft'),
                levels=params.get('levels', 4),
                sigma=None,
                rescale_sigma=True
            )
        else:
            denoised = wavelet_denoise(
                image,
                wavelet=params.get('wavelet', 'db8'),
                method=params.get('method', 'BayesShrink'),
                mode=params.get('mode', 'soft'),
                levels=params.get('levels', 4),
                sigma=None,
                rescale_sigma=True
            )
        
        # Stage 2: Selective grain enhancement
        grain_enhanced = self._enhance_grain_selectively(denoised, image, 
                                                       params.get('sigma_scale', 0.8))
        
        # Stage 3: Gentle sharpening
        sharpened = self._apply_gentle_sharpening(grain_enhanced)
        
        return sharpened
    
    def _process_digital_noise(self, image: np.ndarray, params: Dict, use_gpu: bool) -> np.ndarray:
        """Process digital noise with edge preservation"""
        
        # Stage 1: Non-local means denoising
        denoised = nonlocal_means_denoise(
            image,
            h=params.get('h', 10),
            patch_size=params.get('patch_size', 7),
            patch_distance=params.get('patch_distance', 11),
            multichannel=True,
            fast_mode=True
        )
        
        # Stage 2: Wavelet denoising for residual noise
        if use_gpu:
            final_denoised = gpu_wavelet_denoise(
                denoised,
                wavelet=params.get('wavelet', 'db4'),
                method=params.get('method', 'SureShrink'),
                mode=params.get('mode', 'soft'),
                levels=params.get('levels', 3),
                sigma=None,
                rescale_sigma=True
            )
        else:
            final_denoised = wavelet_denoise(
                denoised,
                wavelet=params.get('wavelet', 'db4'),
                method=params.get('method', 'SureShrink'),
                mode=params.get('mode', 'soft'),
                levels=params.get('levels', 3),
                sigma=None,
                rescale_sigma=True
            )
        
        # Stage 3: Edge enhancement
        edge_enhanced = self._enhance_edges(final_denoised)
        
        return edge_enhanced
    
    def _process_simulated_overlay(self, image: np.ndarray, params: Dict, use_gpu: bool) -> np.ndarray:
        """Process simulated grain overlay by removal"""
        
        # Stage 1: Frequency domain filtering
        filtered = self._apply_frequency_filter(image, params.get('cutoff_freq', 0.1))
        
        # Stage 2: Wavelet denoising
        if use_gpu:
            denoised = gpu_wavelet_denoise(
                filtered,
                wavelet=params.get('wavelet', 'bior2.2'),
                method=params.get('method', 'VisuShrink'),
                mode=params.get('mode', 'soft'),
                levels=params.get('levels', 5),
                sigma=None,
                rescale_sigma=True
            )
        else:
            denoised = wavelet_denoise(
                filtered,
                wavelet=params.get('wavelet', 'bior2.2'),
                method=params.get('method', 'VisuShrink'),
                mode=params.get('mode', 'soft'),
                levels=params.get('levels', 5),
                sigma=None,
                rescale_sigma=True
            )
        
        # Stage 3: Texture preservation
        texture_preserved = self._preserve_texture(denoised, image)
        
        return texture_preserved
    
    def _process_clean_image(self, image: np.ndarray, params: Dict, use_gpu: bool) -> np.ndarray:
        """Process clean image with gentle enhancement"""
        
        # Gentle enhancement without aggressive denoising
        enhanced = self._apply_gentle_enhancement(image, params.get('enhancement_factor', 1.1))
        
        # Contrast enhancement
        contrast_enhanced = self._enhance_contrast(enhanced)
        
        return contrast_enhanced
    
    def _process_mixed_grain(self, image: np.ndarray, params: Dict, use_gpu: bool) -> np.ndarray:
        """Process mixed grain with adaptive approach"""
        
        # Adaptive denoising based on local characteristics
        adaptive_denoised = self._apply_adaptive_denoising(image, params.get('adaptive_strength', 0.7))
        
        # Follow up with wavelet denoising
        if use_gpu:
            final_denoised = gpu_wavelet_denoise(
                adaptive_denoised,
                wavelet=params.get('wavelet', 'db8'),
                method=params.get('method', 'BayesShrink'),
                mode=params.get('mode', 'soft'),
                levels=params.get('levels', 4),
                sigma=None,
                rescale_sigma=True
            )
        else:
            final_denoised = wavelet_denoise(
                adaptive_denoised,
                wavelet=params.get('wavelet', 'db8'),
                method=params.get('method', 'BayesShrink'),
                mode=params.get('mode', 'soft'),
                levels=params.get('levels', 4),
                sigma=None,
                rescale_sigma=True
            )
        
        return final_denoised
    
    def _enhance_grain_selectively(self, denoised: np.ndarray, original: np.ndarray, 
                                 scale: float) -> np.ndarray:
        """Selectively enhance grain based on original image"""
        
        # Extract grain component
        grain = original.astype(np.float32) - denoised.astype(np.float32)
        
        # Apply selective enhancement
        enhanced_grain = grain * scale
        # FIXED: Ensure contiguous after arithmetic
        enhanced_grain = np.ascontiguousarray(enhanced_grain)
        
        # Combine with denoised image
        result = denoised.astype(np.float32) + enhanced_grain
        # FIXED: Ensure contiguous after addition
        result = np.ascontiguousarray(result)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_gentle_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply gentle unsharp masking"""
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges while preserving smooth regions"""
        
        # Convert to grayscale for edge detection
        if image.ndim == 3:
            image = np.ascontiguousarray(image)

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # FIXED: Ensure contiguous array after cv2.cvtColor
            gray = np.ascontiguousarray(gray)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        
        # Create edge mask
        edge_mask = edges.astype(np.float32) / 255.0
        
        # Apply edge enhancement
        if image.ndim == 3:
            edge_mask = np.stack([edge_mask] * 3, axis=2)
        
        # Sharpen edges
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image = np.ascontiguousarray(image)

        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend based on edge mask
        result = image.astype(np.float32) * (1 - edge_mask) + sharpened.astype(np.float32) * edge_mask
        # FIXED: Ensure contiguous after arithmetic blend
        result = np.ascontiguousarray(result)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_frequency_filter(self, image: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Apply frequency domain filtering"""
        
        if image.ndim == 3:
            result_channels = []
            for c in range(image.shape[2]):
                channel = image[:, :, c].astype(np.float32) / 255.0
                
                # FFT
                f_transform = np.fft.fft2(channel)
                f_shifted = np.fft.fftshift(f_transform)
                
                # Create high-pass filter
                rows, cols = channel.shape
                crow, ccol = rows // 2, cols // 2
                
                # Create mask
                mask = np.ones((rows, cols), dtype=np.float32)
                r = int(cutoff_freq * min(rows, cols))
                y, x = np.ogrid[:rows, :cols]
                center_mask = (x - ccol) ** 2 + (y - crow) ** 2 <= r ** 2
                mask[center_mask] = 0
                
                # Apply filter
                filtered = f_shifted * mask
                
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
            
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2
            
            mask = np.ones((rows, cols), dtype=np.float32)
            r = int(cutoff_freq * min(rows, cols))
            y, x = np.ogrid[:rows, :cols]
            center_mask = (x - ccol) ** 2 + (y - crow) ** 2 <= r ** 2
            mask[center_mask] = 0
            
            filtered = f_shifted * mask
            f_ishifted = np.fft.ifftshift(filtered)
            result = np.fft.ifft2(f_ishifted)
            result = np.real(result)
        
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    def _preserve_texture(self, denoised: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Preserve texture details from original image"""
        
        # Calculate texture component
        texture = original.astype(np.float32) - cv2.GaussianBlur(original, (5, 5), 1.0).astype(np.float32)
        
        # Selectively add texture back
        texture_strength = 0.3
        result = denoised.astype(np.float32) + texture * texture_strength
        # FIXED: Ensure contiguous after arithmetic
        result = np.ascontiguousarray(result)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_gentle_enhancement(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Apply gentle enhancement"""
        
        # Gentle contrast adjustment
        enhanced = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        
        return enhanced
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE"""
        
        if image.ndim == 3:
            # Convert to LAB color space
            image = np.ascontiguousarray(image)

            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # FIXED: Ensure contiguous array before CLAHE operation
            l_channel = np.ascontiguousarray(lab[:, :, 0])
            lab[:, :, 0] = clahe.apply(l_channel)
            # FIXED: Ensure contiguous array after slice assignment before color conversion
            lab = np.ascontiguousarray(lab)
            
            # Convert back to RGB
            lab = np.ascontiguousarray(lab)

            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Apply CLAHE directly
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = np.ascontiguousarray(image)

            result = clahe.apply(image)
        
        return result
    
    def _apply_adaptive_denoising(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply adaptive denoising based on local characteristics"""
        
        # Calculate local standard deviation
        if image.ndim == 3:
            image = np.ascontiguousarray(image)

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # FIXED: Ensure contiguous array after cv2.cvtColor
            gray = np.ascontiguousarray(gray)
        else:
            gray = image
        
        # Local variance map
        local_var = cv2.Laplacian(gray, cv2.CV_64F)
        local_var = np.abs(local_var)
        
        # Normalize variance map
        local_var = (local_var - local_var.min()) / (local_var.max() - local_var.min())
        
        # Create adaptive strength map
        adaptive_strength = strength * (1 - local_var)
        
        # Apply bilateral filter with adaptive strength
        if image.ndim == 3:
            result_channels = []
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                # Apply bilateral filter
                channel = np.ascontiguousarray(channel)

                filtered = cv2.bilateralFilter(channel, 9, 75, 75)
                
                # Blend based on adaptive strength
                strength_map = adaptive_strength
                blended = channel.astype(np.float32) * (1 - strength_map) + filtered.astype(np.float32) * strength_map
                # FIXED: Ensure contiguous after arithmetic blend
                blended = np.ascontiguousarray(blended)
                result_channels.append(blended)
            
            result = np.stack(result_channels, axis=2)
        else:
            image = np.ascontiguousarray(image)

            filtered = cv2.bilateralFilter(image, 9, 75, 75)
            result = image.astype(np.float32) * (1 - adaptive_strength) + filtered.astype(np.float32) * adaptive_strength
            # FIXED: Ensure contiguous after arithmetic blend
            result = np.ascontiguousarray(result)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _get_default_recommendations(self, grain_type: str) -> Dict:
        """Get default recommendations for a given grain type"""
        
        defaults = {
            'authentic_film': {
                'parameters': {
                    'wavelet': 'db8',
                    'method': 'BayesShrink',
                    'mode': 'soft',
                    'levels': 4,
                    'sigma_scale': 0.8
                }
            },
            'digital_noise': {
                'parameters': {
                    'h': 10,
                    'patch_size': 7,
                    'patch_distance': 11,
                    'wavelet': 'db4',
                    'method': 'SureShrink',
                    'mode': 'soft',
                    'levels': 3
                }
            },
            'simulated_overlay': {
                'parameters': {
                    'cutoff_freq': 0.1,
                    'wavelet': 'bior2.2',
                    'method': 'VisuShrink',
                    'mode': 'soft',
                    'levels': 5
                }
            },
            'clean': {
                'parameters': {
                    'enhancement_factor': 1.1
                }
            },
            'mixed_grain': {
                'parameters': {
                    'adaptive_strength': 0.7,
                    'wavelet': 'db8',
                    'method': 'BayesShrink',
                    'mode': 'soft',
                    'levels': 4
                }
            }
        }
        
        return defaults.get(grain_type, defaults['mixed_grain'])


# Convenience functions
def analyze_film_grain(image: np.ndarray) -> Dict:
    """Analyze film grain characteristics"""
    analyzer = FilmGrainAnalyzer()
    return analyzer.analyze_grain_type(image)


def process_film_grain(image: np.ndarray, 
                      grain_type: Optional[str] = None,
                      preservation_level: float = 0.7,
                      use_gpu: bool = True) -> Tuple[np.ndarray, Dict]:
    """Process film grain with automatic analysis and recommendations"""
    processor = FilmGrainProcessor()
    return processor.process_grain_aware(image, grain_type=grain_type, 
                                       preservation_level=preservation_level, 
                                       use_gpu=use_gpu)
