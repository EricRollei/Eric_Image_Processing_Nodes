"""
Advanced Film Grain Processing Node for ComfyUI
GPU-accelerated multi-stage grain analysis and intelligent denoising

Author: Eric Hiss (GitHub: EricRollei)
License: See LICENSE file in repository root

Film Grain Processing Approach:
    Custom implementation combining multiple techniques:
    - Wavelet-based grain detection (PyWavelets, MIT License)
    - Frequency domain analysis (SciPy, BSD 3-Clause)
    - GPU-accelerated processing (CuPy, MIT License - optional)
    - Custom neural network for grain classification (optional, trained by author)
    
    Integrates concepts from:
    - Film grain synthesis literature
    - Noise estimation techniques
    - Adaptive filtering methods

Dependencies:
    - PyTorch (BSD 3-Clause License)
    - NumPy (BSD 3-Clause License)
    - PyWavelets (MIT License)
    - SciPy (BSD 3-Clause License)
    - CuPy (MIT License) - optional for GPU acceleration
    - scikit-image (BSD 3-Clause License)
"""
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from ..base_node import BaseImageProcessingNode
from Eric_Image_Processing_Nodes import (
    FilmGrainProcessor,
    FilmGrainAnalyzer,
    gpu_wavelet_denoise,
    gpu_wavelet_denoise_stationary,
    gpu_memory_info,
    cleanup_gpu_memory,
    wavelet_denoise
)


class AdvancedFilmGrainNode(BaseImageProcessingNode):
    """Advanced Film Grain Processing with GPU Acceleration"""
    
    def __init__(self):
        super().__init__()
        self.processor = FilmGrainProcessor()
        self.analyzer = FilmGrainAnalyzer()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "processing_mode": (["auto_analyze", "manual_specify"], {
                    "default": "auto_analyze",
                    "tooltip": "Processing mode:\n• auto_analyze: Automatically detect grain type and optimize settings\n• manual_specify: Use manually specified grain type and settings"
                }),
                "grain_type": (["authentic_film", "digital_noise", "simulated_overlay", "clean", "mixed_grain"], {
                    "default": "authentic_film",
                    "tooltip": "Grain type for manual mode:\n• authentic_film: Real film grain (analog)\n• digital_noise: Digital sensor noise\n• simulated_overlay: Artificial grain overlay\n• clean: High-quality image, minimal noise\n• mixed_grain: Combination of grain types"
                }),
                "denoising_method": (["gpu_wavelet", "gpu_stationary_wavelet", "cpu_wavelet", "adaptive"], {
                    "default": "gpu_wavelet",
                    "tooltip": "Denoising algorithm:\n• gpu_wavelet: Fast GPU-accelerated wavelet denoising\n• gpu_stationary_wavelet: GPU stationary wavelet (better quality, slower)\n• cpu_wavelet: CPU-only wavelet denoising\n• adaptive: Grain-aware adaptive processing"
                }),
                "preservation_level": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Grain preservation level:\n• 0.0: Remove all grain/noise\n• 0.3: Light preservation\n• 0.7: Moderate preservation (recommended)\n• 1.0: Maximum grain preservation"
                }),
                "wavelet_type": (["db8", "db4", "bior2.2", "coif2", "haar"], {
                    "default": "db8",
                    "tooltip": "Wavelet type:\n• db8: Daubechies 8 - Best all-around choice\n• db4: Daubechies 4 - Faster, good for natural images\n• bior2.2: Biorthogonal - Good for images with edges\n• coif2: Coiflets - Good reconstruction\n• haar: Haar - Fastest, simple images"
                }),
                "thresholding_method": (["BayesShrink", "SureShrink", "VisuShrink", "manual"], {
                    "default": "BayesShrink",
                    "tooltip": "Threshold selection:\n• BayesShrink: Adaptive, best for natural images\n• SureShrink: Hybrid SURE-based approach\n• VisuShrink: Conservative, preserves details\n• manual: Use custom sigma value"
                }),
                "thresholding_mode": (["soft", "hard"], {
                    "default": "soft",
                    "tooltip": "Thresholding mode:\n• soft: Gradual transition, smoother (recommended)\n• hard: Sharp cutoff, preserves edges but may create artifacts"
                }),
                "sigma_adjustment": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Noise level adjustment (more sensitive range):\n• 0.1-0.5: Very conservative denoising\n• 0.8-1.2: Normal denoising\n• 1.5-3.0: Aggressive denoising\n• 3.0-5.0: Very aggressive denoising\n• Lower values preserve more detail"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable GPU acceleration (requires CUDA-compatible GPU)"
                }),
                "multi_stage_processing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable multi-stage processing:\n• Edge preservation\n• Grain enhancement\n• Contrast adjustment"
                }),
                "edge_preservation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve edge details from original image during processing"
                }),
                "grain_enhancement": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enhance grain structure while denoising (only with high preservation levels)"
                }),
            },
            "optional": {
                "custom_sigma": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Custom noise level (0 = auto-estimate):\n• 1-10: Light noise\n• 10-25: Moderate noise\n• 25-50: Heavy noise\n• 50+: Extreme noise"
                }),
                "decomposition_levels": ("INT", {
                    "default": 0, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Wavelet decomposition levels (0 = auto):\n• 3-4: Preserve fine details\n• 5-6: Standard denoising\n• 7-8: Heavy denoising (may blur details)"
                }),
                "debug_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print debug information to console (helpful for troubleshooting parameter effects)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("processed_image", "grain_analysis", "processing_info")
    FUNCTION = "process_film_grain"
    CATEGORY = "Eric's Nodes/Advanced Processing"
    
    def process_film_grain(self, image: torch.Tensor, processing_mode: str, grain_type: str,
                          denoising_method: str, preservation_level: float, wavelet_type: str,
                          thresholding_method: str, thresholding_mode: str, sigma_adjustment: float,
                          use_gpu: bool, multi_stage_processing: bool, edge_preservation: bool,
                          grain_enhancement: bool, custom_sigma: float = 0.0, 
                          decomposition_levels: int = 0, debug_output: bool = False) -> Tuple[torch.Tensor, str, str]:
        """
        Process film grain with advanced GPU-accelerated denoising
        """
        
        try:
            # Convert tensor to numpy
            np_image = self.tensor_to_numpy(image)
            
            # Get GPU memory info
            gpu_info = gpu_memory_info()
            
            # Determine processing parameters
            if processing_mode == "auto_analyze":
                # Perform automatic grain analysis
                analysis_result = self.analyzer.analyze_grain_type(np_image)
                detected_grain_type = analysis_result['grain_type']
                recommendations = analysis_result['recommendations']
                
                # Use detected grain type and recommendations
                final_grain_type = detected_grain_type
                processing_params = recommendations.get('parameters', {})
                
                # Override with user preferences
                processing_params.update({
                    'wavelet': wavelet_type,
                    'method': thresholding_method,
                    'mode': thresholding_mode,
                    'sigma_scale': sigma_adjustment,
                })
                
                analysis_info = self._format_analysis_info(analysis_result)
                
            else:
                # Manual specification
                final_grain_type = grain_type
                processing_params = {
                    'wavelet': wavelet_type,
                    'method': thresholding_method,
                    'mode': thresholding_mode,
                    'sigma_scale': sigma_adjustment,
                }
                
                # Apply sigma_adjustment to automatic sigma estimation
                if custom_sigma == 0.0:
                    # Let the denoising method auto-estimate sigma, but scale it
                    processing_params['sigma'] = None
                    processing_params['sigma_scale'] = sigma_adjustment
                else:
                    # Use custom sigma directly, but still apply adjustment
                    processing_params['sigma'] = custom_sigma * sigma_adjustment
                
                analysis_info = f"Manual grain type: {grain_type}\nSigma adjustment: {sigma_adjustment}x"
            
            # Override levels if specified
            if decomposition_levels > 0:
                processing_params['levels'] = decomposition_levels
            
            # Make sure sigma_scale is applied properly
            if 'sigma_scale' in processing_params:
                if debug_output:
                    print(f"Applying sigma scale: {processing_params['sigma_scale']}")
                
                # If we have a custom sigma, apply the scale to it
                if 'sigma' in processing_params and processing_params['sigma'] is not None:
                    original_sigma = processing_params['sigma']
                    processing_params['sigma'] = original_sigma * processing_params['sigma_scale']
                    if debug_output:
                        print(f"Scaled sigma from {original_sigma} to {processing_params['sigma']}")
                else:
                    # For auto-estimation, the scale will be applied in the denoising function
                    if debug_output:
                        print(f"Will apply sigma scale {processing_params['sigma_scale']} to auto-estimated sigma")
            
            # Apply main denoising
            if denoising_method == "gpu_wavelet":
                processed_image = self._apply_gpu_wavelet_denoising(
                    np_image, processing_params, use_gpu, debug_output
                )
            elif denoising_method == "gpu_stationary_wavelet":
                processed_image = self._apply_gpu_stationary_wavelet_denoising(
                    np_image, processing_params, use_gpu, debug_output
                )
            elif denoising_method == "cpu_wavelet":
                processed_image = self._apply_cpu_wavelet_denoising(
                    np_image, processing_params, debug_output
                )
            elif denoising_method == "adaptive":
                processed_image = self._apply_adaptive_processing(
                    np_image, final_grain_type, processing_params, preservation_level, use_gpu, debug_output
                )
            
            # Apply post-processing stages
            if multi_stage_processing:
                processed_image = self._apply_multi_stage_processing(
                    processed_image, np_image, final_grain_type, processing_params,
                    edge_preservation, grain_enhancement, preservation_level, debug_output
                )
            
            # Generate processing info
            processing_info = self._generate_processing_info(
                denoising_method, processing_params, final_grain_type,
                preservation_level, use_gpu, gpu_info
            )
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(processed_image)
            
            # Cleanup GPU memory
            if use_gpu:
                cleanup_gpu_memory()
            
            return (result_tensor, analysis_info, processing_info)
            
        except Exception as e:
            error_msg = f"Error in film grain processing: {str(e)}"
            print(error_msg)
            # Return original image on error
            return (image, error_msg, "Processing failed")
    
    def _apply_gpu_wavelet_denoising(self, image: np.ndarray, params: Dict, use_gpu: bool, debug: bool = False) -> np.ndarray:
        """Apply GPU-accelerated wavelet denoising"""
        
        # Debug: Print parameters being used
        if debug:
            print(f"GPU Wavelet Denoising Parameters: {params}")
        
        return gpu_wavelet_denoise(
            image,
            wavelet=params.get('wavelet', 'db8'),
            sigma=params.get('sigma', None),
            mode=params.get('mode', 'soft'),
            method=params.get('method', 'BayesShrink'),
            levels=params.get('levels', None),
            multichannel=True,
            rescale_sigma=params.get('sigma_scale', 1.0) != 1.0,
            force_cpu=not use_gpu
        )
    
    def _apply_gpu_stationary_wavelet_denoising(self, image: np.ndarray, params: Dict, use_gpu: bool, debug: bool = False) -> np.ndarray:
        """Apply GPU-accelerated stationary wavelet denoising"""
        
        # Debug: Print parameters being used
        if debug:
            print(f"GPU Stationary Wavelet Denoising Parameters: {params}")
        
        return gpu_wavelet_denoise_stationary(
            image,
            wavelet=params.get('wavelet', 'db8'),
            sigma=params.get('sigma', None),
            mode=params.get('mode', 'soft'),
            method=params.get('method', 'BayesShrink'),
            levels=params.get('levels', None),
            multichannel=True,
            rescale_sigma=params.get('sigma_scale', 1.0) != 1.0,
            force_cpu=not use_gpu
        )
    
    def _apply_cpu_wavelet_denoising(self, image: np.ndarray, params: Dict, debug: bool = False) -> np.ndarray:
        """Apply CPU-only wavelet denoising"""
        
        # Debug: Print parameters being used
        if debug:
            print(f"CPU Wavelet Denoising Parameters: {params}")
        
        return wavelet_denoise(
            image,
            wavelet=params.get('wavelet', 'db8'),
            sigma=params.get('sigma', None),
            mode=params.get('mode', 'soft'),
            method=params.get('method', 'BayesShrink'),
            levels=params.get('levels', None),
            multichannel=True,
            rescale_sigma=params.get('sigma_scale', 1.0) != 1.0
        )
    
    def _apply_adaptive_processing(self, image: np.ndarray, grain_type: str, params: Dict, 
                                 preservation_level: float, use_gpu: bool, debug: bool = False) -> np.ndarray:
        """Apply adaptive grain-aware processing"""
        
        # Debug: Print parameters being used
        if debug:
            print(f"Adaptive Processing Parameters: {params}")
            print(f"Grain Type: {grain_type}, Preservation Level: {preservation_level}")
        
        processed_image, _ = self.processor.process_grain_aware(
            image,
            auto_analyze=False,
            grain_type=grain_type,
            preservation_level=preservation_level,
            use_gpu=use_gpu
        )
        return processed_image
    
    def _apply_multi_stage_processing(self, processed_image: np.ndarray, original_image: np.ndarray,
                                    grain_type: str, params: Dict, edge_preservation: bool,
                                    grain_enhancement: bool, preservation_level: float, debug: bool = False) -> np.ndarray:
        """Apply multi-stage post-processing"""
        
        if debug:
            print(f"Multi-stage processing - Grain type: {grain_type}, Preservation level: {preservation_level}")
        
        result = processed_image.copy()
        
        # Stage 1: Edge preservation (make more responsive to settings)
        if edge_preservation:
            edge_strength = 0.2 + (preservation_level * 0.3)  # 0.2 to 0.5 range
            if debug:
                print(f"Applying edge preservation with strength: {edge_strength}")
            result = self._preserve_edges(result, original_image, edge_strength)
        
        # Stage 2: Grain enhancement (if preserving grain)
        if grain_enhancement and preservation_level > 0.2:  # Lower threshold
            enhancement_factor = preservation_level * 0.7  # More responsive
            if debug:
                print(f"Applying grain enhancement with factor: {enhancement_factor}")
            result = self._enhance_grain_structure(result, original_image, enhancement_factor)
        
        # Stage 3: Final contrast adjustment (make more responsive)
        contrast_strength = 0.5 + (preservation_level * 0.5)  # 0.5 to 1.0 range
        if debug:
            print(f"Applying contrast adjustment with strength: {contrast_strength}")
        
        if grain_type in ['authentic_film', 'clean']:
            result = self._adjust_contrast_gentle(result, contrast_strength)
        elif grain_type in ['digital_noise', 'simulated_overlay']:
            result = self._adjust_contrast_aggressive(result, contrast_strength)
        
        return result
    
    def _preserve_edges(self, processed: np.ndarray, original: np.ndarray, edge_strength: float = 0.3) -> np.ndarray:
        """Preserve edge information from original image"""
        import cv2
        
        # Convert to grayscale for edge detection
        if original.ndim == 3:
            gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            gray_proc = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            gray_orig = original
            gray_proc = processed
        
        # Detect edges in original with adaptive thresholds
        lower_thresh = int(50 * edge_strength)
        upper_thresh = int(150 * edge_strength)
        edges = cv2.Canny(gray_orig, lower_thresh, upper_thresh)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        edge_mask = edges.astype(np.float32) / 255.0
        
        # Blend edges from original
        if processed.ndim == 3:
            edge_mask = np.stack([edge_mask] * 3, axis=2)
        
        # Use edge_strength as blend factor
        result = processed.astype(np.float32) * (1 - edge_mask * edge_strength) + \
                original.astype(np.float32) * (edge_mask * edge_strength)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _enhance_grain_structure(self, processed: np.ndarray, original: np.ndarray, 
                               enhancement_factor: float) -> np.ndarray:
        """Enhance grain structure while maintaining denoising"""
        
        # Extract grain component
        grain = original.astype(np.float32) - processed.astype(np.float32)
        
        # Apply selective enhancement based on enhancement factor
        enhanced_grain = grain * enhancement_factor
        
        # Combine with processed image
        result = processed.astype(np.float32) + enhanced_grain
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _adjust_contrast_gentle(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply gentle contrast adjustment"""
        import cv2
        
        # Convert to LAB color space for better contrast control
        if image.ndim == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply gentle CLAHE to L channel with variable strength
            clip_limit = 1.0 + (strength * 1.0)  # 1.0 to 2.0 range
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Apply gentle CLAHE directly
            clip_limit = 1.0 + (strength * 1.0)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            result = clahe.apply(image)
        
        return result
    
    def _adjust_contrast_aggressive(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply more aggressive contrast adjustment"""
        import cv2
        
        if image.ndim == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply stronger CLAHE to L channel with variable strength
            clip_limit = 2.0 + (strength * 2.0)  # 2.0 to 4.0 range
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Apply stronger CLAHE directly
            clip_limit = 2.0 + (strength * 2.0)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            result = clahe.apply(image)
        
        return result
    
    def _format_analysis_info(self, analysis: Dict) -> str:
        """Format analysis information for display"""
        
        info_lines = []
        info_lines.append(f"Detected Grain Type: {analysis['grain_type']}")
        
        if 'noise_statistics' in analysis:
            noise_stats = analysis['noise_statistics']
            info_lines.append(f"Noise Sigma: {noise_stats['global_sigma']:.2f}")
            info_lines.append(f"SNR: {noise_stats['snr_db']:.1f} dB")
        
        if 'grain_structure' in analysis:
            grain_struct = analysis['grain_structure']
            info_lines.append(f"Grain Size: {grain_struct['dominant_grain_size']}")
            info_lines.append(f"Grain Regularity: {grain_struct['grain_regularity']:.2f}")
        
        if 'frequency_analysis' in analysis:
            freq_analysis = analysis['frequency_analysis']
            info_lines.append(f"High Freq Ratio: {freq_analysis['high_freq_ratio']:.3f}")
        
        if 'recommendations' in analysis:
            recommendations = analysis['recommendations']
            info_lines.append(f"Recommended Method: {recommendations.get('primary_method', 'N/A')}")
            info_lines.append(f"Preserve Grain: {recommendations.get('preserve_grain', False)}")
        
        return "\\n".join(info_lines)
    
    def _generate_processing_info(self, method: str, params: Dict, grain_type: str,
                                preservation_level: float, use_gpu: bool, gpu_info: Dict) -> str:
        """Generate processing information"""
        
        info_lines = []
        info_lines.append(f"Processing Method: {method}")
        info_lines.append(f"Grain Type: {grain_type}")
        info_lines.append(f"Preservation Level: {preservation_level:.1f}")
        info_lines.append(f"Wavelet: {params.get('wavelet', 'N/A')}")
        info_lines.append(f"Thresholding: {params.get('method', 'N/A')} ({params.get('mode', 'N/A')})")
        info_lines.append(f"Levels: {params.get('levels', 'auto')}")
        
        if use_gpu and gpu_info.get('available', False):
            info_lines.append(f"GPU Acceleration: Enabled")
            info_lines.append(f"GPU Memory Used: {gpu_info.get('used_bytes', 0) / 1024**2:.1f} MB")
        else:
            info_lines.append(f"GPU Acceleration: Disabled/Unavailable")
        
        return "\\n".join(info_lines)


class GPUWaveletDenoiseNode(BaseImageProcessingNode):
    """Standalone GPU-accelerated wavelet denoising node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "wavelet_type": (["db8", "db4", "bior2.2", "coif2", "haar"], {"default": "db8"}),
                "thresholding_method": (["BayesShrink", "SureShrink", "VisuShrink", "manual"], {"default": "BayesShrink"}),
                "thresholding_mode": (["soft", "hard"], {"default": "soft"}),
                "decomposition_levels": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "sigma_adjustment": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "use_stationary": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("denoised_image", "processing_info")
    FUNCTION = "denoise_wavelet"
    CATEGORY = "Eric's Nodes/Denoising"
    
    def denoise_wavelet(self, image: torch.Tensor, wavelet_type: str, thresholding_method: str,
                       thresholding_mode: str, decomposition_levels: int, sigma_adjustment: float,
                       use_gpu: bool, use_stationary: bool, custom_sigma: float = 0.0) -> Tuple[torch.Tensor, str]:
        """
        Apply GPU-accelerated wavelet denoising
        """
        
        try:
            # Convert tensor to numpy
            np_image = self.tensor_to_numpy(image)
            
            # Prepare parameters
            params = {
                'wavelet': wavelet_type,
                'method': thresholding_method,
                'mode': thresholding_mode,
                'sigma': custom_sigma if custom_sigma > 0 else None,
                'levels': decomposition_levels if decomposition_levels > 0 else None,
                'multichannel': True,
                'rescale_sigma': sigma_adjustment != 1.0,
                'force_cpu': not use_gpu
            }
            
            # Apply denoising
            if use_stationary:
                denoised = gpu_wavelet_denoise_stationary(np_image, **params)
                method_used = "GPU Stationary Wavelet"
            else:
                denoised = gpu_wavelet_denoise(np_image, **params)
                method_used = "GPU Wavelet"
            
            # Generate processing info
            gpu_info = gpu_memory_info()
            info_lines = []
            info_lines.append(f"Method: {method_used}")
            info_lines.append(f"Wavelet: {wavelet_type}")
            info_lines.append(f"Thresholding: {thresholding_method} ({thresholding_mode})")
            info_lines.append(f"Levels: {decomposition_levels if decomposition_levels > 0 else 'auto'}")
            info_lines.append(f"Sigma Adjustment: {sigma_adjustment:.1f}")
            
            if use_gpu and gpu_info.get('available', False):
                info_lines.append(f"GPU: Enabled")
                info_lines.append(f"GPU Memory: {gpu_info.get('used_bytes', 0) / 1024**2:.1f} MB")
            else:
                info_lines.append(f"GPU: Disabled/Unavailable")
            
            processing_info = "\\n".join(info_lines)
            
            # Convert back to tensor
            result_tensor = self.numpy_to_tensor(denoised)
            
            # Cleanup GPU memory
            if use_gpu:
                cleanup_gpu_memory()
            
            return (result_tensor, processing_info)
            
        except Exception as e:
            error_msg = f"Error in wavelet denoising: {str(e)}"
            print(error_msg)
            return (image, error_msg)


class FilmGrainAnalysisNode(BaseImageProcessingNode):
    """Film grain analysis node for diagnostic purposes"""
    
    def __init__(self):
        super().__init__()
        self.analyzer = FilmGrainAnalyzer()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detailed_analysis": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("grain_type", "analysis_summary", "recommendations")
    FUNCTION = "analyze_grain"
    CATEGORY = "Eric's Nodes/Analysis"
    
    def analyze_grain(self, image: torch.Tensor, detailed_analysis: bool) -> Tuple[str, str, str]:
        """
        Analyze film grain characteristics
        """
        
        try:
            # Convert tensor to numpy
            np_image = self.tensor_to_numpy(image)
            
            # Perform analysis
            analysis = self.analyzer.analyze_grain_type(np_image)
            
            # Extract grain type
            grain_type = analysis['grain_type']
            
            # Generate summary
            summary_lines = []
            summary_lines.append(f"Grain Type: {grain_type}")
            
            if detailed_analysis:
                # Noise statistics
                if 'noise_statistics' in analysis:
                    noise_stats = analysis['noise_statistics']
                    summary_lines.append(f"Noise Sigma: {noise_stats['global_sigma']:.2f}")
                    summary_lines.append(f"SNR: {noise_stats['snr_db']:.1f} dB")
                    summary_lines.append(f"Sigma Variation: {noise_stats['sigma_variation']:.2f}")
                
                # Grain structure
                if 'grain_structure' in analysis:
                    grain_struct = analysis['grain_structure']
                    summary_lines.append(f"Dominant Grain Size: {grain_struct['dominant_grain_size']}")
                    summary_lines.append(f"Grain Regularity: {grain_struct['grain_regularity']:.2f}")
                    summary_lines.append(f"Directionality: {grain_struct['directionality']:.2f}")
                
                # Frequency analysis
                if 'frequency_analysis' in analysis:
                    freq_analysis = analysis['frequency_analysis']
                    summary_lines.append(f"High Freq Ratio: {freq_analysis['high_freq_ratio']:.3f}")
                    summary_lines.append(f"Spectral Centroid: {freq_analysis['spectral_centroid']:.2f}")
                
                # Texture analysis
                if 'texture_analysis' in analysis:
                    texture = analysis['texture_analysis']
                    summary_lines.append(f"Local Std Mean: {texture['local_std_mean']:.2f}")
                    summary_lines.append(f"GLCM Contrast: {texture['glcm_contrast']:.2f}")
            
            analysis_summary = "\\n".join(summary_lines)
            
            # Generate recommendations
            recommendations_lines = []
            if 'recommendations' in analysis:
                rec = analysis['recommendations']
                recommendations_lines.append(f"Primary Method: {rec.get('primary_method', 'N/A')}")
                recommendations_lines.append(f"Secondary Method: {rec.get('secondary_method', 'N/A')}")
                recommendations_lines.append(f"Preserve Grain: {rec.get('preserve_grain', False)}")
                recommendations_lines.append(f"Multi-stage: {rec.get('multi_stage', False)}")
                
                # Parameters
                params = rec.get('parameters', {})
                if params:
                    recommendations_lines.append("Parameters:")
                    for key, value in params.items():
                        recommendations_lines.append(f"  {key}: {value}")
                
                # Post-processing
                post_proc = rec.get('post_processing', [])
                if post_proc:
                    recommendations_lines.append(f"Post-processing: {', '.join(post_proc)}")
            
            recommendations_text = "\\n".join(recommendations_lines)
            
            return (grain_type, analysis_summary, recommendations_text)
            
        except Exception as e:
            error_msg = f"Error in grain analysis: {str(e)}"
            print(error_msg)
            return ("error", error_msg, "Analysis failed")


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AdvancedFilmGrainNode": AdvancedFilmGrainNode,
    "GPUWaveletDenoiseNode": GPUWaveletDenoiseNode,
    "FilmGrainAnalysisNode": FilmGrainAnalysisNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedFilmGrainNode": "Advanced Film Grain Processing",
    "GPUWaveletDenoiseNode": "GPU Wavelet Denoising",
    "FilmGrainAnalysisNode": "Film Grain Analysis",
}
