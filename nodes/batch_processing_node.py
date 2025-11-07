"""
Batch image processing node for efficient processing of multiple images
with consistent parameters and optimized memory usage
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import gc
import time

# Import from parent package
try:
    from ..base_node import BaseImageProcessingNode
    from ..scripts.wavelet_denoise import wavelet_denoise
    from ..scripts.nonlocal_means import nonlocal_means_denoise
    from ..scripts.frequency_enhancement import homomorphic_filter
    from ..scripts.richardson_lucy import richardson_lucy_deconvolution
    from ..scripts.wiener_filter import wiener_filter_restoration
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode
    from scripts.wavelet_denoise import wavelet_denoise
    from scripts.nonlocal_means import nonlocal_means_denoise
    from scripts.frequency_enhancement import homomorphic_filter
    from scripts.richardson_lucy import richardson_lucy_deconvolution
    from scripts.wiener_filter import wiener_filter_restoration


class BatchImageProcessingNode(BaseImageProcessingNode):
    """
    Efficient batch processing node for applying the same enhancement
    to multiple images with optimized memory management
    
    Features:
    - Process multiple images with consistent parameters
    - Memory-efficient processing with garbage collection
    - Progress tracking and timing
    - Automatic parameter optimization per image
    - Fallback handling for individual image failures
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of images
                "processing_method": ([
                    "wavelet_denoise",
                    "nonlocal_means", 
                    "homomorphic_filter",
                    "richardson_lucy",
                    "wiener_filter"
                ], {
                    "default": "wavelet_denoise",
                    "tooltip": "Processing method to apply to all images"
                }),
                "consistent_parameters": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use same parameters for all images (faster) or auto-adapt per image"
                }),
            },
            "optional": {
                # Wavelet parameters
                "wavelet_type": (["db8", "db4", "bior2.2", "coif2", "haar"], {
                    "default": "db8",
                    "tooltip": "Wavelet type for wavelet denoising"
                }),
                "wavelet_method": (["BayesShrink", "VisuShrink", "SureShrink", "manual"], {
                    "default": "BayesShrink",
                    "tooltip": "Wavelet thresholding method"
                }),
                "wavelet_sigma": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Wavelet noise level (0 = auto)"
                }),
                
                # Non-local means parameters
                "nlm_h": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "NLM filtering strength (0 = auto)"
                }),
                "nlm_patch_size": ("INT", {
                    "default": 7,
                    "min": 3,
                    "max": 11,
                    "step": 2,
                    "tooltip": "NLM patch size"
                }),
                
                # Homomorphic filter parameters
                "homo_d0": ("FLOAT", {
                    "default": 40.0,
                    "min": 10.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Homomorphic cutoff frequency"
                }),
                "homo_gamma_h": ("FLOAT", {
                    "default": 1.8,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Homomorphic high frequency gain"
                }),
                "homo_gamma_l": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Homomorphic low frequency gain"
                }),
                
                # Richardson-Lucy parameters
                "rl_iterations": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Richardson-Lucy iterations"
                }),
                "rl_blur_size": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Richardson-Lucy blur size"
                }),
                
                # Wiener filter parameters
                "wiener_noise_variance": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Wiener noise variance (0 = auto)"
                }),
                
                # Batch processing options
                "memory_efficient": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use memory-efficient processing (slower but uses less RAM)"
                }),
                "show_progress": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show processing progress in console"
                }),
                "fail_on_error": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stop processing on first error, otherwise skip failed images"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_images",)
    FUNCTION = "process_batch"
    CATEGORY = "Eric's Image Processing/Batch"
    
    def process_batch(
        self,
        images,
        processing_method="wavelet_denoise",
        consistent_parameters=True,
        # Wavelet parameters
        wavelet_type="db8",
        wavelet_method="BayesShrink",
        wavelet_sigma=0.0,
        # NLM parameters
        nlm_h=0.0,
        nlm_patch_size=7,
        # Homomorphic parameters
        homo_d0=40.0,
        homo_gamma_h=1.8,
        homo_gamma_l=0.6,
        # Richardson-Lucy parameters
        rl_iterations=10,
        rl_blur_size=2.0,
        # Wiener parameters
        wiener_noise_variance=0.0,
        # Batch options
        memory_efficient=True,
        show_progress=True,
        fail_on_error=False
    ):
        """Process a batch of images with the specified method"""
        
        try:
            # Validate input
            if len(images.shape) != 4:
                raise ValueError("Expected batch of images with shape [N, H, W, C]")
            
            batch_size = images.shape[0]
            
            if show_progress:
                print(f"Processing batch of {batch_size} images with {processing_method}")
                start_time = time.time()
            
            # Prepare parameters
            params = self._prepare_parameters(
                processing_method,
                wavelet_type, wavelet_method, wavelet_sigma,
                nlm_h, nlm_patch_size,
                homo_d0, homo_gamma_h, homo_gamma_l,
                rl_iterations, rl_blur_size,
                wiener_noise_variance
            )
            
            # Process images
            processed_images = []
            failed_indices = []
            
            for i in range(batch_size):
                try:
                    if show_progress:
                        print(f"Processing image {i+1}/{batch_size}...")
                    
                    # Extract single image
                    single_image = images[i:i+1]
                    
                    # Adapt parameters if needed
                    if not consistent_parameters:
                        adapted_params = self._adapt_parameters_for_image(
                            single_image, processing_method, params
                        )
                    else:
                        adapted_params = params
                    
                    # Process single image
                    processed = self._process_single_image(
                        single_image, processing_method, adapted_params
                    )
                    
                    processed_images.append(processed)
                    
                    # Memory cleanup if requested
                    if memory_efficient:
                        self.cleanup_memory()
                    
                except Exception as e:
                    error_msg = f"Error processing image {i+1}: {str(e)}"
                    print(error_msg)
                    
                    if fail_on_error:
                        raise Exception(error_msg)
                    
                    # Add original image on failure
                    failed_indices.append(i)
                    processed_images.append(single_image)
            
            # Combine results
            if processed_images:
                result = torch.cat(processed_images, dim=0)
            else:
                result = images  # Return original if all failed
            
            if show_progress:
                elapsed = time.time() - start_time
                success_count = batch_size - len(failed_indices)
                print(f"Batch processing completed: {success_count}/{batch_size} successful in {elapsed:.1f}s")
                if failed_indices:
                    print(f"Failed images: {failed_indices}")
            
            return (result,)
            
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            return (images,)
    
    def _prepare_parameters(
        self,
        processing_method: str,
        wavelet_type: str,
        wavelet_method: str,
        wavelet_sigma: float,
        nlm_h: float,
        nlm_patch_size: int,
        homo_d0: float,
        homo_gamma_h: float,
        homo_gamma_l: float,
        rl_iterations: int,
        rl_blur_size: float,
        wiener_noise_variance: float
    ) -> Dict[str, Any]:
        """Prepare parameters for the selected processing method"""
        
        if processing_method == "wavelet_denoise":
            return {
                "wavelet": wavelet_type,
                "method": wavelet_method,
                "sigma": wavelet_sigma if wavelet_sigma > 0 else None,
                "mode": "soft",
                "multichannel": True,
                "rescale_sigma": True
            }
        
        elif processing_method == "nonlocal_means":
            return {
                "h": nlm_h if nlm_h > 0 else None,
                "patch_size": nlm_patch_size,
                "patch_distance": nlm_patch_size + 4,
                "multichannel": True,
                "method": "opencv",
                "fast_mode": True
            }
        
        elif processing_method == "homomorphic_filter":
            return {
                "d0": homo_d0,
                "gamma_h": homo_gamma_h,
                "gamma_l": homo_gamma_l,
                "c": 1.0,
                "filter_type": "gaussian"
            }
        
        elif processing_method == "richardson_lucy":
            return {
                "iterations": rl_iterations,
                "blur_type": "gaussian",
                "blur_size": rl_blur_size,
                "clip": True,
                "regularization": 0.001
            }
        
        elif processing_method == "wiener_filter":
            return {
                "noise_variance": wiener_noise_variance if wiener_noise_variance > 0 else None,
                "blur_type": "gaussian",
                "blur_size": 2.0,
                "method": "parametric"
            }
        
        else:
            raise ValueError(f"Unknown processing method: {processing_method}")
    
    def _adapt_parameters_for_image(
        self,
        image: torch.Tensor,
        processing_method: str,
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt parameters based on individual image characteristics"""
        
        # Convert to numpy for analysis
        img_np = self.tensor_to_numpy(image)
        
        # Basic image analysis
        if len(img_np.shape) == 3:
            gray = np.mean(img_np, axis=2)
        else:
            gray = img_np
        
        # Estimate noise and contrast
        noise_level = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 1.0))
        contrast = np.std(gray) / (np.mean(gray) + 1e-8)
        
        # Adapt parameters based on image characteristics
        adapted_params = base_params.copy()
        
        if processing_method == "wavelet_denoise":
            # Adapt sigma based on noise level
            if adapted_params["sigma"] is None:
                adapted_params["sigma"] = max(5, min(50, noise_level * 0.5))
            
            # Adapt method based on noise level
            if noise_level > 30:
                adapted_params["method"] = "VisuShrink"  # More conservative for high noise
            elif noise_level < 10:
                adapted_params["method"] = "BayesShrink"  # More aggressive for low noise
        
        elif processing_method == "nonlocal_means":
            # Adapt h based on noise level
            if adapted_params["h"] is None:
                adapted_params["h"] = max(3, min(25, noise_level * 0.4))
        
        elif processing_method == "homomorphic_filter":
            # Adapt gamma values based on contrast
            if contrast < 0.3:  # Low contrast
                adapted_params["gamma_h"] = min(3.0, adapted_params["gamma_h"] * 1.5)
                adapted_params["gamma_l"] = max(0.3, adapted_params["gamma_l"] * 0.8)
        
        return adapted_params
    
    def _process_single_image(
        self,
        image: torch.Tensor,
        processing_method: str,
        params: Dict[str, Any]
    ) -> torch.Tensor:
        """Process a single image with the given method and parameters"""
        
        # Define processing function based on method
        if processing_method == "wavelet_denoise":
            def process_func(img_np, **kwargs):
                return wavelet_denoise(img_np, **params)
        
        elif processing_method == "nonlocal_means":
            def process_func(img_np, **kwargs):
                return nonlocal_means_denoise(img_np, **params)
        
        elif processing_method == "homomorphic_filter":
            def process_func(img_np, **kwargs):
                return homomorphic_filter(img_np, **params)
        
        elif processing_method == "richardson_lucy":
            def process_func(img_np, **kwargs):
                return richardson_lucy_deconvolution(img_np, **params)
        
        elif processing_method == "wiener_filter":
            def process_func(img_np, **kwargs):
                return wiener_filter_restoration(img_np, **params)
        
        else:
            raise ValueError(f"Unknown processing method: {processing_method}")
        
        # Process image safely
        return self.process_image_safe(image, process_func)


# Node registration
NODE_CLASS_MAPPINGS = {
    "BatchImageProcessing": BatchImageProcessingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageProcessing": "Batch Image Processing (Eric)"
}
