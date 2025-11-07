"""
BM3D Denoising Integration Script
Block-matching and 3D filtering for advanced image denoising

Integrates the BM3D algorithm which is particularly effective for:
- Additive spatially correlated stationary Gaussian noise
- Color image denoising with proper color space handling
- Deblurring combined with denoising
- Professional image restoration workflows
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List
import warnings

try:
    import bm3d
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False
    warnings.warn("BM3D library not available, BM3D denoising features will be disabled")

try:
    from skimage.restoration import estimate_sigma
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available, some metrics unavailable")


class BM3DProcessor:
    """
    BM3D (Block-matching and 3D filtering) Processor
    
    State-of-the-art denoising algorithm that excels at removing additive
    spatially correlated stationary Gaussian noise through collaborative filtering.
    
    Key advantages:
    - Excellent performance on correlated noise
    - Color image support with proper color space handling
    - Multiple profiles for different use cases
    - Can handle deblurring combined with denoising
    - Professional-grade results
    """
    
    def __init__(self):
        """Initialize BM3D processor"""
        if not BM3D_AVAILABLE:
            raise ImportError("BM3D library not available. Please install: pip install bm3d")
        
        # Available profiles
        self.profiles = {
            'np': 'Normal Profile - Balanced quality/speed',
            'refilter': 'Refiltering - Enhanced quality',
            'vn': 'Very Noisy - For high noise levels',
            'vn_old': 'Very Noisy Old - Legacy high noise',
            'high': 'High Quality - Maximum quality',
            'deb': 'Debug - For development/testing'
        }
        
        # Color spaces for RGB processing
        self.color_spaces = ['YCbCr', 'opp']
    
    def estimate_noise_sigma(self, image: np.ndarray) -> float:
        """
        Estimate noise standard deviation in image
        
        Args:
            image: Input image (H, W) or (H, W, C)
            
        Returns:
            Estimated noise standard deviation
        """
        try:
            if SKIMAGE_AVAILABLE:
                if len(image.shape) == 3:
                    # Multi-channel - use first channel for estimation
                    channel = image[:, :, 0]
                    # FIXED: Ensure contiguous array before estimate_sigma
                    channel = np.ascontiguousarray(channel)
                    return estimate_sigma(channel, channel_axis=None)
                else:
                    return estimate_sigma(image, channel_axis=None)
            else:
                # Fallback estimation using Laplacian
                if len(image.shape) == 3:
                    gray = np.mean(image, axis=2)
                else:
                    gray = image
                
                # Sobel-based noise estimation
                from scipy import ndimage
                laplacian = ndimage.laplace(gray)
                sigma = np.sqrt(2) * np.std(laplacian)
                return sigma
                
        except:
            # Simple fallback
            return np.std(image) * 0.1
    
    def denoise_grayscale(self, image: np.ndarray, sigma: Optional[float] = None,
                         profile: str = 'np', stage: str = 'all') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Denoise grayscale image with BM3D
        
        Args:
            image: Input grayscale image (H, W) in range [0, 1]
            sigma: Noise standard deviation (auto-estimated if None)
            profile: BM3D profile ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb')
            stage: Processing stage ('all', 'hard', 'wiener')
            
        Returns:
            Tuple of (denoised_image, processing_info)
        """
        try:
            # Ensure 2D input
            if len(image.shape) != 2:
                raise ValueError(f"Expected 2D image, got {image.shape}")
            
            # Estimate noise if not provided
            if sigma is None:
                sigma = self.estimate_noise_sigma(image)
            
            # Convert to BM3D expected range [0, 255]
            img_255 = (image * 255.0).astype(np.float64)
            sigma_255 = sigma * 255.0
            
            # Select stage
            if stage == 'all':
                stage_arg = bm3d.BM3DStages.ALL_STAGES
            elif stage == 'hard':
                stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING
            else:  # wiener
                stage_arg = bm3d.BM3DStages.ALL_STAGES  # Will use both stages
            
            # Apply BM3D denoising
            denoised_255 = bm3d.bm3d(img_255, sigma_255, profile=profile, stage_arg=stage_arg)
            
            # Convert back to [0, 1] range
            denoised = np.clip(denoised_255 / 255.0, 0, 1)
            
            # Calculate metrics
            info = {
                'sigma_estimated': sigma,
                'sigma_used': sigma,
                'profile': profile,
                'stage': stage,
                'input_range': [image.min(), image.max()],
                'output_range': [denoised.min(), denoised.max()]
            }
            
            if SKIMAGE_AVAILABLE:
                try:
                    psnr = peak_signal_noise_ratio(image, denoised, data_range=1.0)
                    ssim = structural_similarity(image, denoised, data_range=1.0)
                    info['psnr'] = psnr
                    info['ssim'] = ssim
                except:
                    pass
            
            return denoised, info
            
        except Exception as e:
            raise RuntimeError(f"BM3D grayscale denoising failed: {e}")
    
    def denoise_color(self, image: np.ndarray, sigma: Optional[Union[float, List[float]]] = None,
                     profile: str = 'np', colorspace: str = 'YCbCr') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Denoise color image with BM3D
        
        Args:
            image: Input color image (H, W, 3) in range [0, 1]
            sigma: Noise standard deviation (float or [R, G, B] list)
            profile: BM3D profile ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb')
            colorspace: Color space for processing ('YCbCr' or 'opp')
            
        Returns:
            Tuple of (denoised_image, processing_info)
        """
        try:
            # Ensure 3D input with 3 channels
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected (H, W, 3) color image, got {image.shape}")
            
            # WORKAROUND: 'refilter' profile has NumPy 2.x compatibility issues in BM4D
            # BM3D internally uses BM4D for RGB images with 'refilter' profile, which has
            # a broadcasting bug in get_filtered_residual(). Use 'high' profile instead.
            original_profile = profile
            if profile == 'refilter':
                profile = 'high'
                warnings.warn(
                    "BM3D 'refilter' profile has NumPy 2.x compatibility issues with RGB images. "
                    "Using 'high' profile instead for similar quality results.",
                    UserWarning
                )
            
            # Estimate noise if not provided
            if sigma is None:
                # Estimate noise for each channel
                sigma_list = []
                for c in range(3):
                    sigma_c = self.estimate_noise_sigma(image[:, :, c])
                    sigma_list.append(sigma_c)
                sigma = sigma_list
            elif isinstance(sigma, (int, float)):
                sigma = [sigma, sigma, sigma]
            
            # Convert to BM3D expected range [0, 255]
            img_255 = (image * 255.0).astype(np.float64)
            
            if isinstance(sigma, list):
                sigma_255 = [s * 255.0 for s in sigma]
            else:
                sigma_255 = sigma * 255.0
            
            # Apply BM3D color denoising
            denoised_255 = bm3d.bm3d_rgb(img_255, sigma_255, profile=profile, colorspace=colorspace)
            
            # Convert back to [0, 1] range
            denoised = np.clip(denoised_255 / 255.0, 0, 1)
            
            # Calculate metrics
            info = {
                'sigma_estimated': sigma,
                'sigma_used': sigma,
                'profile': profile,
                'profile_requested': original_profile,
                'colorspace': colorspace,
                'input_range': [image.min(), image.max()],
                'output_range': [denoised.min(), denoised.max()]
            }
            
            # Add note if profile was changed
            if original_profile != profile:
                info['note'] = f"Profile '{original_profile}' replaced with '{profile}' due to NumPy 2.x compatibility"
            
            if SKIMAGE_AVAILABLE:
                try:
                    psnr = peak_signal_noise_ratio(image, denoised, data_range=1.0)
                    ssim = structural_similarity(image, denoised, multichannel=True, 
                                               channel_axis=-1, data_range=1.0)
                    info['psnr'] = psnr
                    info['ssim'] = ssim
                except:
                    pass
            
            return denoised, info
            
        except Exception as e:
            raise RuntimeError(f"BM3D color denoising failed: {e}")
    
    def denoise_with_deblurring(self, image: np.ndarray, psf: np.ndarray,
                               sigma: Optional[float] = None, profile: str = 'np') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Combined deblurring and denoising with BM3D
        
        Args:
            image: Input blurred and noisy image (H, W) - grayscale only
            psf: Point spread function for deblurring
            sigma: Noise standard deviation (auto-estimated if None)
            profile: BM3D profile
            
        Returns:
            Tuple of (denoised_deblurred_image, processing_info)
        """
        try:
            # Ensure grayscale input for deblurring
            if len(image.shape) != 2:
                raise ValueError("BM3D deblurring only supports grayscale images")
            
            # Estimate noise if not provided
            if sigma is None:
                sigma = self.estimate_noise_sigma(image)
            
            # Convert to BM3D expected range [0, 255]
            img_255 = (image * 255.0).astype(np.float64)
            sigma_255 = sigma * 255.0
            
            # Ensure PSF is normalized and float64
            psf_normalized = (psf / np.sum(psf)).astype(np.float64)
            
            # Apply BM3D deblurring with correct parameter order
            # bm3d_deblurring(z, psf, sigma, **kwargs)
            denoised_255 = bm3d.bm3d_deblurring(img_255, psf_normalized, sigma_255, profile=profile)
            
            # Convert back to [0, 1] range
            denoised = np.clip(denoised_255 / 255.0, 0, 1)
            
            # Calculate metrics
            info = {
                'sigma_estimated': sigma,
                'sigma_used': sigma,
                'profile': profile,
                'psf_shape': psf.shape,
                'input_range': [image.min(), image.max()],
                'output_range': [denoised.min(), denoised.max()],
                'operation': 'deblurring + denoising'
            }
            
            if SKIMAGE_AVAILABLE:
                try:
                    psnr = peak_signal_noise_ratio(image, denoised, data_range=1.0)
                    ssim = structural_similarity(image, denoised, data_range=1.0)
                    info['psnr'] = psnr
                    info['ssim'] = ssim
                except:
                    pass
            
            return denoised, info
            
        except Exception as e:
            raise RuntimeError(f"BM3D deblurring failed: {e}")
    
    def create_gaussian_psf(self, size: Tuple[int, int], sigma: float) -> np.ndarray:
        """
        Create Gaussian PSF for deblurring
        
        Args:
            size: PSF size (height, width)
            sigma: Gaussian standard deviation
            
        Returns:
            Normalized Gaussian PSF
        """
        if BM3D_AVAILABLE:
            # CRITICAL FIX: bm3d.gaussian_kernel expects (height, width, sigma) as separate args
            # Not (size_tuple, sigma). Must unpack the tuple.
            return bm3d.gaussian_kernel(size[0], size[1], sigma)
        else:
            # Fallback implementation
            y, x = np.ogrid[:size[0], :size[1]]
            y_center, x_center = (size[0] - 1) / 2, (size[1] - 1) / 2
            
            kernel = np.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma**2))
            return kernel / kernel.sum()
    
    def process_image(self, image: np.ndarray, sigma: Optional[Union[float, List[float]]] = None,
                     profile: str = 'np', stage: str = 'all', colorspace: str = 'YCbCr',
                     psf: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Main processing method with automatic format detection
        
        Args:
            image: Input image (H, W) or (H, W, C) in range [0, 1]
            sigma: Noise standard deviation (auto-estimated if None)
            profile: BM3D profile ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb')
            stage: Processing stage ('all', 'hard', 'wiener')
            colorspace: Color space for RGB processing ('YCbCr' or 'opp')
            psf: Point spread function for deblurring (optional)
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        try:
            info = {
                'method': 'BM3D',
                'input_shape': image.shape,
                'profile': profile,
                'stage': stage
            }
            
            if psf is not None:
                # Deblurring mode
                result, process_info = self.denoise_with_deblurring(image, psf, sigma, profile)
                info.update(process_info)
                info['mode'] = 'deblurring + denoising'
                
            elif len(image.shape) == 2:
                # Grayscale denoising
                result, process_info = self.denoise_grayscale(image, sigma, profile, stage)
                info.update(process_info)
                info['mode'] = 'grayscale denoising'
                
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Color denoising
                result, process_info = self.denoise_color(image, sigma, profile, colorspace)
                info.update(process_info)
                info['mode'] = 'color denoising'
                info['colorspace'] = colorspace
                
            else:
                raise ValueError(f"Unsupported image format: {image.shape}")
            
            return result, info
            
        except Exception as e:
            print(f"❌ BM3D processing error: {e}")
            return None, {'error': str(e), 'method': 'BM3D'}
    
    def get_profile_info(self) -> Dict[str, str]:
        """Get information about available BM3D profiles"""
        return self.profiles.copy()
    
    def compare_profiles(self, image: np.ndarray, 
                        profiles: Optional[List[str]] = None) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Compare different BM3D profiles on the same image
        
        Args:
            image: Input image
            profiles: List of profiles to compare (default: all)
            
        Returns:
            Dictionary with profile results
        """
        if profiles is None:
            profiles = list(self.profiles.keys())
        
        results = {}
        
        for profile in profiles:
            print(f"Testing BM3D profile: {profile}")
            try:
                result, info = self.process_image(image, profile=profile)
                results[profile] = (result, info)
            except Exception as e:
                print(f"Profile {profile} failed: {e}")
                results[profile] = (None, {'error': str(e)})
        
        return results
