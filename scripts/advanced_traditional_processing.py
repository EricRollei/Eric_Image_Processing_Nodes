"""
Advanced Traditional Image Enhancement Methods
Based on the 2024-2025 research findings
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from pathlib import Path
import scipy.ndimage as ndimage
from skimage import restoration, filters, exposure
import warnings
warnings.filterwarnings('ignore')

class LBCLAHEProcessor:
    """Learning-Based CLAHE with automatic hyperparameter tuning"""
    
    def __init__(self):
        self.feature_weights = None
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize feature weights for automatic tuning"""
        # Simplified weights based on common image characteristics
        # In practice, these would be learned from training data
        self.feature_weights = {
            'contrast': 0.3,
            'brightness': 0.2,
            'gradient_magnitude': 0.2,
            'texture_energy': 0.15,
            'edge_density': 0.15
        }
        
    def extract_features(self, image: np.ndarray) -> dict:
        """Extract 15-dimensional feature vectors for automatic tuning"""
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # FIXED: Ensure contiguous array after cv2.cvtColor
            gray = np.ascontiguousarray(gray)
        else:
            gray = image
            
        # Basic statistical features
        features['mean'] = np.mean(gray)
        features['std'] = np.std(gray)
        features['skewness'] = self._calculate_skewness(gray)
        features['kurtosis'] = self._calculate_kurtosis(gray)
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features['hist_entropy'] = self._calculate_entropy(hist)
        features['hist_uniformity'] = np.sum(hist ** 2) / (gray.shape[0] * gray.shape[1]) ** 2
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        # Texture features (simplified GLCM)
        features['texture_energy'] = self._calculate_texture_energy(gray)
        features['texture_contrast'] = self._calculate_texture_contrast(gray)
        features['texture_homogeneity'] = self._calculate_texture_homogeneity(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Local contrast
        features['local_contrast'] = self._calculate_local_contrast(gray)
        
        # Brightness distribution
        features['brightness_range'] = np.max(gray) - np.min(gray)
        features['brightness_ratio'] = np.sum(gray > 128) / (gray.shape[0] * gray.shape[1])
        
        return features
        
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """Calculate skewness of image histogram"""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        return np.mean(((image - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, image: np.ndarray) -> float:
        """Calculate kurtosis of image histogram"""
        mean = np.mean(image)
        std = np.std(image)
        if std == 0:
            return 0
        return np.mean(((image - mean) / std) ** 4) - 3
        
    def _calculate_entropy(self, hist: np.ndarray) -> float:
        """Calculate histogram entropy"""
        hist = hist.flatten()
        hist = hist[hist > 0]
        hist = hist / np.sum(hist)
        return -np.sum(hist * np.log2(hist))
        
    def _calculate_texture_energy(self, image: np.ndarray) -> float:
        """Calculate texture energy using local binary patterns"""
        # Simplified texture energy calculation
        kernel = np.ones((3, 3), np.float32) / 9
        filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)
        diff = np.abs(image.astype(np.float32) - filtered)
        return np.mean(diff)
        
    def _calculate_texture_contrast(self, image: np.ndarray) -> float:
        """Calculate texture contrast"""
        # Use local standard deviation as texture contrast measure
        kernel = np.ones((5, 5), np.float32) / 25
        mean_img = cv2.filter2D(image.astype(np.float32), -1, kernel)
        sqr_diff = (image.astype(np.float32) - mean_img) ** 2
        local_var = cv2.filter2D(sqr_diff, -1, kernel)
        return np.mean(np.sqrt(local_var))
        
    def _calculate_texture_homogeneity(self, image: np.ndarray) -> float:
        """Calculate texture homogeneity"""
        # Simplified homogeneity measure
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return 1.0 / (1.0 + np.mean(gradient_magnitude))
        
    def _calculate_local_contrast(self, image: np.ndarray) -> float:
        """Calculate local contrast using Michelson contrast"""
        kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_max = ndimage.maximum_filter(image, size=9)
        local_min = ndimage.minimum_filter(image, size=9)
        
        # Avoid division by zero
        denominator = local_max + local_min
        denominator[denominator == 0] = 1
        
        local_contrast = (local_max - local_min) / denominator
        return np.mean(local_contrast)
        
    def auto_tune_parameters(self, features: dict, image_shape: tuple = None) -> dict:
        """Automatically tune CLAHE parameters based on image features and size"""
        # Simplified parameter tuning based on features
        params = {}
        
        # Clip limit based on local contrast and gradient magnitude
        if features['local_contrast'] > 0.3:
            params['clip_limit'] = 1.0  # Low clipping for high contrast
        elif features['local_contrast'] < 0.1:
            params['clip_limit'] = 4.0  # High clipping for low contrast
        else:
            params['clip_limit'] = 2.0  # Medium clipping
            
        # FIXED: Scale-aware grid size calculation
        if image_shape is not None:
            height, width = image_shape[:2]
            
            # Calculate optimal tile size (target 64-128 pixels per tile)
            target_tile_size = 80  # pixels
            
            # Calculate grid size based on image dimensions
            grid_h = max(4, min(32, height // target_tile_size))
            grid_w = max(4, min(32, width // target_tile_size))
            
            # Adjust based on image characteristics
            if features['edge_density'] > 0.1:
                # More tiles for edge-rich images (smaller tiles)
                grid_h = min(32, int(grid_h * 1.5))
                grid_w = min(32, int(grid_w * 1.5))
            else:
                # Fewer tiles for smooth images (larger tiles)
                grid_h = max(4, int(grid_h * 0.8))
                grid_w = max(4, int(grid_w * 0.8))
                
            params['grid_size'] = (grid_h, grid_w)
            
        else:
            # Fallback for when image shape isn't provided
            params['grid_size'] = (8, 8)
            
        # Adjust based on brightness distribution
        if features['brightness_ratio'] > 0.8:  # Very bright image
            params['clip_limit'] *= 0.7
        elif features['brightness_ratio'] < 0.2:  # Very dark image
            params['clip_limit'] *= 1.3
            
        return params
        
    def process_image(self, image: np.ndarray, auto_tune: bool = True) -> np.ndarray:
        """Process image with Learning-Based CLAHE"""
        if len(image.shape) == 3:
            # Convert to LAB color space for better color preservation
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            # FIXED: Ensure contiguous array before CLAHE operation
            l_channel = np.ascontiguousarray(l_channel)
            
            if auto_tune:
                # Extract features and auto-tune parameters
                features = self.extract_features(image)
                params = self.auto_tune_parameters(features, image.shape)
                clip_limit = params['clip_limit']
                grid_size = params['grid_size']
            else:
                # Use default parameters with scale-aware sizing
                clip_limit = 2.0
                # Calculate scale-aware grid size even for manual mode
                height, width = image.shape[:2]
                target_tile_size = 80
                grid_h = max(4, min(32, height // target_tile_size))
                grid_w = max(4, min(32, width // target_tile_size))
                grid_size = (grid_h, grid_w)
                
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            l_channel = clahe.apply(l_channel)
            
            # Merge channels back
            lab[:, :, 0] = l_channel
            # FIXED: Ensure contiguous array after slice assignment before color conversion
            lab = np.ascontiguousarray(lab)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
        else:
            # Grayscale image
            if auto_tune:
                features = self.extract_features(image)
                params = self.auto_tune_parameters(features, image.shape)
                clip_limit = params['clip_limit']
                grid_size = params['grid_size']
            else:
                clip_limit = 2.0
                # Calculate scale-aware grid size even for manual mode
                height, width = image.shape[:2]
                target_tile_size = 80
                grid_h = max(4, min(32, height // target_tile_size))
                grid_w = max(4, min(32, width // target_tile_size))
                grid_size = (grid_h, grid_w)
                
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            result = clahe.apply(image)
            
        return result

class MultiScaleRetinexProcessor:
    """Multi-scale Retinex processing for natural color enhancement"""
    
    def __init__(self, scales: List[float] = [15, 80, 250], weights: List[float] = [1/3, 1/3, 1/3]):
        self.base_scales = scales  # Base scales for 1000px reference
        self.scales = scales
        self.weights = weights
        
    def _adapt_scales_for_image(self, image_shape: Tuple[int, int]):
        """Adapt scales based on image dimensions"""
        height, width = image_shape
        reference_size = 1000.0  # Reference size for base scales
        
        # Calculate scale factor based on image diagonal
        diagonal = np.sqrt(height**2 + width**2)
        scale_factor = diagonal / (reference_size * np.sqrt(2))
        
        # Adapt scales proportionally
        self.scales = [scale * scale_factor for scale in self.base_scales]
        
        # Ensure minimum and maximum scale bounds
        self.scales = [max(5, min(scale, min(height, width) // 4)) for scale in self.scales]
        
    def single_scale_retinex(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Apply single-scale Retinex"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), scale)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-6
        image_float = image.astype(np.float32) + epsilon
        blurred_float = blurred + epsilon
        
        # Calculate Retinex
        retinex = np.log(image_float) - np.log(blurred_float)
        
        return retinex
        
    def multi_scale_retinex(self, image: np.ndarray) -> np.ndarray:
        """Apply multi-scale Retinex"""
        # Convert to float
        image_float = image.astype(np.float32)
        
        # Apply SSR at different scales
        retinex_sum = np.zeros_like(image_float)
        
        for scale, weight in zip(self.scales, self.weights):
            ssr = self.single_scale_retinex(image_float, scale)
            retinex_sum += weight * ssr
            
        return retinex_sum
        
    def process_image(self, image: np.ndarray, color_restoration: bool = True, 
                     gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
        """Process image with Multi-scale Retinex"""
        # Adapt scales based on image dimensions
        self._adapt_scales_for_image(image.shape[:2])
        
        if len(image.shape) == 3:
            # Process each channel separately
            result_channels = []
            
            for i in range(3):
                channel = image[:, :, i]
                msr = self.multi_scale_retinex(channel)
                
                # Apply gain and offset
                enhanced = msr * gain + offset
                
                # Normalize to [0, 255]
                enhanced = np.clip(enhanced, 0, 255)
                result_channels.append(enhanced)
                
            result = np.stack(result_channels, axis=2).astype(np.uint8)
            
            if color_restoration:
                # Color restoration using MSRCR
                result = self._color_restoration(image, result)
                
        else:
            # Grayscale image
            msr = self.multi_scale_retinex(image)
            result = msr * gain + offset
            result = np.clip(result, 0, 255).astype(np.uint8)
            
        return result
        
    def _color_restoration(self, original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        """Multi-scale Retinex with Color Restoration (MSRCR)"""
        # Convert to float
        original_float = original.astype(np.float32) + 1e-6
        enhanced_float = enhanced.astype(np.float32) + 1e-6
        
        # Calculate color restoration factor
        sum_original = np.sum(original_float, axis=2, keepdims=True)
        color_restoration = np.log(125.0 * original_float / sum_original)
        
        # Apply color restoration
        result = enhanced_float * color_restoration
        
        # Normalize
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

class BM3DGTADProcessor:
    """BM3D-GT&AD for film grain denoising"""
    
    def __init__(self, sigma: float = 25.0, search_window: int = 39, max_blocks: int = 32, 
                 threshold: float = 2500.0):
        self.sigma = sigma
        self.base_search_window = search_window
        self.max_blocks = max_blocks
        self.threshold = threshold
        
    def _adapt_parameters_for_image(self, image_shape: Tuple[int, int]):
        """Adapt patch parameters based on image dimensions"""
        height, width = image_shape
        reference_size = 1000.0  # Reference size for base parameters
        
        # Calculate scale factor based on smallest dimension
        scale_factor = min(height, width) / reference_size
        
        # For high-resolution images, use more conservative scaling
        if scale_factor > 2.0:  # Very high resolution
            # Cap the scale factor to prevent overly large patches
            scale_factor = 1.0 + (scale_factor - 1.0) * 0.5
        
        # Adapt patch size (base: 7x7 for 1000px images)
        base_patch_size = 7
        self.patch_size = max(3, min(15, int(base_patch_size * scale_factor)))
        
        # Adapt patch distance (base: 11 for 1000px images)
        base_patch_distance = 11
        self.patch_distance = max(5, min(25, int(base_patch_distance * scale_factor)))
        
        # For high-resolution images, increase patch distance more conservatively
        if scale_factor > 1.5:
            # Increase patch distance to improve performance on large images
            self.patch_distance = min(self.patch_distance + 4, 25)
        
        # Ensure odd values for patch_size
        if self.patch_size % 2 == 0:
            self.patch_size += 1
            
        # Ensure patch_distance > patch_size
        if self.patch_distance <= self.patch_size:
            self.patch_distance = self.patch_size + 2
        
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Process image with enhanced BM3D for film grain"""
        # Adapt parameters based on image dimensions
        self._adapt_parameters_for_image(image.shape[:2])
        
        # For very high-resolution images, use more aggressive fast_mode settings
        height, width = image.shape[:2]
        total_pixels = height * width
        use_fast_mode = total_pixels > (2048 * 2048)  # Use fast mode for >4MP images
        
        if len(image.shape) == 3:
            # Process each channel separately
            result_channels = []
            
            for i in range(3):
                channel = image[:, :, i].astype(np.float32) / 255.0
                
                # Apply BM3D (using restoration.denoise_nl_means as approximation)
                denoised = restoration.denoise_nl_means(
                    channel, 
                    patch_size=self.patch_size,
                    patch_distance=self.patch_distance,
                    h=self.sigma/255.0,
                    fast_mode=use_fast_mode
                )
                
                # Convert back to uint8
                denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
                result_channels.append(denoised)
                
            result = np.stack(result_channels, axis=2)
            
        else:
            # Grayscale image
            image_float = image.astype(np.float32) / 255.0
            
            result = restoration.denoise_nl_means(
                image_float,
                patch_size=self.patch_size,
                patch_distance=self.patch_distance,
                h=self.sigma/255.0,
                fast_mode=use_fast_mode
            )
            
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            
        return result

class SmartSharpeningProcessor:
    """Smart Sharpening with artifact detection and adaptive parameters"""
    
    def __init__(self, method: str = 'unsharp_mask'):
        self.method = method
        
    def detect_overshoot(self, original: np.ndarray, sharpened: np.ndarray, 
                        threshold: float = 0.1) -> np.ndarray:
        """Detect overshoot artifacts"""
        # Calculate difference
        diff = np.abs(sharpened.astype(np.float32) - original.astype(np.float32))
        
        # Find edges in original
        if len(original.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            # FIXED: Ensure contiguous array after cv2.cvtColor
            gray = np.ascontiguousarray(gray)
        else:
            gray = original
            
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect overshoot near edges
        kernel = np.ones((3, 3), np.uint8)
        edge_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Create mask for potential overshoot
        if len(diff.shape) == 3:
            overshoot_mask = np.any(diff > threshold * 255, axis=2)
        else:
            overshoot_mask = diff > threshold * 255
            
        # Combine with edge information
        overshoot_mask = overshoot_mask & (edge_dilated > 0)
        
        return overshoot_mask
        
    def adaptive_radius_control(self, image: np.ndarray) -> Tuple[float, float]:
        """Determine optimal radius and amount based on image content"""
        # Calculate scale factor based on image dimensions
        height, width = image.shape[:2]
        reference_size = 1000.0  # Reference size for base parameters
        scale_factor = min(height, width) / reference_size
        
        # Calculate edge density
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # FIXED: Ensure contiguous array after cv2.cvtColor
            gray = np.ascontiguousarray(gray)
        else:
            gray = image
            
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Calculate local variance (detail level)
        kernel = np.ones((5, 5), np.float32) / 25
        mean_img = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        var_img = cv2.filter2D((gray.astype(np.float32) - mean_img) ** 2, -1, kernel)
        detail_level = np.mean(var_img)
        
        # Adaptive base parameters
        if edge_density > 0.1:  # High edge density
            base_radius = 0.5
            amount = 0.5
        elif edge_density < 0.05:  # Low edge density
            base_radius = 2.0
            amount = 1.0
        else:  # Medium edge density
            base_radius = 1.0
            amount = 0.75
            
        # Adjust based on detail level
        if detail_level > 100:  # High detail
            amount *= 0.7
        elif detail_level < 50:  # Low detail
            amount *= 1.3
            
        # Scale radius based on image dimensions
        radius = base_radius * scale_factor
        
        # Ensure radius bounds
        radius = max(0.3, min(radius, 5.0))
            
        return radius, amount
        
    def unsharp_mask_adaptive(self, image: np.ndarray, radius: float, amount: float, 
                            threshold: float = 0.0) -> np.ndarray:
        """Apply adaptive unsharp mask"""
        # Convert to float
        image_float = image.astype(np.float32)
        
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image_float, (0, 0), radius)
        
        # Create mask
        mask = image_float - blurred
        
        # Apply threshold
        if threshold > 0:
            mask = np.where(np.abs(mask) > threshold, mask, 0)
            
        # Apply sharpening
        sharpened = image_float + amount * mask
        
        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 255)
        
        return sharpened.astype(np.uint8)
        
    def process_image(self, image: np.ndarray, auto_params: bool = True, 
                     radius: float = 1.0, amount: float = 0.5) -> np.ndarray:
        """Process image with smart sharpening"""
        if auto_params:
            # Determine optimal parameters
            radius, amount = self.adaptive_radius_control(image)
            
        if self.method == 'unsharp_mask':
            # Apply adaptive unsharp mask
            sharpened = self.unsharp_mask_adaptive(image, radius, amount)
            
            # Detect and correct overshoot
            overshoot_mask = self.detect_overshoot(image, sharpened)
            
            # Blend original and sharpened based on overshoot detection
            if len(image.shape) == 3:
                overshoot_mask = overshoot_mask[:, :, np.newaxis]
                
            result = np.where(overshoot_mask, 
                            image * 0.8 + sharpened * 0.2,  # Reduce sharpening where overshoot
                            sharpened)
            
        else:
            # Fallback to standard unsharp mask
            result = self.unsharp_mask_adaptive(image, radius, amount)
            
        return result.astype(np.uint8)

# Test functions
def test_advanced_methods():
    """Test all advanced methods"""
    # Create test image
    test_image = np.random.rand(256, 256, 3) * 255
    test_image = test_image.astype(np.uint8)
    
    print("Testing Advanced Traditional Methods...")
    
    # Test LB-CLAHE
    print("Testing LB-CLAHE...")
    clahe_processor = LBCLAHEProcessor()
    result_clahe = clahe_processor.process_image(test_image)
    print(f"LB-CLAHE: {test_image.shape} -> {result_clahe.shape}")
    
    # Test Multi-scale Retinex
    print("Testing Multi-scale Retinex...")
    retinex_processor = MultiScaleRetinexProcessor()
    result_retinex = retinex_processor.process_image(test_image)
    print(f"MSR: {test_image.shape} -> {result_retinex.shape}")
    
    # Test BM3D-GT&AD
    print("Testing BM3D-GT&AD...")
    bm3d_processor = BM3DGTADProcessor()
    result_bm3d = bm3d_processor.process_image(test_image)
    print(f"BM3D-GT&AD: {test_image.shape} -> {result_bm3d.shape}")
    
    # Test Smart Sharpening
    print("Testing Smart Sharpening...")
    sharp_processor = SmartSharpeningProcessor()
    result_sharp = sharp_processor.process_image(test_image)
    print(f"Smart Sharpening: {test_image.shape} -> {result_sharp.shape}")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_advanced_methods()
