"""
Advanced preprocessing node combining multiple enhancement techniques
Adaptive approach based on image content analysis
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from skimage import exposure, filters, segmentation, feature
from skimage.restoration import estimate_sigma
from scipy.ndimage import gaussian_filter

# Import from parent package
try:
    from ..base_node import BaseImageProcessingNode
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode


class AdaptiveImageEnhancementNode(BaseImageProcessingNode):
    """
    Intelligent image enhancement that analyzes image content and applies
    appropriate enhancement techniques automatically
    
    Features:
    - Automatic content analysis (documents, photos, artwork)
    - Adaptive enhancement pipeline
    - Quality assessment and adjustment
    - Multi-stage processing with fallbacks
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhancement_level": (["conservative", "moderate", "aggressive"], {
                    "default": "moderate",
                    "tooltip": "Enhancement strength:\n• conservative: Subtle improvements\n• moderate: Balanced enhancement\n• aggressive: Strong enhancement"
                }),
                "auto_detect_type": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically detect image type and apply appropriate enhancement"
                }),
            },
            "optional": {
                "image_type": (["auto", "photograph", "document", "artwork", "mixed"], {
                    "default": "auto",
                    "tooltip": "Image type for targeted enhancement:\n• auto: Detect automatically\n• photograph: Natural images\n• document: Text/line art\n• artwork: Illustrations/drawings\n• mixed: Combine techniques"
                }),
                "noise_reduction": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply adaptive noise reduction"
                }),
                "contrast_enhancement": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply adaptive contrast enhancement"
                }),
                "sharpening": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply adaptive sharpening"
                }),
                "color_correction": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply adaptive color correction"
                }),
                "show_analysis": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print image analysis results to console"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "enhance"
    CATEGORY = "Eric's Image Processing/Enhancement"
    
    def enhance(
        self, 
        image, 
        enhancement_level="moderate",
        auto_detect_type=True,
        image_type="auto",
        noise_reduction=True,
        contrast_enhancement=True,
        sharpening=True,
        color_correction=True,
        show_analysis=False
    ):
        """Apply adaptive image enhancement"""
        
        try:
            def process_func(img_np, **kwargs):
                return self._adaptive_enhance(
                    img_np,
                    enhancement_level=enhancement_level,
                    auto_detect_type=auto_detect_type,
                    image_type=image_type,
                    noise_reduction=noise_reduction,
                    contrast_enhancement=contrast_enhancement,
                    sharpening=sharpening,
                    color_correction=color_correction,
                    show_analysis=show_analysis
                )
            
            result = self.process_image_safe(image, process_func)
            return (result,)
            
        except Exception as e:
            print(f"Error in adaptive enhancement: {str(e)}")
            return (image,)
    
    def _adaptive_enhance(
        self, 
        image: np.ndarray,
        enhancement_level: str,
        auto_detect_type: bool,
        image_type: str,
        noise_reduction: bool,
        contrast_enhancement: bool,
        sharpening: bool,
        color_correction: bool,
        show_analysis: bool
    ) -> np.ndarray:
        """Main adaptive enhancement pipeline"""
        
        # Analyze image content
        analysis = self._analyze_image_content(image)
        
        if show_analysis:
            self._print_analysis(analysis)
        
        # Determine image type if auto-detection is enabled
        if auto_detect_type or image_type == "auto":
            detected_type = self._detect_image_type(analysis)
            if show_analysis:
                print(f"Detected image type: {detected_type}")
        else:
            detected_type = image_type
        
        # Get enhancement parameters based on type and level
        params = self._get_enhancement_parameters(detected_type, enhancement_level)
        
        # Apply enhancement pipeline
        result = image.copy()
        
        # Stage 1: Noise reduction
        if noise_reduction:
            result = self._apply_noise_reduction(result, params, analysis)
        
        # Stage 2: Contrast enhancement
        if contrast_enhancement:
            result = self._apply_contrast_enhancement(result, params, analysis)
        
        # Stage 3: Color correction
        if color_correction and len(result.shape) == 3:
            result = self._apply_color_correction(result, params, analysis)
        
        # Stage 4: Sharpening
        if sharpening:
            result = self._apply_sharpening(result, params, analysis)
        
        return result
    
    def _analyze_image_content(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image content to determine appropriate enhancement strategy"""
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to 0-1 range
        gray_norm = gray.astype(np.float32) / 255.0
        
        analysis = {}
        
        # Basic statistics
        analysis['mean_brightness'] = np.mean(gray_norm)
        analysis['std_brightness'] = np.std(gray_norm)
        analysis['contrast'] = np.std(gray_norm) / (np.mean(gray_norm) + 1e-8)
        
        # Noise estimation
        analysis['noise_level'] = estimate_sigma(gray_norm, channel_axis=None)
        
        # Edge density (indicates text/line art vs photographs)
        edges = feature.canny(gray_norm, sigma=1.0)
        analysis['edge_density'] = np.mean(edges)
        
        # Texture analysis - use gray_co_matrix (newer API) or graycomatrix (older API)
        try:
            # Try newer API first
            from skimage.feature import gray_co_matrix, gray_co_props
            glcm = gray_co_matrix(
                (gray_norm * 255).astype(np.uint8), 
                [1], [0, 45, 90, 135], 
                symmetric=True, normed=True
            )
            analysis['texture_contrast'] = np.mean(gray_co_props(glcm, 'contrast'))
            analysis['texture_homogeneity'] = np.mean(gray_co_props(glcm, 'homogeneity'))
        except (ImportError, AttributeError):
            # Fallback to older API or skip texture analysis
            try:
                from skimage.feature import graycomatrix, graycoprops
                glcm = graycomatrix(
                    (gray_norm * 255).astype(np.uint8), 
                    [1], [0, 45, 90, 135], 
                    symmetric=True, normed=True
                )
                analysis['texture_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
                analysis['texture_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            except (ImportError, AttributeError):
                # If neither API works, use simple fallback metrics
                analysis['texture_contrast'] = np.std(gray_norm) * 4  # Simple contrast estimate
                analysis['texture_homogeneity'] = 1.0 / (1.0 + analysis['texture_contrast'])
        
        # Frequency content
        f_transform = np.fft.fft2(gray_norm)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # High frequency energy
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        high_freq_mask = np.zeros_like(magnitude)
        high_freq_mask[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8] = 1
        analysis['high_freq_energy'] = np.sum(magnitude * (1 - high_freq_mask)) / np.sum(magnitude)
        
        # Color analysis if color image
        if len(image.shape) == 3:
            # Color saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            analysis['color_saturation'] = np.mean(hsv[:, :, 1]) / 255.0
            
            # Color distribution
            analysis['color_diversity'] = self._calculate_color_diversity(image)
        else:
            analysis['color_saturation'] = 0.0
            analysis['color_diversity'] = 0.0
        
        return analysis
    
    def _calculate_color_diversity(self, image: np.ndarray) -> float:
        """Calculate color diversity metric"""
        # Reduce image to representative colors
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = min(8, len(data))
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Calculate color diversity based on cluster distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        
        # Shannon entropy as diversity measure
        diversity = -np.sum(probabilities * np.log2(probabilities + 1e-8))
        return diversity / np.log2(k)  # Normalize to [0,1]
    
    def _detect_image_type(self, analysis: Dict[str, Any]) -> str:
        """Detect image type based on content analysis"""
        
        # Decision thresholds
        high_edge_threshold = 0.1
        high_texture_contrast = 100
        low_color_sat = 0.1
        
        # Document detection (high edge density, low color saturation)
        if (analysis['edge_density'] > high_edge_threshold and 
            analysis['color_saturation'] < low_color_sat):
            return "document"
        
        # Artwork detection (high texture contrast, high color diversity)
        if (analysis['texture_contrast'] > high_texture_contrast and 
            analysis['color_diversity'] > 0.6):
            return "artwork"
        
        # Photograph detection (moderate edge density, natural color distribution)
        if (analysis['edge_density'] < high_edge_threshold and 
            analysis['color_saturation'] > low_color_sat):
            return "photograph"
        
        # Default to mixed if unclear
        return "mixed"
    
    def _get_enhancement_parameters(self, image_type: str, enhancement_level: str) -> Dict[str, Any]:
        """Get enhancement parameters based on image type and enhancement level"""
        
        # Base parameters
        params = {
            'noise_reduction_strength': 0.5,
            'contrast_strength': 0.5,
            'sharpening_strength': 0.5,
            'color_correction_strength': 0.5
        }
        
        # Adjust based on enhancement level
        level_multipliers = {
            'conservative': 0.5,
            'moderate': 1.0,
            'aggressive': 1.5
        }
        multiplier = level_multipliers.get(enhancement_level, 1.0)
        
        # Adjust based on image type
        if image_type == "document":
            params.update({
                'noise_reduction_strength': 0.3 * multiplier,
                'contrast_strength': 0.8 * multiplier,
                'sharpening_strength': 0.9 * multiplier,
                'color_correction_strength': 0.2 * multiplier
            })
        elif image_type == "photograph":
            params.update({
                'noise_reduction_strength': 0.7 * multiplier,
                'contrast_strength': 0.6 * multiplier,
                'sharpening_strength': 0.4 * multiplier,
                'color_correction_strength': 0.8 * multiplier
            })
        elif image_type == "artwork":
            params.update({
                'noise_reduction_strength': 0.4 * multiplier,
                'contrast_strength': 0.7 * multiplier,
                'sharpening_strength': 0.6 * multiplier,
                'color_correction_strength': 0.9 * multiplier
            })
        
        # Clamp values
        for key in params:
            params[key] = np.clip(params[key], 0.0, 2.0)
        
        return params
    
    def _apply_noise_reduction(self, image: np.ndarray, params: Dict[str, Any], analysis: Dict[str, Any]) -> np.ndarray:
        """Apply adaptive noise reduction"""
        
        strength = params['noise_reduction_strength']
        noise_level = analysis['noise_level']
        
        # More aggressive noise reduction - lowered the threshold
        if noise_level < 0.005:  # Very low noise
            return image
        
        # Use bilateral filter for light noise
        if noise_level < 0.03:  # Lowered threshold for more aggressive treatment
            d = max(5, int(7 * strength))  # Increased base diameter
            sigma_color = max(50, 100 * strength)  # Increased strength
            sigma_space = max(50, 100 * strength)  # Increased strength
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        # Use NLM for moderate to higher noise
        else:
            h = max(8, min(30, 25 * strength * noise_level))  # Increased minimum h value
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    
    def _apply_contrast_enhancement(self, image: np.ndarray, params: Dict[str, Any], analysis: Dict[str, Any]) -> np.ndarray:
        """Apply adaptive contrast enhancement"""
        
        strength = params['contrast_strength']
        current_contrast = analysis['contrast']
        
        # Skip if contrast is already good
        if current_contrast > 0.8:
            return image
        
        # Use CLAHE for adaptive contrast
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            clip_limit = 2.0 * strength
            tile_grid_size = (8, 8)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            clip_limit = 2.0 * strength
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def _apply_color_correction(self, image: np.ndarray, params: Dict[str, Any], analysis: Dict[str, Any]) -> np.ndarray:
        """Apply adaptive color correction"""
        
        strength = params['color_correction_strength']
        
        # Simple white balance correction
        result = image.copy().astype(np.float32)
        
        # Gray world assumption
        avg_r = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])
        
        gray_avg = (avg_r + avg_g + avg_b) / 3
        
        # Apply correction with strength parameter
        r_gain = 1.0 + strength * (gray_avg / avg_r - 1.0)
        g_gain = 1.0 + strength * (gray_avg / avg_g - 1.0)
        b_gain = 1.0 + strength * (gray_avg / avg_b - 1.0)
        
        result[:, :, 0] *= r_gain
        result[:, :, 1] *= g_gain
        result[:, :, 2] *= b_gain
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_sharpening(self, image: np.ndarray, params: Dict[str, Any], analysis: Dict[str, Any]) -> np.ndarray:
        """Apply adaptive sharpening"""
        
        strength = params['sharpening_strength']
        
        # Create unsharp mask
        gaussian_blur = cv2.GaussianBlur(image, (5, 5), 1.0)
        unsharp_mask = cv2.addWeighted(image, 1.0 + strength, gaussian_blur, -strength, 0)
        
        return np.clip(unsharp_mask, 0, 255).astype(np.uint8)
    
    def _print_analysis(self, analysis: Dict[str, Any]):
        """Print analysis results to console"""
        print("\n=== Image Analysis Results ===")
        print(f"Mean brightness: {analysis['mean_brightness']:.3f}")
        print(f"Contrast: {analysis['contrast']:.3f}")
        print(f"Noise level: {analysis['noise_level']:.3f}")
        print(f"Edge density: {analysis['edge_density']:.3f}")
        print(f"Texture contrast: {analysis['texture_contrast']:.1f}")
        print(f"High freq energy: {analysis['high_freq_energy']:.3f}")
        print(f"Color saturation: {analysis['color_saturation']:.3f}")
        print(f"Color diversity: {analysis['color_diversity']:.3f}")
        print("===============================\n")


# Node registration
NODE_CLASS_MAPPINGS = {
    "AdaptiveImageEnhancement": AdaptiveImageEnhancementNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveImageEnhancement": "Adaptive Image Enhancement (Eric)"
}
