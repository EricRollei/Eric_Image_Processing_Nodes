"""
Image Quality Assessment Node for evaluating enhancement results
Provides metrics and visual feedback for processed images
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.restoration import estimate_sigma
import torch

# Import from parent package
try:
    from ..base_node import BaseImageProcessingNode
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode


class ImageQualityAssessmentNode(BaseImageProcessingNode):
    """
    Comprehensive image quality assessment node for evaluating enhancement results
    
    Provides:
    - No-reference quality metrics (doesn't need original)
    - Full-reference metrics (compares to original)
    - Content analysis metrics
    - Enhancement effectiveness scores
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_image": ("IMAGE",),
            },
            "optional": {
                "original_image": ("IMAGE",),
                "assessment_type": (["no_reference", "full_reference", "both"], {
                    "default": "no_reference",
                    "tooltip": "Type of quality assessment:\n• no_reference: Analyze processed image only\n• full_reference: Compare to original\n• both: Perform both types of assessment"
                }),
                "show_detailed_analysis": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show detailed analysis in console"
                }),
                "output_metrics": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Output metrics as text overlay on image"
                }),
                "create_comparison": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Create side-by-side comparison image (requires original)"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("assessed_image", "quality_report")
    FUNCTION = "assess_quality"
    CATEGORY = "Eric's Image Processing/Analysis"
    
    def assess_quality(
        self,
        processed_image,
        original_image=None,
        assessment_type="no_reference",
        show_detailed_analysis=True,
        output_metrics=False,
        create_comparison=False
    ):
        """Assess image quality and provide comprehensive analysis"""
        
        try:
            # Convert images to numpy
            processed_np = self.tensor_to_numpy(processed_image)
            original_np = None
            
            if original_image is not None:
                original_np = self.tensor_to_numpy(original_image)
            
            # Perform assessment
            assessment_results = self._perform_assessment(
                processed_np,
                original_np,
                assessment_type,
                show_detailed_analysis
            )
            
            # Create output image
            output_image = processed_image
            
            if output_metrics:
                output_image = self._add_metrics_overlay(processed_np, assessment_results)
                output_image = self.numpy_to_tensor(output_image)
            
            if create_comparison and original_np is not None:
                comparison_image = self._create_comparison_image(original_np, processed_np)
                output_image = self.numpy_to_tensor(comparison_image)
            
            # Generate quality report
            quality_report = self._generate_quality_report(assessment_results)
            
            return (output_image, quality_report)
            
        except Exception as e:
            print(f"Error in quality assessment: {str(e)}")
            return (processed_image, "Quality assessment failed")
    
    def _perform_assessment(
        self,
        processed_image: np.ndarray,
        original_image: Optional[np.ndarray],
        assessment_type: str,
        show_detailed: bool
    ) -> Dict[str, Any]:
        """Perform comprehensive quality assessment"""
        
        results = {
            'no_reference': {},
            'full_reference': {},
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # No-reference assessment (always performed)
        if assessment_type in ['no_reference', 'both']:
            results['no_reference'] = self._no_reference_assessment(processed_image)
        
        # Full-reference assessment (if original available)
        if original_image is not None and assessment_type in ['full_reference', 'both']:
            results['full_reference'] = self._full_reference_assessment(
                original_image, processed_image
            )
        
        # Calculate overall score
        results['overall_score'] = self._calculate_overall_score(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Print detailed analysis if requested
        if show_detailed:
            self._print_detailed_analysis(results)
        
        return results
    
    def _no_reference_assessment(self, image: np.ndarray) -> Dict[str, float]:
        """Perform no-reference quality assessment"""
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        gray_norm = gray.astype(np.float32) / 255.0
        
        metrics = {}
        
        # 1. Sharpness metrics
        # Laplacian variance (edge sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['laplacian_variance'] = np.var(laplacian)
        
        # Gradient magnitude (overall sharpness)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        metrics['gradient_magnitude'] = np.mean(gradient_magnitude)
        
        # 2. Noise metrics
        # Noise estimation
        metrics['noise_level'] = estimate_sigma(gray_norm, channel_axis=None)
        
        # High frequency noise
        f_transform = np.fft.fft2(gray_norm)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        h, w = magnitude.shape
        
        # High frequency energy (potential noise indicator)
        center_h, center_w = h // 2, w // 2
        high_freq_region = magnitude.copy()
        high_freq_region[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8] = 0
        metrics['high_freq_energy'] = np.sum(high_freq_region) / np.sum(magnitude)
        
        # 3. Contrast metrics
        # RMS contrast
        metrics['rms_contrast'] = np.std(gray_norm)
        
        # Michelson contrast
        metrics['michelson_contrast'] = (np.max(gray_norm) - np.min(gray_norm)) / (np.max(gray_norm) + np.min(gray_norm) + 1e-8)
        
        # 4. Brightness and exposure metrics
        metrics['mean_brightness'] = np.mean(gray_norm)
        metrics['brightness_std'] = np.std(gray_norm)
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / np.sum(hist)
        
        # Histogram entropy (dynamic range)
        metrics['histogram_entropy'] = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
        
        # Under/overexposure detection
        metrics['underexposed_pixels'] = np.sum(gray < 25) / gray.size
        metrics['overexposed_pixels'] = np.sum(gray > 230) / gray.size
        
        # 5. Color metrics (if color image)
        if len(image.shape) == 3:
            # Color saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            metrics['color_saturation'] = np.mean(hsv[:, :, 1]) / 255.0
            
            # Color balance
            r_mean = np.mean(image[:, :, 0])
            g_mean = np.mean(image[:, :, 1])
            b_mean = np.mean(image[:, :, 2])
            gray_mean = (r_mean + g_mean + b_mean) / 3
            
            metrics['color_balance_r'] = abs(r_mean - gray_mean) / gray_mean
            metrics['color_balance_g'] = abs(g_mean - gray_mean) / gray_mean
            metrics['color_balance_b'] = abs(b_mean - gray_mean) / gray_mean
        
        return metrics
    
    def _full_reference_assessment(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Perform full-reference quality assessment"""
        
        # Ensure images are same size
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        metrics = {}
        
        # Convert to grayscale for some metrics
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original.copy()
            proc_gray = processed.copy()
        
        # 1. Structural Similarity Index (SSIM)
        if len(original.shape) == 3:
            ssim_value = structural_similarity(original, processed, multichannel=True, channel_axis=2)
        else:
            ssim_value = structural_similarity(original, processed)
        metrics['ssim'] = ssim_value
        
        # 2. Peak Signal-to-Noise Ratio (PSNR)
        try:
            psnr_value = peak_signal_noise_ratio(original, processed)
            metrics['psnr'] = psnr_value
        except:
            metrics['psnr'] = 0.0
        
        # 3. Mean Squared Error (MSE)
        mse = np.mean((original.astype(np.float32) - processed.astype(np.float32))**2)
        metrics['mse'] = mse
        
        # 4. Mean Absolute Error (MAE)
        mae = np.mean(np.abs(original.astype(np.float32) - processed.astype(np.float32)))
        metrics['mae'] = mae
        
        # 5. Edge preservation metric
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        proc_edges = cv2.Canny(proc_gray, 50, 150)
        
        edge_intersection = np.logical_and(orig_edges, proc_edges)
        edge_union = np.logical_or(orig_edges, proc_edges)
        
        if np.sum(edge_union) > 0:
            metrics['edge_preservation'] = np.sum(edge_intersection) / np.sum(edge_union)
        else:
            metrics['edge_preservation'] = 1.0
        
        # 6. Noise reduction effectiveness
        orig_noise = estimate_sigma(orig_gray.astype(np.float32) / 255.0, channel_axis=None)
        proc_noise = estimate_sigma(proc_gray.astype(np.float32) / 255.0, channel_axis=None)
        
        if orig_noise > 0:
            metrics['noise_reduction'] = max(0, (orig_noise - proc_noise) / orig_noise)
        else:
            metrics['noise_reduction'] = 0.0
        
        # 7. Contrast enhancement
        orig_contrast = np.std(orig_gray.astype(np.float32) / 255.0)
        proc_contrast = np.std(proc_gray.astype(np.float32) / 255.0)
        
        if orig_contrast > 0:
            metrics['contrast_enhancement'] = (proc_contrast - orig_contrast) / orig_contrast
        else:
            metrics['contrast_enhancement'] = 0.0
        
        return metrics
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)"""
        
        score = 0.0
        weights = 0.0
        
        # No-reference metrics
        if 'no_reference' in results and results['no_reference']:
            nr_metrics = results['no_reference']
            
            # Sharpness component (0-25 points)
            sharpness_score = min(25, nr_metrics.get('laplacian_variance', 0) / 1000 * 25)
            score += sharpness_score
            weights += 25
            
            # Noise component (0-25 points)
            noise_level = nr_metrics.get('noise_level', 0)
            noise_score = max(0, 25 - noise_level * 500)  # Lower noise = higher score
            score += noise_score
            weights += 25
            
            # Contrast component (0-25 points)
            contrast_score = min(25, nr_metrics.get('rms_contrast', 0) * 100)
            score += contrast_score
            weights += 25
            
            # Exposure component (0-25 points)
            underexp = nr_metrics.get('underexposed_pixels', 0)
            overexp = nr_metrics.get('overexposed_pixels', 0)
            exposure_score = max(0, 25 - (underexp + overexp) * 500)
            score += exposure_score
            weights += 25
        
        # Full-reference metrics (if available)
        if 'full_reference' in results and results['full_reference']:
            fr_metrics = results['full_reference']
            
            # SSIM component (0-40 points)
            ssim_score = fr_metrics.get('ssim', 0) * 40
            score += ssim_score
            weights += 40
            
            # PSNR component (0-30 points)
            psnr = fr_metrics.get('psnr', 0)
            psnr_score = min(30, max(0, (psnr - 20) / 20 * 30))  # PSNR 20-40 maps to 0-30
            score += psnr_score
            weights += 30
            
            # Edge preservation (0-30 points)
            edge_score = fr_metrics.get('edge_preservation', 0) * 30
            score += edge_score
            weights += 30
        
        # Normalize score
        if weights > 0:
            return score / weights * 100
        else:
            return 0.0
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate enhancement recommendations based on assessment"""
        
        recommendations = []
        
        # No-reference recommendations
        if 'no_reference' in results:
            nr = results['no_reference']
            
            # Sharpness recommendations
            if nr.get('laplacian_variance', 0) < 500:
                recommendations.append("Image appears soft - consider applying sharpening or deconvolution")
            
            # Noise recommendations
            if nr.get('noise_level', 0) > 0.05:
                recommendations.append("High noise detected - consider wavelet or NLM denoising")
            
            # Contrast recommendations
            if nr.get('rms_contrast', 0) < 0.15:
                recommendations.append("Low contrast - consider CLAHE or homomorphic filtering")
            
            # Exposure recommendations
            if nr.get('underexposed_pixels', 0) > 0.05:
                recommendations.append("Underexposure detected - consider brightness/gamma adjustment")
            
            if nr.get('overexposed_pixels', 0) > 0.05:
                recommendations.append("Overexposure detected - consider highlight recovery")
            
            # Color recommendations
            if nr.get('color_saturation', 0) < 0.1:
                recommendations.append("Low color saturation - consider color enhancement")
        
        # Full-reference recommendations
        if 'full_reference' in results:
            fr = results['full_reference']
            
            if fr.get('ssim', 0) < 0.8:
                recommendations.append("Structural similarity is low - processing may be too aggressive")
            
            if fr.get('edge_preservation', 0) < 0.7:
                recommendations.append("Edge information lost - consider gentler processing or edge-preserving methods")
            
            if fr.get('noise_reduction', 0) < 0.3:
                recommendations.append("Noise reduction is minimal - consider stronger denoising parameters")
        
        if not recommendations:
            recommendations.append("Image quality appears good - no specific recommendations")
        
        return recommendations
    
    def _print_detailed_analysis(self, results: Dict[str, Any]):
        """Print detailed analysis to console"""
        
        print("\n=== IMAGE QUALITY ASSESSMENT ===")
        print(f"Overall Quality Score: {results['overall_score']:.1f}/100")
        
        if results['no_reference']:
            print("\n--- No-Reference Metrics ---")
            nr = results['no_reference']
            
            print(f"Sharpness:")
            print(f"  Laplacian Variance: {nr.get('laplacian_variance', 0):.1f}")
            print(f"  Gradient Magnitude: {nr.get('gradient_magnitude', 0):.1f}")
            
            print(f"Noise:")
            print(f"  Noise Level: {nr.get('noise_level', 0):.4f}")
            print(f"  High Freq Energy: {nr.get('high_freq_energy', 0):.3f}")
            
            print(f"Contrast:")
            print(f"  RMS Contrast: {nr.get('rms_contrast', 0):.3f}")
            print(f"  Michelson Contrast: {nr.get('michelson_contrast', 0):.3f}")
            
            print(f"Exposure:")
            print(f"  Mean Brightness: {nr.get('mean_brightness', 0):.3f}")
            print(f"  Underexposed: {nr.get('underexposed_pixels', 0)*100:.1f}%")
            print(f"  Overexposed: {nr.get('overexposed_pixels', 0)*100:.1f}%")
        
        if results['full_reference']:
            print("\n--- Full-Reference Metrics ---")
            fr = results['full_reference']
            
            print(f"SSIM: {fr.get('ssim', 0):.3f}")
            print(f"PSNR: {fr.get('psnr', 0):.1f} dB")
            print(f"MSE: {fr.get('mse', 0):.1f}")
            print(f"Edge Preservation: {fr.get('edge_preservation', 0):.3f}")
            print(f"Noise Reduction: {fr.get('noise_reduction', 0)*100:.1f}%")
            print(f"Contrast Enhancement: {fr.get('contrast_enhancement', 0)*100:.1f}%")
        
        print("\n--- Recommendations ---")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("=" * 35)
    
    def _generate_quality_report(self, results: Dict[str, Any]) -> str:
        """Generate text quality report"""
        
        report = f"Quality Score: {results['overall_score']:.1f}/100\n\n"
        
        if results['no_reference']:
            report += "No-Reference Metrics:\n"
            nr = results['no_reference']
            report += f"- Sharpness: {nr.get('laplacian_variance', 0):.1f}\n"
            report += f"- Noise Level: {nr.get('noise_level', 0):.4f}\n"
            report += f"- Contrast: {nr.get('rms_contrast', 0):.3f}\n\n"
        
        if results['full_reference']:
            report += "Full-Reference Metrics:\n"
            fr = results['full_reference']
            report += f"- SSIM: {fr.get('ssim', 0):.3f}\n"
            report += f"- PSNR: {fr.get('psnr', 0):.1f} dB\n"
            report += f"- Edge Preservation: {fr.get('edge_preservation', 0):.3f}\n\n"
        
        report += "Recommendations:\n"
        for i, rec in enumerate(results['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def _add_metrics_overlay(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Add metrics overlay to image"""
        
        overlay_image = image.copy()
        
        # Create overlay text
        overlay_text = f"Quality: {results['overall_score']:.1f}/100"
        
        if results['no_reference']:
            nr = results['no_reference']
            overlay_text += f"\nSharpness: {nr.get('laplacian_variance', 0):.0f}"
            overlay_text += f"\nNoise: {nr.get('noise_level', 0):.3f}"
            overlay_text += f"\nContrast: {nr.get('rms_contrast', 0):.3f}"
        
        # Add text to image
        y_offset = 30
        for line in overlay_text.split('\n'):
            cv2.putText(overlay_image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay_image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_offset += 25
        
        return overlay_image
    
    def _create_comparison_image(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison image"""
        
        # Ensure same size
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        # Create side-by-side comparison
        comparison = np.hstack([original, processed])
        
        # Add labels
        h, w = original.shape[:2]
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Processed", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return comparison


# Node registration
NODE_CLASS_MAPPINGS = {
    "ImageQualityAssessment": ImageQualityAssessmentNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageQualityAssessment": "Image Quality Assessment (Eric)"
}
