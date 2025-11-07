"""
Comprehensive Image Processing Comparison Node
Compare multiple processing methods side-by-side with detailed analysis
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List

# Import from main package
try:
    from ..base_node import BaseImageProcessingNode
except ImportError:
    # Fallback for direct testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode

# Import all available processors for comparison
try:
    from Eric_Image_Processing_Nodes import (
        SCUNetProcessor,
        SwinIRProcessor,
        RealESRGANProcessor,
        FrequencyEnhancementProcessor,
        WaveletDenoiseProcessor
    )
except ImportError:
    try:
        from scripts.scunet_processing import SCUNetProcessor
        from scripts.swinir_processing import SwinIRProcessor
        from scripts.real_esrgan_processing import RealESRGANProcessor
        from scripts.frequency_enhancement import FrequencyEnhancementProcessor
        from scripts.wavelet_denoise import WaveletDenoiseProcessor
    except ImportError:
        print("Warning: Some processors not available for comparison")
        SCUNetProcessor = None
        SwinIRProcessor = None
        RealESRGANProcessor = None
        FrequencyEnhancementProcessor = None
        WaveletDenoiseProcessor = None


class ComprehensiveComparisonNode(BaseImageProcessingNode):
    """
    Comprehensive Image Processing Comparison
    
    Compare multiple processing methods side-by-side:
    
    AI METHODS:
    - SCUNet: Realistic restoration with blind denoising
    - SwinIR: Transformer-based precision processing
    - Real-ESRGAN: Real-world photo enhancement
    
    TRADITIONAL METHODS:
    - Wavelet Denoising: Mathematical noise reduction
    - Frequency Enhancement: Spectral domain processing
    - Bilateral Filtering: Edge-preserving smoothing
    
    COMPARISON FEATURES:
    - Side-by-side visual results
    - Quantitative quality metrics
    - Processing time comparison
    - Memory usage analysis
    - Method recommendations
    
    ANALYSIS METRICS:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity)
    - Processing time per method
    - Memory efficiency
    - Visual quality assessment
    
    Perfect for:
    - Method evaluation and selection
    - Research and benchmarking
    - Understanding method strengths
    - Quality comparison studies
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "methods_to_compare": ("STRING", {
                    "default": "scunet,swinir,real_esrgan,wavelet",
                    "tooltip": "Comma-separated list of methods:\nscunet, swinir, real_esrgan, wavelet, frequency, bilateral\nOr 'all' for complete comparison"
                }),
                "comparison_scale": ([1, 2, 3, 4], {
                    "default": 2,
                    "tooltip": "Upscaling factor for comparison:\n• 1: Enhancement only\n• 2-4: Super-resolution comparison"
                }),
            },
            "optional": {
                "enable_metrics": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Calculate quantitative comparison metrics"
                }),
                "time_analysis": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include processing time analysis"
                }),
                "create_grid": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Create comparison grid image"
                }),
                "device_preference": (["auto", "cpu", "cuda"], {
                    "default": "auto",
                    "tooltip": "Processing device preference"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("comparison_grid", "best_result", "detailed_report")
    FUNCTION = "compare_methods"
    
    def compare_methods(self, image: torch.Tensor, methods_to_compare: str = "scunet,swinir,real_esrgan,wavelet",
                       comparison_scale: int = 2, enable_metrics: bool = True, time_analysis: bool = True,
                       create_grid: bool = True, device_preference: str = "auto") -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Compare multiple image processing methods
        """
        try:
            # Convert input
            img_np = self.tensor_to_numpy(image)
            
            # Parse methods to compare
            if methods_to_compare.lower() == "all":
                methods = ["scunet", "swinir", "real_esrgan", "wavelet", "frequency", "bilateral"]
            else:
                methods = [m.strip().lower() for m in methods_to_compare.split(",")]
            
            # Initialize comparison report
            report_lines = [
                "=== COMPREHENSIVE METHOD COMPARISON ===",
                f"Input: {img_np.shape} pixels",
                f"Methods: {', '.join(methods)}",
                f"Target scale: {comparison_scale}x",
                f"Device: {device_preference}",
                "",
            ]
            
            # Process with each method
            results = {}
            processing_times = {}
            
            for method in methods:
                print(f"Processing with {method}...")
                result, process_time = self._process_with_timing(
                    img_np, method, comparison_scale, device_preference
                )
                
                if result is not None:
                    results[method] = result
                    processing_times[method] = process_time
                    report_lines.append(f"✅ {method.upper()}: Success ({process_time:.2f}s)")
                else:
                    report_lines.append(f"❌ {method.upper()}: Failed")
            
            # Calculate metrics if enabled
            if enable_metrics and results:
                report_lines.extend(["", "=== QUALITY METRICS ==="])
                metrics = self._calculate_metrics(img_np, results, comparison_scale)
                
                for method, metric_data in metrics.items():
                    report_lines.append(f"{method.upper()}:")
                    for metric, value in metric_data.items():
                        if isinstance(value, float):
                            report_lines.append(f"  • {metric}: {value:.4f}")
                        else:
                            report_lines.append(f"  • {metric}: {value}")
                    report_lines.append("")
            
            # Processing time analysis
            if time_analysis and processing_times:
                report_lines.extend(["=== PERFORMANCE ANALYSIS ==="])
                
                sorted_times = sorted(processing_times.items(), key=lambda x: x[1])
                report_lines.append("Processing times (fastest to slowest):")
                
                for i, (method, time_taken) in enumerate(sorted_times, 1):
                    speedup = sorted_times[0][1] / time_taken if time_taken > 0 else 1
                    report_lines.append(f"{i}. {method.upper()}: {time_taken:.2f}s (1.0x)" if i == 1 
                                      else f"{i}. {method.upper()}: {time_taken:.2f}s ({speedup:.1f}x slower)")
                
                report_lines.append("")
            
            # Method recommendations
            report_lines.extend(self._generate_method_recommendations(results, processing_times, metrics if enable_metrics else None))
            
            # Create visual comparison
            if create_grid and results:
                grid_image = self._create_comparison_grid(img_np, results)
                grid_tensor = self.numpy_to_tensor(grid_image)
            else:
                grid_tensor = image
            
            # Select best result
            if results:
                best_method, best_result = self._select_best_result(results, metrics if enable_metrics else None)
                best_tensor = self.numpy_to_tensor(best_result)
                report_lines.append(f"🏆 RECOMMENDED: {best_method.upper()}")
            else:
                best_tensor = image
                report_lines.append("⚠️ No successful processing results")
            
            final_report = "\n".join(report_lines)
            print("Comprehensive comparison completed!")
            print(final_report)
            
            return (grid_tensor, best_tensor, final_report)
            
        except Exception as e:
            error_report = f"Comparison error: {e}\nReturning original image."
            print(error_report)
            import traceback
            traceback.print_exc()
            return (image, image, error_report)
    
    def _process_with_timing(self, img_np: np.ndarray, method: str, scale: int, device: str) -> Tuple[Optional[np.ndarray], float]:
        """Process image with a specific method and measure time"""
        import time
        
        start_time = time.time()
        result = None
        
        try:
            if method == "scunet" and SCUNetProcessor is not None:
                processor = SCUNetProcessor(device=device)
                result = processor.process_image(img_np)
                
                # Handle upscaling for SCUNet
                if result is not None and scale > 1:
                    from PIL import Image
                    pil_img = Image.fromarray((result * 255).astype(np.uint8))
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    result = np.array(upscaled).astype(np.float32) / 255.0
            
            elif method == "swinir" and SwinIRProcessor is not None:
                task = "classical_sr" if scale > 1 else "color_dn"
                processor = SwinIRProcessor(task=task, scale=scale, device=device)
                result = processor.process_image(img_np)
            
            elif method == "real_esrgan" and RealESRGANProcessor is not None:
                processor = RealESRGANProcessor(scale=scale, device=device)
                result = processor.process_image(img_np)
            
            elif method == "wavelet" and WaveletDenoiseProcessor is not None:
                processor = WaveletDenoiseProcessor()
                denoised = processor.process_image(img_np, sigma=15, wavelet='db8')
                
                if denoised is not None and scale > 1:
                    # Simple upscaling after denoising
                    from PIL import Image
                    pil_img = Image.fromarray((denoised * 255).astype(np.uint8))
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    result = np.array(upscaled).astype(np.float32) / 255.0
                else:
                    result = denoised
            
            elif method == "frequency" and FrequencyEnhancementProcessor is not None:
                processor = FrequencyEnhancementProcessor()
                enhanced = processor.process_image(img_np, method="Adaptive_Frequency")
                
                if enhanced is not None and scale > 1:
                    # Simple upscaling after enhancement
                    from PIL import Image
                    pil_img = Image.fromarray((enhanced * 255).astype(np.uint8))
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    result = np.array(upscaled).astype(np.float32) / 255.0
                else:
                    result = enhanced
            
            elif method == "bilateral":
                # Bilateral filtering (traditional method)
                from skimage.restoration import denoise_bilateral
                denoised = denoise_bilateral(img_np, sigma_color=0.1, sigma_spatial=15, multichannel=True)
                
                if scale > 1:
                    # Simple upscaling
                    from PIL import Image
                    pil_img = Image.fromarray((denoised * 255).astype(np.uint8))
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    upscaled = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    result = np.array(upscaled).astype(np.float32) / 255.0
                else:
                    result = denoised
            
        except Exception as e:
            print(f"Error processing with {method}: {e}")
            result = None
        
        processing_time = time.time() - start_time
        return result, processing_time
    
    def _calculate_metrics(self, original: np.ndarray, results: Dict[str, np.ndarray], scale: int) -> Dict[str, Dict]:
        """Calculate quality metrics for comparison"""
        metrics = {}
        
        try:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            
            # Create reference (upscaled original for fair comparison)
            if scale > 1:
                from PIL import Image
                pil_ref = Image.fromarray((original * 255).astype(np.uint8))
                new_size = (pil_ref.width * scale, pil_ref.height * scale)
                upscaled_ref = pil_ref.resize(new_size, Image.Resampling.LANCZOS)
                reference = np.array(upscaled_ref).astype(np.float32) / 255.0
            else:
                reference = original
            
            for method, result in results.items():
                try:
                    # Ensure same size for comparison
                    if result.shape != reference.shape:
                        # Resize result to match reference
                        from PIL import Image
                        pil_result = Image.fromarray((result * 255).astype(np.uint8))
                        resized = pil_result.resize((reference.shape[1], reference.shape[0]), Image.Resampling.LANCZOS)
                        result = np.array(resized).astype(np.float32) / 255.0
                    
                    # Calculate PSNR
                    psnr = peak_signal_noise_ratio(reference, result, data_range=1.0)
                    
                    # Calculate SSIM
                    ssim = structural_similarity(reference, result, multichannel=True, channel_axis=-1, data_range=1.0)
                    
                    # Calculate enhancement ratio
                    enhancement_ratio = result.shape[0] / original.shape[0]
                    
                    # Dynamic range analysis
                    orig_range = original.max() - original.min()
                    result_range = result.max() - result.min()
                    range_preservation = result_range / orig_range
                    
                    metrics[method] = {
                        "PSNR": psnr,
                        "SSIM": ssim,
                        "Enhancement_Ratio": enhancement_ratio,
                        "Range_Preservation": range_preservation,
                        "Output_Size": f"{result.shape[0]}x{result.shape[1]}"
                    }
                    
                except Exception as e:
                    print(f"Error calculating metrics for {method}: {e}")
                    metrics[method] = {"Error": str(e)}
            
        except ImportError:
            print("Warning: scikit-image not available for metric calculation")
            # Fallback basic metrics
            for method, result in results.items():
                enhancement_ratio = result.shape[0] / original.shape[0]
                metrics[method] = {
                    "Enhancement_Ratio": enhancement_ratio,
                    "Output_Size": f"{result.shape[0]}x{result.shape[1]}"
                }
        
        return metrics
    
    def _create_comparison_grid(self, original: np.ndarray, results: Dict[str, np.ndarray]) -> np.ndarray:
        """Create a visual comparison grid"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Convert results to PIL images
            images = []
            labels = []
            
            # Add original
            orig_pil = Image.fromarray((original * 255).astype(np.uint8))
            images.append(orig_pil)
            labels.append("Original")
            
            # Add results
            for method, result in results.items():
                result_pil = Image.fromarray((result * 255).astype(np.uint8))
                images.append(result_pil)
                labels.append(method.upper())
            
            # Calculate grid dimensions
            num_images = len(images)
            cols = min(3, num_images)
            rows = (num_images + cols - 1) // cols
            
            # Find maximum dimensions
            max_width = max(img.width for img in images)
            max_height = max(img.height for img in images)
            
            # Add space for labels
            label_height = 30
            
            # Create grid
            grid_width = cols * max_width
            grid_height = rows * (max_height + label_height)
            grid = Image.new('RGB', (grid_width, grid_height), 'white')
            
            # Place images
            for i, (img, label) in enumerate(zip(images, labels)):
                row = i // cols
                col = i % cols
                
                x = col * max_width
                y = row * (max_height + label_height)
                
                # Resize image to fit
                img_resized = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
                grid.paste(img_resized, (x, y))
                
                # Add label
                try:
                    draw = ImageDraw.Draw(grid)
                    # Try to use a font, fallback to default
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    
                    text_x = x + max_width // 2
                    text_y = y + max_height + 5
                    
                    # Get text size
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    
                    draw.text((text_x - text_width // 2, text_y), label, fill='black', font=font)
                except:
                    pass  # Skip labeling if there's an issue
            
            # Convert back to numpy
            grid_np = np.array(grid).astype(np.float32) / 255.0
            return grid_np
            
        except Exception as e:
            print(f"Error creating comparison grid: {e}")
            # Return original as fallback
            return original
    
    def _select_best_result(self, results: Dict[str, np.ndarray], metrics: Optional[Dict] = None) -> Tuple[str, np.ndarray]:
        """Select the best result based on metrics or heuristics"""
        if not results:
            return "none", None
        
        if metrics:
            # Select based on highest SSIM score
            best_method = None
            best_score = -1
            
            for method in results.keys():
                if method in metrics and "SSIM" in metrics[method]:
                    ssim = metrics[method]["SSIM"]
                    if ssim > best_score:
                        best_score = ssim
                        best_method = method
            
            if best_method:
                return best_method, results[best_method]
        
        # Fallback: prefer AI methods in order of sophistication
        preference_order = ["swinir", "real_esrgan", "scunet", "frequency", "wavelet", "bilateral"]
        
        for preferred in preference_order:
            if preferred in results:
                return preferred, results[preferred]
        
        # Return first available result
        first_method = list(results.keys())[0]
        return first_method, results[first_method]
    
    def _generate_method_recommendations(self, results: Dict, times: Dict, metrics: Optional[Dict] = None) -> List[str]:
        """Generate method recommendations based on analysis"""
        recommendations = [
            "",
            "=== METHOD RECOMMENDATIONS ===",
        ]
        
        if not results:
            recommendations.append("⚠️ No successful results to analyze")
            return recommendations
        
        # Speed recommendations
        if times:
            fastest = min(times.items(), key=lambda x: x[1])
            slowest = max(times.items(), key=lambda x: x[1])
            
            recommendations.append(f"🚀 FASTEST: {fastest[0].upper()} ({fastest[1]:.2f}s)")
            recommendations.append(f"🐌 SLOWEST: {slowest[0].upper()} ({slowest[1]:.2f}s)")
        
        # Quality recommendations
        if metrics:
            # Find highest SSIM
            best_quality = None
            best_ssim = -1
            
            for method, metric_data in metrics.items():
                if "SSIM" in metric_data:
                    if metric_data["SSIM"] > best_ssim:
                        best_ssim = metric_data["SSIM"]
                        best_quality = method
            
            if best_quality:
                recommendations.append(f"🏆 BEST QUALITY: {best_quality.upper()} (SSIM: {best_ssim:.4f})")
        
        # Method-specific recommendations
        recommendations.extend([
            "",
            "📋 METHOD ANALYSIS:",
        ])
        
        if "scunet" in results:
            recommendations.append("• SCUNet: Best for realistic restoration and noise reduction")
        if "swinir" in results:
            recommendations.append("• SwinIR: Excellent transformer precision, best for clean upscaling")
        if "real_esrgan" in results:
            recommendations.append("• Real-ESRGAN: Great for natural photos and real-world images")
        if "wavelet" in results:
            recommendations.append("• Wavelet: Fast traditional denoising, good baseline")
        if "frequency" in results:
            recommendations.append("• Frequency: Good for detail enhancement and sharpening")
        if "bilateral" in results:
            recommendations.append("• Bilateral: Edge-preserving smoothing, fastest traditional method")
        
        return recommendations


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "ComprehensiveComparison": ComprehensiveComparisonNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComprehensiveComparison": "Comprehensive Method Comparison",
}
