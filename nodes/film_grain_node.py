"""
Film Grain Processing Node for ComfyUI
Specialized processing for different types of film grain
"""

import torch
import numpy as np
from typing import Optional

# Import from parent package
try:
    from ..base_node import BaseImageProcessingNode
    from ..scripts.film_grain_processing import (
        denoise_film_grain,
        analyze_grain_type,
        get_grain_processing_recommendations
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode
    from scripts.film_grain_processing import (
        denoise_film_grain,
        analyze_grain_type,
        get_grain_processing_recommendations
    )


class FilmGrainProcessingNode(BaseImageProcessingNode):
    """
    Specialized film grain processing node that analyzes grain type
    and applies appropriate denoising algorithms
    
    Excellent for:
    - Authentic film grain from scanned negatives
    - Different film formats (35mm, 16mm, 8mm)
    - Push-processed or high-ISO film
    - Simulated film grain removal
    - Digital noise vs. film grain distinction
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grain_type": ([
                    "auto_detect",
                    "fine_film_grain",
                    "medium_film_grain", 
                    "coarse_film_grain",
                    "simulated_regular",
                    "digital_noise",
                    "minimal_grain"
                ], {
                    "default": "auto_detect",
                    "tooltip": "Type of grain to process:\n"
                              "• auto_detect: Automatically analyze and determine grain type\n"
                              "• fine_film_grain: High-quality 35mm film grain\n"
                              "• medium_film_grain: Standard 16mm film grain\n"
                              "• coarse_film_grain: Heavy 8mm or push-processed grain\n"
                              "• simulated_regular: Artificially added grain patterns\n"
                              "• digital_noise: High ISO sensor noise\n"
                              "• minimal_grain: Very low noise levels"
                }),
                "processing_strength": (["minimal", "light", "moderate", "strong", "adaptive"], {
                    "default": "adaptive",
                    "tooltip": "Processing strength:\n"
                              "• minimal: Very conservative, preserves maximum texture\n"
                              "• light: Gentle processing, maintains film character\n"
                              "• moderate: Balanced noise reduction and texture preservation\n"
                              "• strong: Aggressive denoising, may reduce film character\n"
                              "• adaptive: Automatically adjust based on grain analysis"
                }),
            },
            "optional": {
                "preserve_texture": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve film texture characteristics during processing"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration if available (requires CuPy)"
                }),
                "show_analysis": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show grain analysis results in console"
                }),
                "show_recommendations": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show processing recommendations based on grain type"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "grain_analysis")
    FUNCTION = "process_film_grain"
    CATEGORY = "Eric's Image Processing/Film Grain"
    
    def process_film_grain(
        self,
        image,
        grain_type="auto_detect",
        processing_strength="adaptive",
        preserve_texture=True,
        use_gpu=True,
        show_analysis=False,
        show_recommendations=False
    ):
        """Process film grain with specialized algorithms"""
        
        try:
            def process_func(img_np, **kwargs):
                # Analyze grain type if auto-detection is enabled
                if grain_type == "auto_detect":
                    analysis = analyze_grain_type(img_np, show_analysis)
                    detected_grain_type = analysis['grain_type']
                    
                    if show_analysis:
                        print(f"Auto-detected grain type: {detected_grain_type}")
                    
                    # Show recommendations if requested
                    if show_recommendations:
                        recommendations = get_grain_processing_recommendations(detected_grain_type)
                        print(f"\nRecommendations for {detected_grain_type}:")
                        print(f"Description: {recommendations['description']}")
                        print(f"Processing: {recommendations['processing']}")
                        print(f"Strength: {recommendations['strength']}")
                        print(f"Notes: {recommendations['notes']}\n")
                else:
                    detected_grain_type = grain_type
                    analysis = analyze_grain_type(img_np, show_analysis)
                    
                    if show_recommendations:
                        recommendations = get_grain_processing_recommendations(detected_grain_type)
                        print(f"\nRecommendations for {detected_grain_type}:")
                        print(f"Description: {recommendations['description']}")
                        print(f"Processing: {recommendations['processing']}")
                        print(f"Strength: {recommendations['strength']}")
                        print(f"Notes: {recommendations['notes']}\n")
                
                # Adjust processing based on strength setting
                if processing_strength == "adaptive":
                    # Use grain type to determine optimal processing
                    effective_preserve_texture = preserve_texture
                elif processing_strength == "minimal":
                    effective_preserve_texture = True
                elif processing_strength == "light":
                    effective_preserve_texture = True
                elif processing_strength == "moderate":
                    effective_preserve_texture = preserve_texture
                elif processing_strength == "strong":
                    effective_preserve_texture = False
                else:
                    effective_preserve_texture = preserve_texture
                
                # Process the image
                result = denoise_film_grain(
                    img_np,
                    grain_type=detected_grain_type,
                    preserve_texture=effective_preserve_texture,
                    use_gpu=use_gpu,
                    show_analysis=False  # Already shown above
                )
                
                # Create analysis report
                analysis_report = f"Grain Type: {detected_grain_type}\n"
                analysis_report += f"Noise Level: {analysis['noise_level']:.4f}\n"
                analysis_report += f"Estimated Grain Size: {analysis['estimated_grain_size']}\n"
                analysis_report += f"Processing Strength: {processing_strength}\n"
                analysis_report += f"Texture Preservation: {effective_preserve_texture}\n"
                analysis_report += f"GPU Acceleration: {use_gpu}"
                
                return result, analysis_report
            
            # Process image safely
            result, analysis_report = self.process_image_safe_with_report(image, process_func)
            
            return (result, analysis_report)
            
        except Exception as e:
            print(f"Error in film grain processing: {str(e)}")
            return (image, f"Processing failed: {str(e)}")
    
    def process_image_safe_with_report(self, image: torch.Tensor, processing_func):
        """Process image and return both result and analysis report"""
        try:
            # Validate input
            self.validate_image_tensor(image)
            
            # Convert to numpy for processing
            img_np = self.tensor_to_numpy(image)
            
            # Apply processing
            processed_np, analysis_report = processing_func(img_np)
            
            # Convert back to tensor
            result = self.numpy_to_tensor(processed_np)
            
            return result, analysis_report
            
        except Exception as e:
            print(f"Error in image processing: {str(e)}")
            raise
        finally:
            # Clean up memory
            self.cleanup_memory()
            
            # Clean up GPU memory if available
            try:
                from Eric_Image_Processing_Nodes.scripts.gpu_utils import cleanup_gpu_memory
                cleanup_gpu_memory()
            except ImportError:
                pass


class FilmGrainAnalysisNode(BaseImageProcessingNode):
    """
    Film grain analysis node that provides detailed grain characteristics
    without processing the image
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "show_detailed_analysis": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show detailed grain analysis in console"
                }),
                "show_recommendations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show processing recommendations"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("original_image", "detailed_analysis")
    FUNCTION = "analyze_grain"
    CATEGORY = "Eric's Image Processing/Film Grain"
    
    def analyze_grain(
        self,
        image,
        show_detailed_analysis=True,
        show_recommendations=True
    ):
        """Analyze film grain characteristics without processing"""
        
        try:
            def analyze_func(img_np, **kwargs):
                # Perform grain analysis
                analysis = analyze_grain_type(img_np, show_detailed_analysis)
                grain_type = analysis['grain_type']
                
                # Create detailed report
                report = "=== FILM GRAIN ANALYSIS REPORT ===\n\n"
                
                report += f"Detected Grain Type: {grain_type}\n"
                report += f"Noise Level: {analysis['noise_level']:.4f}\n"
                report += f"High Frequency Energy: {analysis['high_freq_energy']:.3f}\n"
                report += f"Mid Frequency Energy: {analysis['mid_freq_energy']:.3f}\n"
                report += f"Pattern Regularity: {analysis['pattern_regularity']:.3f}\n"
                report += f"Estimated Grain Size: {analysis['estimated_grain_size']}\n\n"
                
                # Add grain responses
                report += "Grain Response Profile:\n"
                for i, response in enumerate(analysis['grain_responses']):
                    report += f"  Kernel {3 + i*2}: {response:.4f}\n"
                
                # Add recommendations
                if show_recommendations:
                    recommendations = get_grain_processing_recommendations(grain_type)
                    report += f"\n=== PROCESSING RECOMMENDATIONS ===\n"
                    report += f"Description: {recommendations['description']}\n"
                    report += f"Recommended Processing: {recommendations['processing']}\n"
                    report += f"Suggested Strength: {recommendations['strength']}\n"
                    report += f"Preserve Texture: {recommendations['preserve_texture']}\n"
                    report += f"Notes: {recommendations['notes']}\n"
                
                # Analysis interpretation
                report += f"\n=== ANALYSIS INTERPRETATION ===\n"
                
                if grain_type == "fine_film_grain":
                    report += "This appears to be high-quality film grain, likely from 35mm film.\n"
                    report += "Use gentle processing to maintain film character.\n"
                elif grain_type == "medium_film_grain":
                    report += "This appears to be standard film grain, likely from 16mm film.\n"
                    report += "Moderate processing should work well.\n"
                elif grain_type == "coarse_film_grain":
                    report += "This appears to be heavy film grain, possibly 8mm or push-processed.\n"
                    report += "More aggressive processing may be needed.\n"
                elif grain_type == "simulated_regular":
                    report += "This appears to be artificially added grain with regular patterns.\n"
                    report += "Targeted processing to remove artificial patterns is recommended.\n"
                elif grain_type == "digital_noise":
                    report += "This appears to be digital sensor noise rather than film grain.\n"
                    report += "Frequency domain processing is most effective.\n"
                else:
                    report += "Very little grain detected in this image.\n"
                    report += "Minimal processing recommended to avoid over-smoothing.\n"
                
                return img_np, report
            
            # Analyze image
            result, detailed_analysis = self.process_image_safe_with_report(image, analyze_func)
            
            return (result, detailed_analysis)
            
        except Exception as e:
            print(f"Error in grain analysis: {str(e)}")
            return (image, f"Analysis failed: {str(e)}")
    
    def process_image_safe_with_report(self, image: torch.Tensor, processing_func):
        """Process image and return both result and analysis report"""
        try:
            # Validate input
            self.validate_image_tensor(image)
            
            # Convert to numpy for processing
            img_np = self.tensor_to_numpy(image)
            
            # Apply processing
            processed_np, analysis_report = processing_func(img_np)
            
            # Convert back to tensor
            result = self.numpy_to_tensor(processed_np)
            
            return result, analysis_report
            
        except Exception as e:
            print(f"Error in image processing: {str(e)}")
            raise
        finally:
            # Clean up memory
            self.cleanup_memory()


# Node registration
NODE_CLASS_MAPPINGS = {
    "FilmGrainProcessing": FilmGrainProcessingNode,
    "FilmGrainAnalysis": FilmGrainAnalysisNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilmGrainProcessing": "Film Grain Processing (Eric)",
    "FilmGrainAnalysis": "Film Grain Analysis (Eric)"
}
