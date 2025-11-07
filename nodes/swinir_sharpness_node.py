"""
Enhanced SwinIR node exposing detail sharpening controls.
"""

from typing import Tuple
import numpy as np

try:
    from ..base_node import BaseImageProcessingNode
except ImportError:  # pragma: no cover - direct script testing fallback
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode

try:
    from Eric_Image_Processing_Nodes import SwinIRProcessor
except ImportError:
    try:
        from ..scripts.swinir_processing import SwinIRProcessor
    except ImportError:
        SwinIRProcessor = None


class SwinIRSharpnessBoostNode(BaseImageProcessingNode):
    """SwinIR restoration with user-adjustable sharpness boosting."""

    CATEGORY = "Eric's Image Processing/Transformer Enhancements"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "enhance_sharpness"

    _TASK_OPTIONS = [
        "classical_sr",
        "lightweight_sr",
        "real_sr",
        "color_dn",
        "gray_dn",
        "jpeg_car",
    ]

    _MODEL_VARIANTS = [
        "auto",
        "2x",
        "3x",
        "4x",
        "8x",
        "light_noise",
        "medium_noise",
        "heavy_noise",
        "jpeg_q10",
        "jpeg_q20",
        "jpeg_q30",
        "jpeg_q40",
    ]

    _SHARPNESS_PRESETS = {
        "balanced": {"strength": 0.9, "detail_boost": 0.18, "detail_radius": 1.2},
        "detail_boost": {"strength": 1.0, "detail_boost": 0.28, "detail_radius": 1.0},
        "denoise_priority": {"strength": 0.75, "detail_boost": 0.08, "detail_radius": 1.4},
        "film_preserve": {"strength": 0.85, "detail_boost": 0.12, "detail_radius": 1.6},
        "superres_edges": {"strength": 1.0, "detail_boost": 0.32, "detail_radius": 0.9},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "task": (cls._TASK_OPTIONS, {
                    "default": "classical_sr",
                    "tooltip": "Select SwinIR restoration task"
                }),
                "model_variant": (cls._MODEL_VARIANTS, {
                    "default": "auto",
                    "tooltip": "Choose pretrained weight profile"
                }),
                "sharpness_profile": ((["custom"] + list(cls._SHARPNESS_PRESETS.keys())), {
                    "default": "balanced",
                    "tooltip": "Preset tuning curves for restoration vs. sharpness"
                }),
            },
            "optional": {
                "processing_strength": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Blending weight between original and restored output"
                }),
                "detail_boost": ("FLOAT", {
                    "default": 0.18,
                    "min": 0.0,
                    "max": 0.6,
                    "step": 0.02,
                    "tooltip": "High frequency emphasis after SwinIR processing"
                }),
                "detail_radius": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.3,
                    "max": 2.5,
                    "step": 0.1,
                    "tooltip": "Gaussian radius used when extracting detail layer"
                }),
                "device_preference": (("auto", "cpu", "cuda"), {
                    "default": "auto",
                    "tooltip": "Device override for SwinIR inference"
                }),
            }
        }

    def _parse_model_variant(self, task: str, model_variant: str):
        scale = 2
        noise_level = 25
        jpeg_quality = 40

        if model_variant == "auto":
            if task in ["classical_sr", "lightweight_sr", "real_sr"]:
                scale = 4
            elif task in ["gray_dn", "color_dn"]:
                scale = 1
                noise_level = 25
            elif task == "jpeg_car":
                scale = 1
                jpeg_quality = 40
        elif model_variant in {"2x", "3x", "4x", "8x"}:
            scale = int(model_variant[0]) if model_variant != "8x" else 8
        elif model_variant == "light_noise":
            scale = 1
            noise_level = 15
        elif model_variant == "medium_noise":
            scale = 1
            noise_level = 25
        elif model_variant == "heavy_noise":
            scale = 1
            noise_level = 50
        elif model_variant.startswith("jpeg_q"):
            scale = 1
            jpeg_quality = int(model_variant.replace("jpeg_q", ""))

        return scale, noise_level, jpeg_quality

    def enhance_sharpness(
        self,
        image,
        task: str = "classical_sr",
        model_variant: str = "auto",
        sharpness_profile: str = "balanced",
        processing_strength: float = 0.9,
        detail_boost: float = 0.18,
        detail_radius: float = 1.2,
        device_preference: str = "auto"
    ) -> Tuple[np.ndarray]:
        if SwinIRProcessor is None:
            print("SwinIR processor unavailable; returning original image")
            return (image,)

        profile = sharpness_profile or "balanced"
        if profile != "custom" and profile in self._SHARPNESS_PRESETS:
            preset = self._SHARPNESS_PRESETS[profile]
            processing_strength = preset["strength"]
            detail_boost = preset["detail_boost"]
            detail_radius = preset["detail_radius"]

        processing_strength = float(np.clip(processing_strength, 0.0, 1.0))
        detail_boost = float(np.clip(detail_boost, 0.0, 0.6))
        detail_radius = max(0.3, float(detail_radius))

        try:
            np_image = self.tensor_to_numpy(image)
            scale, noise_level, jpeg_quality = self._parse_model_variant(task, model_variant)

            processor = SwinIRProcessor(
                task=task,
                scale=scale,
                noise=noise_level,
                jpeg=jpeg_quality,
                device=device_preference
            )

            result_np = processor.process_image(
                np_image,
                strength=processing_strength,
                detail_boost=detail_boost,
                detail_radius=detail_radius
            )

            if result_np is None:
                print("SwinIR processing failed; returning original image")
                return (image,)

            result_tensor = self.numpy_to_tensor(result_np)
            self.cleanup_memory()
            return (result_tensor,)

        except Exception as exc:  # pragma: no cover - runtime safeguard
            print(f"SwinIR sharpness boost error: {exc}")
            import traceback
            traceback.print_exc()
            self.cleanup_memory()
            return (image,)


NODE_CLASS_MAPPINGS = {
    "SwinIRSharpnessBoostNode": SwinIRSharpnessBoostNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwinIRSharpnessBoostNode": "SwinIR Sharpness Boost (Eric)",
}
