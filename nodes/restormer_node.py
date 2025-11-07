"""ComfyUI node exposing Restormer transformer restoration."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    from ..base_node import BaseImageProcessingNode
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode

try:
    from Eric_Image_Processing_Nodes import RestormerProcessor, MODEL_CONFIGS as RESTORMER_MODEL_CONFIGS
except ImportError:
    try:
        from ..scripts.restormer_processing import RestormerProcessor, MODEL_CONFIGS as RESTORMER_MODEL_CONFIGS
    except ImportError:  # pragma: no cover
        RestormerProcessor = None
        RESTORMER_MODEL_CONFIGS = {}


class RestormerRestorationNode(BaseImageProcessingNode):
    """Restormer restoration with optional sharpness boosting."""

    CATEGORY = "Eric's Image Processing/Transformer Enhancements"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "restore"

    _PRESETS: Dict[str, Dict[str, float]] = {
        "balanced": {"strength": 0.85, "detail_boost": 0.12, "detail_radius": 1.3},
        "detail_boost": {"strength": 0.95, "detail_boost": 0.22, "detail_radius": 1.1},
        "texture_preserve": {"strength": 0.8, "detail_boost": 0.08, "detail_radius": 1.5},
        "denoise_priority": {"strength": 0.7, "detail_boost": 0.05, "detail_radius": 1.6},
        "superres_edges": {"strength": 1.0, "detail_boost": 0.3, "detail_radius": 1.0},
    }

    @classmethod
    def INPUT_TYPES(cls):  # pragma: no cover - UI metadata
        task_options = list(RESTORMER_MODEL_CONFIGS.keys()) or [
            "real_sr_x4",
            "denoise_sigma15",
            "denoise_sigma25",
            "denoise_sigma50",
            "real_denoise",
            "motion_deblur",
            "defocus_deblur",
            "derain_indoor",
            "derain_outdoor",
            "dehaze",
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "task": (task_options, {
                    "default": task_options[0],
                    "tooltip": "Select pre-trained Restormer model"
                }),
                "sharpness_profile": ((["custom"] + list(cls._PRESETS.keys())), {
                    "default": "balanced",
                    "tooltip": "Preset balance between restoration and detail boost"
                }),
            },
            "optional": {
                "processing_strength": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Blend ratio between original and Restormer output"
                }),
                "detail_boost": ("FLOAT", {
                    "default": 0.12,
                    "min": 0.0,
                    "max": 0.6,
                    "step": 0.02,
                    "tooltip": "Amount of high-frequency enhancement after restoration"
                }),
                "detail_radius": ("FLOAT", {
                    "default": 1.3,
                    "min": 0.3,
                    "max": 2.5,
                    "step": 0.1,
                    "tooltip": "Gaussian radius when computing detail layer"
                }),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Tile size for high-resolution inference (set 0 for auto)"
                }),
                "tile_overlap": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 256,
                    "step": 16,
                    "tooltip": "Overlap between tiles to avoid seams"
                }),
                "device_preference": (("auto", "cpu", "cuda"), {
                    "default": "auto",
                    "tooltip": "Device preference for Restormer"
                }),
            }
        }

    def restore(
        self,
        image,
        task: str = "real_sr_x4",
        sharpness_profile: str = "balanced",
        processing_strength: float = 0.85,
        detail_boost: float = 0.12,
        detail_radius: float = 1.3,
        tile_size: int = 512,
        tile_overlap: int = 64,
        device_preference: str = "auto",
    ) -> Tuple[np.ndarray]:
        if RestormerProcessor is None:
            print("Restormer processor not available; returning original image")
            return (image,)

        preset = sharpness_profile or "balanced"
        if preset != "custom" and preset in self._PRESETS:
            preset_values = self._PRESETS[preset]
            processing_strength = preset_values["strength"]
            detail_boost = preset_values["detail_boost"]
            detail_radius = preset_values["detail_radius"]

        processing_strength = float(np.clip(processing_strength, 0.0, 1.0))
        detail_boost = float(np.clip(detail_boost, 0.0, 0.6))
        detail_radius = max(0.3, float(detail_radius))
        tile_overlap = int(max(0, tile_overlap))
        tile_size = int(tile_size)
        tile_size = None if tile_size <= 0 else tile_size

        try:
            np_image = self.tensor_to_numpy(image)
            processor = RestormerProcessor(
                task=task,
                device=device_preference,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            result_np = processor.process_image(
                np_image,
                strength=processing_strength,
                detail_boost=detail_boost,
                detail_radius=detail_radius,
            )
            if result_np is None:
                return (image,)
            result_tensor = self.numpy_to_tensor(result_np)
            self.cleanup_memory()
            return (result_tensor,)
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"Restormer restoration error: {exc}")
            import traceback

            traceback.print_exc()
            self.cleanup_memory()
            return (image,)


NODE_CLASS_MAPPINGS = {
    "RestormerRestorationNode": RestormerRestorationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RestormerRestorationNode": "Restormer Restoration (Eric)",
}
