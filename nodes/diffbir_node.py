"""ComfyUI node that wraps the official DiffBIR inference pipeline."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from ..base_node import BaseImageProcessingNode
except ImportError:  # pragma: no cover - direct script testing fallback
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from base_node import BaseImageProcessingNode  # type: ignore

try:
    from Eric_Image_Processing_Nodes import DiffBIRConfig, DiffBIRProcessor
except ImportError:  # pragma: no cover - fallback when package not installed
    try:
        from ..scripts.diffbir_processing import DiffBIRConfig, DiffBIRProcessor
    except ImportError:  # pragma: no cover - runtime safeguard
        DiffBIRConfig = None  # type: ignore
        DiffBIRProcessor = None  # type: ignore


class DiffBIRRestorationNode(BaseImageProcessingNode):
    """Diffusion-based blind restoration using DiffBIR."""

    CATEGORY = "Eric's Image Processing/Transformer Enhancements"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "run_diffbir"

    _TASKS = ["sr", "denoise", "face", "unaligned_face"]
    _VERSIONS = ["v2.1", "v2", "v1"]
    _SAMPLERS = [
        "edm_dpm++_3m_sde",
        "edm_dpm++_2m_sde",
        "edm_dpm++_2m",
        "edm_dpm++_2s_a",
        "edm_dpm++_sde",
        "edm_dpm_2_a",
        "edm_dpm_2",
        "edm_euler_a",
        "edm_euler",
        "edm_lms",
        "edm_heun",
        "dpm++_m2",
        "ddim",
        "spaced",
    ]
    _CAPTIONERS = ["none", "llava", "ram"]
    _PRECISIONS = ["fp16", "fp32", "bf16"]
    _START_POINTS = ["noise", "cond"]

    @classmethod
    def INPUT_TYPES(cls):  # pragma: no cover - UI metadata
        return {
            "required": {
                "image": ("IMAGE",),
                "task": (cls._TASKS, {
                    "default": "sr",
                    "tooltip": "Select DiffBIR task"
                }),
                "version": (cls._VERSIONS, {
                    "default": "v2.1",
                    "tooltip": "Choose DiffBIR checkpoint family"
                }),
                "sampler": (cls._SAMPLERS, {
                    "default": "edm_dpm++_3m_sde",
                    "tooltip": "Sampling strategy for the diffusion stage"
                }),
                "steps": ("INT", {
                    "default": 12,
                    "min": 4,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Number of diffusion steps"
                }),
                "upscale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.5,
                    "tooltip": "Output scaling factor (used for SR and face modes)"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 6.0,
                    "min": 0.0,
                    "max": 12.0,
                    "step": 0.25,
                    "tooltip": "Classifier-free guidance scale"
                }),
            },
            "optional": {
                "captioner": (cls._CAPTIONERS, {
                    "default": "none",
                    "tooltip": "Automatic captioner (requires extra weights for llava/ram)"
                }),
                "noise_aug": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 40,
                    "step": 1,
                    "tooltip": "Noise augmentation level"
                }),
                "start_point_type": (cls._START_POINTS, {
                    "default": "noise",
                    "tooltip": "Initial latent for sampling (cond recommended for v1/v2 when stability is needed)"
                }),
                "rescale_cfg": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Gradually ramp the CFG scale"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "ControlNet conditioning strength"
                }),
                "cleaner_tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiling for the stage-1 cleaner"
                }),
                "cleaner_tile_size": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Tile size for cleaner when tiling is enabled"
                }),
                "cleaner_tile_stride": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 512,
                    "step": 16,
                    "tooltip": "Overlap between cleaner tiles"
                }),
                "cldm_tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiling for the diffusion U-Net"
                }),
                "cldm_tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Tile size for diffusion tiles"
                }),
                "cldm_tile_stride": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 512,
                    "step": 16,
                    "tooltip": "Overlap between diffusion tiles"
                }),
                "guidance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable restoration guidance loss"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Strength of restoration guidance"
                }),
                "guidance_loss": (("w_mse", "mse"), {
                    "default": "w_mse",
                    "tooltip": "Guidance loss variant"
                }),
                "guidance_space": (("rgb", "y"), {
                    "default": "rgb",
                    "tooltip": "Color space for guidance loss"
                }),
                "guidance_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Fraction of steps before guidance activates"
                }),
                "guidance_stop": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Fraction of steps after which guidance stops"
                }),
                "seed": ("INT", {
                    "default": 231,
                    "min": 0,
                    "max": 2 ** 31 - 1,
                    "tooltip": "Random seed"
                }),
                "device_preference": (("auto", "cuda", "cpu", "mps"), {
                    "default": "auto",
                    "tooltip": "Device override"
                }),
                "precision": (cls._PRECISIONS, {
                    "default": "fp16",
                    "tooltip": "Computation precision"
                }),
                "positive_prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional positive prompt override"
                }),
                "negative_prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional negative prompt override"
                }),
            }
        }

    def run_diffbir(
        self,
        image,
        task: str = "sr",
        version: str = "v2.1",
        sampler: str = "edm_dpm++_3m_sde",
        steps: int = 12,
        upscale: float = 4.0,
        cfg_scale: float = 6.0,
        captioner: str = "none",
        noise_aug: int = 0,
        start_point_type: str = "noise",
        rescale_cfg: bool = False,
        strength: float = 1.0,
        cleaner_tiled: bool = False,
        cleaner_tile_size: int = 512,
        cleaner_tile_stride: int = 256,
        cldm_tiled: bool = False,
        cldm_tile_size: int = 512,
        cldm_tile_stride: int = 256,
        guidance: bool = False,
        guidance_scale: float = 0.0,
        guidance_loss: str = "w_mse",
        guidance_space: str = "rgb",
        guidance_start: float = 0.0,
        guidance_stop: float = 1.0,
        seed: int = 231,
        device_preference: str = "auto",
        precision: str = "fp16",
        positive_prompt_override: str = "",
        negative_prompt_override: str = "",
    ) -> Tuple[np.ndarray]:
        if DiffBIRProcessor is None or DiffBIRConfig is None:
            print("DiffBIR dependencies not available; returning original image")
            return (image,)

        np_image = self.tensor_to_numpy(image)

        config = DiffBIRConfig(
            task=task,
            version=version,
            sampler=sampler,
            steps=steps,
            upscale=upscale,
            captioner=captioner,
            cfg_scale=cfg_scale,
            noise_aug=noise_aug,
            start_point_type=start_point_type,
            rescale_cfg=rescale_cfg,
            strength=strength,
            cleaner_tiled=cleaner_tiled,
            cleaner_tile_size=cleaner_tile_size,
            cleaner_tile_stride=cleaner_tile_stride,
            cldm_tiled=cldm_tiled,
            cldm_tile_size=cldm_tile_size,
            cldm_tile_stride=cldm_tile_stride,
            guidance=guidance,
            guidance_scale=guidance_scale,
            guidance_loss=guidance_loss,
            guidance_space=guidance_space,
            guidance_start=guidance_start,
            guidance_stop=guidance_stop,
            seed=seed,
            device_preference=device_preference,
            precision=precision,
            pos_prompt=positive_prompt_override.strip() or None,
            neg_prompt=negative_prompt_override.strip() or None,
        )

        try:
            processor = DiffBIRProcessor(config)
            result_np = processor.process_image(np_image)
            result_tensor = self.numpy_to_tensor(result_np)
            self.cleanup_memory()
            return (result_tensor,)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print(f"DiffBIR inference failed: {exc}")
            import traceback

            traceback.print_exc()
            self.cleanup_memory()
            return (image,)


NODE_CLASS_MAPPINGS = {
    "DiffBIRRestorationNode": DiffBIRRestorationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffBIRRestorationNode": "DiffBIR Restoration (Eric)",
}
