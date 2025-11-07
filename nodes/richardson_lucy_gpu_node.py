"""ComfyUI node for GPU-accelerated Richardson-Lucy deconvolution."""

from __future__ import annotations

from typing import Dict

from ..base_node import BaseImageProcessingNode
from ..scripts.richardson_lucy import get_blur_presets, estimate_motion_blur
from ..scripts.richardson_lucy_gpu import richardson_lucy_deconvolution_gpu


class RichardsonLucyGPUNode(BaseImageProcessingNode):
    CATEGORY = "Eric's Image Processing/Restoration"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        blur_types = ["gaussian", "motion"]
        device_options = ["auto", "cuda", "cpu", "mps"]
        precision_options = ["fp32", "fp16"]

        return {
            "required": {
                "image": ("IMAGE",),
                "blur_type": (blur_types, {
                    "default": "gaussian",
                    "tooltip": "Blur model to invert"
                }),
                "iterations": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of Richardson-Lucy iterations"
                }),
                "device_preference": (device_options, {
                    "default": "auto",
                    "tooltip": "Processing device (falls back to CPU when unavailable)"
                }),
                "precision": (precision_options, {
                    "default": "fp32",
                    "tooltip": "Internal tensor precision (fp16 requires CUDA)"
                }),
            },
            "optional": {
                "blur_size": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 25.0,
                    "step": 0.05,
                    "tooltip": "Gaussian sigma when blur_type is gaussian"
                }),
                "motion_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Motion blur direction in degrees"
                }),
                "motion_length": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.5,
                    "max": 150.0,
                    "step": 0.5,
                    "tooltip": "Motion blur distance in pixels"
                }),
                "regularization": ("FLOAT", {
                    "default": 0.0025,
                    "min": 0.0,
                    "max": 0.25,
                    "step": 0.0005,
                    "tooltip": "Laplacian regularization strength"
                }),
                "clip_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp restored image to valid range"
                }),
                "use_preset": (["none"] + list(get_blur_presets().keys()), {
                    "default": "none",
                    "tooltip": "Preset parameter bundles"
                }),
                "estimate_motion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Estimate motion parameters from the image"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore_gpu"

    def restore_gpu(
        self,
        image,
        blur_type="gaussian",
        iterations=15,
        device_preference="auto",
        precision="fp32",
        blur_size=3.0,
        motion_angle=0.0,
        motion_length=15.0,
        regularization=0.0025,
        clip_output=True,
        use_preset="none",
        estimate_motion=False,
    ):
        if use_preset != "none":
            presets = get_blur_presets()
            preset = presets.get(use_preset)
            if preset:
                blur_type = preset.get("blur_type", blur_type)
                blur_size = preset.get("blur_size", blur_size)
                motion_angle = preset.get("motion_angle", motion_angle)
                motion_length = preset.get("motion_length", motion_length)
                iterations = preset.get("iterations", iterations)
                regularization = preset.get("regularization", regularization)

        def process_func(img_np, **_kwargs):
            local_angle = motion_angle
            local_length = motion_length
            if estimate_motion and blur_type == "motion":
                estimated_angle, estimated_length = estimate_motion_blur(img_np)
                print(f"Estimated motion (GPU RL): angle={estimated_angle:.1f}°, length={estimated_length:.1f}px")
                local_angle = estimated_angle
                local_length = estimated_length

            return richardson_lucy_deconvolution_gpu(
                img_np,
                iterations=iterations,
                clip=clip_output,
                blur_type=blur_type,
                blur_size=blur_size,
                motion_angle=local_angle,
                motion_length=local_length,
                regularization=regularization,
                device=device_preference,
                precision=precision,
            )

        print(
            f"Richardson-Lucy GPU: {blur_type} blur, {iterations} iterations, device={device_preference}, precision={precision}"
        )
        result = self.process_image_safe(image, process_func)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "RichardsonLucyGPU": RichardsonLucyGPUNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RichardsonLucyGPU": "Richardson-Lucy Deconvolution GPU (Eric)"
}
