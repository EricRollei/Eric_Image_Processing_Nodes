"""ComfyUI node wrapping the external DeepInv denoising service."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from ..base_node import BaseImageProcessingNode
from ..scripts.deepinv_client import (
    DeepInvServiceError,
    DeepInvServiceClient,
    get_client,
    load_tensor,
    save_tensor,
)


class DeepInvDenoiseNode(BaseImageProcessingNode):
    """Denoise images using DeepInv pretrained models via the helper service."""

    CATEGORY = "Eric's Image Processing/DeepInv"

    _MODEL_PRESETS = [
        (
            "DRUNet – general purpose",
            "drunet",
            "Balanced denoiser. Works well around σ ≈ 0.08 for 1 MP medium noise; slider 0.04–0.15 useful.",
        ),
        (
            "DnCNN – light clean up",
            "dncnn_light",
            "Residual clean-up for faint noise (trained around σ ≈ 0.02). Expect subtle changes only.",
        ),
        (
            "DnCNN – Lipschitz variant",
            "dncnn_lipschitz",
            "Stability-focused weights with tighter Lipschitz constraint. Better for moderate noise with fewer artifacts.",
        ),
        (
            "DiffUNet – FFHQ diffusion",
            "diffunet_ffhq",
            "Diffusion UNet trained on FFHQ faces. Strong restoration for portraits; try σ ≈ 0.05–0.12.",
        ),
        (
            "DiffUNet – ImageNet large",
            "diffunet_imagenet",
            "Large DiffUNet (ImageNet). Sharper detail but heavy VRAM usage; works best around σ ≈ 0.05–0.10.",
        ),
        (
            "RAM – foundation denoiser",
            "ram_denoise",
            "Reconstruct Anything Model. Handles mixed noise; respects σ input and auto-manages Poisson gain.",
        ),
        (
            "SCUNet – real image",
            "scunet_real_psnr",
            "Real-image denoiser. Sigma input ignored by model (preset weights).",
        ),
        (
            "SwinIR – noise level 15",
            "swinir_noise15",
            "Transformer denoiser trained for σ ≈ 0.06 (noise 15/255). Good for mild color noise.",
        ),
        (
            "SwinIR – noise level 25",
            "swinir_noise25",
            "Same architecture with noise 25 weights (σ ≈ 0.10). Use for stronger noise.",
        ),
        (
            "SwinIR – noise level 50",
            "swinir_noise50",
            "Noisiest SwinIR preset (σ ≈ 0.20). Useful for heavy ISO grain, expect detail smoothing.",
        ),
    ]

    _DISPLAY_TO_KEY = {display: key for display, key, _ in _MODEL_PRESETS}
    _DISPLAY_TO_NOTE = {display: note for display, _, note in _MODEL_PRESETS}
    _KEY_TO_DISPLAY = {key: display for display, key, _ in _MODEL_PRESETS}
    _LEGACY_MODEL_KEY_ALIASES = {
        "dncnn": "dncnn_light",
        "dncnn_light": "dncnn_light",
        "dncnn_lipschitz": "dncnn_lipschitz",
        "drunet": "drunet",
        "scunet": "scunet_real_psnr",
        "scunet_real": "scunet_real_psnr",
        "scunet_real_psnr": "scunet_real_psnr",
        "swinir": "swinir_noise25",
        "swinir_noise15": "swinir_noise15",
        "swinir_noise25": "swinir_noise25",
        "swinir_noise50": "swinir_noise50",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    [display for display, _, _ in cls._MODEL_PRESETS],
                    {"default": cls._MODEL_PRESETS[0][0]},
                ),
                "sigma": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "prefer_gpu": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "service_url": ("STRING", {"default": os.environ.get("DEEPINV_SERVICE_URL", "http://127.0.0.1:6112")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "denoise"

    def denoise(
        self,
        image: torch.Tensor,
        model: str = "drunet",
        sigma: float = 0.05,
        prefer_gpu: bool = True,
        service_url: str | None = None,
    ) -> Tuple[torch.Tensor, str]:
        self.validate_image_tensor(image)

        if sigma < 0.0:
            sigma = 0.0
        if sigma > 1.0:
            sigma = 1.0

        display_value = model
        preset_key = self._DISPLAY_TO_KEY.get(display_value)
        if preset_key is None:
            if model in self._KEY_TO_DISPLAY:
                preset_key = model
                display_value = self._KEY_TO_DISPLAY[model]
            else:
                alias_key = self._LEGACY_MODEL_KEY_ALIASES.get(model)
                if alias_key and alias_key in self._KEY_TO_DISPLAY:
                    preset_key = alias_key
                    display_value = self._KEY_TO_DISPLAY[alias_key]
                else:
                    raise ValueError(f"Unknown DeepInv preset: {model}")

        batch = image.detach().cpu().numpy().astype(np.float32)
        batch = np.ascontiguousarray(batch)

        info_lines = [
            f"DeepInv preset: {display_value}",
            f"Sigma: {sigma:.4f}",
            f"Prefer GPU: {'yes' if prefer_gpu else 'no'}",
        ]
        info_lines.append("Preset note:")
        info_lines.append(f"  {self._DISPLAY_TO_NOTE.get(display_value, 'No preset note available.')}")
        info_lines.append("Other presets:")
        for display, _, note in self._MODEL_PRESETS:
            if display == display_value:
                continue
            info_lines.append(f"  {display}: {note}")

        with tempfile.TemporaryDirectory(prefix="deepinv_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input.npy"
            output_path = tmpdir_path / "output.npy"

            save_tensor(input_path, batch)

            client = self._resolve_client(service_url)

            try:
                response = client.denoise(
                    input_path=input_path,
                    output_path=output_path,
                    model=preset_key,
                    sigma=float(sigma),
                    prefer_gpu=prefer_gpu,
                )
            except DeepInvServiceError as exc:
                info_lines.append(f"Error: {exc}")
                self.cleanup_memory()
                return image, "\n".join(info_lines)

            if not output_path.exists():
                info_lines.append("Error: DeepInv service did not produce output file")
                self.cleanup_memory()
                return image, "\n".join(info_lines)

            output = load_tensor(output_path)

        output_tensor = torch.from_numpy(output.astype(np.float32))
        if output_tensor.ndim == 3:
            output_tensor = output_tensor.unsqueeze(0)

        info_lines.append(f"Device: {response.get('device', 'unknown')}")
        stats = response.get("stats", {})
        if stats:
            info_lines.append("Output stats:")
            info_lines.append(f"  min: {stats.get('min', 0.0):.4f}")
            info_lines.append(f"  max: {stats.get('max', 0.0):.4f}")
            info_lines.append(f"  mean: {stats.get('mean', 0.0):.4f}")

        self.cleanup_memory()
        return output_tensor, "\n".join(info_lines)

    def _resolve_client(self, service_url: str | None) -> DeepInvServiceClient:
        if service_url:
            return DeepInvServiceClient(base_url=service_url)
        return get_client()