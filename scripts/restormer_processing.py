"""
Restormer transformer-based restoration processor with tiling and sharpness controls.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from ..network_restormer import Restormer
except ImportError:  # pragma: no cover - allow standalone execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from network_restormer import Restormer


@dataclass(frozen=True)
class ModelConfig:
    identifier: str
    scale: int
    filename_candidates: Tuple[str, ...]
    download_url: Optional[str]
    description: str


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "real_sr_x4": ModelConfig(
        identifier="real_sr_x4",
        scale=4,
        filename_candidates=(
            "Restormer_RealSR_DFOWM_x4.pth",
            "Restormer_RealSR_DFOWM_x4_GAN.pth",
            "Restormer_RealSR_DFOWM_x4_PSNR.pth",
        ),
        download_url="https://github.com/swz30/Restormer/releases/download/v1.0/Restormer_RealSR_DFOWM_x4.pth",
        description="Real-world super-resolution x4 (DFOWM dataset)",
    ),
    "denoise_sigma15": ModelConfig(
        identifier="denoise_sigma15",
        scale=1,
        filename_candidates=(
            "Restormer_Denoising_Color_sigma15.pth",
            "Restormer_GaussianColorSigma15.pth",
            "gaussian_color_denoising_sigma15.pth",
        ),
        download_url="https://github.com/swz30/Restormer/releases/download/v1.0/Restormer_Denoising_Color_sigma15.pth",
        description="Gaussian color denoising σ=15",
    ),
    "denoise_sigma25": ModelConfig(
        identifier="denoise_sigma25",
        scale=1,
        filename_candidates=(
            "Restormer_Denoising_Color_sigma25.pth",
            "Restormer_GaussianColorSigma25.pth",
            "gaussian_color_denoising_sigma25.pth",
        ),
        download_url="https://github.com/swz30/Restormer/releases/download/v1.0/Restormer_Denoising_Color_sigma25.pth",
        description="Gaussian color denoising σ=25",
    ),
    "denoise_sigma50": ModelConfig(
        identifier="denoise_sigma50",
        scale=1,
        filename_candidates=(
            "Restormer_Denoising_Color_sigma50.pth",
            "Restormer_GaussianColorSigma50.pth",
            "gaussian_color_denoising_sigma50.pth",
        ),
        download_url="https://github.com/swz30/Restormer/releases/download/v1.0/Restormer_Denoising_Color_sigma50.pth",
        description="Gaussian color denoising σ=50",
    ),
    "real_denoise": ModelConfig(
        identifier="real_denoise",
        scale=1,
        filename_candidates=(
            "Restormer_RealDenoising.pth",
            "Restormer_SIDD.pth",
            "real_denoising.pth",
            "restormer_real_denoising.pth",
            "gaussian_color_denoising_blind.pth",
        ),
        download_url="https://github.com/swz30/Restormer/releases/download/v1.0/Restormer_RealDenoising.pth",
        description="Real image denoising (SIDD)",
    ),
    "motion_deblur": ModelConfig(
        identifier="motion_deblur",
        scale=1,
        filename_candidates=(
            "Restormer_MotionDeblurring.pth",
            "Restormer_GoPro.pth",
        ),
        download_url="https://github.com/swz30/Restormer/releases/download/v1.0/Restormer_MotionDeblurring.pth",
        description="Motion deblurring (GoPro)",
    ),
    "defocus_deblur": ModelConfig(
        identifier="defocus_deblur",
        scale=1,
        filename_candidates=(
            "Restormer_DefocusDeblurring_x4.pth",
            "Restormer_DPDD.pth",
            "dual_pixel_defocus_deblurring.pth",
            "single_image_defocus_deblurring.pth",
        ),
        download_url="https://github.com/swz30/Restormer/releases/download/v1.0/Restormer_DefocusDeblurring_x4.pth",
        description="Defocus deblurring (DPDD)",
    ),
    "derain_indoor": ModelConfig(
        identifier="derain_indoor",
        scale=1,
        filename_candidates=(
            "Restormer_Deraining_Indoor.pth",
            "Restormer_Rain100H.pth",
            "deraining.pth",
        ),
        download_url="https://github.com/swz30/Restormer/releases/download/v1.0/Restormer_Deraining_Indoor.pth",
        description="Deraining for heavy indoor streaks",
    ),
    "derain_outdoor": ModelConfig(
        identifier="derain_outdoor",
        scale=1,
        filename_candidates=(
            "Restormer_Deraining_Outdoor.pth",
            "Restormer_SPA-Data.pth",
            "deraining.pth",
        ),
        download_url="https://github.com/swz30/Restormer/releases/download/v1.0/Restormer_Deraining_Outdoor.pth",
        description="Deraining for outdoor scenes (SPA-Data)",
    ),
    "dehaze": ModelConfig(
        identifier="dehaze",
        scale=1,
        filename_candidates=(
            "Restormer_Dehazing.pth",
            "Restormer_ITS.pth",
        ),
        download_url="https://github.com/swz30/Restormer/releases/download/v1.0/Restormer_Dehazing.pth",
        description="Dehazing (ITS dataset)",
    ),
}


class RestormerProcessor:
    """Wrapper around Restormer with tiling support and detail boosting."""

    def __init__(
        self,
        task: str = "real_sr_x4",
        device: str = "auto",
        models_dir: Optional[str] = None,
        tile_size: Optional[int] = None,
        tile_overlap: int = 48,
    ) -> None:
        if task not in MODEL_CONFIGS:
            raise ValueError(f"Unknown Restormer task: {task}")

        self.config = MODEL_CONFIGS[task]
        self.scale = self.config.scale
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = device

        default_model_dir = Path(__file__).resolve().parent.parent / "models"
        self.models_dir = Path(models_dir).expanduser().resolve() if models_dir else default_model_dir
        # Cache candidate directories and lazy-load model weights on this instance.
        self._weight_search_dirs = self._build_weight_search_dirs(self.models_dir)
        self._weights_loaded = False
        self._last_weight_error: Optional[str] = None
        self.model = self._load_model().to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _build_weight_search_dirs(self, base_dir: Path) -> Tuple[Path, ...]:
        """Construct candidate directories for locating Restormer weights."""
        repo_root = Path(__file__).resolve().parent.parent
        candidates = [base_dir]

        # Support nested pretrained_weights directory alongside base dir
        if base_dir.name != "pretrained_weights":
            candidates.append(base_dir / "pretrained_weights")

        # Fall back to project-level models directories
        repo_models = repo_root / "models"
        if repo_models not in candidates:
            candidates.append(repo_models)
        repo_models_pretrained = repo_models / "pretrained_weights"
        if repo_models_pretrained not in candidates:
            candidates.append(repo_models_pretrained)

        # Remove duplicates while preserving order
        unique_candidates = []
        for path in candidates:
            resolved = path.resolve()
            if resolved not in unique_candidates:
                unique_candidates.append(resolved)
        return tuple(unique_candidates)

    def _load_model(self) -> Restormer:
        model = Restormer(inp_channels=3, out_channels=3)
        weight_path = self._resolve_weight_path()

        if weight_path is None:
            message = (
                f"Restormer weights for task '{self.config.identifier}' were not found.\n"
                f"Expected one of: {self.config.filename_candidates}.\n"
                "Place the file in one of the following directories: \n"
                + "\n".join(f"  • {path}" for path in self._weight_search_dirs)
                + "\n"
                f"Suggested download URL: {self.config.download_url}"
            )
            self._last_weight_error = message
            self._weights_loaded = False
            print(message)
            return model

        try:
            checkpoint = torch.load(weight_path, map_location="cpu")
            state_dict = self._extract_state_dict(checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warning: missing parameters while loading Restormer weights: {missing}")
            if unexpected:
                print(f"Warning: unexpected parameters while loading Restormer weights: {unexpected}")
            print(f"Loaded Restormer weights from {weight_path}")
            self._weights_loaded = True
            self._last_weight_error = None
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"Failed to load Restormer weights: {exc}")
            import traceback

            traceback.print_exc()
            self._last_weight_error = str(exc)
            self._weights_loaded = False
        return model

    def _resolve_weight_path(self) -> Optional[str]:
        for directory in self._weight_search_dirs:
            if not directory.exists():
                continue
            for candidate in self.config.filename_candidates:
                candidate_path = directory / candidate
                if candidate_path.exists():
                    return str(candidate_path)
        return None

    @staticmethod
    def _extract_state_dict(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("params", "state_dict", "model", "net_g"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        return checkpoint

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process_image(
        self,
        image: np.ndarray,
        strength: float = 1.0,
        detail_boost: float = 0.0,
        detail_radius: float = 1.2,
    ) -> np.ndarray:
        if not self._weights_loaded:
            error_message = self._last_weight_error or "Restormer weights are not loaded."
            raise RuntimeError(error_message)

        # Support optional batch dimension
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("Restormer processor expects a single image (batch size 1)")
            image = image[0]

        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("Restormer expects RGB images with 3 channels")

        # Drop alpha channel if present
        if image.shape[2] == 4:
            image = image[..., :3]

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

        original = image.copy()
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.tile_size is None:
                output = self._forward(tensor)
            else:
                output = self._forward_with_tiling(tensor)

        result = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        result = np.clip(result, 0.0, 1.0)

        if self.scale > 1:
            # Upscale original for blending when performing super-resolution
            from scipy.ndimage import zoom

            upscaled = zoom(original, (self.scale, self.scale, 1), order=3)
            original = upscaled

        result = original * (1.0 - strength) + result * strength

        if detail_boost > 0:
            detail_boost = float(np.clip(detail_boost, 0.0, 1.0))
            sigma = max(0.1, float(detail_radius))
            base = np.ascontiguousarray(result)
            blurred = cv2.GaussianBlur(base, (0, 0), sigma)
            blurred = np.ascontiguousarray(blurred)
            detail = np.ascontiguousarray(base - blurred)
            result = np.ascontiguousarray(base + detail_boost * detail)

        result = np.clip(result, 0.0, 1.0)
        result = np.ascontiguousarray(result)
        return result

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)

    def _forward_with_tiling(self, tensor: torch.Tensor) -> torch.Tensor:
        tile = int(self.tile_size)
        overlap = int(self.tile_overlap)
        if tile <= overlap:
            raise ValueError("tile_size must be larger than tile_overlap")

        tensor, pad_h, pad_w = self._pad_to_tile(tensor, tile)
        b, c, h, w = tensor.shape
        stride = tile - overlap

        h_steps = self._compute_steps(h, tile, stride)
        w_steps = self._compute_steps(w, tile, stride)

        if self.scale > 1:
            out_shape = (b, c, h * self.scale, w * self.scale)
        else:
            out_shape = (b, c, h, w)

        dtype = tensor.dtype
        output_accum = torch.zeros(out_shape, device=self.device, dtype=dtype)
        weight_accum = torch.zeros(out_shape, device=self.device, dtype=dtype)

        for y in h_steps:
            for x in w_steps:
                patch = tensor[:, :, y : y + tile, x : x + tile]
                patch_out = self._forward(patch)

                if self.scale > 1:
                    ys = y * self.scale
                    xs = x * self.scale
                    ye = ys + patch_out.shape[2]
                    xe = xs + patch_out.shape[3]
                else:
                    ys, ye = y, y + patch_out.shape[2]
                    xs, xe = x, x + patch_out.shape[3]

                output_accum[:, :, ys:ye, xs:xe] += patch_out
                weight_accum[:, :, ys:ye, xs:xe] += 1.0

        output = output_accum / torch.clamp(weight_accum, min=1e-6)

        if pad_h or pad_w:
            if self.scale > 1:
                cut_h = (tensor.shape[2] - pad_h) * self.scale
                cut_w = (tensor.shape[3] - pad_w) * self.scale
            else:
                cut_h = tensor.shape[2] - pad_h
                cut_w = tensor.shape[3] - pad_w
            output = output[:, :, :cut_h, :cut_w]
        return output

    @staticmethod
    def _compute_steps(length: int, tile: int, stride: int) -> Iterable[int]:
        positions: List[int] = []
        pos = 0
        while True:
            positions.append(pos)
            if pos + tile >= length:
                break
            pos += stride
            if pos + tile > length:
                pos = length - tile
        return positions

    @staticmethod
    def _pad_to_tile(tensor: torch.Tensor, tile: int) -> Tuple[torch.Tensor, int, int]:
        b, c, h, w = tensor.shape
        pad_h = (tile - h % tile) % tile
        pad_w = (tile - w % tile) % tile
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        return tensor, pad_h, pad_w

    # ------------------------------------------------------------------
    # Convenience interface
    # ------------------------------------------------------------------
    def __call__(
        self,
        image: np.ndarray,
        strength: float = 1.0,
        detail_boost: float = 0.0,
        detail_radius: float = 1.2,
    ) -> np.ndarray:
        return self.process_image(
            image,
            strength=strength,
            detail_boost=detail_boost,
            detail_radius=detail_radius,
        )


__all__ = ["RestormerProcessor", "MODEL_CONFIGS"]
