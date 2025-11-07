"""DiffBIR processing helpers for ComfyUI integration.

This module wraps the official DiffBIR reference implementation so that it can be
invoked from a lightweight ComfyUI node without duplicating the training codebase.
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Type

import numpy as np
from PIL import Image

try:  # Optional dependency check
    import torch
except ImportError as exc:  # pragma: no cover - runtime guard
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:  # DiffBIR relies on accelerate for deterministic seeding
    from accelerate.utils import set_seed  # type: ignore[import-error]
except ImportError as exc:  # pragma: no cover - runtime guard
    set_seed = None  # type: ignore
    _ACCELERATE_IMPORT_ERROR = exc
else:
    _ACCELERATE_IMPORT_ERROR = None


# Defaults copied from transformer_lab/third_party/diffbir/inference.py to avoid
# importing face-specific dependencies at module import time.
DEFAULT_POS_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, "
    "skin pore detailing, hyper sharpness, perfect without deformations."
)

DEFAULT_NEG_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
    "signature, jpeg artifacts, deformed, lowres, over-smooth."
)


def _safe_check_device(device: str) -> str:
    """Mirror DiffBIR's device selection without pulling heavy dependencies."""

    if torch is None:
        print("PyTorch not available; defaulting to CPU")
        return "cpu"

    if device == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not built with CUDA enabled."
            )
            device = "cpu"
    elif device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if not mps_backend or not mps_backend.is_available():
            if mps_backend and not mps_backend.is_built():
                print(
                    "MPS not available because the current PyTorch install was not built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or the device does not support MPS."
                )
            device = "cpu"
    else:
        # fall back to cpu for any unexpected value
        device = "cpu"

    print(f"using device {device}")
    return device


ROOT_DIR = Path(__file__).resolve().parent.parent
TRANSFORMER_LAB_DIR = ROOT_DIR / "transformer_lab"
DIFFBIR_REPO_DIR = TRANSFORMER_LAB_DIR / "third_party" / "diffbir"

# Manual weight cache locations (mirrors bootstrap.py search order)
LOCAL_WEIGHT_DIRS: Tuple[Path, ...] = (
    TRANSFORMER_LAB_DIR / "models" / "pretrained_weights",
    TRANSFORMER_LAB_DIR / "models",
    ROOT_DIR / "models" / "pretrained_weights",
    ROOT_DIR / "models",
)

# Files referenced by diffbir/inference/pretrained_models.py
REQUIRED_WEIGHT_FILES: Dict[str, str] = {
    "DiffBIR_v2.1.pt": "DiffBIR v2.1 (IRControlNet)",
    "v2.pth": "DiffBIR v2 (IRControlNet)",
    "v1_general.pth": "DiffBIR v1 general (IRControlNet)",
    "v1_face.pth": "DiffBIR v1 face (IRControlNet)",
    "realesrgan_s4_swinir_100k.pth": "SwinIR RealESRGAN cleaner",
    "general_swinir_v1.ckpt": "SwinIR general cleaner",
    "face_swinir_v1.ckpt": "SwinIR face cleaner",
    "BSRNet.pth": "BSRNet cleaner",
    "scunet_color_real_psnr.pth": "SCUNet cleaner",
    "codeformer_swinir.ckpt": "CodeFormer SwinIR cleaner",
    "v2-1_512-ema-pruned.ckpt": "Stable Diffusion 2.1 base",
    "sd2.1-base-zsnr-laionaes5.ckpt": "Stable Diffusion 2.1 ZSNR variant",
}

# Subset mapping for per-version requirements
VERSION_SPECIFIC_FILES: Dict[str, Iterable[str]] = {
    "v1": (
        "v1_general.pth",
        "v1_face.pth",
        "general_swinir_v1.ckpt",
        "face_swinir_v1.ckpt",
        "v2-1_512-ema-pruned.ckpt",
    ),
    "v2": (
        "v2.pth",
        "bsrnet.pth",
        "scunet_color_real_psnr.pth",
        "realesrgan_s4_swinir_100k.pth",
        "general_swinir_v1.ckpt",
        "codeformer_swinir.ckpt",
        "v2-1_512-ema-pruned.ckpt",
    ),
    "v2.1": (
        "DiffBIR_v2.1.pt",
        "realesrgan_s4_swinir_100k.pth",
        "general_swinir_v1.ckpt",
        "face_swinir_v1.ckpt",
        "scunet_color_real_psnr.pth",
        "codeformer_swinir.ckpt",
        "sd2.1-base-zsnr-laionaes5.ckpt",
    ),
}

# Version mapping above contains a typo (bsrnet casing). Normalise keys.
VERSION_SPECIFIC_FILES["v2"] = tuple(
    "BSRNet.pth" if name.lower() == "bsrnet.pth" else name
    for name in VERSION_SPECIFIC_FILES["v2"]
)

# Additional cleaner dependencies shared across tasks
SHARED_CLEANER_FILES = {
    "BSRNet.pth",
    "scunet_color_real_psnr.pth",
    "realesrgan_s4_swinir_100k.pth",
    "general_swinir_v1.ckpt",
    "face_swinir_v1.ckpt",
    "codeformer_swinir.ckpt",
}


@dataclass
class DiffBIRConfig:
    """Configuration payload controlling the DiffBIR inference call."""

    task: str = "sr"
    version: str = "v2.1"
    sampler: str = "edm_dpm++_3m_sde"
    steps: int = 12
    upscale: float = 4.0
    captioner: str = "none"
    cfg_scale: float = 6.0
    noise_aug: int = 0
    start_point_type: str = "noise"
    rescale_cfg: bool = False
    strength: float = 1.0
    cleaner_tiled: bool = False
    cleaner_tile_size: int = 512
    cleaner_tile_stride: int = 256
    vae_encoder_tiled: bool = False
    vae_encoder_tile_size: int = 256
    vae_decoder_tiled: bool = False
    vae_decoder_tile_size: int = 256
    cldm_tiled: bool = False
    cldm_tile_size: int = 512
    cldm_tile_stride: int = 256
    guidance: bool = False
    guidance_loss: str = "w_mse"
    guidance_scale: float = 0.0
    guidance_start: float = 0.0
    guidance_stop: float = 1.0
    guidance_space: str = "rgb"
    guidance_repeat: int = 1
    batch_size: int = 1
    n_samples: int = 1
    seed: int = 231
    device_preference: str = "auto"
    precision: str = "fp16"
    llava_bit: str = "4"
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = 300.0
    s_noise: float = 1.0
    eta: float = 1.0
    order: int = 1
    pos_prompt: Optional[str] = None
    neg_prompt: Optional[str] = None


class DiffBIRProcessor:
    """Thin wrapper that executes the official DiffBIR inference loops."""

    def __init__(self, config: DiffBIRConfig):
        self.config = config
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError(
                "PyTorch is required for DiffBIR inference"
            ) from _TORCH_IMPORT_ERROR
        if _ACCELERATE_IMPORT_ERROR is not None:
            raise RuntimeError(
                "`accelerate` is required for DiffBIR inference (set_seed)"
            ) from _ACCELERATE_IMPORT_ERROR
        if not DIFFBIR_REPO_DIR.exists():
            raise FileNotFoundError(
                "DiffBIR repository not found. Run `python transformer_lab\\bootstrap_models.py diffbir` first."
            )

        # Ensure the repo root is importable
        if str(DIFFBIR_REPO_DIR) not in sys.path:
            sys.path.insert(0, str(DIFFBIR_REPO_DIR))

        # Copy weights into the expected `weights/` directory inside the repo
        self._prepare_weights()

        # Late imports (after sys.path update)
        from diffbir.inference.bid_loop import BIDInferenceLoop  # type: ignore
        from diffbir.inference.bsr_loop import BSRInferenceLoop  # type: ignore

        self._DEFAULT_POS_PROMPT = DEFAULT_POS_PROMPT
        self._DEFAULT_NEG_PROMPT = DEFAULT_NEG_PROMPT
        self._check_device = _safe_check_device

        self._loops: Dict[str, Optional[Type[Any]]] = {
            "sr": BSRInferenceLoop,
            "denoise": BIDInferenceLoop,
        }

        self._face_dependency_error: Optional[BaseException] = None

        try:
            from diffbir.inference.bfr_loop import BFRInferenceLoop  # type: ignore
        except ModuleNotFoundError as exc:
            if exc.name == "facexlib":
                self._face_dependency_error = exc
                self._loops["face"] = None
            else:
                raise
        else:
            self._loops["face"] = BFRInferenceLoop

        try:
            from diffbir.inference.unaligned_bfr_loop import UnAlignedBFRInferenceLoop  # type: ignore
        except ModuleNotFoundError as exc:
            if exc.name == "facexlib":
                self._face_dependency_error = exc
                self._loops["unaligned_face"] = None
            else:
                raise
        else:
            self._loops["unaligned_face"] = UnAlignedBFRInferenceLoop

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_weights(self) -> None:
        weights_dir = DIFFBIR_REPO_DIR / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        required = set(REQUIRED_WEIGHT_FILES.keys())
        # Include version-specific list if defined
        version = self.config.version
        if version in VERSION_SPECIFIC_FILES:
            required |= set(VERSION_SPECIFIC_FILES[version])
        # cleaners are shared across tasks; keep them available to avoid lazy downloads
        required |= SHARED_CLEANER_FILES

        missing: list[str] = []
        for filename in sorted(required):
            src = self._find_local_weight(filename)
            if src is None:
                missing.append(filename)
                continue
            dst = weights_dir / filename
            try:
                if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(src, dst)
            except PermissionError:
                # Fall back to copying via temporary file if the destination is locked
                tmp_path = dst.with_suffix(dst.suffix + ".tmp")
                shutil.copy2(src, tmp_path)
                tmp_path.replace(dst)
        if missing:
            pretty = ", ".join(missing)
            raise FileNotFoundError(
                "The following DiffBIR weights are missing. Place them under "
                "`transformer_lab/models/pretrained_weights` and retry: " + pretty
            )

    @staticmethod
    def _find_local_weight(filename: str) -> Optional[Path]:
        for directory in LOCAL_WEIGHT_DIRS:
            candidate = directory / filename
            if candidate.exists():
                return candidate
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("DiffBIR expects an HWC image with 3 or 4 channels")

        # Drop alpha channel if present
        if image.shape[2] == 4:
            image = image[..., :3]

        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

        temp_root = TRANSFORMER_LAB_DIR / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="diffbir_", dir=str(temp_root)) as workdir:
            workdir_path = Path(workdir)
            input_dir = workdir_path / "input"
            output_dir = workdir_path / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            input_path = input_dir / "input.png"
            Image.fromarray(image_uint8).save(input_path)

            args = self._build_args(
                input_dir=input_dir,
                output_dir=output_dir,
                file_stem=input_path.stem,
            )

            args.device = self._check_device(args.device)
            if set_seed is not None:
                set_seed(args.seed)

            loop_cls = self._loops.get(args.task)
            if loop_cls is None:
                raise ValueError(f"Unsupported DiffBIR task: {args.task}")

            loop = loop_cls(args)
            loop.run()

            output_file = output_dir / f"{input_path.stem}.png"
            if not output_file.exists():
                raise FileNotFoundError(
                    f"DiffBIR did not produce an output image at {output_file}"
                )

            result = Image.open(output_file).convert("RGB")
            return np.array(result)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _resolve_prompts(self) -> Tuple[str, str]:
        pos = self.config.pos_prompt if self.config.pos_prompt is not None else self._DEFAULT_POS_PROMPT
        neg = self.config.neg_prompt if self.config.neg_prompt is not None else self._DEFAULT_NEG_PROMPT
        return pos, neg

    def _resolve_device(self) -> str:
        pref = (self.config.device_preference or "auto").lower()
        if pref == "auto":
            if torch is not None:
                if torch.cuda.is_available():
                    return "cuda"
                mps_backend = getattr(torch.backends, "mps", None)
                if mps_backend and mps_backend.is_available():
                    return "mps"
            return "cpu"
        return pref

    def _build_args(self, *, input_dir: Path, output_dir: Path, file_stem: str) -> Namespace:
        pos_prompt, neg_prompt = self._resolve_prompts()
        return Namespace(
            task=self.config.task,
            upscale=float(self.config.upscale),
            version=self.config.version,
            train_cfg="",
            ckpt="",
            sampler=self.config.sampler,
            steps=int(self.config.steps),
            start_point_type=self.config.start_point_type,
            cleaner_tiled=bool(self.config.cleaner_tiled),
            cleaner_tile_size=int(self.config.cleaner_tile_size),
            cleaner_tile_stride=int(self.config.cleaner_tile_stride),
            vae_encoder_tiled=bool(self.config.vae_encoder_tiled),
            vae_encoder_tile_size=int(self.config.vae_encoder_tile_size),
            vae_decoder_tiled=bool(self.config.vae_decoder_tiled),
            vae_decoder_tile_size=int(self.config.vae_decoder_tile_size),
            cldm_tiled=bool(self.config.cldm_tiled),
            cldm_tile_size=int(self.config.cldm_tile_size),
            cldm_tile_stride=int(self.config.cldm_tile_stride),
            captioner=self.config.captioner,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            cfg_scale=float(self.config.cfg_scale),
            rescale_cfg=bool(self.config.rescale_cfg),
            noise_aug=int(self.config.noise_aug),
            s_churn=float(self.config.s_churn),
            s_tmin=float(self.config.s_tmin),
            s_tmax=float(self.config.s_tmax),
            s_noise=float(self.config.s_noise),
            eta=float(self.config.eta),
            order=int(self.config.order),
            strength=float(self.config.strength),
            batch_size=int(self.config.batch_size),
            guidance=bool(self.config.guidance),
            g_loss=self.config.guidance_loss,
            g_scale=float(self.config.guidance_scale),
            g_start=float(self.config.guidance_start),
            g_stop=float(self.config.guidance_stop),
            g_space=self.config.guidance_space,
            g_repeat=int(self.config.guidance_repeat),
            input=str(input_dir),
            n_samples=int(self.config.n_samples),
            output=str(output_dir),
            seed=int(self.config.seed),
            device=self._resolve_device(),
            precision=self.config.precision,
            llava_bit=str(self.config.llava_bit),
            file_stem=file_stem,
        )


__all__ = [
    "DiffBIRConfig",
    "DiffBIRProcessor",
]
