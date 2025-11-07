"""FastAPI wrapper around DeepInv models for use with ComfyUI nodes.

This service is designed to run inside the dedicated DeepInv virtual
environment so that heavy dependencies stay isolated from the main
ComfyUI runtime. Nodes communicate with this service over HTTP by
posting JSON payloads containing disk paths to the tensors they want to
process. The service responds with status metadata after writing the
results back to disk.
"""
from __future__ import annotations

import gc
import inspect
from contextlib import suppress
import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import deepinv as dinv

LOGGER = logging.getLogger("deepinv-service")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_SEARCH_DIRS: Tuple[Path, ...] = tuple(
    path
    for path in (
        _PROJECT_ROOT / "models" / "pretrained_weights",
        _PROJECT_ROOT / "models",
        Path(os.environ.get("DEEPINV_PRETRAINED_DIR", "")).expanduser() if os.environ.get("DEEPINV_PRETRAINED_DIR") else None,
    )
    if path is not None
)


def _locate_pretrained(filename: str) -> str | None:
    for directory in _MODEL_SEARCH_DIRS:
        try:
            candidate = (directory / filename).resolve()
        except FileNotFoundError:  # pragma: no cover - network shares may error on resolve
            continue
        if candidate.exists():
            return str(candidate)
    return None


def _diffunet_kwargs(filename: str, *, large_model: bool = False) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if large_model:
        kwargs["large_model"] = True
    weight_path = _locate_pretrained(filename)
    kwargs["pretrained"] = weight_path or "download"
    return kwargs


def _ram_kwargs(filename: str) -> Dict[str, Any]:
    weight_path = _locate_pretrained(filename)
    return {"pretrained": weight_path or True}


SUPPORTED_DENOISERS: Dict[str, Tuple[type[torch.nn.Module], Dict[str, Any]]] = {
    "drunet": (dinv.models.DRUNet, {"pretrained": "download"}),
    "dncnn_light": (dinv.models.DnCNN, {"pretrained": "download"}),
    "dncnn_lipschitz": (dinv.models.DnCNN, {"pretrained": "download_lipschitz"}),
    "diffunet_ffhq": (dinv.models.DiffUNet, _diffunet_kwargs("diffusion_ffhq_10m.pt")),
    "diffunet_imagenet": (dinv.models.DiffUNet, _diffunet_kwargs("diffusion_openai.pt", large_model=True)),
    "ram_denoise": (dinv.models.RAM, _ram_kwargs("ram.pth")),
    "scunet_real_psnr": (dinv.models.SCUNet, {"pretrained": "download"}),
    "swinir_noise15": (dinv.models.SwinIR, {"pretrained": "download", "pretrained_noise_level": 15}),
    "swinir_noise25": (dinv.models.SwinIR, {"pretrained": "download", "pretrained_noise_level": 25}),
    "swinir_noise50": (dinv.models.SwinIR, {"pretrained": "download", "pretrained_noise_level": 50}),
}


def _resolve_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        try:
            device_name = dinv.utils.get_freer_gpu()
            return torch.device(device_name)
        except Exception:  # pragma: no cover - fallback when helper fails
            return torch.device("cuda")
    return torch.device("cpu")


def _empty_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            with suppress(Exception):  # pragma: no cover - ipc_collect optional
                torch.cuda.ipc_collect()
    gc.collect()


def _is_indexing_error(exc: RuntimeError) -> bool:
    return "canUse32BitIndexMath" in str(exc)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc)
    return "CUDA out of memory" in message or "CUDA error: out of memory" in message


class DenoiseRequest(BaseModel):
    input_path: Path = Field(..., description="Path to .npy file containing NHWC float32 tensor in [0,1].")
    output_path: Path = Field(..., description="Destination .npy file that will receive NHWC float32 tensor in [0,1].")
    model: Literal[
        "drunet",
        "dncnn_light",
        "dncnn_lipschitz",
        "diffunet_ffhq",
        "diffunet_imagenet",
        "ram_denoise",
        "scunet_real_psnr",
        "swinir_noise15",
        "swinir_noise25",
        "swinir_noise50",
    ] = Field("drunet", description="DeepInv denoiser preset key")
    sigma: float = Field(0.1, ge=0.0, le=1.0, description="Noise level, typically standard deviation in [0,1].")
    prefer_gpu: bool = Field(True, description="Attempt to run on CUDA if available.")

    @field_validator("input_path", "output_path", mode="before")
    def _expand(cls, value: str | os.PathLike[str]) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("input_path")
    def _ensure_exists(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Input path does not exist: {value}")
        return value


class DenoiseResponse(BaseModel):
    model: str
    device: str
    sigma: float
    output_path: Path
    stats: Dict[str, float]


class DeepInvModelCache:
    def __init__(self) -> None:
        self._cache: Dict[str, torch.nn.Module] = {}

    def get_denoiser(self, key: str, device: torch.device) -> torch.nn.Module:
        # Cache models per device so multi-GPU systems keep isolated copies.
        device_suffix = "cpu" if device.type == "cpu" else device.index if device.index is not None else 0
        cache_key = f"{key}-{device.type}-{device_suffix}"
        if cache_key not in self._cache:
            model_cls, default_kwargs = SUPPORTED_DENOISERS[key]
            init_kwargs = dict(default_kwargs)
            LOGGER.info("Loading DeepInv model %s on %s", model_cls.__name__, device)

            signature = inspect.signature(model_cls.__init__)
            if "device" in signature.parameters:
                init_kwargs.setdefault("device", device)

            try:
                model = model_cls(**init_kwargs)
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    LOGGER.warning(
                        "Failed to load %s on %s due to OOM; clearing cache and re-raising", model_cls.__name__, device
                    )
                    _empty_cuda(device)
                raise
            if "device" not in signature.parameters:
                model.to(device)
            model.eval()
            self._cache[cache_key] = model
        return self._cache[cache_key]

    def clear(self) -> None:
        for model in self._cache.values():
            try:
                device = next(model.parameters()).device  # type: ignore[arg-type]
            except StopIteration:
                continue
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    with suppress(Exception):
                        torch.cuda.ipc_collect()
        self._cache.clear()


app = FastAPI(title="DeepInv Service", version="0.1.0")
MODEL_CACHE = DeepInvModelCache()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/cache/clear")
def cache_clear() -> Dict[str, str]:
    MODEL_CACHE.clear()
    return {"status": "cleared"}


@app.post("/denoise", response_model=DenoiseResponse)
def denoise(req: DenoiseRequest) -> DenoiseResponse:
    device = _resolve_device(req.prefer_gpu)
    try:
        model = MODEL_CACHE.get_denoiser(req.model, device)
    except RuntimeError as exc:
        if not (_is_cuda_oom(exc) and device.type == "cuda"):
            raise
        LOGGER.warning(
            "Unable to load model %s on %s due to OOM, retrying on CPU", req.model, device
        )
        _empty_cuda(device)
        device = torch.device("cpu")
        model = MODEL_CACHE.get_denoiser(req.model, device)

    LOGGER.info("Denoising %s with %s (sigma=%s) on %s", req.input_path, req.model, req.sigma, device)

    image = np.load(req.input_path, allow_pickle=False)
    if image.ndim == 3:
        image = image[None, ...]
    if image.ndim != 4:
        raise HTTPException(status_code=400, detail="Expected NHWC tensor with optional batch dimension.")

    image = np.array(image, dtype=np.float32, copy=True)
    input_reference = image.copy()
    tensor = torch.from_numpy(image).permute(0, 3, 1, 2).contiguous()
    if device.type == "cuda":
        tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        tensor = tensor.to(device=device, dtype=torch.float32)

    def _run_model(current_model: torch.nn.Module, current_tensor: torch.Tensor, target_sigma: float) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(current_model, dinv.models.DiffUNet):
                original_height = current_tensor.shape[2]
                original_width = current_tensor.shape[3]
                pad_h = (32 - original_height % 32) % 32
                pad_w = (32 - original_width % 32) % 32
                padded_tensor = current_tensor
                if pad_h or pad_w:
                    padded_tensor = F.pad(current_tensor, (0, pad_w, 0, pad_h), mode="reflect")
                sigma_tensor = torch.full(
                    (padded_tensor.shape[0], 1, 1, 1),
                    float(target_sigma),
                    dtype=padded_tensor.dtype,
                    device=padded_tensor.device,
                )
                denoised = current_model.forward_denoise(padded_tensor, sigma_tensor)
                if pad_h or pad_w:
                    denoised = denoised[..., :original_height, :original_width]
                return denoised
            if isinstance(current_model, dinv.models.RAM):
                sigma_value = float(max(target_sigma, 0.0))
                return current_model(current_tensor, sigma=sigma_value, gain=0.0)
            return current_model(current_tensor, sigma=target_sigma)

    try:
        output = _run_model(model, tensor, req.sigma)
    except RuntimeError as exc:
        if not (_is_indexing_error(exc) or _is_cuda_oom(exc)):
            raise
        fallback_reason = "indexing" if _is_indexing_error(exc) else "oom"
        LOGGER.warning(
            "Model %s encountered %s issue on %s, retrying on CPU", req.model, fallback_reason, device
        )
        tensor_cpu = tensor.detach().cpu()
        if tensor.device.type == "cuda":
            del tensor
            _empty_cuda(device)
        device = torch.device("cpu")
        model = MODEL_CACHE.get_denoiser(req.model, device)
        output = _run_model(model, tensor_cpu, req.sigma)
        tensor = tensor_cpu

    output_cpu = output.detach().cpu()
    if output.device.type == "cuda":
        output_device = output.device
        del output
        _empty_cuda(output_device)

    output_nhwc = output_cpu.permute(0, 2, 3, 1).contiguous().numpy()
    output_nhwc = np.clip(output_nhwc, 0.0, 1.0)
    output_nhwc = np.ascontiguousarray(output_nhwc)

    req.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(req.output_path, output_nhwc, allow_pickle=False)

    reference = input_reference[
        : output_nhwc.shape[0],
        : output_nhwc.shape[1],
        : output_nhwc.shape[2],
        : output_nhwc.shape[3],
    ]
    delta = np.abs(output_nhwc - reference)

    stats = {
        "min": float(output_nhwc.min()),
        "max": float(output_nhwc.max()),
        "mean": float(output_nhwc.mean()),
        "delta_max": float(delta.max()),
        "delta_mean": float(delta.mean()),
    }

    tensor_device = tensor.device
    del tensor
    if tensor_device.type == "cuda":
        _empty_cuda(tensor_device)

    return DenoiseResponse(
        model=req.model,
        device=str(device),
        sigma=req.sigma,
        output_path=req.output_path,
        stats=stats,
    )


def main() -> None:
    import uvicorn

    port = int(os.environ.get("DEEPINV_SERVICE_PORT", "6112"))
    LOGGER.info("Starting DeepInv service on port %d", port)
    uvicorn.run("service:app", host="127.0.0.1", port=port, factory=False)


if __name__ == "__main__":
    main()
