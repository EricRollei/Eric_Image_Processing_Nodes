"""Torch-accelerated Richardson-Lucy deconvolution."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:  # Optional dependency
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - runtime guard
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _select_device(device_preference: str) -> "torch.device":
    if torch is None:  # pragma: no cover - guard for static analyzers
        raise RuntimeError("PyTorch is required for GPU Richardson-Lucy") from _TORCH_IMPORT_ERROR

    pref = (device_preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(pref)
    except (TypeError, ValueError):
        print(f"Unknown device '{device_preference}', falling back to CPU")
        return torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if not (mps_backend and mps_backend.is_available()):
            print("MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
    return device


def _generate_psf_tensor(
    blur_type: str,
    blur_size: float,
    motion_angle: float,
    motion_length: float,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    blur_type = blur_type.lower()
    if blur_type == "gaussian":
        sigma = max(blur_size, 1e-3)
        size = int(2 * math.ceil(3 * sigma) + 1)
        size = max(size, 3)
        coords = torch.arange(size, device=device, dtype=dtype)
        coords = coords - coords.mean()
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        psf = torch.exp(-((xx ** 2 + yy ** 2) / (2 * sigma ** 2)))
    elif blur_type == "motion":
        length = max(1, int(round(motion_length)))
        size = max(length, 3)
        if size % 2 == 0:
            size += 1
        psf = torch.zeros((size, size), device=device, dtype=dtype)
        center = size // 2
        angle_rad = math.radians(motion_angle)
        dx = math.cos(angle_rad) * (length - 1) / 2.0
        dy = math.sin(angle_rad) * (length - 1) / 2.0
        x0, y0 = center - dx, center - dy
        x1, y1 = center + dx, center + dy
        for step in range(length):
            t = step / max(length - 1, 1)
            x = round(x0 + (x1 - x0) * t)
            y = round(y0 + (y1 - y0) * t)
            if 0 <= x < size and 0 <= y < size:
                psf[y, x] = 1.0
    else:
        raise ValueError(f"Unsupported blur type '{blur_type}' for GPU Richardson-Lucy")

    psf_sum = psf.sum()
    if psf_sum <= 0:
        psf.fill_(0)
        psf[psf.shape[0] // 2, psf.shape[1] // 2] = 1.0
        psf_sum = psf.sum()
    return psf / psf_sum


def _depthwise_conv(image: "torch.Tensor", kernel: "torch.Tensor") -> "torch.Tensor":
    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2
    padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    return F.conv2d(padded, kernel, groups=image.shape[1])


def _apply_laplacian_regularization(
    image: "torch.Tensor", strength: float, epsilon: float
) -> "torch.Tensor":
    if strength <= 0:
        return image
    laplacian_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    laplacian_kernel = laplacian_kernel.repeat(image.shape[1], 1, 1, 1)
    laplacian = _depthwise_conv(image, laplacian_kernel)
    return torch.clamp(image - strength * laplacian, min=epsilon)


def richardson_lucy_deconvolution_gpu(
    image: np.ndarray,
    psf: Optional[np.ndarray] = None,
    iterations: int = 10,
    clip: bool = True,
    filter_epsilon: Optional[float] = None,
    blur_type: str = "gaussian",
    blur_size: float = 2.0,
    motion_angle: float = 0.0,
    motion_length: float = 10.0,
    regularization: float = 0.0,
    device: str = "auto",
    precision: str = "fp32",
) -> np.ndarray:
    """Accelerated Richardson-Lucy deconvolution using PyTorch."""

    if torch is None:
        raise RuntimeError("PyTorch is required for GPU Richardson-Lucy") from _TORCH_IMPORT_ERROR

    if image.ndim not in (2, 3):
        raise ValueError("Image must be 2D or 3D array")
    if iterations < 1:
        raise ValueError("Iterations must be >= 1")

    image = np.ascontiguousarray(image)

    alpha: Optional[np.ndarray] = None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[..., 3:]
        image = image[..., :3]

    input_float = image.astype(np.float32) / 255.0
    if input_float.ndim == 2:
        input_tensor = torch.from_numpy(input_float).unsqueeze(0).unsqueeze(0)
    else:
        input_tensor = torch.from_numpy(input_float).permute(2, 0, 1).unsqueeze(0)

    target_device = _select_device(device)
    compute_dtype = torch.float16 if precision == "fp16" and target_device.type == "cuda" else torch.float32
    if compute_dtype == torch.float16 and target_device.type != "cuda":
        compute_dtype = torch.float32

    observed = input_tensor.to(device=target_device, dtype=compute_dtype)
    estimate = observed.clone()

    eps = filter_epsilon if filter_epsilon is not None else torch.finfo(observed.dtype).eps

    if psf is not None:
        if psf.ndim != 2:
            raise ValueError("Custom PSF must be 2D")
        psf_tensor = torch.from_numpy(np.ascontiguousarray(psf.astype(np.float32)))
        psf_tensor = psf_tensor.to(device=target_device, dtype=compute_dtype)
        psf_tensor = psf_tensor / psf_tensor.sum().clamp(min=eps)
    else:
        psf_tensor = _generate_psf_tensor(
            blur_type, blur_size, motion_angle, motion_length, target_device, compute_dtype
        )

    kernel = psf_tensor.view(1, 1, psf_tensor.shape[0], psf_tensor.shape[1])
    kernel = kernel.repeat(observed.shape[1], 1, 1, 1)
    kernel_flipped = torch.flip(kernel, dims=(-2, -1))

    for _ in range(iterations):
        convolved = _depthwise_conv(estimate, kernel)
        convolved = torch.clamp(convolved, min=eps)
        ratio = observed / convolved
        correlation = _depthwise_conv(ratio, kernel_flipped)
        estimate = estimate * correlation
        estimate = _apply_laplacian_regularization(estimate, regularization, eps)
        estimate = torch.clamp(estimate, min=eps)

    if clip:
        estimate = torch.clamp(estimate, 0.0, 1.0)

    estimate = estimate.squeeze(0)
    if estimate.ndim == 3:
        result = estimate.permute(1, 2, 0).to(torch.float32).cpu().numpy()
        result = np.clip(result * 255.0 + 0.5, 0, 255).astype(np.uint8)
        if alpha is not None:
            result = np.concatenate([result, alpha], axis=2)
    else:
        result = estimate.to(torch.float32).cpu().numpy()
        result = np.clip(result * 255.0 + 0.5, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(result)


__all__ = ["richardson_lucy_deconvolution_gpu"]
