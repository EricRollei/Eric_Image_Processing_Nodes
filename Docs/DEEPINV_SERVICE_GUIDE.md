# DeepInv Service Setup & Usage Guide

This guide explains how to run the external DeepInv service that powers the **DeepInv Denoise (Eric)** node and how to work with its presets, including DiffUNet and RAM.

## 1. Overview


## 2. Prerequisites

- **Python environment**: Use the dedicated DeepInv virtual environment included with this project (recommended) or any environment with Python 3.10+.
- **Dependencies**: Install DeepInv and the service requirements.

  ```powershell
  # From the project root
  & .venv\Scripts\Activate.ps1                # or activate your own env
  pip install deepinv fastapi uvicorn pydantic
  ```

- **Weights**: Place optional pretrained weights in one of the locations the service scans:
  1. `models/pretrained_weights/`
  2. `models/`
  3. Directory pointed to by `DEEPINV_PRETRAINED_DIR`

  The service automatically detects filenames such as:
  - `diffusion_ffhq_10m.pt`, `diffusion_openai.pt` (DiffUNet)
  - `ram.pth` (Reconstruct Anything Model)
  - All SwinIR, DnCNN, DRUNet, and SCUNet checkpoints

If a weight is missing, DeepInv will download it on demand when possible.

## 3. Launching the Service

From the repository root:

```powershell
& .venv\Scripts\Activate.ps1
python -m external_tools.deepinv_service.service
```

Environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `DEEPINV_SERVICE_PORT` | Port the service listens on | `6112` |
| `DEEPINV_PRETRAINED_DIR` | Extra directory to search for weights | _unset_ |

> **Tip:** Run the service before starting ComfyUI so the node can connect immediately.

## 4. HTTP Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Quick readiness check; returns `{ "status": "ok" }`. |
| `POST` | `/denoise` | Main inference endpoint (see payload below). |
| `POST` | `/cache/clear` | Flushes cached models and releases GPU memory (clears CUDA caches for each model). |

### 4.1 `/denoise` Payload

```json
{
  "input_path": "C:/Temp/input.npy",
  "output_path": "C:/Temp/output.npy",
  "model": "diffunet_ffhq",
  "sigma": 0.08,
  "prefer_gpu": true
}
```

- `input_path` / `output_path`: Absolute paths to `.npy` tensors in NHWC float32 `[0, 1]` format.
- `model`: One of the registered presets (`drunet`, `dncnn_light`, `dncnn_lipschitz`, `diffunet_ffhq`, `diffunet_imagenet`, `ram_denoise`, `scunet_real_psnr`, `swinir_noise15`, `swinir_noise25`, `swinir_noise50`).
- `sigma`: Noise level in `[0, 1]`. Models such as DiffUNet and RAM will use this value directly; fixed-weight models (SCUNet, SwinIR) ignore it.
- `prefer_gpu`: When `true`, the service tries CUDA first and falls back to CPU if a memory error occurs.

The response includes the model name, device used, sigma, output path, and basic statistics about the result.

## 5. VRAM Management & Fallbacks

- The service caches each model per device (e.g., `diffunet_imagenet-cuda:0`). Use `POST /cache/clear` to drop cached models and free VRAM without restarting the server.
- Large checkpoints (e.g., `diffunet_imagenet`) can exceed 20 GB of VRAM during inference. If CUDA raises an out-of-memory error, the service automatically retries on CPU.
- Inputs are internally padded to multiples of 32 pixels for DiffUNet, then cropped back to their original size.
- For extremely tight memory budgets you can run the service with `prefer_gpu=false` to stay on CPU from the start.

## 6. Using the ComfyUI Node

1. **Start the service** (`python -m external_tools.deepinv_service.service`).
2. **Launch ComfyUI** and add the `DeepInv Denoise (Eric)` node.
3. Choose a preset from the dropdown:
   - Legacy models: DRUNet, DnCNN (light & Lipschitz)
   - New diffusion/foundation models: DiffUNet (FFHQ & ImageNet) and RAM
   - SwinIR noise-specific models
4. Adjust the sigma slider (0–0.5 recommended). DiffUNet/RAM respond to sigma; SwinIR/SCUNet do not.
5. Toggle `prefer_gpu` when you want CUDA execution.
6. Optional: Specify a custom `service_url` input if the service runs on another machine or port.

When processing completes, the node returns the denoised tensor and a multi-line info string summarizing the run.

## 7. Troubleshooting

- **"Weight not found"**: Ensure the file is in one of the search directories or set `DEEPINV_PRETRAINED_DIR` before launching the service.
- **Timeouts / connection errors**: Confirm the service is running and that the port is reachable. Reboot if Windows firewall blocks the local port.
- **High VRAM usage after multiple runs**: Invoke `POST /cache/clear` or restart the service to drop cached models and flush CUDA allocator buffers.
- **DiffUNet fails silently**: Check the service console for padding or resolution warnings. The node’s info output also mirrors service errors.

With the server running and weights in place, the DeepInv presets integrate seamlessly with your ComfyUI workflows.```}
