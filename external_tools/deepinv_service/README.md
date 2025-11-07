# DeepInv Service

This folder hosts the FastAPI microservice that exposes selected
DeepInv models to ComfyUI nodes. The service is intended to run inside
the dedicated virtual environment at `A:\Comfy25\envs\DeepInv` so it can
install heavy dependencies without touching the primary ComfyUI venv.

## Quick start

```powershell
# Activate the DeepInv environment
& A:\Comfy25\envs\DeepInv\Scripts\Activate.ps1

# Launch the API
python external_tools\deepinv_service\service.py
```

The service listens on `http://127.0.0.1:6112` by default. Override the
port by setting the `DEEPINV_SERVICE_PORT` environment variable before
launch.

## API surface

- `GET /health` — quick liveness probe.
- `POST /denoise` — run a DeepInv denoiser on an image saved as a `.npy`
  tensor (NHWC, float32, range `[0,1]`). The request body must contain:
  - `input_path`: path to the tensor to process.
  - `output_path`: destination `.npy` file that will store the result.
  - `model`: one of `drunet`, `dncnn`, `restormer`, `scunet`, `swinir`.
  - `sigma`: optional noise level (defaults to `0.1`).
  - `prefer_gpu`: optional flag to request CUDA execution.

Example payload:

```json
{
  "input_path": "A:/tmp/input_image.npy",
  "output_path": "A:/tmp/output_image.npy",
  "model": "drunet",
  "sigma": 0.05,
  "prefer_gpu": true
}
```

## Next steps

- Add endpoints for deblurring, super-resolution, and other recovery
  tasks using DeepInv's physics + reconstructor stack.
- Integrate an auto-starter from the ComfyUI launch batch so the service
  comes up automatically.
- Build ComfyUI nodes that call the API (e.g., via `requests` or
  asynchronous HTTP) and exchange tensors through the temporary `.npy`
  files.
