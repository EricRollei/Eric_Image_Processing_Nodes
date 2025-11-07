# Transformer Lab Sandbox

This directory contains helper scripts and notes for testing DiffBIR, DiffIR, and KVDegformer
outside of the main ComfyUI runtime. The goal is to evaluate quality/performance in an isolated
virtual environment before integrating the models as ComfyUI nodes.

## 1. Activate the sandbox environment

```powershell
# From the repo root
.\.venv\Scripts\Activate.ps1
```

If you prefer a separate environment (e.g. `a:\Comfy25\envs\transformer-lab`), activate that
instead. The helper scripts only require that the interpreter has `torch`, `einops`, `timm`,
`omegaconf`, `opencv-python` and `scikit-image` available.

## 2. Download sample inputs

Copy a few test images into `transformer_lab/samples/`. You can use the provided placeholder
directory:

```powershell
New-Item -ItemType Directory -Path transformer_lab/samples -Force | Out-Null
# Copy your own files
Copy-Item portrait.jpg transformer_lab/samples/
```

## 3. One-time repository bootstrap

Run the setup helper to fetch the official inference code and pretrained weights for each model.
The script downloads repos into `transformer_lab/third_party` and weights into
`transformer_lab/models`.

```powershell
python transformer_lab\bootstrap_models.py --all
```

Available flags:

- `--diffbir` – download DiffBIR repo + pretrained weights (blind SR x4)
- `--diffir` – download DiffIR repo + weights (promptable restoration)
- `--kvdegformer` – download KV-Deformer repo + motion/gaussian checkpoints
- `--all` – fetch everything in one go (default)

The downloads are cached; rerunning the command is safe.

## 4. Run quick inference trials

The repositories fetched in step 3 contain their own inference entrypoints. After bootstrapping,
explore the `transformer_lab/third_party/<repo>` directories. Example commands:

```powershell
# DiffBIR blind super-resolution (see official README for more options)
python transformer_lab\third_party\DiffBIR-main\inference\inference_diffbir.py `
    --input_path transformer_lab\samples\portrait.jpg `
    --output_dir transformer_lab\outputs\diffbir `
    --model_path transformer_lab\models\DiffBIR_SR_x4.pth

# DiffIR restoration with textual prior
python transformer_lab\third_party\DiffIR-main\scripts\inference_diffir.py `
    --input transformer_lab\samples\noisy_scene.png `
    --prompt "sharp cinematic photo" `
    --checkpoint transformer_lab\models\DiffIR_s128_nf64_bs32.pth `
    --output transformer_lab\outputs\diffir

# KV-Deformer motion deblur
python transformer_lab\third_party\KV-Deformer-main\inference.py `
    --input transformer_lab\samples\motion_blur.png `
    --checkpoint transformer_lab\models\kvdeformer_motion.pth `
    --output transformer_lab\outputs\kvdeformer
```

If the upstream repository changes filenames, consult their README files for the latest CLI
arguments. The cached weights live in `transformer_lab/models/` so you can reference them without
re-downloading.

## 5. Inspect results and iterate

Compare outputs across different presets, note GPU memory usage, and decide which workflows are
worth porting back into ComfyUI. When you are satisfied with a given model, we can translate the
corresponding script into a custom node with trimmed dependencies.

## 6. Clean up

To reclaim disk space after experimentation, remove the third-party directories:

```powershell
Remove-Item transformer_lab\third_party -Recurse -Force
Remove-Item transformer_lab\models -Recurse -Force
```

---

If you hit issues with downloads or inference, capture the stack trace and we can refine the
bootstrapping scripts before bringing the models into the main workflow.
