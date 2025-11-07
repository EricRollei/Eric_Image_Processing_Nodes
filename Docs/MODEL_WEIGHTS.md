# Model Weights Guide

This document provides information on obtaining pretrained model weights for Eric's Image Processing Nodes.

## Automatic Downloads

**Most models will download automatically** when you first use a node. The download happens only once, and weights are cached for future use.

**Cache Location:** `ComfyUI/custom_nodes/Eric_Image_Processing_Nodes/models/pretrained_weights/`

## Model Weights by Node

### 1. DnCNN Denoise Node

**What it does:** Classic CNN denoiser for various noise levels

**Pretrained Models:**

| Model | Noise Level | Channels | Auto-Download | Size |
|-------|-------------|----------|---------------|------|
| `color_blind` | Unknown (blind) | RGB | ✅ Yes | ~670 KB |
| `grayscale_blind` | Unknown (blind) | Grayscale | ✅ Yes | ~670 KB |
| `grayscale_sigma15` | σ=15 | Grayscale | ✅ Yes | ~670 KB |
| `grayscale_sigma25` | σ=25 | Grayscale | ✅ Yes | ~670 KB |
| `grayscale_sigma50` | σ=50 | Grayscale | ✅ Yes | ~670 KB |

**Source:** <https://github.com/cszn/DnCNN>

**Citation:**

```bibtex
@article{zhang2017dncnn,
  title={Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
```

**License:** Academic research use (no explicit license stated)

---

### 2. SCUNet Node

**What it does:** Practical blind denoising with Swin Transformer + CNN

**Pretrained Models:**

| Model | Type | Purpose | Auto-Download | Size |
|-------|------|---------|---------------|------|
| `scunet_color_real_psnr` | PSNR-optimized | Best quality | ✅ Yes | ~18 MB |
| `scunet_color_real_gan` | GAN-trained | Perceptual quality | ✅ Yes | ~18 MB |

**Source:** <https://github.com/cszn/SCUNet>

**HuggingFace Mirror:** <https://huggingface.co/datasets/eugenesiow/SCUNet/>

**Citation:**

```bibtex
@inproceedings{zhang2022scunet,
  title={Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis},
  author={Zhang, Kai and Li, Yawei and Liang, Jingyun and Cao, Jiezhang and Zhang, Yulun and Tang, Hao and Timofte, Radu and Van Gool, Luc},
  booktitle={European Conference on Computer Vision},
  pages={593--610},
  year={2022},
  organization={Springer}
}
```

**License:** Not explicitly stated (academic research)

---

### 3. SwinIR Node

**What it does:** Image restoration using Swin Transformer

**Pretrained Models:**

| Model | Task | Scale | Auto-Download | Size |
|-------|------|-------|---------------|------|
| `001_classicalSR_DF2K_s64w8_SwinIR-M_x2` | Super-resolution | 2x | ✅ Yes | ~11 MB |
| `001_classicalSR_DF2K_s64w8_SwinIR-M_x4` | Super-resolution | 4x | ✅ Yes | ~11 MB |
| `001_classicalSR_DIV2K_s48w8_SwinIR-M_x8` | Super-resolution | 8x | ✅ Yes | ~11 MB |
| `002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4` | Lightweight SR | 4x | ✅ Yes | ~1 MB |
| `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN` | Real-world SR (GAN) | 4x | ✅ Yes | ~12 MB |
| `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR` | Real-world SR (PSNR) | 4x | ✅ Yes | ~12 MB |
| `004_grayDN_DFWB_s128w8_SwinIR-M_noise15` | Grayscale denoise | σ=15 | ✅ Yes | ~11 MB |
| `004_grayDN_DFWB_s128w8_SwinIR-M_noise25` | Grayscale denoise | σ=25 | ✅ Yes | ~11 MB |
| `004_grayDN_DFWB_s128w8_SwinIR-M_noise50` | Grayscale denoise | σ=50 | ✅ Yes | ~11 MB |
| `005_colorDN_DFWB_s128w8_SwinIR-M_noise15` | Color denoise | σ=15 | ✅ Yes | ~11 MB |
| `005_colorDN_DFWB_s128w8_SwinIR-M_noise25` | Color denoise | σ=25 | ✅ Yes | ~11 MB |
| `005_colorDN_DFWB_s128w8_SwinIR-M_noise50` | Color denoise | σ=50 | ✅ Yes | ~11 MB |

**Source:** <https://github.com/JingyunLiang/SwinIR>

**Download Link:** <https://github.com/JingyunLiang/SwinIR/releases/>

**Citation:**

```bibtex
@inproceedings{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  pages={1833--1844},
  year={2021}
}
```

**License:** Apache 2.0

---

### 4. NAFNet Node

**What it does:** Simple baseline for image restoration

**Pretrained Models:**

| Model | Task | Purpose | Auto-Download | Size |
|-------|------|---------|---------------|------|
| `NAFNet-width32.pth` | Denoising | General denoising | ✅ Yes | ~6 MB |
| `NAFNet-width64.pth` | Deblurring | Motion/defocus blur | ✅ Yes | ~23 MB |

**Source:** <https://github.com/megvii-research/NAFNet>

**Citation:**

```bibtex
@inproceedings{chen2022nafnet,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  booktitle={European Conference on Computer Vision},
  pages={17--33},
  year={2022},
  organization={Springer}
}
```

**License:** MIT License

---

### 5. Noise-DA Node

**What it does:** Noise-aware diffusion for denoising and deblurring

**Pretrained Models:**

| Model | Task | Status | Size |
|-------|------|--------|------|
| `noise_da_denoise.pth` | Denoising | Custom-trained | ~45 MB |
| `noise_da_deblur.pth` | Deblurring | Custom-trained | ~45 MB |

**Note:** These models are trained specifically for this project. They are **NOT** automatically downloaded as they are not from a public source. You can train your own or request pre-trained weights from the author.

**Training:** See `Docs/TRAINING_GUIDE.md` for instructions on training your own Noise-DA models.

---

### 6. Film Grain Analysis NN

**What it does:** Neural network for film grain detection and classification

**Pretrained Models:**

| Model | Purpose | Status | Size |
|-------|---------|--------|------|
| `fga_nn_model.pth` | Grain analysis | Custom-trained | ~12 MB |

**Note:** This model is trained on a custom dataset of film grain samples. It's **NOT** automatically downloaded. You can train your own using the training script at `scripts/train_film_grain_models.py`.

**Training Data Requirements:**

- Authentic film grain samples
- Digital noise samples
- Synthetic overlay samples
- Clean reference images

---

### 7. BM3D Node (No Weights Required)

**What it does:** Block-matching 3D filtering algorithm

**No pretrained weights needed!** BM3D is an algorithmic approach that doesn't require neural network training.

**Implementation:** Uses `pytorch-bm3d` library (<https://github.com/lizhihao6/pytorch-bm3d>)

**Citation:**

```bibtex
@article{dabov2007bm3d,
  title={Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering},
  author={Dabov, Kostadin and Foi, Alessandro and Katkovnik, Vladimir and Egiazarian, Karen},
  journal={IEEE Transactions on Image Processing},
  volume={16},
  number={8},
  pages={2080--2095},
  year={2007},
  publisher={IEEE}
}
```

**License:** MIT License (pytorch-bm3d implementation)

---

### 8. DeepInv Service (External Models)

**What it does:** Diffusion-based denoising via external service

**Models Used by Service:**

- **DiffUNet**: Diffusion-based U-Net
- **RAM**: Recognize Anything Model (for image understanding)
- **SwinIR**: Swin Transformer for restoration
- **DRUNet**: Deep Residual U-Net
- **DnCNN**: Denoising CNN (via DeepInv wrapper)

**Note:** These models are loaded by the external DeepInv service, not by ComfyUI directly. See `Docs/DEEPINV_SERVICE_GUIDE.md` for setup instructions.

**Source:** <https://github.com/deepinv/deepinv>

**License:** BSD 3-Clause License

---

## Manual Download Instructions

If automatic download fails, you can manually download weights:

### Step 1: Download the Model

Visit the source repository and download the `.pth` or `.pt` file.

### Step 2: Place in Correct Directory

```text
ComfyUI/
└── custom_nodes/
    └── Eric_Image_Processing_Nodes/
        └── models/
            └── pretrained_weights/
                ├── dncnn/
                │   ├── color_blind.pth
                │   ├── grayscale_blind.pth
                │   └── ...
                ├── scunet/
                │   ├── scunet_color_real_psnr.pth
                │   └── scunet_color_real_gan.pth
                ├── swinir/
                │   ├── 001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth
                │   └── ...
                └── nafnet/
                    ├── NAFNet-width32.pth
                    └── NAFNet-width64.pth
```

### Step 3: Restart ComfyUI

The nodes will detect the manually placed weights on next launch.

---

## Troubleshooting

### Download Fails

**Symptoms:** Error message about failed download, timeout, or connection refused.

**Solutions:**

1. **Check internet connection**: Ensure you can access GitHub and HuggingFace
2. **Check firewall**: Some corporate firewalls block GitHub raw content
3. **Use VPN**: If in a country with restricted access
4. **Manual download**: Follow instructions above
5. **Check disk space**: Ensure enough space in `models/` directory

### Model Not Found

**Symptoms:** Error message about missing model file.

**Solutions:**

1. **Check file path**: Ensure model is in correct subdirectory
2. **Check filename**: Must match expected name exactly (case-sensitive on Linux)
3. **Check file corruption**: Re-download if file size is wrong
4. **Check permissions**: Ensure ComfyUI can read the file

### Out of Memory

**Symptoms:** CUDA out of memory error when loading model.

**Solutions:**

1. **Use smaller model**: Choose a lighter variant (e.g., `NAFNet-width32` instead of `width64`)
2. **Reduce batch size**: Process images one at a time
3. **Close other GPU applications**: Free up VRAM
4. **Use CPU mode**: Set device to CPU in node settings (slower but uses system RAM)

---

## Storage Requirements

**Minimum Storage:** ~200 MB (basic models)

**Recommended Storage:** 1-2 GB (all models)

**Full Collection:** 3-5 GB (all models + variants)

**Breakdown by Model Family:**

- DnCNN: ~3 MB total (5 models)
- SCUNet: ~36 MB (2 models)
- SwinIR: ~140 MB (12 models)
- NAFNet: ~30 MB (2 models)
- Noise-DA: ~90 MB (2 models, if trained)
- Film Grain NN: ~12 MB (if trained)

---

## License Compliance

When using these pretrained models, please ensure you:

1. ✅ **Cite the original papers** in any research publications
2. ✅ **Respect academic use restrictions** (most models are research-only)
3. ✅ **Check commercial use permissions** before using in commercial products
4. ✅ **Provide attribution** in derivative works
5. ✅ **Comply with individual model licenses** (see each model's source repository)

**For commercial use**, contact the original model authors for licensing:

- **DnCNN/SCUNet/SwinIR**: Kai Zhang (cskaizhang@gmail.com)
- **NAFNet**: Megvii Research (see their GitHub)
- **DeepInv models**: Check individual model sources

---

## Additional Resources

- **Main Documentation:** `Docs/README.md`
- **DeepInv Service Setup:** `Docs/DEEPINV_SERVICE_GUIDE.md`
- **Training Custom Models:** `Docs/TRAINING_GUIDE.md`
- **GPU Acceleration:** `Docs/GPU_BM3D_QUICK_START.md`
- **BM3D Technical Details:** `Docs/BM3D_Resolution_Scaling_Guide.md`

---

**Questions?** Contact Eric Hiss at eric@historic.camera or eric@rollei.us
