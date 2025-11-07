# Film Grain Denoising Model Training Guide

This comprehensive guide will walk you through training your own film grain denoising models, even if you've never trained a neural network before.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Training Configuration](#training-configuration)
- [Running Training](#running-training)
- [Monitoring Progress](#monitoring-progress)
- [Using Trained Models](#using-trained-models)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Prerequisites

### Hardware Requirements

**Minimum (CPU Training):**
- 8GB RAM
- 20GB free disk space
- Training will be slow (~1 hour per epoch)

**Recommended (GPU Training):**
- NVIDIA GPU with 4GB+ VRAM (GTX 1050 Ti or better)
- 16GB RAM
- 50GB free disk space
- Training will be fast (~2-5 minutes per epoch)

### Software Requirements

1. **Python 3.8+** (you likely already have this from ComfyUI)

2. **Required Python packages:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install opencv-python
   pip install tensorboard
   pip install tqdm
   pip install scipy
   ```

3. **Verify GPU is available** (optional but recommended):
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print: True
   print(torch.cuda.get_device_name(0))  # Shows your GPU name
   ```

---

## Dataset Preparation

You need images to train on. There are two modes:

### Mode 1: Paired Dataset (Recommended for Best Results)

You need **clean images** and their corresponding **noisy versions**.

**Directory structure:**
```
my_training_data/
├── clean/
│   ├── image_001.png
│   ├── image_002.png
│   ├── image_003.png
│   └── ...
└── noisy/
    ├── image_001.png
    ├── image_002.png
    ├── image_003.png
    └── ...
```

**Important:** File names must match between clean/ and noisy/ folders!

#### Where to get paired data:

**Option A: Create Your Own**
1. Take clean digital photos (or use stock images)
2. Add film grain using:
   - Your existing film grain node in ComfyUI
   - Photo editing software (Photoshop, GIMP)
   - The provided synthetic noise generator (see below)

**Option B: Use Existing Datasets**
- **BSD68**: Classic denoising dataset
  - Download: https://github.com/clausmichele/CBSD68-dataset
  - Contains 68 natural images with multiple noise levels
  
- **DIV2K**: High-quality dataset
  - Download: https://data.vision.ee.ethz.ch/cvl/DIV2K/
  - 800 high-resolution images (you'll need to add noise)

**Option C: Film Scans**
- If you have access to film scans and their originals
- Professional restoration workflows

#### Quick synthetic noise generator:

Create `scripts/add_synthetic_grain.py`:

```python
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def add_film_grain(image, grain_strength=0.05):
    """Add realistic film grain to image"""
    noise = np.random.normal(0, grain_strength, image.shape)
    
    # Add spatial correlation
    kernel = cv2.getGaussianKernel(3, 0.5)
    kernel = kernel @ kernel.T
    
    for i in range(image.shape[2]):
        noise[:, :, i] = cv2.filter2D(noise[:, :, i], -1, kernel)
    
    # Intensity-dependent noise
    intensity_factor = 1.0 + 0.5 * (image / 255.0)
    noise = noise * intensity_factor
    
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy

# Process all images
clean_dir = Path("my_training_data/clean")
noisy_dir = Path("my_training_data/noisy")
noisy_dir.mkdir(exist_ok=True)

for img_path in tqdm(list(clean_dir.glob("*.png"))):
    img = cv2.imread(str(img_path))
    noisy = add_film_grain(img, grain_strength=15.0)  # Adjust strength as needed
    cv2.imwrite(str(noisy_dir / img_path.name), noisy)

print("✓ Synthetic grain added to all images!")
```

Run it:
```bash
python scripts/add_synthetic_grain.py
```

### Mode 2: Self-Supervised (Easier, But Less Accurate)

You only need **clean images**. The script will add synthetic noise during training.

**Directory structure:**
```
my_training_data/
└── images/
    ├── image_001.png
    ├── image_002.png
    ├── image_003.png
    └── ...
```

This is easier but produces slightly less accurate results because synthetic noise doesn't perfectly match real film grain.

### Dataset Size Recommendations

- **Minimum:** 50-100 images (will work, but may overfit)
- **Good:** 500-1000 images (decent generalization)
- **Excellent:** 2000+ images (best results)

### Image Requirements

- **Format:** PNG or JPG
- **Resolution:** Any size (patches are extracted during training)
  - Larger images = more training patches = better learning
  - Minimum: 256×256 pixels
  - Recommended: 512×512 or larger
- **Color:** RGB (3 channels)

---

## Training Configuration

### Understanding the Training Script Arguments

Here's what each parameter means:

```bash
--model              # Which model to train: 'fgann' or 'progressive_cnn'
--data_root          # Path to your dataset folder
--mode               # 'paired' or 'self_supervised'
--noise_level        # For self-supervised: how much noise to add (0.01-0.1)

--epochs             # How many times to go through all data (default: 100)
--batch_size         # How many images to process at once (default: 16)
                     #   Larger = faster but needs more VRAM
                     #   If you get "out of memory" errors, reduce this
--patch_size         # Size of image crops during training (default: 128)
--lr                 # Learning rate (default: 0.0001)
                     #   Higher = faster learning but less stable

--use_perceptual     # Use advanced perceptual loss (slower but better quality)
--output_dir         # Where to save checkpoints and logs
--save_freq          # Save checkpoint every N epochs
--resume             # Path to checkpoint to continue training from
```

### Recommended Configurations

**For Beginners (Fast Training):**
```bash
python scripts/train_film_grain_models.py \
    --model progressive_cnn \
    --data_root ./my_training_data \
    --mode paired \
    --epochs 50 \
    --batch_size 8 \
    --patch_size 128 \
    --output_dir ./trained_models
```

**For Best Quality:**
```bash
python scripts/train_film_grain_models.py \
    --model fgann \
    --data_root ./my_training_data \
    --mode paired \
    --epochs 100 \
    --batch_size 16 \
    --patch_size 128 \
    --use_perceptual \
    --output_dir ./trained_models
```

**For Self-Supervised (Clean Images Only):**
```bash
python scripts/train_film_grain_models.py \
    --model progressive_cnn \
    --data_root ./my_training_data \
    --mode self_supervised \
    --noise_level 0.05 \
    --epochs 80 \
    --batch_size 16 \
    --output_dir ./trained_models
```

---

## Running Training

### Step-by-Step Process

**Step 1: Navigate to your ComfyUI custom nodes directory**
```bash
cd A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_Image_Processing_Nodes
```

**Step 2: Activate your Python environment (if using one)**
```bash
# If you're using ComfyUI's embedded Python:
path\to\ComfyUI\python_embeded\python.exe scripts/train_film_grain_models.py ...

# OR if using conda/venv:
conda activate comfyui  # or: source venv/bin/activate
python scripts/train_film_grain_models.py ...
```

**Step 3: Start training!**
```bash
python scripts/train_film_grain_models.py \
    --model progressive_cnn \
    --data_root ./my_training_data \
    --mode paired \
    --epochs 100 \
    --batch_size 16 \
    --output_dir ./trained_models
```

### What You'll See

The training will output:
```
Using device: cuda
============================================================
Initializing PROGRESSIVE_CNN model
============================================================
Total parameters: 487,232
Trainable parameters: 487,232
Model size: 1.86 MB
============================================================
Loading dataset from: ./my_training_data
Mode: paired
============================================================
✓ Loaded 800 paired images
Training samples: 720
Validation samples: 80
============================================================
Starting training for 100 epochs
============================================================

Epoch 1/100:
Epoch 1: 100%|██████████| 45/45 [00:23<00:00,  1.91it/s, loss=0.0234, psnr=32.45dB]

Train Loss: 0.0234 | Train PSNR: 32.45dB
Validation: 100%|██████████| 5/5 [00:02<00:00,  2.34it/s]
Val Loss: 0.0198 | Val PSNR: 33.12dB

...
```

### Training Time Estimates

**Progressive CNN (Lightweight):**
- GPU (GTX 1060): ~3 minutes per epoch
- GPU (RTX 3060): ~1 minute per epoch
- CPU: ~45 minutes per epoch

**FGA-NN (Larger):**
- GPU (GTX 1060): ~8 minutes per epoch
- GPU (RTX 3060): ~3 minutes per epoch
- CPU: ~2 hours per epoch

**Total training time** (100 epochs):
- Progressive CNN on GPU: ~3-5 hours
- FGA-NN on GPU: ~8-12 hours
- Either model on CPU: Multiple days (not recommended)

---

## Monitoring Progress

### TensorBoard (Visual Monitoring)

TensorBoard provides real-time graphs and images of your training progress.

**Step 1: Start TensorBoard**

In a NEW terminal window:
```bash
cd A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_Image_Processing_Nodes

tensorboard --logdir=trained_models/progressive_cnn
```

**Step 2: Open in browser**

Navigate to: `http://localhost:6006`

You'll see:
- **Scalars tab:** Loss and PSNR curves over time
- **Images tab:** Sample denoising results every 5 epochs
- **Graphs tab:** Model architecture visualization

### What to Look For

**Healthy Training:**
- Loss steadily decreases
- PSNR steadily increases
- Validation metrics follow training (not too far behind)

**Warning Signs:**
- Loss suddenly spikes → learning rate too high
- Validation worse than training → overfitting (need more data)
- No improvement after 20 epochs → learning rate too low or dataset issues

### Checkpoints

The script automatically saves:
- `checkpoint_epoch_5.pth` (every 5 epochs)
- `checkpoint_epoch_10.pth`
- ...
- `best_checkpoint_epoch_XX.pth` (best validation PSNR)
- `final_model.pth` (after training completes)

---

## Using Trained Models

### Step 1: Locate Your Best Model

After training, find the best checkpoint:
```
trained_models/
└── progressive_cnn/
    └── 20250110_143022/
        ├── best_checkpoint_epoch_47.pth  ← This one!
        ├── checkpoint_epoch_50.pth
        ├── final_model.pth
        └── tensorboard/
```

### Step 2: Load in ComfyUI

In your ComfyUI workflow:
1. Add your denoising node (FGA-NN or Progressive CNN)
2. In the `model_path` field, enter the **full path** to your best checkpoint:
   ```
   A:\Comfy25\...\Eric_Image_Processing_Nodes\trained_models\progressive_cnn\20250110_143022\best_checkpoint_epoch_47.pth
   ```

### Step 3: Test and Compare

Try your trained model on images it hasn't seen before to verify it generalizes well.

---

## Troubleshooting

### Problem: "CUDA out of memory"

**Solution 1:** Reduce batch size
```bash
--batch_size 4  # or even 2
```

**Solution 2:** Reduce patch size
```bash
--patch_size 64
```

**Solution 3:** Use CPU (slower but works)
- The script automatically uses CPU if GPU unavailable

### Problem: "RuntimeError: DataLoader worker ... is killed by signal: Killed"

**Solution:** Reduce number of workers

Edit line in `train_film_grain_models.py`:
```python
num_workers=4  →  num_workers=0  # or 1
```

### Problem: Loss is NaN or Inf

**Causes:**
- Learning rate too high
- Bad data (corrupted images)
- Numerical instability

**Solutions:**
```bash
--lr 1e-5  # Lower learning rate
```

Check your dataset for corrupted images:
```python
from PIL import Image
from pathlib import Path

for img_path in Path("my_training_data/clean").glob("*.png"):
    try:
        img = Image.open(img_path)
        img.verify()
    except Exception as e:
        print(f"Corrupted: {img_path}")
```

### Problem: Training is very slow

**Solutions:**
1. Check if GPU is being used:
   - Look for "Using device: cuda" at start
   - If it says "cpu", install CUDA-enabled PyTorch
   
2. Reduce image resolution:
   - Resize images to 512×512 before training
   
3. Use fewer data augmentations:
   - Edit dataset class to skip some augmentations

### Problem: Model doesn't denoise well

**Possible causes:**
1. **Insufficient training:**
   - Train for more epochs (100-200)
   
2. **Dataset mismatch:**
   - Synthetic noise doesn't match real grain
   - Use paired real data instead
   
3. **Overfitting:**
   - Get more training data
   - Add data augmentation
   
4. **Wrong model:**
   - Try the other model (FGA-NN vs Progressive CNN)

---

## Advanced Topics

### Fine-Tuning a Pre-trained Model

If you want to improve an existing model:

```bash
python scripts/train_film_grain_models.py \
    --model progressive_cnn \
    --data_root ./new_training_data \
    --mode paired \
    --epochs 50 \
    --resume ./trained_models/progressive_cnn/20250110_143022/best_checkpoint_epoch_47.pth \
    --lr 1e-5  # Lower learning rate for fine-tuning
    --output_dir ./finetuned_models
```

### Custom Loss Functions

To modify the loss function, edit the `CombinedLoss` class in `train_film_grain_models.py`.

Example: Add SSIM loss:
```python
from pytorch_msssim import ssim

class CombinedLoss(nn.Module):
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        
        loss = 0.4 * mse + 0.4 * l1 + 0.2 * ssim_loss
        return loss
```

### Training on Multiple GPUs

If you have multiple GPUs:

```python
# In train_film_grain_models.py, wrap model:
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

### Custom Data Augmentation

Edit the `augment_data` method in `FilmGrainDataset`:

```python
def augment_data(self, clean, noisy):
    # Add color jittering
    if np.random.rand() > 0.5:
        # Adjust brightness
        factor = np.random.uniform(0.8, 1.2)
        clean = np.clip(clean * factor, 0, 1)
        noisy = np.clip(noisy * factor, 0, 1)
    
    # Add rotation
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10, 10)
        h, w = clean.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        clean = cv2.warpAffine(clean, M, (w, h))
        noisy = cv2.warpAffine(noisy, M, (w, h))
    
    return clean, noisy
```

---

## Quick Reference Commands

**Basic training:**
```bash
python scripts/train_film_grain_models.py \
    --model progressive_cnn \
    --data_root ./my_data \
    --mode paired \
    --epochs 100
```

**Resume training:**
```bash
python scripts/train_film_grain_models.py \
    --model progressive_cnn \
    --data_root ./my_data \
    --mode paired \
    --epochs 100 \
    --resume ./trained_models/.../best_checkpoint_epoch_XX.pth
```

**Self-supervised:**
```bash
python scripts/train_film_grain_models.py \
    --model progressive_cnn \
    --data_root ./my_data \
    --mode self_supervised \
    --noise_level 0.05 \
    --epochs 100
```

**Monitor with TensorBoard:**
```bash
tensorboard --logdir=trained_models/progressive_cnn
```

---

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review this guide's troubleshooting section
3. Check TensorBoard for training curves
4. Verify your dataset structure and image files
5. Try training with a small subset (10-20 images) first to test

## Additional Resources

- **PyTorch Tutorials:** https://pytorch.org/tutorials/
- **Image Denoising Papers:** https://paperswithcode.com/task/image-denoising
- **ComfyUI Custom Nodes:** https://github.com/comfyanonymous/ComfyUI

---

**Good luck with your training! 🚀**