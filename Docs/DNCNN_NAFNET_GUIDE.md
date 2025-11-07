# DnCNN and NAFNet Pre-trained Denoising Nodes - Quick Start Guide

# DnCNN and NAFNet Pre-trained Denoising Nodes

## ✅ Installation Complete!

You now have **two state-of-the-art pre-trained denoising nodes** that work immediately without any training!

---

## 📦 What Was Installed

### Files Created:
- `models/dncnn_architecture.py` - DnCNN model and processor
- `models/nafnet_architecture.py` - NAFNet model and processor  
- `nodes/dncnn_node.py` - DnCNN ComfyUI node
- `nodes/nafnet_node.py` - NAFNet ComfyUI node
- Both `__init__.py` files updated for registration

---

## 🚀 Quick Start

### Step 1: Restart ComfyUI
Completely close and restart ComfyUI to load the new nodes.

### Step 2: Find the Nodes
Look under: **Eric's Image Processing → Pre-trained Denoisers**
- **DnCNN Denoise (Pre-trained)**
- **NAFNet Denoise (Pre-trained)**

### Step 3: Use Them!
Just connect an image and run - they work immediately!

**First use**: Weights will download automatically (~2-10MB per model). This happens once, then they're cached.

---

## 🎯 DnCNN Denoise Node

### Overview
Classic deep learning denoiser from 2017 that still performs excellently.

### Available Models:
1. **dncnn_25** (Default, Recommended)
   - Noise level: σ=25
   - Best for moderate noise
   - Grayscale processing

2. **dncnn_15**
   - Noise level: σ=15
   - Best for light noise
   - Grayscale processing

3. **dncnn_50**
   - Noise level: σ=50
   - Best for heavy noise
   - Grayscale processing

4. **dncnn3**
   - Blind denoising (any noise level)
   - Grayscale processing
   - More flexible but slightly less accurate

5. **dncnn_color_blind**
   - Blind denoising for RGB images
   - Color processing
   - Most versatile

### When to Use DnCNN:
- ✅ Fast processing needed
- ✅ Film grain or moderate noise
- ✅ You know the approximate noise level
- ✅ Smaller file sizes preferred

### Model Size: ~2-3 MB per model
### Speed: ⚡⚡⚡ Very Fast

---

## 🚀 NAFNet Denoise Node

### Overview
State-of-the-art 2022 architecture with no activation functions - more efficient!

### Available Models:
1. **nafnet-width32** (Default, Recommended)
   - Lighter, faster model
   - ~2M parameters
   - Best for speed

2. **nafnet-width64**
   - Balanced quality/speed
   - ~8M parameters
   - Best all-around

3. **nafnet-sidd**
   - Trained on real-world noise (SIDD dataset)
   - Best for camera noise, ISO noise
   - Handles complex real-world scenarios

### When to Use NAFNet:
- ✅ Best quality needed
- ✅ Real-world camera noise
- ✅ Complex noise patterns
- ✅ Latest technology preferred

### Model Size: ~8-30 MB per model
### Speed: ⚡⚡ Fast

---

## 📊 Comparison: DnCNN vs NAFNet

| Feature | DnCNN | NAFNet |
|---------|-------|--------|
| **Year** | 2017 | 2022 |
| **Speed** | Faster ⚡⚡⚡ | Fast ⚡⚡ |
| **Quality** | Excellent ⭐⭐⭐⭐ | State-of-Art ⭐⭐⭐⭐⭐ |
| **Size** | Smaller (2-3MB) | Larger (8-30MB) |
| **Best For** | Known noise levels | Real-world noise |
| **Film Grain** | Very Good | Excellent |
| **Color** | Limited | Full RGB |

---

## 💡 Usage Tips

### For Film Grain:
```
1. Try DnCNN-25 first (fast, good results)
2. If not enough → Try NAFNet-width32
3. For best quality → Use NAFNet-sidd
```

### For Camera Noise:
```
1. Start with NAFNet-sidd (trained for real-world)
2. Alternative: NAFNet-width64 (more general)
3. Fast option: DnCNN-color-blind
```

### For Synthetic Noise:
```
1. Know noise level? → Use matching DnCNN model
2. Unknown noise? → Use NAFNet-width32
3. Very noisy? → Try DnCNN-50
```

### Batch Processing:
Both nodes support batch processing automatically:
- Connect multiple images
- All processed with same settings
- Faster than processing one-by-one

---

## 🔧 Technical Details

### DnCNN Architecture:
- 17-layer CNN with residual learning
- Predicts noise, not clean image
- Batch normalization for stability
- ~670K parameters

### NAFNet Architecture:
- No activation functions (unique!)
- Simple gating mechanisms instead
- U-Net style encoder-decoder
- Layer normalization
- 2M-8M parameters

### Automatic Weight Management:
- Weights download automatically on first use
- Cached in: `models/pretrained_weights/`
- Shared across all uses
- No manual downloading needed

---

## 🎨 Example Workflows

### Simple Denoising:
```
Load Image → DnCNN/NAFNet → Save Image
```

### Compare Models:
```
Load Image → DnCNN-25 → Preview
          ↘ NAFNet-32 → Preview
```

### Denoise + Enhance:
```
Load Image → NAFNet → BM3D → Sharpen → Save
```

---

## 🐛 Troubleshooting

### "Weights not downloading"
- Check internet connection
- Check firewall settings
- Weights come from GitHub - ensure access
- Manual download possible from GitHub releases

### "Out of memory"
- Use DnCNN (smaller memory footprint)
- Use NAFNet-width32 instead of width64
- Process images at lower resolution
- Enable CPU mode if GPU insufficient

### "Slow processing"
- Normal on first use (downloading weights)
- Subsequent uses are much faster
- Consider DnCNN for speed priority
- GPU highly recommended

---

## 📚 References

### DnCNN:
**Paper**: "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"  
**Authors**: Zhang et al.  
**Published**: IEEE TIP 2017  
**Weights Source**: https://github.com/cszn/KAIR

### NAFNet:
**Paper**: "Simple Baselines for Image Restoration"  
**Authors**: Chen et al.  
**Published**: ECCV 2022  
**Weights Source**: https://github.com/megvii-research/NAFNet

---

## ✨ Summary

You now have **professional-grade denoising** that:
- ✅ Works immediately - no training required
- ✅ Automatically downloads weights
- ✅ Handles multiple noise types
- ✅ Processes batches efficiently
- ✅ Offers speed/quality options

**Quick Recommendation**: Start with **DnCNN-25** for most images. If you need better quality or have real camera noise, upgrade to **NAFNet-width32** or **NAFNet-sidd**.

---

Enjoy your new denoising superpowers! 🎉
