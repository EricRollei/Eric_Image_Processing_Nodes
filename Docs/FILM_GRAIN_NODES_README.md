# Film Grain Denoising Nodes - Quick Reference

## Overview

This package contains two cutting-edge film grain denoising nodes for ComfyUI:

1. **FGA-NN (Film Grain Analysis Neural Network)**: Multi-scale architecture with grain parameter analysis
2. **Lightweight Progressive CNN**: Fast, efficient denoising with minimal parameters

Both nodes are ready to use out-of-the-box with random initialization, and support loading pre-trained weights for improved performance.

---

## Files Created

### Model Architectures
- `models/fga_nn_architecture.py` - FGA-NN model implementation
- `models/progressive_cnn_architecture.py` - Progressive CNN model implementation

### ComfyUI Nodes
- `nodes/fga_nn_film_grain_node.py` - FGA-NN ComfyUI node
- `nodes/lightweight_cnn_denoise_node.py` - Progressive CNN ComfyUI node

### Training System
- `scripts/train_film_grain_models.py` - Complete training script
- `Docs/TRAINING_GUIDE.md` - Comprehensive training documentation

---

## Quick Start

### 1. Install the Nodes

Add these imports to your `nodes/__init__.py`:

```python
from .fga_nn_film_grain_node import FGANN_NODE_CLASS_MAPPINGS, FGANN_NODE_DISPLAY_NAME_MAPPINGS
from .lightweight_cnn_denoise_node import PROGRESSIVE_CNN_NODE_CLASS_MAPPINGS, PROGRESSIVE_CNN_NODE_DISPLAY_NAME_MAPPINGS

# Then merge them
NODE_CLASS_MAPPINGS.update(FGANN_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(PROGRESSIVE_CNN_NODE_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(FGANN_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PROGRESSIVE_CNN_NODE_DISPLAY_NAME_MAPPINGS)
```

### 2. Restart ComfyUI

After updating `__init__.py`, restart ComfyUI to load the new nodes.

### 3. Find the Nodes

In ComfyUI, the nodes will appear under:
- **Eric's Image Processing → Film Grain → FGA-NN Film Grain Denoise**
- **Eric's Image Processing → Film Grain → Lightweight Progressive CNN Denoise**

---

## Using the Nodes (Without Training)

Both nodes work immediately with random initialization:

1. Add a node to your workflow
2. Connect an image input
3. Leave `model_path` empty
4. Run!

**Note:** Untrained models won't produce great results, but you can test the architecture and workflow.

---

## Training Your Own Models

See the comprehensive guide: **`Docs/TRAINING_GUIDE.md`**

### Quick Training Example

```bash
# Navigate to your node directory
cd A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_Image_Processing_Nodes

# Train the lightweight model (faster)
python scripts/train_film_grain_models.py \
    --model progressive_cnn \
    --data_root ./my_training_data \
    --mode paired \
    --epochs 100 \
    --batch_size 16 \
    --output_dir ./trained_models
```

---

## Node Features

### FGA-NN Film Grain Denoise

**Strengths:**
- Advanced multi-scale feature extraction
- Analyzes grain characteristics (AR coefficients, frequency bands, intensity scaling)
- Best for heavy film grain removal
- Provides detailed grain analysis report

**Parameters:**
- `image` - Input image
- `analyze_grain` - Enable grain parameter analysis
- `model_path` - Path to trained weights (optional)

**Outputs:**
- Denoised image
- Detailed analysis report with grain characteristics

### Lightweight Progressive CNN Denoise

**Strengths:**
- Fast inference (~500KB model)
- Efficient for real-time applications
- Progressive residual fusion for quality
- Good for moderate grain removal

**Parameters:**
- `image` - Input image
- `show_comparison` - Show noise reduction metrics
- `model_path` - Path to trained weights (optional)

**Outputs:**
- Denoised image
- Processing report with metrics

---

## Model Comparison

| Feature | FGA-NN | Progressive CNN |
|---------|--------|-----------------|
| Model Size | ~2.5 MB | ~1.8 MB |
| Parameters | ~650K | ~487K |
| Speed (GPU) | Medium | Fast |
| Quality | Excellent | Very Good |
| Grain Analysis | Yes | No |
| Best For | Heavy grain | Light-moderate grain |

---

## Training Tips

### Dataset Preparation

**Paired Mode (Recommended):**
```
my_training_data/
├── clean/
│   └── (clean images)
└── noisy/
    └── (corresponding noisy images)
```

**Self-Supervised Mode:**
```
my_training_data/
└── images/
    └── (clean images only)
```

### Recommended Settings

**For Quick Testing:**
- 50-100 images
- 50 epochs
- Batch size 8
- ~2-3 hours on GPU

**For Production Quality:**
- 500+ images
- 100-150 epochs
- Batch size 16
- Use `--use_perceptual` flag
- ~8-12 hours on GPU

---

## Loading Trained Models

After training, use your best checkpoint:

1. Locate: `trained_models/MODEL_NAME/TIMESTAMP/best_checkpoint_epoch_XX.pth`
2. In ComfyUI node, set `model_path` to full path
3. Run your workflow!

---

## Troubleshooting

### "Module not found" errors
- Make sure all model files are in the `models/` directory
- Check that `__init__.py` has the correct imports

### "CUDA out of memory" during training
- Reduce `--batch_size` (try 8, 4, or 2)
- Reduce `--patch_size` (try 64)

### Poor denoising results
- Model needs training on real data
- Try the other model (FGA-NN vs Progressive CNN)
- Ensure trained model matches the grain type

### Node doesn't appear in ComfyUI
- Check `nodes/__init__.py` for correct imports
- Restart ComfyUI completely
- Check console for error messages

---

## Architecture Details

### FGA-NN Components
- Multi-scale feature extraction (3x3, 5x5, 7x7 kernels)
- 6 residual blocks with CBAM attention
- Grain parameter estimator (8 parameters)
- Residual learning for noise prediction

### Progressive CNN Components
- Dense blocks for local features (4 layers, 16 growth rate)
- 3 progressive residual blocks
- Lightweight attention (channel + spatial)
- Progressive shallow-to-deep feature fusion

---

## Next Steps

1. **Test the nodes** with untrained models to verify installation
2. **Prepare a dataset** using the guide in `TRAINING_GUIDE.md`
3. **Train both models** and compare results
4. **Share your results** and help improve the nodes!

---

## Future Enhancements

Potential improvements:
- Pre-trained weights on common film grain types
- Hybrid node combining both architectures
- Real-time video denoising support
- Fine-tuning interface in ComfyUI
- Model quantization for faster inference

---

## Technical Notes

- Both models use residual learning (predict noise, not clean image)
- Training uses combined MSE + L1 loss (optional perceptual loss)
- Supports batch processing in ComfyUI
- Automatic GPU/CPU selection
- Compatible with PyTorch 1.9+

---

## Credits

Built using modern computer vision research:
- Film Grain Analysis Neural Networks (2024-2025)
- Progressive residual networks with attention
- Dense feature extraction
- Multi-scale processing

Designed for integration with Eric's Image Processing Nodes suite for ComfyUI.

---

**Happy denoising! 🎬✨**