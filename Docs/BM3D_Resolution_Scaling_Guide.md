# BM3D Film Grain Denoising - Resolution Scaling Guide

## Overview

The BM3D Film Grain Denoising node has been optimized for high-resolution images (1-20MP) with automatic parameter adjustment based on image resolution. This addresses the original issue where the denoising was too strong for large images.

## Key Improvements

### 1. Resolution-Aware Sigma Scaling

The node now automatically adjusts the denoising strength based on image resolution:

| Resolution | Scale Factor | Effective Sigma Reduction |
|------------|--------------|---------------------------|
| ≤ 2MP      | 1.0          | No reduction (full strength) |
| 2-4MP      | 0.9          | 10% reduction |
| 4-8MP      | 0.75         | 25% reduction |
| > 8MP      | 0.6          | 40% reduction |

### 2. Updated Grain Strength Presets

Base sigma values have been optimized for high-resolution processing:

| Preset | Base Sigma | Best Use Case |
|--------|------------|---------------|
| `ultra_light` | 5.0 | High-resolution images (>8MP) |
| `light` | 8.0 | Subtle grain removal |
| `medium` | 15.0 | Balanced denoising (default) |
| `heavy` | 25.0 | Strong grain removal |
| `custom` | 1.0-80.0 | User-defined value |

### 3. Adaptive Patch Size Scaling

The patch size automatically scales based on image dimensions:

- **Small images (512x512)**: 3x3 patches
- **Medium images (1024x1024)**: 7x7 patches  
- **Large images (2048x2048)**: 11x11 patches
- **Very large images (4096x4096)**: 15x15 patches (capped)

### 4. Performance Optimizations

- **Fast Mode**: Automatically enabled for images > 4MP
- **Conservative Scaling**: Prevents overly aggressive processing on very large images
- **Texture Preservation**: Enhanced texture preservation factor (0.8 vs 0.7)

## Usage Recommendations

### For High-Resolution Images (>8MP)
- Use `ultra_light` or `light` grain strength
- Enable `preserve_texture` for natural film look
- Consider custom sigma values between 3-8 for very large images

### For Medium Resolution Images (2-8MP)
- Use `light` to `medium` grain strength
- Standard settings work well with automatic scaling

### For Low Resolution Images (<2MP)
- Use `medium` to `heavy` grain strength
- No automatic scaling applied (full strength)

## Test Results

The scaling improvements were validated with test images:

```
Testing 512x512 (0.26MP)
  ultra_light: Effective Sigma: 4.0, Patch Size: 3x3
  light:       Effective Sigma: 6.4, Patch Size: 3x3
  medium:      Effective Sigma: 12.0, Patch Size: 3x3

Testing 1024x1024 (1.05MP)
  ultra_light: Effective Sigma: 4.0, Patch Size: 7x7
  light:       Effective Sigma: 6.4, Patch Size: 7x7
  medium:      Effective Sigma: 12.0, Patch Size: 7x7

Testing 2048x2048 (4.19MP)
  ultra_light: Effective Sigma: 3.6, Patch Size: 11x11 (0.9 scale factor)
  light:       Effective Sigma: 5.8, Patch Size: 11x11 (0.9 scale factor)
  medium:      Effective Sigma: 10.8, Patch Size: 11x11 (0.9 scale factor)
```

## Technical Implementation

### Sigma Calculation Formula
```python
effective_sigma = base_sigma * resolution_scale_factor * texture_preservation_factor
```

Where:
- `base_sigma`: Preset-specific value (5.0, 8.0, 15.0, 25.0, or custom)
- `resolution_scale_factor`: 1.0, 0.9, 0.75, or 0.6 based on megapixels
- `texture_preservation_factor`: 0.8 if enabled, 1.0 if disabled

### Patch Size Calculation
```python
scale_factor = min(height, width) / 1000.0
# Conservative scaling for very high resolution
if scale_factor > 2.0:
    scale_factor = 1.0 + (scale_factor - 1.0) * 0.5
    
patch_size = max(3, min(15, int(7 * scale_factor)))
```

## Migration Notes

### From Previous Version
- **Old `light` (sigma=15)** → **New `medium` (sigma=15)** for same strength
- **Old `medium` (sigma=25)** → **New `heavy` (sigma=25)** for same strength
- **New `ultra_light` (sigma=5)** → Best for high-resolution images

### Custom Sigma Recommendations
- **1.0-5.0**: Very gentle (high-res images)
- **8.0-12.0**: Light denoising
- **15.0-20.0**: Medium denoising
- **25.0-35.0**: Heavy denoising

## Troubleshooting

### If Denoising is Too Strong
- Try `ultra_light` preset
- Use custom sigma between 3-8
- Ensure `preserve_texture` is enabled

### If Denoising is Too Weak
- Try `heavy` preset
- Use custom sigma between 20-35
- Disable `preserve_texture` for maximum removal

### Performance Issues
- Fast mode automatically enables for >4MP images
- Consider downscaling extremely large images before processing
- Patch distance automatically increases for large images to improve performance

## Future Enhancements

Potential improvements for future versions:
- Content-aware sigma adjustment based on noise level detection
- GPU acceleration for very large images
- Batch processing optimization
- Integration with other denoising methods for hybrid approaches
