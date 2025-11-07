# BM3D Film Grain Denoising - Summary of Improvements

## Problem Solved

The original BM3D node was too aggressive for high-resolution images (1-20MP), causing over-processing and loss of detail. Film grain that should be subtle was being completely removed.

## Solution Implemented

### 1. **Resolution-Aware Sigma Scaling**
- Automatically reduces denoising strength for larger images
- 8MP+ images get 40% reduction in processing strength
- 4-8MP images get 25% reduction  
- 2-4MP images get 10% reduction
- ≤2MP images get full strength (no change)

### 2. **New Ultra-Light Preset**
- Added `ultra_light` preset with sigma=5.0
- Ideal for high-resolution images
- Provides minimal grain removal while preserving texture

### 3. **Adaptive Patch Sizing**
- Patch size automatically scales with image dimensions
- Larger images get larger patches for better performance
- Conservative scaling prevents overly aggressive processing

### 4. **Enhanced Texture Preservation**
- Improved texture preservation factor (0.8 instead of 0.7)
- Better maintains natural film look
- Automatic fast mode for images >4MP

## Test Results Summary

**Before:** All images used same parameters regardless of size
**After:** Parameters automatically adjust based on resolution

```
512x512 (0.26MP):   No scaling needed (patch: 3x3)
1024x1024 (1.05MP): No scaling needed (patch: 7x7)  
2048x2048 (4.19MP): 0.9x scale factor (patch: 11x11)
4096x4096 (16.78MP): 0.6x scale factor (patch: 15x15)
```

## Usage Recommendations

- **High-res images (>8MP)**: Use `ultra_light` or `light`
- **Medium-res images (2-8MP)**: Use `light` or `medium`
- **Low-res images (<2MP)**: Use `medium` or `heavy`

## Migration Guide

| Old Setting | New Equivalent | Notes |
|-------------|---------------|-------|
| `light` | `medium` | Same sigma value (15.0) |
| `medium` | `heavy` | Same sigma value (25.0) |
| N/A | `ultra_light` | New option for high-res |

## Custom Sigma Ranges

- **3-5**: Very gentle (high-resolution)
- **8-12**: Light denoising
- **15-20**: Medium denoising  
- **25-35**: Heavy denoising

The node now provides appropriate denoising for all image sizes while maintaining the natural film grain aesthetic!
