# Richardson-Lucy Settings for AI Images

## Problem
The Richardson-Lucy node with default settings (blur_size=3.0) is not effective for tightening soft AI-generated images.

## Why Richardson-Lucy Wasn't Working

### The Issue:
**AI image softness ≠ Camera blur**

- **Camera blur**: Image was sharp, then blurred by lens/motion (blur_size 2-10 pixels)
- **AI softness**: Model never generated fine details in the first place (equivalent blur_size < 1 pixel)

### Default Settings Were Wrong:
```
blur_size: 3.0 pixels  ❌ WAY TOO LARGE for AI softness
iterations: 15         ✅ OK
regularization: 0.005  ✅ OK
```

## Test Results

Tested on soft AI-generated portrait (1800×1440):

| Method | Sharpness Improvement | Settings |
|--------|----------------------|----------|
| **Smart Sharpen 2.5** | **+1078%** ⭐ | strength=2.5, radius=0.8, threshold=0.05 |
| Smart Sharpen 2.0 | +769% | strength=2.0, radius=0.8, threshold=0.05 |
| Smart Sharpen 1.5 | +506% | strength=1.5, radius=0.8, threshold=0.05 |
| Smart Sharpen 1.0 | +290% | strength=1.0, radius=0.8, threshold=0.05 |
| HiRaLoAm 5.0 | +124% | radius_ratio=5.0, amount=0.3 |

## Recommendations

### ⭐ BEST FOR AI IMAGES: Smart Sharpening Node

**Why it's better:**
- Designed for enhancing existing edges
- Adaptive radius based on local content
- Overshoot protection prevents artifacts
- Works on soft/smooth areas, not just edges

**Optimal settings for AI images:**
```
Strength: 1.5-2.5
Radius: 0.8-1.2 (smaller than default)
Threshold: 0.03-0.08 (lower than default)
Overshoot Protection: ON
```

### 🎨 ALTERNATIVE: HiRaLoAm Sharpening

**For more photographic/subtle look:**
```
Radius Ratio: 4.0-5.0
Amount Ratio: 0.3-0.4
Blur Type: mixed
Frequency Bands: 3
```

**Result**: Less "digital" looking, more natural

### 🔬 WHEN TO USE RICHARDSON-LUCY:

**ONLY if your AI images have visible blur artifacts:**
- Motion blur lines
- Defocus circles
- Lens aberrations

**Correct settings for subtle AI softness:**
```
Blur Type: gaussian
Blur Size: 0.3-0.8 (NOT 3.0!)  ⚠️ CRITICAL
Iterations: 15-25
Regularization: 0.001-0.005
```

## Why Richardson-Lucy Has Minimal Impact on AI Images

1. **Wrong blur model**: RL assumes image = sharp_image ⊗ blur_kernel
   - AI images: There was never a "sharp_image" to recover!
   
2. **Blur size mismatch**: 
   - Default 3.0 pixels = moderate camera defocus
   - AI softness ≈ 0.3-0.8 pixels = model smoothing

3. **Wrong tool for the job**:
   - RL = deconvolution (remove known blur)
   - AI needs = enhancement (add detail that wasn't there)

## What Changed in the Fix

### Smart Sharpening Improvements:
1. ✅ Removed 30% strength reduction
2. ✅ Added soft falloff (30% minimum everywhere)
3. ✅ Now produces visible results

### Richardson-Lucy:
- No code changes needed
- Just need correct parameters for AI images
- Default settings are correct for real camera blur

## Recommendations Summary

| Image Type | Best Method | Settings |
|------------|-------------|----------|
| **Soft AI images** | Smart Sharpening | strength=2.0, radius=0.8, threshold=0.05 |
| **AI with slight blur** | Richardson-Lucy | blur_size=0.5, iterations=20 |
| **Camera defocus** | Richardson-Lucy | blur_size=2-5, iterations=15 |
| **Motion blur** | Richardson-Lucy | motion mode, length=10-30 |
| **Photographic look** | HiRaLoAm | radius_ratio=4-5, amount=0.3 |

## Usage Guide

### For Soft AI Images (DALL-E, Midjourney, Stable Diffusion):

1. **First try: Smart Sharpening**
   ```
   Strength: 2.0
   Radius: 0.8
   Threshold: 0.05
   Overshoot Protection: ON
   ```

2. **If too aggressive: HiRaLoAm**
   ```
   Radius Ratio: 4.5
   Amount Ratio: 0.35
   Blur Type: mixed
   ```

3. **If still soft: Increase Smart Sharpening**
   ```
   Strength: 2.5-3.0
   Radius: 0.6-0.8 (smaller = sharper details)
   Threshold: 0.03 (lower = affects more pixels)
   ```

### For AI Images with Actual Blur Artifacts:

```
Richardson-Lucy with:
Blur Size: 0.5-1.0 (much smaller than default!)
Iterations: 20-25
Regularization: 0.002-0.005
```

## Common Mistakes

❌ Using Richardson-Lucy blur_size=3.0 on AI images
❌ Expecting RL to "add detail" that AI never generated
❌ Using very high strengths and creating artifacts
❌ Not testing Smart Sharpening first

✅ Smart Sharpening for general AI softness
✅ Small blur_size (0.3-0.8) if using RL on AI
✅ Lower threshold for broader sharpening
✅ Test at strength 1.5-2.0 first, then adjust

## Technical Explanation

**Richardson-Lucy assumes:**
```
observed_image = true_image ⊗ PSF + noise
```

**For AI images:**
```
ai_image = model_output (already "smoothed")
```

There's no hidden sharp image to recover! The AI model generated smooth output directly.

**Solution:**
Use enhancement (sharpening) not deconvolution (RL).

## Date
October 10, 2025
