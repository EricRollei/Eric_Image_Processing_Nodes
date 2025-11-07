# Quick Reference: Non-Contiguous Array Fixes Applied

## ✅ COMPLETE - All 15 Files Fixed (32 locations)

### High Priority (8 files, 22 fixes)
1. **advanced_sharpening.py** - 7 fixes - LAB conversions, arithmetic ops, denoise_bilateral
2. **learning_based_clahe.py** - 6 fixes - Color conversions, CLAHE operations  
3. **auto_denoise.py** - 4 fixes - Tensor conversions, deep learning preprocessing
4. **sfhformer_processing.py** - 1 fix - CLAHE L channel
5. **advanced_film_grain.py** - 1 fix - CLAHE L channel
6. **perceptual_color_processing.py** - 1 fix - CLAHE L channel
7. **advanced_traditional_processing.py** - 1 fix - CLAHE L channel
8. **adaptive_frequency_decomposition.py** - 1 fix - Canny edge detection

### Medium Priority (7 files, 10 fixes)
9. **wavelet_denoise.py** - 2 fixes - Channel arithmetic, Laplacian
10. **bm3d_denoise.py** - 1 fix - Sigma estimation
11. **nonlocal_means.py** - 3 fixes - Multiple cvtColor operations
12. **richardson_lucy.py** - 1 fix - Grayscale conversion
13. **frequency_enhancement.py** - 1 fix - Debug grayscale
14. **film_grain_processing.py** - 1 fix - Grain analysis
15. **gpu_utils.py** - 2 fixes - Wrapper functions (defensive)

## The Fix Pattern

```python
# Before OpenCV/scikit-image operations, add:
array = np.ascontiguousarray(array)
```

## Why This Happened

- **Old environment**: NumPy 1.x → More contiguous copies
- **New environment**: NumPy 2.1.3 → More views (non-contiguous)
- **Result**: OpenCV C++ requires contiguous → crashes on views

## Next Steps

1. **Restart ComfyUI** to load the fixed code
2. **Test high-priority nodes** (CLAHE-based ones first)
3. **Report any remaining issues**

All fixes have **zero performance cost** when arrays are already contiguous!
