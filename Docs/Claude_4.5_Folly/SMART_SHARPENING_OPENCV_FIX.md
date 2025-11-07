# Smart Sharpening OpenCV Memory Layout Fix

## Problem
Error when using Smart Sharpening node:
```
Error in Smart Sharpening: OpenCV(4.12.0) D:\a\opencv-python\opencv-python\opencv\modules\core\src\matrix.cpp:1097: 
error: (-215:Assertion failed) dims <= 2 && step[0] > 0 in function 'cv::Mat::locateROI'
```

## Root Cause
The error `cv::Mat::locateROI` is triggered when OpenCV receives a numpy array that is **not contiguous in memory**. This happens because:

1. **LAB Color Space Conversion**: When processing color images, the code converts RGB → LAB → extracts L channel
2. **Array Slicing**: The operation `lab_image[:, :, 0]` creates a view/slice that may not be contiguous in memory
3. **OpenCV Requirement**: OpenCV's `bilateralFilter` and other C++ operations require contiguous arrays for direct memory access

### What is a Contiguous Array?
- **Contiguous**: Data stored in sequential memory addresses (can be accessed in one memory block)
- **Non-contiguous**: Data scattered across memory (requires multiple memory accesses)
- NumPy operations like slicing can create non-contiguous views for efficiency

## Solution
Added `np.ascontiguousarray()` to all array operations before passing to OpenCV or using with OpenCV-dependent functions.

### Changes Made in `scripts/advanced_sharpening.py`

#### 1. Smart Sharpening Method (line ~55-65)
```python
# BEFORE:
work_image = luminance

# AFTER:
work_image = np.ascontiguousarray(luminance)
```

#### 2. HiRaLoAm Sharpening Method (line ~165-175)
```python
# BEFORE:
work_image = luminance

# AFTER:
work_image = np.ascontiguousarray(luminance)
```

#### 3. Edge Directional Sharpening Method (line ~275-285)
```python
# BEFORE:
work_image = luminance

# AFTER:
work_image = np.ascontiguousarray(luminance)
```

#### 4. Multi-scale Laplacian Sharpening Method (line ~365-375)
```python
# BEFORE:
work_image = luminance

# AFTER:
work_image = np.ascontiguousarray(luminance)
```

#### 5. Guided Filter Sharpening Method (line ~450-460)
```python
# BEFORE:
work_image = luminance

# AFTER:
work_image = np.ascontiguousarray(luminance)
```

#### 6. Bilateral Filter Calls (line ~195 and ~210)
```python
# BEFORE:
img_8bit = (work_image * 255).astype(np.uint8)

# AFTER:
img_8bit = (work_image * 255).astype(np.uint8)
img_8bit = np.ascontiguousarray(img_8bit)  # Ensure contiguous for OpenCV
```

## Why This Fixes the Problem
1. **Memory Layout**: `np.ascontiguousarray()` ensures data is stored in C-contiguous order (row-major)
2. **OpenCV Compatibility**: OpenCV can now directly access the memory without stride/step issues
3. **No Performance Penalty**: If array is already contiguous, `np.ascontiguousarray()` returns the original (no copy)
4. **Safe for All Cases**: Works for both grayscale and color images

## Testing
After restart, test Smart Sharpening with:
- ✅ Color images (RGB → LAB processing)
- ✅ Grayscale images
- ✅ All sharpening methods (smart, hiraloam, directional, multiscale, guided)
- ✅ Different image sizes (512x512 to 4K+)

## Related Files
- `scripts/advanced_sharpening.py` - Main fix location
- `nodes/advanced_sharpening_node.py` - Node wrapper (no changes needed)
- `.github/copilot-instructions.md` - Updated with this knowledge

## Best Practice for Future
When working with OpenCV in this codebase:
1. Always use `np.ascontiguousarray()` before passing arrays to OpenCV functions
2. Pay special attention after color space conversions (RGB↔LAB, RGB↔HSV)
3. After array slicing operations that extract channels
4. When converting data types (especially to uint8 for OpenCV)

## Performance Impact
- Minimal: `np.ascontiguousarray()` is a no-op if array is already contiguous
- Small copy cost if not contiguous (one-time per image)
- Much better than catching exceptions and retrying
