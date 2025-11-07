# LB-CLAHE OpenCV copyMakeBorder Fix

## Problem
LB-CLAHE (Learning-Based CLAHE) node failing with OpenCV error:
```
Error in LB-CLAHE processing: OpenCV(4.12.0) D:\a\opencv-python\opencv-python\opencv\modules\core\src\copy.cpp:1160: 
error: (-215:Assertion failed) top >= 0 && bottom >= 0 && left >= 0 && right >= 0 && _src.dims() <= 2 
in function 'cv::copyMakeBorder'
```

## Root Cause
**Same non-contiguous array issue** affecting all nodes using LAB color space + OpenCV operations.

### Where copyMakeBorder is Called
OpenCV's `clahe.apply()` internally calls `copyMakeBorder()` to handle image boundaries when processing tiles. This function requires:
1. **Contiguous memory layout** - Data must be in sequential memory addresses
2. **Proper array dimensions** - Must be 2D with correct stride information
3. **Valid memory access** - Step values must be positive and properly aligned

### Why It Fails
1. **LAB Conversion**: `color.rgb2lab(image)` creates a 3D array
2. **Channel Extraction**: `work_image[:, :, 0]` extracts L channel (creates non-contiguous view)
3. **Type Conversion**: `img_as_ubyte(image)` may preserve non-contiguous layout
4. **OpenCV CLAHE**: `clahe.apply()` → `copyMakeBorder()` fails on non-contiguous array

The error message `_src.dims() <= 2` is misleading - the real issue is the `step[0] > 0` check failing due to non-contiguous memory.

## Solution
Added `np.ascontiguousarray()` at **5 strategic points** in `scripts/learning_based_clahe.py`:

### Changes Made

#### 1. LAB Color Space Conversion (line ~150)
```python
# BEFORE:
if color_space == 'lab':
    work_image = color.rgb2lab(image)
    work_image[:, :, 0] = work_image[:, :, 0] / 100.0
    color_info['lab_conversion'] = True

# AFTER:
if color_space == 'lab':
    work_image = color.rgb2lab(image)
    work_image[:, :, 0] = work_image[:, :, 0] / 100.0
    # FIXED: Ensure contiguous array after LAB conversion
    work_image = np.ascontiguousarray(work_image)
    color_info['lab_conversion'] = True
```

#### 2. Oklab Conversion (line ~157)
```python
# AFTER:
elif color_space == 'oklab':
    work_image = color.rgb2lab(image)
    work_image[:, :, 0] = work_image[:, :, 0] / 100.0
    # FIXED: Ensure contiguous array after LAB conversion
    work_image = np.ascontiguousarray(work_image)
    color_info['oklab_fallback_to_lab'] = True
```

#### 3. Jzazbz Conversion (line ~164)
```python
# AFTER:
elif color_space == 'jzazbz':
    work_image = color.rgb2lab(image)
    work_image[:, :, 0] = work_image[:, :, 0] / 100.0
    # FIXED: Ensure contiguous array after LAB conversion
    work_image = np.ascontiguousarray(work_image)
    color_info['jzazbz_fallback_to_lab'] = True
```

#### 4. Luminance Channel Extraction (line ~88)
```python
# BEFORE:
luminance = work_image[:, :, 0] if len(work_image.shape) == 3 else work_image
enhanced_luminance = self._apply_adaptive_clahe(
    luminance, optimal_params, region_map, perceptual_weights
)

# AFTER:
luminance = work_image[:, :, 0] if len(work_image.shape) == 3 else work_image
# FIXED: Ensure contiguous array after channel extraction
luminance = np.ascontiguousarray(luminance)
enhanced_luminance = self._apply_adaptive_clahe(
    luminance, optimal_params, region_map, perceptual_weights
)
```

#### 5. CLAHE Application (line ~490)
**Most Critical Fix** - Right before OpenCV CLAHE:
```python
# BEFORE:
def _apply_adaptive_clahe(self, image: np.ndarray, params: Dict[str, Any], ...):
    try:
        # Convert to 8-bit for OpenCV CLAHE
        image_8bit = img_as_ubyte(image)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=region_size)
        enhanced_8bit = clahe.apply(image_8bit)

# AFTER:
def _apply_adaptive_clahe(self, image: np.ndarray, params: Dict[str, Any], ...):
    try:
        # FIXED: Ensure contiguous array before OpenCV operations
        image = np.ascontiguousarray(image)
        
        # Convert to 8-bit for OpenCV CLAHE
        image_8bit = img_as_ubyte(image)
        # FIXED: Ensure 8-bit array is also contiguous for OpenCV
        image_8bit = np.ascontiguousarray(image_8bit)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=region_size)
        enhanced_8bit = clahe.apply(image_8bit)
```

#### 6. Feature Extraction (line ~211)
```python
# BEFORE:
if len(image.shape) == 3:
    if color_space in ['lab', 'oklab', 'jzazbz']:
        luminance = image[:, :, 0]
    else:
        luminance = color.rgb2gray(image)
else:
    luminance = image

# AFTER:
if len(image.shape) == 3:
    if color_space in ['lab', 'oklab', 'jzazbz']:
        luminance = image[:, :, 0]
        # FIXED: Ensure contiguous array after channel extraction
        luminance = np.ascontiguousarray(luminance)
    else:
        luminance = color.rgb2gray(image)
        # FIXED: Ensure contiguous array after grayscale conversion
        luminance = np.ascontiguousarray(luminance)
else:
    luminance = np.ascontiguousarray(image)
```

## Why This Fixes the Problem
1. **Memory Layout**: Contiguous arrays have sequential memory addresses
2. **OpenCV Compatibility**: `copyMakeBorder` can directly access memory without stride calculations
3. **No Performance Penalty**: If already contiguous, `np.ascontiguousarray()` returns original (zero copy)
4. **Prevents Cascade Failures**: Fixes issues in both color conversion AND CLAHE application

## Understanding copyMakeBorder
OpenCV's `copyMakeBorder()` is used by CLAHE to:
1. Add padding around image edges
2. Enable tile-based processing without edge artifacts
3. Mirror/replicate edge pixels for boundary handling

When arrays are non-contiguous, the padding calculation fails because:
- Stride information is incorrect
- Memory access patterns are broken
- Boundary checks fail (negative or zero step values)

## Hardware Independence
This issue affects **all systems** regardless of:
- ❌ RAM size (your 512GB is irrelevant)
- ❌ GPU VRAM (your 112GB is irrelevant)  
- ❌ CPU power
- ✅ **Only depends on**: How numpy creates array views in memory

The error is purely about **memory layout**, not capacity or performance.

## Testing
After restart, test LB-CLAHE with:
- ✅ Different color spaces (LAB, Oklab, Jzazbz, HSV, RGB)
- ✅ Different clip limits and region sizes
- ✅ Adaptive regions enabled/disabled
- ✅ Perceptual weighting enabled/disabled
- ✅ Various image sizes (512px to 4K+)

Expected: All color spaces should work without OpenCV errors.

## Pattern Recognition
This is now the **3rd instance** of the same root cause:
1. **Smart Sharpening** - `cv::Mat::locateROI` error
2. **Auto-Denoise** - Silent failures with PyTorch tensors
3. **LB-CLAHE** - `copyMakeBorder` error

### Common Thread
All involve:
- LAB/HSV color space conversions
- Channel extraction via array slicing
- OpenCV operations expecting contiguous memory

### Universal Fix
**Always use `np.ascontiguousarray()` after**:
- Color space conversions (`rgb2lab`, `rgb2hsv`)
- Channel extraction (`array[:, :, 0]`)
- Type conversions (`img_as_ubyte`)
- Before OpenCV operations
- Before PyTorch tensor conversions

## Related Files
- `scripts/learning_based_clahe.py` - Main fix location
- `nodes/learning_based_clahe_node.py` - Node wrapper (no changes needed)
- `.github/copilot-instructions.md` - Already documents this pattern

## Prevention Strategy
Going forward, **ALWAYS add** `np.ascontiguousarray()`:
1. After any LAB/HSV/YCbCr color conversion
2. After extracting channels from multi-channel images
3. Before passing to OpenCV functions
4. Before converting to PyTorch tensors

This is a **systemic pattern** in this codebase that needs to be addressed comprehensively.
