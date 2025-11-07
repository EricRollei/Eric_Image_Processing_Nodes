# Multi-Node Bug Fixes - October 12, 2025

## Issues Found and Fixes Applied

### 1. Smart Sharpening - OpenCV locateROI Error ✓ FIXED
**Error**: `OpenCV(4.11.0) error: (-215:Assertion failed) dims <= 2 && step[0] > 0 in function 'cv::Mat::locateROI'`

**Root Cause**: `color.rgb2lab()` itself can return non-contiguous arrays

**Fix Applied**: Added `np.ascontiguousarray(lab_image)` immediately after ALL `color.rgb2lab()` calls in `scripts/advanced_sharpening.py`

### 2. LB-CLAHE - copyMakeBorder Error ✓ FIXED
**Error**: `OpenCV(4.11.0) error: (-215:Assertion failed) top >= 0 && bottom >= 0 && left >= 0 && right >= 0 && _src.dims() <= 2 in function 'cv::copyMakeBorder'`

**Root Cause**: Channel slicing `work_image[:, :, c]` creates non-contiguous views

**Fix Applied**: In `scripts/learning_based_clahe.py` line ~109, make channel contiguous before passing to `_apply_adaptive_clahe`

### 3. Film Grain Processing - OpenCV locateROI Error
**Error**: `OpenCV(4.11.0) error: (-215:Assertion failed) dims <= 2 && step[0] > 0 in function 'cv::Mat::locateROI'`

**Status**: NEEDS FIX - Same pattern as Smart Sharpening

**Fix Needed**: Check `scripts/advanced_film_grain.py` for:
- Non-contiguous arrays after `cv2.cvtColor`
- Non-contiguous arrays after channel slicing
- Non-contiguous arrays after color space conversions

### 4. BM3D Denoise - Broadcast Shape Error
**Error**: `operands could not be broadcast together with shapes (3599,2880,1) (1799,1440,1)`

**Root Cause**: Image dimensions are being doubled somewhere (1799→3599, 1440→2880)

**Status**: NEEDS INVESTIGATION

**Likely Issue**: Resolution scaling logic in `scripts/bm3d_denoise.py` is doubling dimensions

### 5. BM3D Deblurring - Multiple Arguments Error
**Error**: `bm3d_deblurring() got multiple values for argument 'sigma_psd'`

**Root Cause**: Function signature mismatch - passing `sigma_psd` both positionally and as keyword

**Status**: NEEDS FIX

**Fix Needed**: Check function call in `scripts/bm3d_denoise.py` or node wrapper

### 6. BM3D Deblurring - Tuple Index Error
**Error**: `tuple index out of range`

**Root Cause**: Trying to access index that doesn't exist in a tuple/list

**Status**: NEEDS FIX

**Fix Needed**: Add bounds checking in `scripts/bm3d_denoise.py`

### 7. Deep Image Prior - Processing Failed
**Error**: Generic "Deep Image Prior processing failed"

**Status**: NEEDS TRACEBACK

**Fix Needed**: Add exception handling with traceback in `scripts/auto_denoise.py`

### 8. Auto-Denoise - Processing Failed
**Error**: Using deep_image_prior but fails

**Status**: RELATED TO #7

**Fix Needed**: Fix Deep Image Prior first

### 9. NAFNet - 404 Download Error
**Error**: `HTTP Error 404: Not Found`

**Root Cause**: Model download URL is broken or model moved

**Status**: NEEDS URL UPDATE

**Fix Needed**: Update model URL in `scripts/nafnet_denoise.py` or disable auto-download

### 10. Noise2Void - Tensor Size Mismatch
**Error**: `Sizes of tensors must match except in dimension 1. Expected size 898 but got size 899`

**Root Cause**: U-Net architecture padding/cropping mismatch

**Status**: NEEDS FIX

**Fix Needed**: Fix padding/cropping logic in `scripts/auto_denoise.py` Noise2Void processor

## Priority Order

### CRITICAL (Prevent OpenCV crashes):
1. ✓ Smart Sharpening - FIXED
2. ✓ LB-CLAHE - FIXED
3. Film Grain Processing - NEEDS FIX

### HIGH (Core functionality broken):
4. BM3D Denoise - Shape mismatch
5. Deep Image Prior - Generic failure
6. Auto-Denoise - Depends on DIP

### MEDIUM (Specific features):
7. BM3D Deblurring - Argument error
8. BM3D Deblurring - Index error
9. Noise2Void - Tensor mismatch

### LOW (External dependency):
10. NAFNet - Download URL

## Next Steps

1. Test Smart Sharpening and LB-CLAHE (should be fixed)
2. Fix Film Grain Processing (same pattern)
3. Investigate BM3D dimension doubling issue
4. Add proper error handling to get full tracebacks for remaining issues
