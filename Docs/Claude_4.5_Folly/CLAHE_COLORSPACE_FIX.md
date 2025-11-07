# CLAHE Color Space Conversion Fix

## Problem
When using Learning-Based CLAHE with **Oklab** or **Jzazbz** color spaces, the output looked like a **color negative** (inverted/corrupted colors). LAB color space worked fine.

## Root Cause
**Flag Name Mismatch** in the color conversion code.

### The Bug:

**Forward Conversion** (RGB → Color Space):
```python
elif color_space == 'oklab':
    work_image = color.rgb2lab(image)
    work_image[:, :, 0] = work_image[:, :, 0] / 100.0
    color_info['oklab_approximation'] = True  # ❌ Wrong flag name!
```

**Backward Conversion** (Color Space → RGB):
```python
if color_space == 'lab' or color_info.get('oklab_fallback_to_lab') or color_info.get('jzazbz_fallback_to_lab'):
    # Convert back from LAB
    lab_image = image.copy()
    lab_image[:, :, 0] = lab_image[:, :, 0] * 100.0
    result = color.lab2rgb(lab_image)
```

**The Mismatch:**
- Forward set: `oklab_approximation` and `jzazbz_approximation`
- Backward checked: `oklab_fallback_to_lab` and `jzazbz_fallback_to_lab`

**Result:**
- When converting back, the condition failed
- Code didn't convert back from LAB to RGB
- Returned LAB values as if they were RGB → color negative!

## The Fix

### Before:
```python
elif color_space == 'oklab':
    work_image = color.rgb2lab(image)
    work_image[:, :, 0] = work_image[:, :, 0] / 100.0
    color_info['oklab_approximation'] = True  # Wrong flag
    
elif color_space == 'jzazbz':
    work_image = color.rgb2lab(image)
    work_image[:, :, 0] = work_image[:, :, 0] / 100.0
    color_info['jzazbz_approximation'] = True  # Wrong flag
```

### After:
```python
elif color_space == 'oklab':
    work_image = color.rgb2lab(image)
    work_image[:, :, 0] = work_image[:, :, 0] / 100.0
    color_info['oklab_fallback_to_lab'] = True  # Matches back-conversion check!
    
elif color_space == 'jzazbz':
    work_image = color.rgb2lab(image)
    work_image[:, :, 0] = work_image[:, :, 0] / 100.0
    color_info['jzazbz_fallback_to_lab'] = True  # Matches back-conversion check!
```

## Test Results

### Before Fix:
```
LAB:    ✅ Works (native support)
Oklab:  ❌ Color negative
Jzazbz: ❌ Color negative
HSV:    ✅ Works (native support)
RGB:    ✅ Works (no conversion)
```

### After Fix:
```
LAB:    ✅ Works (native support)
Oklab:  ✅ Works (LAB fallback with correct conversion back)
Jzazbz: ✅ Works (LAB fallback with correct conversion back)
HSV:    ✅ Works (native support)
RGB:    ✅ Works (no conversion)
```

**Test Results:**
- Original mean: R=0.486, G=0.486, B=0.486
- LAB result:    R=0.498, G=0.498, B=0.498 ✅
- Oklab result:  R=0.498, G=0.498, B=0.498 ✅
- Jzazbz result: R=0.498, G=0.498, B=0.498 ✅

All color spaces now produce normal-looking output!

## Why Oklab/Jzazbz Use LAB

The code uses **LAB as an approximation** for Oklab and Jzazbz because:

1. **True Oklab/Jzazbz conversion** requires specialized libraries (colorspacious, colour-science)
2. **LAB is a good approximation** - all three are perceptually uniform color spaces
3. **For CLAHE purposes**, LAB works similarly since we're only processing the luminance channel
4. **Simplified dependencies** - doesn't require additional packages

This is a reasonable trade-off for this use case.

## Impact

### Before:
- ❌ Oklab and Jzazbz produced corrupted output
- ⚠️ Users would get color negative images
- 😕 Confusing user experience

### After:
- ✅ All 5 color spaces work correctly
- ✅ Oklab and Jzazbz produce valid output using LAB fallback
- ✅ No more color negatives
- 😊 Consistent, expected behavior

## Files Modified

**File:** `scripts/learning_based_clahe.py`

**Lines Modified:**
- Line 154-159: Oklab conversion - fixed flag name
- Line 161-166: Jzazbz conversion - fixed flag name

**Changes:**
- Changed `oklab_approximation` → `oklab_fallback_to_lab`
- Changed `jzazbz_approximation` → `jzazbz_fallback_to_lab`

## Technical Notes

### Why This Bug Existed:

The original code had two different naming conventions:
1. **Intent-based naming**: `oklab_approximation` (describes what's happening)
2. **Implementation-based naming**: `oklab_fallback_to_lab` (describes the technical approach)

The developer used intent-based names when setting the flag but implementation-based names when checking it.

### Lesson Learned:

**Consistency matters!** Pick one naming convention and stick to it:
- Either: `oklab_approximation` everywhere
- Or: `oklab_fallback_to_lab` everywhere

We chose `oklab_fallback_to_lab` because:
1. It matches the existing back-conversion check
2. It's more explicit about the implementation
3. Makes it clear these aren't "true" Oklab/Jzazbz conversions

## Date
October 10, 2025

## Summary

**Problem:** Oklab and Jzazbz color spaces produced color negatives

**Cause:** Flag name mismatch prevented proper RGB conversion back

**Fix:** Use consistent flag names (`oklab_fallback_to_lab` and `jzazbz_fallback_to_lab`)

**Result:** All color spaces now work correctly! ✅
