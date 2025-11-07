# CLAHE Visibility Issues Found

## Problem
The Learning-Based CLAHE node is **reducing contrast** instead of enhancing it!

Test results on portrait image:
- Default (clip=2.0): **-2.2% contrast** ❌
- Higher (clip=3.0): **-3.3% contrast** ❌ 
- Maximum (clip=4.0): **-4.2% contrast** ❌

## Root Causes

### 1. Enhancement Strength Reduction
**Location:** `scripts/learning_based_clahe.py` line 368-374

```python
# Adaptive enhancement strength
if local_variation < 0.05:  # Smooth regions
    enhancement_strength = 1.2
elif local_variation > 0.2:  # Textured regions
    enhancement_strength = 0.8  # ❌ REDUCES CLAHE effect by 20%!
else:
    enhancement_strength = 1.0
```

**Problem:** Textured regions (which are common in photos) get their CLAHE effect reduced to 80%.

### 2. Perceptual Weighting
**Location:** `scripts/learning_based_clahe.py` line 512-514

```python
# Apply perceptual weighting if provided
if perceptual_weights is not None:
    # Blend based on perceptual importance
    enhanced = image + perceptual_weights * (enhanced - image)  # ❌ Further reduction
```

**Problem:** Perceptual weights range from 0-1, so non-salient areas get 0-50% of the CLAHE effect.

### 3. Clip Limit Reduction for High Contrast
**Location:** `scripts/learning_based_clahe.py` line 344-347

```python
# Adjust for contrast
if contrast < 0.1:  # Low contrast
    clip_limit *= 1.3
elif contrast > 0.3:  # High contrast  
    clip_limit *= 0.7  # ❌ REDUCES by 30%!
```

**Problem:** Images that already have good contrast get their clip limit reduced by 30%.

## Combined Effect

For a typical photo with:
- High contrast: clip limit reduced to 0.7x
- Textured regions: enhancement strength = 0.8
- Non-salient areas: perceptual weight = 0.5

**Total effective strength = base × 0.7 × 0.8 × 0.5 = 0.28 (28%!)**

No wonder the effect is barely visible!

## Why This Happens

The code was designed to be "intelligent" and "adaptive" but became **too conservative**:

1. **Philosophy**: "Don't over-enhance already good images"
2. **Problem**: What if the user WANTS more enhancement?
3. **Result**: The "learning-based" system actually prevents effective enhancement

## Recommendations

### For Immediate Use:

**Don't use Learning-Based CLAHE with default settings!**

Instead:
1. **Use Smart Sharpening** for contrast enhancement on AI images
2. **Use basic CLAHE** if you have one (without the "learning-based" reductions)
3. **Increase base_clip_limit to 4.0-5.0** to compensate for reductions

### For Code Fixes:

Similar to the sharpening fix, the CLAHE needs:

1. **Remove or reduce enhancement_strength penalty**
   ```python
   # OPTION 1: Always use full strength
   enhancement_strength = 1.0
   
   # OPTION 2: More aggressive for all regions
   if local_variation < 0.05:
       enhancement_strength = 1.3
   else:
       enhancement_strength = 1.1  # Instead of 0.8 or 1.0
   ```

2. **Make perceptual weighting optional or less aggressive**
   ```python
   # Use minimum 50% effect everywhere instead of 0-100%
   enhanced = image + (0.5 + 0.5 * perceptual_weights) * (enhanced - image)
   ```

3. **Don't reduce clip limit for high contrast**
   ```python
   # Remove the 0.7x reduction - if user sets clip=3.0, use clip=3.0!
   # The user KNOWS their image and what they want
   ```

## Comparison to Sharpening Issue

| Aspect | Smart Sharpening | CLAHE |
|--------|------------------|-------|
| **Strength reduction** | 0.7x (30% less) | 0.8x (20% less) |
| **Selective application** | Threshold-based | Perceptual weights |
| **Adaptive penalties** | Edge-based | Contrast + texture |
| **Combined effect** | ~30% of requested | ~28% of requested |
| **User visibility** | Minimal | **Negative!** |

## Why CLAHE is Worse

With sharpening:
- At least it was still increasing sharpness
- Just by less than expected

With CLAHE:
- It's actually DECREASING contrast!
- The "adaptive" logic is working against the goal
- The perceptual weighting is too aggressive

## Test Results Detail

The test image (portrait.jpg):
- Original contrast: 0.2716
- After CLAHE (clip=2.0): 0.2655 (-2.2%)
- After CLAHE (clip=4.0): 0.2603 (-4.2%)

**Higher clip limit = MORE contrast reduction!**

This is backwards. CLAHE should increase local contrast, not reduce it.

## Likely Cause of Negative Effect

The perceptual weighting might be inverting the effect:
```python
perceptual_weights = 0.6 * edge_strength + 0.4 * center_bias
```

If most of the image has low perceptual weights (0.2-0.4), and the formula is:
```python
enhanced = image + perceptual_weights * (enhanced - image)
```

Then only 20-40% of the CLAHE effect is applied, which combined with the 0.8 enhancement_strength and 0.7 clip_limit reduction, results in:
- **Effective strength = 0.7 × 0.8 × 0.3 (avg weight) = 0.168 = 17%**

This explains why we see contrast REDUCTION - the global histogram is being redistributed but only 17% of the local enhancement is applied, creating a net flattening effect.

## What To Do

### Option 1: Fix the code (like we did for sharpening)
Remove the excessive reductions and make the effect actually visible.

### Option 2: Use different nodes
For AI image enhancement:
1. **Smart Sharpening** - Already fixed, works great
2. **Unsharp Mask** - If you have a basic version
3. **Curves/Levels** - Manual contrast adjustment
4. **Basic CLAHE** - Without the "learning-based" penalties

### Option 3: Extreme settings to compensate
```
base_clip_limit: 5.0 (maximum)
adaptive_regions: False
perceptual_weighting: False
```

This might give you 5.0 × 1.0 × 1.0 = actual visible enhancement.

## Date
October 10, 2025
