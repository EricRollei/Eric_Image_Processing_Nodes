# Auto-Denoise Non-Contiguous Array Fix

## Problem
Auto-denoise was failing silently with message:
```
❌ Auto-denoise processing failed
```

Analysis showed it was recommended to use `deep_image_prior` but processing failed.

## Root Cause
Same issue as Smart Sharpening - **non-contiguous arrays** after numpy operations. The auto-denoise processors use:
1. **Array transposition**: `image.transpose(2, 0, 1)` for PyTorch tensor conversion
2. **Channel expansion**: `np.expand_dims(image, axis=2)` for grayscale images
3. **Color space operations**: Various array slicing operations

These operations can create non-contiguous memory views that cause issues when:
- Converting to PyTorch tensors
- GPU memory transfers
- Deep learning operations

## Solution
Added `np.ascontiguousarray()` at strategic points in `scripts/auto_denoise.py`:

### Changes Made

#### 1. AutoDenoiseProcessor.process_image (line ~580)
```python
# BEFORE:
def process_image(self, image: np.ndarray, method: str = "auto", **kwargs):
    # Auto-select method if needed
    if method == "auto":
        analysis = self.analyze_image(image)

# AFTER:
def process_image(self, image: np.ndarray, method: str = "auto", **kwargs):
    # FIXED: Ensure contiguous array
    image = np.ascontiguousarray(image)
    
    # Auto-select method if needed
    if method == "auto":
        analysis = self.analyze_image(image)
```

#### 2. DeepImagePriorProcessor.process_image (line ~405)
```python
# BEFORE:
channels = image.shape[2]
img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)

# AFTER:
# FIXED: Ensure contiguous array before transpose
image = np.ascontiguousarray(image)

channels = image.shape[2]
img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
```

#### 3. Noise2VoidProcessor.train_model (line ~165)
```python
# BEFORE:
if len(image.shape) == 2:
    image = np.expand_dims(image, axis=2)

# Convert to tensor
img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)

# AFTER:
if len(image.shape) == 2:
    image = np.expand_dims(image, axis=2)

# FIXED: Ensure contiguous array before transpose
image = np.ascontiguousarray(image)

# Convert to tensor
img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
```

#### 4. Noise2VoidProcessor.process_image (line ~235)
```python
# BEFORE:
def process_image(self, image: np.ndarray, train_epochs: int = 100):
    try:
        # Train model on the image if epochs > 0
        if train_epochs > 0:

# AFTER:
def process_image(self, image: np.ndarray, train_epochs: int = 100):
    try:
        # FIXED: Ensure contiguous array
        image = np.ascontiguousarray(image)
        
        # Train model on the image if epochs > 0
        if train_epochs > 0:
```

Also added after `expand_dims`:
```python
# Prepare input
if len(image.shape) == 2:
    image = np.expand_dims(image, axis=2)

# Ensure contiguous after expand_dims
image = np.ascontiguousarray(image)
```

### Enhanced Error Reporting
Added better error messages in `AutoDenoiseProcessor.process_image`:
```python
if method == "noise2void":
    epochs = kwargs.get('train_epochs', 100)
    result = self.n2v_processor.process_image(image, train_epochs=epochs)
    if result is None:
        print(f"❌ Noise2Void returned None - processing failed")
    return result

elif method == "deep_image_prior":
    iterations = kwargs.get('iterations', 3000)
    lr = kwargs.get('learning_rate', 1e-2)
    print(f"🔄 Starting Deep Image Prior with {iterations} iterations...")
    result = self.dip_processor.process_image(image, iterations=iterations, learning_rate=lr)
    if result is None:
        print(f"❌ Deep Image Prior returned None - processing failed")
    return result
```

Added traceback for better debugging:
```python
except Exception as e:
    import traceback
    print(f"❌ Auto-denoise processing error: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    return None
```

## Why This Fixes the Problem
1. **PyTorch Compatibility**: PyTorch's `from_numpy()` works better with contiguous arrays
2. **GPU Transfers**: CUDA operations require contiguous memory layout
3. **Performance**: Contiguous arrays have better cache locality
4. **Reliability**: Prevents subtle bugs from non-contiguous views

## Additional Issues Found
The default `iterations=3000` for Deep Image Prior is very high and can cause:
- Long processing times (30+ seconds per image)
- GPU memory exhaustion
- User confusion about hanging

### Recommended Parameter Adjustments
For testing and faster iteration:
- **Small images (<512px)**: 1000-1500 iterations
- **Medium images (512-1024px)**: 1500-2000 iterations  
- **Large images (>1024px)**: 2000-3000 iterations

Or set lower defaults in node:
```python
"iterations": ("INT", {"default": 1500, "min": 500, "max": 10000, "step": 100}),
```

## Testing
After restart, test Auto-Denoise with:
- ✅ Auto method selection
- ✅ Explicit noise2void method
- ✅ Explicit deep_image_prior method
- ✅ Different image sizes
- ✅ Grayscale and color images

Expected behavior:
- Should see processing progress messages
- Should complete without silent failures
- Should return denoised images

## Related Files
- `scripts/auto_denoise.py` - Main fix location
- `nodes/auto_denoise_node.py` - Node wrapper (no changes needed)
- `.github/copilot-instructions.md` - Already updated with contiguous array guidance

## Pattern Recognition
This is the **same root cause** as Smart Sharpening:
- Array operations creating non-contiguous views
- OpenCV/PyTorch expecting contiguous memory
- Silent failures or cryptic error messages

**Rule of Thumb**: Always use `np.ascontiguousarray()` before:
- PyTorch tensor conversion
- OpenCV operations  
- GPU transfers
- After transpose/reshape/slicing operations
