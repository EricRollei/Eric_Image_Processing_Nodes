# Batch Processing Fix for SmartSharpeningNode

## Problem Identified

The `SmartSharpeningNode` (and all other advanced sharpening nodes) were not handling batches of images correctly. When processing a batch of images, only the first image was being returned.

## Root Cause

The issue was in the `base_node.py` file's `tensor_to_numpy()` method:

```python
# OLD CODE (INCORRECT)
@staticmethod
def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    # Take first image if batch
    if len(tensor.shape) == 4:
        img = tensor[0]  # ❌ Only takes first image!
    else:
        img = tensor
    
    img_np = img.cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return img_np
```

This method was explicitly taking only the first image from the batch (`tensor[0]`), which caused all subsequent images in the batch to be discarded.

## Solution

Updated the `tensor_to_numpy()` method to preserve the batch dimension:

```python
# NEW CODE (CORRECT)
@staticmethod
def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI image tensor (NHWC) to numpy array (NHWC or HWC)
    
    Args:
        tensor: ComfyUI image tensor in format [N, H, W, C] or [H, W, C] with values 0-1
        
    Returns:
        numpy array in format [N, H, W, C] or [H, W, C] with values 0-255 (uint8)
    """
    # Keep batch dimension if present
    img_np = tensor.cpu().numpy()
    
    # Convert to numpy and scale to 0-255
    img_np = (img_np * 255).astype(np.uint8)
    
    return img_np
```

## Changes Made

### 1. `base_node.py` - `tensor_to_numpy()` method
- **Before:** Took only the first image from batch (`tensor[0]`)
- **After:** Preserves full batch dimension
- Now correctly handles both single images `[H, W, C]` and batches `[N, H, W, C]`

### 2. `base_node.py` - `numpy_to_tensor()` method
- Updated documentation to reflect that it handles batches
- No functional change needed - already handled batches correctly

## How It Works Now

The nodes in `advanced_sharpening_node.py` already had the correct batch processing logic:

```python
def process_image(self, image: torch.Tensor, ...):
    # Convert tensor to numpy (now preserves batch: [N, H, W, C])
    img_np = self.tensor_to_numpy(image)
    
    processed_images = []
    processing_info = []
    
    # Process each image in the batch
    for i in range(img_np.shape[0]):
        single_img = img_np[i]  # Extract single image [H, W, C]
        result, info = processor.smart_sharpening(single_img, ...)
        processed_images.append(result)
        processing_info.append(info)
    
    # Stack processed images back into batch [N, H, W, C]
    processed_tensor = self.numpy_to_tensor(np.stack(processed_images))
    
    return (processed_tensor, info_str)
```

This pattern now works correctly because:
1. `tensor_to_numpy()` returns the full batch `[N, H, W, C]`
2. The loop processes each image individually `img_np[i]` → `[H, W, C]`
3. Results are stacked back into batch format `np.stack()` → `[N, H, W, C]`
4. `numpy_to_tensor()` converts back to tensor format

## Affected Nodes

All nodes in `advanced_sharpening_node.py` are now fixed:
- ✅ `AdvancedSharpeningNode`
- ✅ `SmartSharpeningNode`
- ✅ `HiRaLoAmSharpeningNode`
- ✅ `EdgeDirectionalSharpeningNode`
- ✅ `MultiScaleLaplacianSharpeningNode`
- ✅ `GuidedFilterSharpeningNode`

## Testing

A test file `test_batch_processing.py` has been created to verify the fix. It tests:
1. **Batch processing:** 3 images with different intensities
2. **Single image processing:** 1 image (regression test)

To run the test:
```bash
python test_batch_processing.py
```

Expected output:
- ✅ Batch size preserved correctly (3 images in → 3 images out)
- ✅ Image dimensions preserved
- ✅ Each image processed independently
- ✅ Single image processing still works

## Impact

This fix affects **all nodes that inherit from `BaseImageProcessingNode`**, not just the advanced sharpening nodes. Any custom node using the base class will now correctly handle batches.

## Backward Compatibility

The fix is **backward compatible**:
- Single images `[1, H, W, C]` continue to work correctly
- Batches `[N, H, W, C]` now work as expected
- No breaking changes to the API

## Summary

The issue was a simple bug in the base conversion function that was discarding batch information. By preserving the batch dimension through the conversion process, all nodes now correctly handle multiple images in a single processing pass, which is essential for efficient ComfyUI workflows.
