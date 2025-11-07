"""
GPU-Accelerated BM3D Denoise Node for ComfyUI
15-30x faster than CPU BM3D implementation

Author: Eric Hiss (GitHub: EricRollei)
License: See LICENSE file in repository root

Original BM3D Algorithm:
    Paper: "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering"
    Authors: Dabov et al., IEEE TIP 2007
    
    Citation:
    @article{dabov2007bm3d,
      title={Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering},
      author={Dabov, Kostadin and Foi, Alessandro and Katkovnik, Vladimir and Egiazarian, Karen},
      journal={IEEE Transactions on Image Processing},
      volume={16},
      number={8},
      pages={2080--2095},
      year={2007}
    }

GPU Implementation:
    Library: pytorch-bm3d (https://github.com/lizhihao6/pytorch-bm3d)
    Author: lizhihao6
    License: MIT License

Dependencies:
    - PyTorch (BSD 3-Clause License)
    - NumPy (BSD 3-Clause License)
    - pytorch-bm3d (MIT License)
"""

from ..base_node import BaseImageProcessingNode

try:
    from ..scripts.bm3d_gpu_denoise import (
        bm3d_gpu_denoise, 
        is_available, 
        get_preset_parameters
    )
    GPU_BM3D_AVAILABLE = True
except ImportError:
    GPU_BM3D_AVAILABLE = False


class BM3DGPUDenoiseNode(BaseImageProcessingNode):
    """
    GPU-accelerated BM3D denoising node.
    
    Requires:
    - pytorch-bm3d library installed
    - Compiled CUDA extension (bm3d_cuda)
    - CUDA-capable GPU
    
    Performance: 15-30x faster than CPU BM3D
    - 256x256: ~0.08s (vs ~2.5s CPU)
    - 1080p: ~0.14s (vs ~5-10s CPU)
    - 4K: ~0.5s (vs ~15-20s CPU)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        
        # Get available presets
        if GPU_BM3D_AVAILABLE:
            presets = list(get_preset_parameters().keys())
        else:
            presets = ["medium_noise"]  # Default fallback
        
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (presets, {
                    "default": "medium_noise"
                }),
                "sigma": ("INT", {
                    "default": 25,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "two_step": ("BOOLEAN", {
                    "default": True,
                    "label_on": "High Quality (Two-Step)",
                    "label_off": "Fast (Single-Step)"
                }),
                "use_preset": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Use Preset",
                    "label_off": "Manual Settings"
                }),
            },
            "optional": {
                "gpu_device": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 7,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_image",)
    FUNCTION = "denoise"
    CATEGORY = "Eric's Image Processing/GPU Denoisers"
    
    def denoise(self, image, preset, sigma, two_step, use_preset, gpu_device=0):
        """
        Apply GPU-accelerated BM3D denoising.
        
        Args:
            image: Input image tensor [N,H,W,C]
            preset: Preset name
            sigma: Noise standard deviation (0-100)
            two_step: Use two-step refinement
            use_preset: Whether to use preset or manual settings
            gpu_device: GPU device index
            
        Returns:
            tuple: (denoised_image_tensor,)
        """
        # Check if GPU BM3D is available
        if not GPU_BM3D_AVAILABLE:
            raise RuntimeError(
                "GPU BM3D is not available. Please install pytorch-bm3d:\n"
                "1. Clone: git clone https://github.com/lizhihao6/pytorch-bm3d.git\n"
                "2. Compile CUDA extension (see documentation)\n"
                "3. Install to ComfyUI's Python environment"
            )
        
        available, reason = is_available()
        if not available:
            raise RuntimeError(f"GPU BM3D not available: {reason}")
        
        # Get device info
        device_info = self.get_device_info()
        
        # Convert tensor to numpy
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            # Convert single image: [H,W,C] float [0-1]
            # FIXED: Remove batch dimension by using [i] instead of [i:i+1]
            img_np = self.tensor_to_numpy(image[i])
            
            # Apply preset if requested
            if use_preset:
                presets = get_preset_parameters()
                if preset in presets:
                    preset_params = presets[preset]
                    sigma = preset_params["sigma"]
                    two_step = preset_params["two_step"]
            
            # Set device
            import torch
            device = torch.device(f"cuda:{gpu_device}")
            
            try:
                # Denoise
                denoised, info = bm3d_gpu_denoise(
                    img_np,
                    sigma=sigma,
                    two_step=two_step,
                    device=device
                )
                
                # Print info for first image
                if i == 0:
                    print(f"GPU BM3D Denoising:")
                    print(f"  Sigma: {info['sigma']}")
                    print(f"  Mode: {'Two-step (High Quality)' if info['two_step'] else 'Single-step (Fast)'}")
                    print(f"  Device: {info['device']}")
                    print(f"  Time: {info['processing_time']}")
                    if 'psnr' in info:
                        print(f"  PSNR: {info['psnr']}")
                
                results.append(denoised)
                
            except Exception as e:
                print(f"GPU BM3D failed on image {i+1}/{batch_size}: {e}")
                # Fallback: return original image
                results.append(img_np)
        
        # Stack results into single array [N, H, W, C]
        import numpy as np
        if len(results) == 1:
            # Single image - already correct shape
            stacked_results = results[0]
        else:
            # Multiple images - stack along batch dimension
            stacked_results = np.stack(results, axis=0)
        
        # Convert back to tensor
        output_tensor = self.numpy_to_tensor(stacked_results)
        
        # Cleanup
        self.cleanup_memory()
        
        return (output_tensor,)


# For compatibility checking
NODE_CLASS = BM3DGPUDenoiseNode
NODE_DISPLAY_NAME = "BM3D GPU Denoise (Eric)"
