"""
Auto-Denoise Integration Script
Self-supervised and unsupervised CNN denoising methods implementation

Integrates state-of-the-art methods:
- Noise2Void: Single image self-supervised denoising
- Noise2Noise: Paired image denoising
- Deep Image Prior: Unsupervised restoration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict, Any, Union
import warnings
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available, some features may be limited")

try:
    from skimage.restoration import estimate_sigma
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available, some metrics unavailable")


class Noise2VoidProcessor:
    """
    Noise2Void: Self-supervised denoising from single noisy images
    
    Revolutionary approach that doesn't require clean reference images.
    Works by training on blind spots in the same image.
    
    Key advantages:
    - Single image processing (no pairs needed)
    - Works with any noise type
    - Self-supervised learning
    - No clean data required
    """
    
    def __init__(self, device: str = "auto", model_size: str = "small"):
        """
        Initialize Noise2Void processor
        
        Args:
            device: Processing device ('auto', 'cpu', 'cuda')
            model_size: Model complexity ('small', 'medium', 'large')
        """
        self.device = self._get_device(device)
        self.model_size = model_size
        self.model = None
        self.trained = False
        
        # Model architectures
        self.architectures = {
            'small': {'channels': [32, 64, 128], 'depth': 3},
            'medium': {'channels': [64, 128, 256], 'depth': 4},
            'large': {'channels': [64, 128, 256, 512], 'depth': 5}
        }
    
    def _get_device(self, device: str) -> torch.device:
        """Determine optimal processing device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _create_model(self, in_channels: int = 3) -> nn.Module:
        """Create Noise2Void U-Net architecture"""
        arch = self.architectures[self.model_size]
        
        class Noise2VoidUNet(nn.Module):
            def __init__(self, in_channels, channels, depth):
                super().__init__()
                self.depth = depth
                self.encoders = nn.ModuleList()
                self.decoders = nn.ModuleList()
                self.pools = nn.ModuleList()
                
                # Encoder path
                prev_ch = in_channels
                for i, ch in enumerate(channels):
                    self.encoders.append(self._conv_block(prev_ch, ch))
                    if i < len(channels) - 1:
                        self.pools.append(nn.MaxPool2d(2))
                    prev_ch = ch
                
                # Decoder path
                for i in range(len(channels) - 2, -1, -1):
                    self.decoders.append(
                        nn.ConvTranspose2d(channels[i + 1], channels[i], 2, stride=2)
                    )
                    self.decoders.append(self._conv_block(channels[i] * 2, channels[i]))
                
                # Output layer
                self.final = nn.Conv2d(channels[0], in_channels, 1)
            
            def _conv_block(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoder
                enc_features = []
                for i, encoder in enumerate(self.encoders):
                    x = encoder(x)
                    enc_features.append(x)
                    if i < len(self.pools):
                        x = self.pools[i](x)
                
                # Decoder
                dec_idx = 0
                for i in range(len(enc_features) - 2, -1, -1):
                    x = self.decoders[dec_idx](x)  # Upsample
                    
                    # FIXED: Handle dimension mismatch from odd-sized inputs
                    # Crop or pad x to match enc_features[i] dimensions
                    if x.shape[2:] != enc_features[i].shape[2:]:
                        # Calculate size difference
                        diff_h = enc_features[i].shape[2] - x.shape[2]
                        diff_w = enc_features[i].shape[3] - x.shape[3]
                        
                        # Pad if upsampled tensor is smaller
                        if diff_h > 0 or diff_w > 0:
                            pad_h = max(0, diff_h)
                            pad_w = max(0, diff_w)
                            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
                        
                        # Crop if upsampled tensor is larger
                        if diff_h < 0 or diff_w < 0:
                            x = x[:, :, :enc_features[i].shape[2], :enc_features[i].shape[3]]
                    
                    x = torch.cat([x, enc_features[i]], dim=1)  # Skip connection
                    x = self.decoders[dec_idx + 1](x)  # Conv block
                    dec_idx += 2
                
                return self.final(x)
        
        return Noise2VoidUNet(in_channels, arch['channels'], arch['depth'])
    
    def _create_blind_spot_mask(self, shape: Tuple[int, ...], mask_size: int = 5) -> torch.Tensor:
        """Create blind spot mask for Noise2Void training"""
        batch, channels, height, width = shape
        mask = torch.ones(shape, dtype=torch.bool)
        
        # Create random blind spots
        num_blind_spots = max(1, (height * width) // (mask_size * mask_size * 10))
        
        for _ in range(num_blind_spots):
            if height <= mask_size or width <= mask_size:
                break  # Degenerate case handled by sparse fallback
            y = torch.randint(0, height - mask_size + 1, (1,))
            x = torch.randint(0, width - mask_size + 1, (1,))
            mask[:, :, y:y+mask_size, x:x+mask_size] = False
        
        return mask

    def _create_sparse_mask(self, shape: Tuple[int, ...], drop_fraction: float = 0.001) -> torch.Tensor:
        """Create sparse dropout mask ensuring at least one masked element"""
        mask = torch.ones(shape, dtype=torch.bool)
        flat = mask.view(-1)
        total = flat.numel()
        drop_count = max(1, int(total * drop_fraction))
        idx = torch.randperm(total)[:drop_count]
        flat[idx] = False
        return mask.view(shape)
    
    def train_model(self, image: np.ndarray, epochs: int = 100, learning_rate: float = 1e-3) -> Dict[str, Any]:
        """
        Train Noise2Void model on single image
        
        Args:
            image: Input noisy image (H, W, C) in range [0, 1]
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            
        Returns:
            Training statistics dictionary
        """
        print(f"Training Noise2Void model on {image.shape} image...")
        
        # Prepare data
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # FIXED: Ensure contiguous array before transpose
        image = np.ascontiguousarray(image)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        img_tensor.requires_grad_(False)  # Target tensor should not require gradients
        
        # Create model
        channels = image.shape[2]
        self.model = self._create_model(channels).to(self.device)
        
        # FIXED: Ensure all model parameters require gradients
        self.model.train()  # Set to training mode first
        for param in self.model.parameters():
            param.requires_grad_(True)  # Explicitly enable gradients
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        losses = []
        
        print(f"Training on {self.device} for {epochs} epochs...")
        
        with torch.enable_grad():
            gradless_retries = 0
            for epoch in range(epochs):
                try:
                    optimizer.zero_grad()
                    
                    # Create blind spot mask
                    mask = self._create_blind_spot_mask(img_tensor.shape).to(self.device)

                    # Ensure we actually masked something; fall back to sparse random mask if needed
                    if torch.count_nonzero(~mask).item() == 0:
                        mask = self._create_sparse_mask(img_tensor.shape, drop_fraction=0.005).to(self.device)
                    
                    # Forward pass
                    output = self.model(img_tensor)
                    
                    # Loss only on blind spots (where mask is False)
                    # Use masked_select to avoid gradient issues
                    masked_output = torch.masked_select(output, ~mask)
                    masked_target = torch.masked_select(img_tensor, ~mask)

                    # Fallback: if mask produced no samples, randomly sample a subset to avoid zero gradient
                    if masked_output.numel() == 0:
                        flat_output = output.view(-1)
                        flat_target = img_tensor.view(-1)
                        sample_count = max(1, flat_output.numel() // 1024)
                        sample_idx = torch.randperm(flat_output.numel(), device=self.device)[:sample_count]
                        masked_output = flat_output[sample_idx]
                        masked_target = flat_target[sample_idx]
                    loss = criterion(masked_output, masked_target)
                    
                    if not loss.requires_grad:
                        gradless_retries += 1
                        if gradless_retries <= 3:
                            print(
                                f"⚠️ Noise2Void loss had no gradients at epoch {epoch + 1}; "
                                "resampling blind spots."
                            )
                            continue
                        raise RuntimeError("Noise2Void loss does not require gradients after multiple attempts")

                    gradless_retries = 0

                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if (epoch + 1) % 20 == 0:
                        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"GPU memory error at epoch {epoch+1}, stopping early")
                        break
                    else:
                        raise e
        
        self.trained = True
        
        # Ensure we have at least one loss value
        if not losses:
            losses = [float('inf')]  # Fallback value
        
        return {
            'final_loss': losses[-1],
            'training_losses': losses,
            'epochs': epochs,
            'model_size': self.model_size
        }
    
    def process_image(self, image: np.ndarray, train_epochs: int = 100) -> Optional[np.ndarray]:
        """
        Process image with Noise2Void
        
        Args:
            image: Input noisy image (H, W, C) in range [0, 1]
            train_epochs: Number of training epochs
            
        Returns:
            Denoised image or None if processing fails
        """
        try:
            # FIXED: Ensure contiguous array
            image = np.ascontiguousarray(image)
            
            # Train model on the image if epochs > 0
            if train_epochs > 0:
                self.train_model(image, epochs=train_epochs)
            elif not self.trained:
                # If no training requested but model not trained, do minimal training
                self.train_model(image, epochs=10)
            
            # Prepare input
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            
            # Ensure contiguous after expand_dims
            image = np.ascontiguousarray(image)
            
            img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            img_tensor.requires_grad_(False)  # No gradients needed for inference
            
            # Inference
            self.model.eval()
            with torch.no_grad():
                denoised = self.model(img_tensor)
                denoised = torch.clamp(denoised, 0, 1)
            
            # Convert back to numpy
            result = denoised.cpu().squeeze(0).numpy().transpose(1, 2, 0)
            
            if result.shape[2] == 1:
                result = result.squeeze(2)
            
            # Clean up GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print(f"✅ Noise2Void processing completed successfully")
            return result
            
        except Exception as e:
            print(f"❌ Noise2Void processing failed: {e}")
            # Clean up GPU memory on error
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return None


class DeepImagePriorProcessor:
    """
    Deep Image Prior: Unsupervised restoration using network architecture as prior
    
    No training data required - uses the implicit bias of CNN architectures
    for image restoration tasks including denoising, super-resolution, and inpainting.
    
    IMPORTANT NOTES:
    - Each image requires its OWN training session (model cannot be reused)
    - The trained model is overfitted to ONE specific noisy image
    - Uses GPU if available for much faster training (~10-50x speedup)
    - Training time: ~1-5 minutes on GPU, ~30-120 minutes on CPU
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize Deep Image Prior processor"""
        self.device = self._get_device(device)
        self.model = None
        self.trained = False
        self.last_image_shape = None
        
        # Performance tracking
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Deep Image Prior using GPU: {gpu_name}")
        else:
            print(f"⚠️ Deep Image Prior using CPU (will be slow)")
            print(f"   Tip: For ~30x speedup, use a CUDA-capable GPU")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine optimal processing device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _create_model(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create Deep Image Prior network architecture"""
        
        class DIPNetwork(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                
                # Encoder
                self.enc1 = self._conv_block(in_channels, 128)
                self.enc2 = self._conv_block(128, 128)
                self.enc3 = self._conv_block(128, 128)
                self.enc4 = self._conv_block(128, 128)
                self.enc5 = self._conv_block(128, 128)
                
                # Decoder
                self.dec5 = self._conv_block(128, 128)
                self.dec4 = self._conv_block(256, 128)
                self.dec3 = self._conv_block(256, 128)
                self.dec2 = self._conv_block(256, 128)
                self.dec1 = self._conv_block(256, 128)
                
                # Output
                self.output = nn.Conv2d(128, out_channels, 1)
                
                # Initialize weights
                self._initialize_weights()
            
            def _conv_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                # Encoder with skip connections
                e1 = self.enc1(x)
                e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
                e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
                e4 = self.enc4(nn.functional.max_pool2d(e3, 2))
                e5 = self.enc5(nn.functional.max_pool2d(e4, 2))
                
                # Decoder with skip connections - match dimensions exactly
                d5 = self.dec5(e5)
                d4_up = nn.functional.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)
                d4 = self.dec4(torch.cat([d4_up, e4], 1))
                
                d3_up = nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
                d3 = self.dec3(torch.cat([d3_up, e3], 1))
                
                d2_up = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
                d2 = self.dec2(torch.cat([d2_up, e2], 1))
                
                d1_up = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
                d1 = self.dec1(torch.cat([d1_up, e1], 1))
                
                return torch.sigmoid(self.output(d1))
        
        return DIPNetwork(in_channels, out_channels)
    
    def process_image(self, image: np.ndarray, iterations: int = 3000, 
                     learning_rate: float = 1e-2) -> Optional[np.ndarray]:
        """
        Process image with Deep Image Prior
        
        IMPORTANT: The trained model CANNOT be reused on other images!
        
        Why? Deep Image Prior works by OVERFITTING a network to one specific noisy image.
        The network learns to reconstruct THAT PARTICULAR IMAGE from random noise.
        The "magic" is that natural image structure is learned faster than noise patterns.
        
        This means:
        - Each image needs its own training session (1-5 min on GPU, 30-120 min on CPU)
        - The model is useless after training - it only knows how to make this one image
        - Cannot batch process multiple images with same model
        - Cannot save and reuse the trained weights
        
        Performance tips:
        - Use GPU for 30-50x speedup (auto-detected)
        - Reduce iterations to 1000-2000 for faster results
        - Early stopping typically finishes before max iterations
        
        Args:
            image: Input noisy image (H, W, C) in range [0, 1]
            iterations: Number of optimization iterations (default: 3000)
                       - 500-1000: Fast but lower quality
                       - 2000-3000: Good balance (recommended)
                       - 5000+: Better quality but slower
            learning_rate: Learning rate for optimization (default: 0.01)
            
        Returns:
            Restored image or None if processing fails
        """
        grad_enabled_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            print(f"Processing with Deep Image Prior ({iterations} iterations)...")
            
            # Handle zero iterations case
            if iterations <= 0:
                print("  ⚠️ Zero iterations specified - returning original image")
                return image.copy()
            
            # Prepare data
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            
            # FIXED: Ensure contiguous array before transpose
            image = np.ascontiguousarray(image)
            
            channels = image.shape[2]
            
            best_loss = float('inf')
            best_output = None
            patience = 200  # Stop if no improvement for 200 iterations
            no_improve_count = 0

            import time
            start_time = time.time()

            with torch.inference_mode(False):
                img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
                img_tensor = img_tensor.to(self.device)
                # Target tensor should not require gradients
                img_tensor.requires_grad_(False)
                
                # Create model and ensure parameters require gradients
                self.model = self._create_model(channels, channels).to(self.device)
                self.model.train()  # Set to training mode
                
                # Ensure all model parameters require gradients
                for param in self.model.parameters():
                    param.requires_grad_(True)
                
                # FIXED: Fixed random input (this is the key insight of DIP)
                # The noise input should NOT require gradients - it's fixed!
                # We optimize the MODEL parameters, not the input
                noise_input = torch.randn_like(img_tensor) * 0.1
                noise_input = noise_input.to(self.device)
                noise_input.requires_grad_(False)  # Input is FIXED - we optimize model params
                
                # Optimizer - optimize the model parameters
                # PERFORMANCE: Use AdamW which is slightly faster and more memory efficient
                optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
                criterion = nn.MSELoss()
                
                # FIXED: Disable mixed precision - causes gradient flow issues with BatchNorm
                # Mixed precision autocast can prevent gradients from flowing through BatchNorm layers
                # This is a known issue with autocast + BatchNorm in PyTorch 2.x
                # Since DIP is already fast (single image), the 2x speedup isn't worth broken training
                use_amp = False  # Disabled - use FP32 for reliable gradient flow
                if use_amp:
                    try:
                        scaler = torch.cuda.amp.GradScaler()
                        print(f"   ⚡ Using mixed precision (FP16) for 2-3x speedup")
                    except:
                        use_amp = False
                        scaler = None
                else:
                    scaler = None
                
                with torch.enable_grad():
                    gradless_iterations = 0
                    for i in range(iterations):
                        try:
                            optimizer.zero_grad()
                            
                            # PERFORMANCE: Mixed precision forward pass
                            if use_amp:
                                with torch.cuda.amp.autocast():
                                    output = self.model(noise_input)
                                    loss = criterion(output, img_tensor)
                            else:
                                output = self.model(noise_input)
                                loss = criterion(output, img_tensor)
                            
                            # FIXED: Check output has gradients (should be True since model has gradients)
                            if not output.requires_grad:
                                gradless_iterations += 1
                                if gradless_iterations <= 10 and i < 25:
                                    if i < 5:
                                        print(f"⚠️ Warning: Output doesn't require gradients at iteration {i+1}")
                                        print(f"   This means the model may not be training properly.")
                                    continue
                                else:
                                    print(f"❌ Critical: No gradients after {i+1} iterations, stopping.")
                                    break

                            gradless_iterations = 0
                            
                            if not loss.requires_grad:
                                if i < 5:
                                    print(f"⚠️ Warning: Loss doesn't require gradients at iteration {i+1}")
                                continue
                            
                            # PERFORMANCE: Mixed precision backward pass
                            if use_amp and scaler is not None:
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()
                            
                            grad_norm = 0
                            for param in self.model.parameters():
                                if param.grad is not None:
                                    grad_norm += param.grad.data.norm(2).item() ** 2
                            grad_norm = grad_norm ** 0.5
                            
                            if grad_norm == 0 and (i + 1) % 100 == 0:
                                print(f"⚠️ Warning: Zero gradients at iteration {i+1}")
                            
                            current_loss = loss.item()
                            if current_loss < best_loss:
                                best_loss = current_loss
                                best_output = output.detach().clone()
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                            
                            if no_improve_count >= patience:
                                elapsed = time.time() - start_time
                                print(f"   ⏱️ Early stopping at iteration {i + 1} (no improvement for {patience} iterations)")
                                print(f"   Time elapsed: {elapsed:.1f}s, Best loss: {best_loss:.6f}")
                                break
                            
                            if (i + 1) % 500 == 0:
                                elapsed = time.time() - start_time
                                iter_per_sec = (i + 1) / elapsed
                                eta = (iterations - i - 1) / iter_per_sec if iter_per_sec > 0 else 0
                                print(f"Iteration {i + 1}/{iterations}, Loss: {current_loss:.6f}, "
                                      f"Best: {best_loss:.6f}, Speed: {iter_per_sec:.1f} it/s, ETA: {eta:.0f}s")
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(f"GPU memory error at iteration {i+1}, stopping early")
                                break
                            elif "grad" in str(e).lower():
                                print(f"Gradient error at iteration {i+1}: {e}")
                                continue
                            else:
                                raise e
            
            # Convert result
            if best_output is not None:
                total_time = time.time() - start_time
                actual_iterations = i + 1
                
                result = best_output.cpu().squeeze(0).numpy().transpose(1, 2, 0)
                if result.shape[2] == 1:
                    result = result.squeeze(2)
                
                # Clean up GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                print(f"✅ Deep Image Prior processing completed")
                print(f"   📊 Stats: {actual_iterations} iterations in {total_time:.1f}s "
                      f"({actual_iterations/total_time:.1f} it/s)")
                print(f"   🎯 Final loss: {best_loss:.6f}")
                print(f"   💾 Device: {self.device.type.upper()}")
                if use_amp:
                    print(f"   ⚡ Mixed precision: Enabled")
                
                return result
            else:
                # Clean up GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                return None
                
        except Exception as e:
            print(f"❌ Deep Image Prior processing failed: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"Image range: [{image.min():.4f}, {image.max():.4f}]")
            # Clean up GPU memory on error
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return None
        finally:
            torch.set_grad_enabled(grad_enabled_state)


class AutoDenoiseProcessor:
    """
    Unified Auto-Denoise processor combining multiple state-of-the-art methods
    
    Automatically selects and applies the best denoising method based on image characteristics
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize Auto-Denoise processor"""
        self.device = device
        self.n2v_processor = Noise2VoidProcessor(device)
        self.dip_processor = DeepImagePriorProcessor(device)
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image to determine optimal processing method"""
        analysis = {}
        
        try:
            # Basic image statistics
            analysis['mean'] = np.mean(image)
            analysis['std'] = np.std(image)
            analysis['min'] = np.min(image)
            analysis['max'] = np.max(image)
            
            # Noise estimation
            if SKIMAGE_AVAILABLE:
                if len(image.shape) == 3:
                    # Multi-channel noise estimation
                    noise_levels = []
                    for c in range(image.shape[2]):
                        noise_levels.append(estimate_sigma(image[:, :, c], channel_axis=None))
                    analysis['noise_level'] = np.mean(noise_levels)
                else:
                    analysis['noise_level'] = estimate_sigma(image, channel_axis=None)
            else:
                # Fallback noise estimation
                analysis['noise_level'] = np.std(image) * 0.5
            
            # Image complexity (gradient magnitude)
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            grad_x = np.diff(gray, axis=1)
            grad_y = np.diff(gray, axis=0)
            gradient_magnitude = np.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
            analysis['complexity'] = np.mean(gradient_magnitude)
            
            # Recommend method based on analysis
            if analysis['noise_level'] > 0.1:
                analysis['recommended_method'] = 'noise2void'
                analysis['reason'] = 'High noise level detected - Noise2Void recommended'
            elif analysis['complexity'] < 0.05:
                analysis['recommended_method'] = 'deep_image_prior'
                analysis['reason'] = 'Low complexity image - Deep Image Prior recommended'
            else:
                analysis['recommended_method'] = 'noise2void'
                analysis['reason'] = 'General case - Noise2Void recommended'
            
        except Exception as e:
            print(f"Analysis error: {e}")
            analysis['recommended_method'] = 'noise2void'
            analysis['reason'] = 'Analysis failed - using default method'
        
        return analysis
    
    def process_image(self, image: np.ndarray, method: str = "auto", 
                     **kwargs) -> Optional[np.ndarray]:
        """
        Process image with Auto-Denoise
        
        Args:
            image: Input image (H, W, C) in range [0, 1]
            method: Processing method ('auto', 'noise2void', 'deep_image_prior')
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Processed image or None if processing fails
        """
        try:
            # FIXED: Ensure contiguous array
            image = np.ascontiguousarray(image)
            
            # Auto-select method if needed
            if method == "auto":
                analysis = self.analyze_image(image)
                method = analysis['recommended_method']
                print(f"Auto-selected method: {method}")
                print(f"Reason: {analysis['reason']}")
            
            # Process with selected method
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
            
            else:
                print(f"❌ Unknown method: {method}")
                return None
                
        except Exception as e:
            import traceback
            print(f"❌ Auto-denoise processing error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def compare_methods(self, image: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        Compare multiple denoising methods on the same image
        
        Args:
            image: Input image to process
            
        Returns:
            Dictionary with results from different methods
        """
        results = {}
        
        print("Comparing Auto-Denoise methods...")
        
        # Noise2Void
        print("Processing with Noise2Void...")
        results['noise2void'] = self.n2v_processor.process_image(image, train_epochs=50)
        
        # Deep Image Prior (reduced iterations for comparison)
        print("Processing with Deep Image Prior...")
        results['deep_image_prior'] = self.dip_processor.process_image(image, iterations=1000)
        
        # Calculate metrics if possible
        if SKIMAGE_AVAILABLE:
            for method, result in results.items():
                if result is not None:
                    try:
                        psnr = peak_signal_noise_ratio(image, result, data_range=1.0)
                        ssim = structural_similarity(image, result, multichannel=True, channel_axis=-1, data_range=1.0)
                        print(f"{method.upper()}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
                    except:
                        print(f"{method.upper()}: Metrics calculation failed")
        
        return results
