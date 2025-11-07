"""
NAFNet: Nonlinear Activation Free Network for Image Restoration
State-of-the-art denoising architecture with pre-trained weights

Paper: "Simple Baselines for Image Restoration"
Chen et al., ECCV 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import urllib.request
import urllib.error
import shutil


class LayerNormFunction(torch.autograd.Function):
    """Custom LayerNorm for better performance"""
    
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """2D Layer Normalization"""
    
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps
    
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Simple gating mechanism - no activation functions needed"""
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    NAF Block - Nonlinear Activation Free block
    Uses simple gating instead of complex activations
    """
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        # Layer normalization
        self.norm1 = LayerNorm2d(c)
        
        # Depthwise convolution branch
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simple gate
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()
        
        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        
        # Feed-forward network
        self.norm2 = LayerNorm2d(c)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    
    def forward(self, inp):
        x = inp
        
        # First branch: depthwise convolution with gating
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        
        y = inp + x * self.beta
        
        # Second branch: feed-forward network
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet Model - Nonlinear Activation Free Network
    
    Simple yet effective architecture without ReLU or other activations.
    Uses gating mechanisms and layer normalization instead.
    """
    
    def __init__(self, img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
        
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
        
        self.padder_size = 2 ** len(self.encoders)
    
    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        x = self.intro(inp)
        
        encs = []
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.ending(x)
        x = x + inp
        
        return x[:, :, :H, :W]
    
    def check_image_size(self, x):
        """Pad image to be divisible by padder_size"""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetProcessor:
    """High-level processor for NAFNet with automatic weight downloading"""
    
    PRETRAINED_MODELS = {
        'nafnet-width32': {
            'url': 'https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-GoPro-width32.pth',
            'width': 32,
            'description': 'NAFNet width-32 (lighter, faster)',
            'task': 'image denoising'
        },
        'nafnet-width64': {
            'url': 'https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-GoPro-width64.pth',
            'width': 64,
            'description': 'NAFNet width-64 (balanced)',
            'task': 'image denoising'
        },
        'nafnet-sidd': {
            'url': 'https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-SIDD-width64.pth',
            'width': 64,
            'description': 'NAFNet trained on SIDD (real-world noise)',
            'task': 'real-world denoising'
        }
    }
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.current_model_name = None
        self.weights_dir = Path(__file__).parent / 'pretrained_weights'
        self.weights_dir.mkdir(exist_ok=True)
    
    def download_weights(self, model_name: str) -> Path:
        """Download pre-trained weights if not already cached"""
        if model_name not in self.PRETRAINED_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.PRETRAINED_MODELS.keys())}")
        
        model_info = self.PRETRAINED_MODELS[model_name]
        weights_path = self.weights_dir / f"{model_name}.pth"
        url_filename = Path(model_info['url']).name
        candidate_paths = [weights_path]
        if url_filename:
            candidate_paths.append(self.weights_dir / url_filename)

        for candidate in candidate_paths:
            if candidate.exists():
                if candidate != weights_path and not weights_path.exists():
                    try:
                        shutil.copy2(candidate, weights_path)
                        print(f"✓ Using manually provided weights: {candidate}")
                        return weights_path
                    except OSError:
                        print(f"✓ Using manually provided weights: {candidate}")
                        return candidate
                print(f"✓ Using cached weights: {candidate}")
                return candidate
        
        print(f"📥 Downloading {model_name} weights...")
        print(f"   URL: {model_info['url']}")
        print(f"   This may take a few minutes...")
        
        try:
            # Add headers to mimic browser request
            req = urllib.request.Request(
                model_info['url'],
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            # Download with progress indication
            with urllib.request.urlopen(req, timeout=60) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 8192
                
                with open(weights_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
                
                print()  # New line after progress
            
            print(f"✓ Downloaded successfully to: {weights_path}")
            return weights_path
            
        except urllib.error.HTTPError as e:
            print(f"\n✗ HTTP Error {e.code}: {e.reason}")
            print(f"   URL may be invalid or model not available.")
            print(f"   Please check: {model_info['url']}")
            print(f"\n   Alternative: Download manually and place at:")
            print(f"   {weights_path}")
            raise
        except urllib.error.URLError as e:
            print(f"\n✗ Network error: {e.reason}")
            print(f"   Check your internet connection and firewall settings.")
            raise
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _infer_architecture_from_state(self, state_dict: Dict) -> Tuple[list, list, int]:
        """Infer block configuration from a NAFNet state dict."""
        params = state_dict.get('params', state_dict)
        encoder_blocks: Dict[int, set] = {}
        decoder_blocks: Dict[int, set] = {}
        middle_blocks: set = set()

        for key in params.keys():
            if key.startswith('encoders') and key.endswith('beta'):
                parts = key.split('.')
                if len(parts) >= 3:
                    stage = int(parts[1])
                    block = int(parts[2])
                    encoder_blocks.setdefault(stage, set()).add(block)
            elif key.startswith('decoders') and key.endswith('beta'):
                parts = key.split('.')
                if len(parts) >= 3:
                    stage = int(parts[1])
                    block = int(parts[2])
                    decoder_blocks.setdefault(stage, set()).add(block)
            elif key.startswith('middle_blks') and key.endswith('beta'):
                parts = key.split('.')
                if len(parts) >= 2:
                    middle_blocks.add(int(parts[1]))

        def to_counts(mapping: Dict[int, set]) -> list:
            if not mapping:
                return []
            return [len(mapping[idx]) for idx in sorted(mapping.keys())]

        enc_blk_nums = to_counts(encoder_blocks)
        dec_blk_nums = to_counts(decoder_blocks)
        middle_blk_num = len(middle_blocks) if middle_blocks else 0

        return enc_blk_nums, dec_blk_nums, middle_blk_num

    def load_model(self, model_name: str = 'nafnet-width32'):
        """Load NAFNet model with pre-trained weights"""
        if self.current_model_name == model_name and self.model is not None:
            return  # Already loaded
        
        # Download weights if needed
        weights_path = self.download_weights(model_name)
        
        # Get model configuration
        model_info = self.PRETRAINED_MODELS[model_name]
        width = model_info['width']
        
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            
            # Handle different state dict formats
            if 'params' in state_dict:
                state_dict = state_dict['params']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            enc_blk_nums, dec_blk_nums, middle_blk_num = self._infer_architecture_from_state(state_dict)
            if not enc_blk_nums:
                enc_blk_nums = model_info.get('enc_blk_nums', [2, 2, 4, 8])
            if not dec_blk_nums:
                dec_blk_nums = model_info.get('dec_blk_nums', [2, 2, 2, 2])
            if middle_blk_num == 0:
                middle_blk_num = model_info.get('middle_blk_num', 12)

            self.model = NAFNet(
                img_channel=3,
                width=width,
                middle_blk_num=middle_blk_num,
                enc_blk_nums=enc_blk_nums,
                dec_blk_nums=dec_blk_nums,
            )
            
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            
            self.current_model_name = model_name
            print(f"✓ Loaded {model_name}: {model_info['description']}")
            
        except Exception as e:
            print(f"✗ Error loading weights: {e}")
            raise
    
    def process_image(self, image: np.ndarray, model_name: str = 'nafnet-width32') -> Tuple[np.ndarray, Dict]:
        """
        Process image with NAFNet
        
        Args:
            image: Input image [H, W, C] or [H, W] in range [0, 1]
            model_name: Which pre-trained model to use
            
        Returns:
            denoised_image: Denoised result
            info: Processing information
        """
        # Load model if needed
        self.load_model(model_name)
        
        # Prepare input (NAFNet expects RGB)
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=2)
        
        # Convert to tensor [1, C, H, W]
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
        img_tensor = img_tensor.to(self.device)
        
        if self.model is None:
            raise RuntimeError("NAFNet model not loaded. Call load_model first.")

        # Process
        with torch.no_grad():
            denoised_tensor = self.model(img_tensor)
        
        # Convert back to numpy
        denoised = denoised_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        
        # Clip to valid range
        denoised = np.clip(denoised, 0, 1)
        
        # Calculate metrics
        model_key: str = self.current_model_name if self.current_model_name is not None else model_name
        info = {
            'model': model_name,
            'description': self.PRETRAINED_MODELS[model_key]['description'],
            'psnr': self._calculate_psnr(image, denoised),
            'output_range': [float(denoised.min()), float(denoised.max())]
        }
        
        return denoised, info
    
    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate PSNR between two images"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    def get_model_info(self) -> Dict:
        """Get information about current model"""
        if self.model is None:
            return {"status": "Not loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        current_name = self.current_model_name if self.current_model_name is not None else "unknown"
        return {
            "model_name": current_name,
            "description": self.PRETRAINED_MODELS.get(current_name, {}).get('description', 'Unknown model'),
            "total_parameters": total_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "device": str(self.device),
            "status": "Loaded"
        }
    
    @classmethod
    def list_available_models(cls) -> Dict:
        """List all available pre-trained models"""
        return cls.PRETRAINED_MODELS
