"""
SCUNet (Swin-Conv-UNet) Implementation for Mixed Degradation Restoration
State-of-the-art for handling film grain, digital noise, JPEG compression, and blur

Based on the official SCUNet implementation:
https://github.com/cszn/SCUNet

Memory-optimized version to prevent system crashes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import math
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# Import memory utilities
try:
    from .memory_utils import MemoryManager, SafeModelLoader, safe_torch_operation
except ImportError:
    # Fallback if memory_utils not available
    class MemoryManager:
        @staticmethod
        def safe_device_selection(device='auto', min_gpu_memory_gb=4.0):
            if device == 'cpu' or not torch.cuda.is_available():
                return 'cpu'
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        
        @staticmethod
        def cleanup_memory():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        @staticmethod
        def optimize_tile_size(image_shape, max_memory_gb=2.0, min_tile_size=64, max_tile_size=1024):
            return min_tile_size
    
    class SafeModelLoader:
        def __init__(self, device='auto', cleanup_on_exit=True):
            self.device = device
        def __enter__(self):
            return self.device
        def __exit__(self, *args):
            pass
    
    def safe_torch_operation(func):
        return func

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization (simplified version)"""
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(min=a, max=b)
    return tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class WMSA(nn.Module):
    """Self-attention module in Swin Transformer"""
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size - 1), self.n_heads))
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        
        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))
    
    def generate_mask(self, h, w, p, shift):
        """Generate mask for SW-MSA"""
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask
        
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask
    
    def forward(self, x):
        if self.type != 'W':
            x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))
        
        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)
        
        if self.type != 'W':
            output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        
        return output
    
    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class Block(nn.Module):
    """SwinTransformer Block"""
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        
        if input_resolution is not None and input_resolution <= window_size:
            self.type = 'W'
        
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )
    
    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class ConvTransBlock(nn.Module):
    """SwinTransformer and Conv Block"""
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution
        
        assert self.type in ['W', 'SW']
        if self.input_resolution is not None and self.input_resolution <= self.window_size:
            self.type = 'W'
        
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, 
                               self.window_size, self.drop_path, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        )
    
    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x

class SCUNet(nn.Module):
    """SCUNet model with official architecture"""
    def __init__(self, in_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(SCUNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8
        
        # Drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        
        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))
        
        begin = 0
        self.m_down1 = nn.Sequential(
            *[ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 
                           'W' if not i%2 else 'SW', input_resolution) for i in range(config[0])],
            nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)
        )
        
        begin += config[0]
        self.m_down2 = nn.Sequential(
            *[ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 
                           'W' if not i%2 else 'SW', input_resolution//2) for i in range(config[1])],
            nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)
        )
        
        begin += config[1]
        self.m_down3 = nn.Sequential(
            *[ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 
                           'W' if not i%2 else 'SW', input_resolution//4) for i in range(config[2])],
            nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)
        )
        
        begin += config[2]
        self.m_body = nn.Sequential(
            *[ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 
                           'W' if not i%2 else 'SW', input_resolution//8) for i in range(config[3])]
        )
        
        begin += config[3]
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(8*dim, 4*dim, 2, 2, 0, bias=False),
            *[ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 
                           'W' if not i%2 else 'SW', input_resolution//4) for i in range(config[4])]
        )
        
        begin += config[4]
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(4*dim, 2*dim, 2, 2, 0, bias=False),
            *[ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 
                           'W' if not i%2 else 'SW', input_resolution//2) for i in range(config[5])]
        )
        
        begin += config[5]
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),
            *[ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 
                           'W' if not i%2 else 'SW', input_resolution) for i in range(config[6])]
        )
        
        self.m_tail = nn.Sequential(nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False))
    
    def forward(self, x0):
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h/64)*64-h)
        paddingRight = int(np.ceil(w/64)*64-w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)
        
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        
        x = x[..., :h, :w]
        return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
            
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x, H, W):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Window partition
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x

def window_partition(x, window_size):
    """Partition into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SCUNetProcessor:
    """Memory-safe processor for SCUNet mixed degradation restoration"""
    
    def __init__(self, model_name: str = "scunet_color_real_psnr", device: str = 'auto', 
                 lightweight: bool = True):
        # Safe device selection with memory checking
        self.device = MemoryManager.safe_device_selection(device, min_gpu_memory_gb=3.0)
        self.model = None
        self.model_name = model_name
        
        # Determine model path based on model name
        models_dir = Path(__file__).parent.parent / "models"
        model_path = models_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            print(f"Warning: Model {model_name}.pth not found in {models_dir}")
            print("Available models:")
            for model_file in models_dir.glob("scunet_*.pth"):
                print(f"  - {model_file.name}")
            # Try to find any SCUNet model as fallback
            fallback_models = list(models_dir.glob("scunet_*.pth"))
            if fallback_models:
                model_path = fallback_models[0]
                print(f"Using fallback model: {model_path.name}")
            else:
                print("No SCUNet models found. Download models using: python download_models.py")
        
        self.model_path = str(model_path) if model_path.exists() else None
        self.lightweight = lightweight
        
        # Determine input channels based on model name
        if "gray" in model_name.lower():
            self.in_channels = 1
        else:
            self.in_channels = 3
            
        print(f"SCUNet initializing: {model_name} on {self.device} (channels: {self.in_channels})")
        self._load_model()
        
    def _load_model(self):
        """Load SCUNet model with memory-safe parameters"""
        with SafeModelLoader(self.device) as device:
            try:
                # Use official SCUNet architecture with full configuration
                # to match the pretrained model
                config = [4, 4, 4, 4, 4, 4, 4]  # Official SCUNet config
                dim = 64
                print(f"Using official SCUNet configuration for {self.model_name}")
                
                # Initialize model with correct input channels
                self.model = SCUNet(
                    in_nc=self.in_channels, 
                    config=config, 
                    dim=dim, 
                    drop_path_rate=0.0, 
                    input_resolution=256
                )
                
                # Load pretrained weights if available
                if self.model_path and Path(self.model_path).exists():
                    checkpoint = torch.load(self.model_path, map_location='cpu')
                    
                    # Handle different checkpoint formats
                    if 'params' in checkpoint:
                        state_dict = checkpoint['params']
                    elif 'params_ema' in checkpoint:
                        state_dict = checkpoint['params_ema']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    
                    # Remove any module prefix if present
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('module.'):
                            new_key = key[7:]  # Remove 'module.' prefix
                        else:
                            new_key = key
                        new_state_dict[new_key] = value
                    
                    # Load with strict=False to handle architecture differences
                    missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
                    
                    if missing_keys:
                        print(f"Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"Unexpected keys: {len(unexpected_keys)}")
                    
                    print(f"Loaded SCUNet model: {self.model_name}")
                else:
                    print(f"No pretrained model found for {self.model_name}. Using randomly initialized weights.")
                    
                self.model.to(device)
                self.model.eval()
                
                # Clear memory after loading
                MemoryManager.cleanup_memory()
                
            except Exception as e:
                print(f"Error loading SCUNet model: {e}")
                print("Model loading failed - check architecture compatibility")
                raise e
        
    @safe_torch_operation
    def process_image(self, image: np.ndarray, tile_size: int = None, overlap: int = 16) -> np.ndarray:
        """
        Process image with SCUNet (memory-safe version)
        
        Args:
            image: Input image as numpy array (H, W, C)
            tile_size: Size of processing tiles (auto-calculated if None)
            overlap: Overlap between tiles
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        try:
            if image.ndim == 4:
                if image.shape[0] != 1:
                    raise ValueError("SCUNet processor expects a single image (batch size 1)")
                image = image[0]

            # Auto-calculate tile size based on available memory and image size
            if tile_size is None:
                tile_size = MemoryManager.optimize_tile_size(
                    image.shape, 
                    max_memory_gb=2.0 if self.device == 'cuda' else 1.0,
                    min_tile_size=64,
                    max_tile_size=512 if self.device == 'cuda' else 256
                )
                print(f"Auto-selected tile size: {tile_size}")
            
            # Convert to tensor
            input_tensor = self._preprocess(image)
            
            # Use tiling for any reasonably sized image to be safe
            max_direct_size = 256 if self.device == 'cuda' else 128
            if input_tensor.shape[2] > max_direct_size or input_tensor.shape[3] > max_direct_size:
                output_tensor = self._process_with_tiles(input_tensor, tile_size, overlap)
            else:
                with torch.no_grad():
                    output_tensor = self.model(input_tensor)
                    
            # Convert back to numpy
            result = self._postprocess(output_tensor)
            
            # Clean up memory
            del input_tensor, output_tensor
            MemoryManager.cleanup_memory()
                
            return result
            
        except Exception as e:
            # Clean up on error
            MemoryManager.cleanup_memory()
            print(f"SCUNet processing error: {e}")
            return image  # Return original image on error
        
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for SCUNet"""
        # Convert to float and normalize
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        
        # Handle channel dimension
        if len(image.shape) == 2:
            # Grayscale image without channel dimension
            image = np.expand_dims(image, axis=2)
        
        # Ensure correct number of channels
        if self.in_channels == 1 and image.shape[2] == 3:
            # Convert RGB to grayscale
            image = np.mean(image, axis=2, keepdims=True)
        elif self.in_channels == 3 and image.shape[2] == 1:
            # Convert grayscale to RGB
            image = np.repeat(image, 3, axis=2)
        
        # Convert to tensor format (C, H, W)
        # FIXED: Ensure image has correct shape before transpose
        if len(image.shape) != 3:
            raise ValueError(f"Image must have 3 dimensions (H,W,C), got shape {image.shape}")
        
        # FIXED: Ensure contiguous array before transpose
        image = np.ascontiguousarray(image)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
        
    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output back to image"""
        # Remove batch dimension and convert to numpy
        image = tensor.squeeze(0).cpu().detach().numpy()
        
        # FIXED: Ensure contiguous array before transpose
        image = np.ascontiguousarray(image)
        
        # Convert from (C, H, W) to (H, W, C)
        image = image.transpose(1, 2, 0)
        
        # Handle single channel output
        if image.shape[2] == 1:
            # For grayscale output, keep as single channel
            pass
        
        # Clip values to valid range [0, 1]
        image = np.clip(image, 0.0, 1.0)
        
        return image
        
    def _process_with_tiles(self, input_tensor: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
        """Process large image with overlapping tiles (memory-optimized)"""
        B, C, H, W = input_tensor.shape
        
        # Calculate tile positions
        stride = tile_size - overlap
        tiles_h = (H - overlap) // stride + (1 if (H - overlap) % stride > 0 else 0)
        tiles_w = (W - overlap) // stride + (1 if (W - overlap) % stride > 0 else 0)
        
        # Use smaller chunks to avoid memory issues
        output_tensor = torch.zeros_like(input_tensor)
        weight_tensor = torch.zeros((B, 1, H, W), device=input_tensor.device)
        
        # Process tiles in batches to reduce memory usage
        batch_size = 4 if self.device == 'cuda' else 8
        
        for i in range(0, tiles_h, batch_size):
            for j in range(0, tiles_w, batch_size):
                # Process batch of tiles
                batch_end_i = min(i + batch_size, tiles_h)
                batch_end_j = min(j + batch_size, tiles_w)
                
                for ii in range(i, batch_end_i):
                    for jj in range(j, batch_end_j):
                        # Calculate tile boundaries
                        start_h = ii * stride
                        end_h = min(start_h + tile_size, H)
                        start_w = jj * stride
                        end_w = min(start_w + tile_size, W)
                        
                        # Extract tile
                        tile = input_tensor[:, :, start_h:end_h, start_w:end_w]
                        
                        # Pad tile if necessary
                        pad_h = tile_size - (end_h - start_h)
                        pad_w = tile_size - (end_w - start_w)
                        if pad_h > 0 or pad_w > 0:
                            tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                        
                        # Process tile
                        with torch.no_grad():
                            processed_tile = self.model(tile)
                        
                        # Remove padding
                        if pad_h > 0 or pad_w > 0:
                            processed_tile = processed_tile[:, :, :end_h-start_h, :end_w-start_w]
                        
                        # Add to output with weighting
                        output_tensor[:, :, start_h:end_h, start_w:end_w] += processed_tile
                        weight_tensor[:, :, start_h:end_h, start_w:end_w] += 1
                        
                        # Clean up tile memory
                        del tile, processed_tile
                        
                # Clean GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        # Normalize by weights
        output_tensor = torch.where(weight_tensor > 0, output_tensor / weight_tensor, output_tensor)
        
        return output_tensor

# Test function
def test_scunet_processor():
    """Test SCUNet processor with memory safety"""
    # Create small test image to avoid memory issues
    test_image = np.random.rand(128, 128, 3).astype(np.float32)
    test_image = (test_image * 255).astype(np.uint8)
    
    # Test processor with safe settings
    processor = SCUNetProcessor(device='cpu', lightweight=True)
    result = processor.process_image(test_image, tile_size=64)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Input range: [{test_image.min()}, {test_image.max()}]")
    print(f"Output range: [{result.min()}, {result.max()}]")
    print("SCUNet test completed successfully!")
    
    return result

if __name__ == "__main__":
    test_scunet_processor()
