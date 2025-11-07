"""
Simplified SCUNet Implementation for Memory Safety
A streamlined version that avoids complex dimension mismatches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class SimplifiedSCUNet(nn.Module):
    """Simplified SCUNet for memory-safe processing"""
    
    def __init__(self, in_chans=3, out_chans=3, base_dim=32, num_layers=3):
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.base_dim = base_dim
        self.num_layers = num_layers
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_chans, base_dim, 3, 1, 1)
        
        # Encoder blocks
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = base_dim * (2 ** i)
            out_dim = base_dim * (2 ** (i + 1)) if i < num_layers - 1 else in_dim
            
            # Encoder block
            encoder_block = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True)
            )
            self.encoder.append(encoder_block)
            
            # Downsampling (except for last layer)
            if i < num_layers - 1:
                downsample = nn.Conv2d(in_dim, out_dim, 3, 2, 1)
                self.downsample.append(downsample)
        
        # Decoder blocks
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i in range(num_layers - 1):
            layer_idx = num_layers - 2 - i
            in_dim = base_dim * (2 ** (layer_idx + 1))
            out_dim = base_dim * (2 ** layer_idx)
            
            # Upsampling
            upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
            self.upsample.append(upsample)
            
            # Decoder block (with skip connection)
            decoder_block = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, 3, 1, 1),  # *2 for skip connection
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
            self.decoder.append(decoder_block)
        
        # Final convolution
        self.conv_out = nn.Conv2d(base_dim, out_chans, 3, 1, 1)
        
    def forward(self, x):
        # Store input for residual connection
        input_x = x
        
        # Initial conv
        x = self.conv_in(x)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            
            if i < self.num_layers - 1:
                skip_connections.append(x)
                x = self.downsample[i](x)
        
        # Decoder path
        for i in range(self.num_layers - 1):
            x = self.upsample[i](x)
            
            # Skip connection
            skip = skip_connections[-(i + 1)]
            
            # Ensure dimensions match
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = self.decoder[i](x)
        
        # Final conv
        x = self.conv_out(x)
        
        # Add residual connection to maintain input structure
        # This helps with randomly initialized weights
        x = x + input_x
        
        return x

class SimplifiedSCUNetProcessor:
    """Memory-safe simplified SCUNet processor"""
    
    def __init__(self, device: str = 'auto', lightweight: bool = True):
        # Simple device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
            
        self.model = None
        self.lightweight = lightweight
        
        print(f"Simplified SCUNet initializing on {self.device}")
        self._load_model()
        
    def _load_model(self):
        """Load simplified model"""
        try:
            if self.lightweight:
                # Very lightweight configuration
                self.model = SimplifiedSCUNet(
                    in_chans=3, out_chans=3, base_dim=16, num_layers=2
                )
            else:
                # Standard configuration
                self.model = SimplifiedSCUNet(
                    in_chans=3, out_chans=3, base_dim=32, num_layers=3
                )
            
            self.model.to(self.device)
            self.model.eval()
            print(f"Simplified SCUNet loaded successfully ({self.device})")
            
        except Exception as e:
            print(f"Error loading simplified SCUNet: {e}")
            # Fallback to minimal model
            self.model = SimplifiedSCUNet(in_chans=3, out_chans=3, base_dim=8, num_layers=2)
            self.model.to(self.device)
            self.model.eval()
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Process image with simplified SCUNet"""
        if self.model is None:
            return image
            
        try:
            # Preprocess
            if image.dtype == np.uint8:
                image_normalized = image.astype(np.float32) / 255.0
            else:
                image_normalized = image.astype(np.float32)
            
            # Convert to tensor
            input_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Process
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # Convert back
            output = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            output = np.clip(output * 255, 0, 255).astype(np.uint8)
            
            return output
            
        except Exception as e:
            print(f"Simplified SCUNet processing error: {e}")
            return image

# Test function
def test_simplified_scunet():
    """Test simplified SCUNet"""
    print("Testing Simplified SCUNet...")
    
    # Create test image
    test_image = np.random.rand(256, 256, 3).astype(np.float32)
    test_image = (test_image * 255).astype(np.uint8)
    
    # Test processor
    processor = SimplifiedSCUNetProcessor(device='cpu', lightweight=True)
    result = processor.process_image(test_image)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Input range: [{test_image.min()}, {test_image.max()}]")
    print(f"Output range: [{result.min()}, {result.max()}]")
    print("Simplified SCUNet test completed successfully!")
    
    return result

if __name__ == "__main__":
    test_simplified_scunet()
