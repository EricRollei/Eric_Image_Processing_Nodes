"""
Training Script for Film Grain Denoising Models
Supports both FGA-NN and Lightweight Progressive CNN

This script provides a complete training pipeline with:
- Dataset loading and augmentation
- Training loop with validation
- Model checkpointing
- TensorBoard logging
- Progress visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.fga_nn_architecture import FGANNModel
from models.progressive_cnn_architecture import ProgressiveCNNModel


class FilmGrainDataset(Dataset):
    """
    Dataset for film grain denoising training
    
    Expects directory structure:
    dataset_root/
        clean/
            image1.png
            image2.png
            ...
        noisy/
            image1.png
            image2.png
            ...
    
    OR for self-supervised (clean images only):
    dataset_root/
        images/
            image1.png
            image2.png
            ...
    """
    
    def __init__(self, root_dir: str, mode: str = 'paired', patch_size: int = 128, 
                 noise_level: float = 0.0, augment: bool = True):
        """
        Args:
            root_dir: Root directory containing images
            mode: 'paired' (clean+noisy) or 'self_supervised' (clean only, add synthetic noise)
            patch_size: Size of random crops during training
            noise_level: Noise level for self-supervised mode (0.0-1.0)
            augment: Whether to apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.noise_level = noise_level
        self.augment = augment
        
        if mode == 'paired':
            self.clean_dir = self.root_dir / 'clean'
            self.noisy_dir = self.root_dir / 'noisy'
            
            # Get list of image files
            self.clean_images = sorted(list(self.clean_dir.glob('*.png')) + 
                                      list(self.clean_dir.glob('*.jpg')))
            self.noisy_images = sorted(list(self.noisy_dir.glob('*.png')) + 
                                      list(self.noisy_dir.glob('*.jpg')))
            
            assert len(self.clean_images) == len(self.noisy_images), \
                "Number of clean and noisy images must match"
            
            print(f"✓ Loaded {len(self.clean_images)} paired images")
            
        else:  # self_supervised
            self.images_dir = self.root_dir / 'images'
            self.clean_images = sorted(list(self.images_dir.glob('*.png')) + 
                                      list(self.images_dir.glob('*.jpg')))
            
            print(f"✓ Loaded {len(self.clean_images)} images for self-supervised training")
            print(f"  Synthetic noise level: {noise_level}")
    
    def __len__(self):
        return len(self.clean_images)
    
    def add_film_grain_noise(self, image: np.ndarray, noise_level: float) -> np.ndarray:
        """Add realistic film grain noise"""
        h, w, c = image.shape
        
        # Generate base Gaussian noise
        noise = np.random.normal(0, noise_level, (h, w, c))
        
        # Add spatially correlated component (film grain characteristic)
        kernel_size = 3
        kernel = cv2.getGaussianKernel(kernel_size, 0.5)
        kernel = kernel @ kernel.T
        
        for i in range(c):
            noise[:, :, i] = cv2.filter2D(noise[:, :, i], -1, kernel)
        
        # Add intensity-dependent noise (brighter areas have more grain)
        intensity_factor = 1.0 + 0.5 * image
        noise = noise * intensity_factor
        
        # Add noise to image
        noisy = image + noise
        noisy = np.clip(noisy, 0, 1)
        
        return noisy.astype(np.float32)
    
    def random_crop(self, img1: np.ndarray, img2: np.ndarray = None) -> tuple:
        """Random crop to patch_size"""
        h, w = img1.shape[:2]
        
        if h < self.patch_size or w < self.patch_size:
            # If image is smaller than patch_size, resize
            scale = max(self.patch_size / h, self.patch_size / w)
            new_h, new_w = int(h * scale) + 1, int(w * scale) + 1
            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if img2 is not None:
                img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = img1.shape[:2]
        
        # Random crop
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)
        
        img1_crop = img1[top:top+self.patch_size, left:left+self.patch_size]
        
        if img2 is not None:
            img2_crop = img2[top:top+self.patch_size, left:left+self.patch_size]
            return img1_crop, img2_crop
        
        return img1_crop
    
    def augment_data(self, clean: np.ndarray, noisy: np.ndarray) -> tuple:
        """Apply random augmentations"""
        # Random flip
        if np.random.rand() > 0.5:
            clean = np.fliplr(clean)
            noisy = np.fliplr(noisy)
        
        if np.random.rand() > 0.5:
            clean = np.flipud(clean)
            noisy = np.flipud(noisy)
        
        # Random rotation (90, 180, 270)
        k = np.random.randint(0, 4)
        clean = np.rot90(clean, k)
        noisy = np.rot90(noisy, k)
        
        return clean.copy(), noisy.copy()
    
    def __getitem__(self, idx):
        # Load clean image
        clean_path = self.clean_images[idx]
        clean = cv2.imread(str(clean_path))
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        clean = clean.astype(np.float32) / 255.0
        
        if self.mode == 'paired':
            # Load corresponding noisy image
            noisy_path = self.noisy_images[idx]
            noisy = cv2.imread(str(noisy_path))
            noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
            noisy = noisy.astype(np.float32) / 255.0
            
            # Random crop
            clean, noisy = self.random_crop(clean, noisy)
            
            # Augmentation
            if self.augment:
                clean, noisy = self.augment_data(clean, noisy)
        
        else:  # self_supervised
            # Random crop
            clean = self.random_crop(clean)
            
            # Add synthetic noise
            noisy = self.add_film_grain_noise(clean, self.noise_level)
            
            # Augmentation
            if self.augment:
                clean, noisy = self.augment_data(clean, noisy)
        
        # Convert to torch tensors [C, H, W]
        clean = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        noisy = torch.from_numpy(noisy.transpose(2, 0, 1)).float()
        
        return noisy, clean


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use VGG16 features
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
        
        # Extract feature layers
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # Extract features
        pred_features = self.features(pred_norm)
        target_features = self.features(target_norm)
        
        # Calculate loss
        loss = nn.functional.mse_loss(pred_features, target_features)
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function for training"""
    
    def __init__(self, use_perceptual: bool = True, perceptual_weight: float = 0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        
        if use_perceptual:
            try:
                self.perceptual_loss = PerceptualLoss()
                print("✓ Using perceptual loss")
            except Exception as e:
                print(f"⚠ Could not initialize perceptual loss: {e}")
                print("  Falling back to pixel-wise loss only")
                self.use_perceptual = False
    
    def forward(self, pred, target):
        # Pixel-wise losses
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        
        loss = 0.5 * mse + 0.5 * l1
        
        # Add perceptual loss if enabled
        if self.use_perceptual:
            perceptual = self.perceptual_loss(pred, target)
            loss = loss + self.perceptual_weight * perceptual
        
        return loss


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_psnr = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (noisy, clean) in enumerate(pbar):
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Handle different model outputs (FGA-NN can return params)
        output = model(noisy)
        if isinstance(output, tuple):
            output = output[0]  # Get denoised image only
        
        # Calculate loss
        loss = criterion(output, clean)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        psnr = calculate_psnr(output.detach(), clean)
        
        total_loss += loss.item()
        total_psnr += psnr.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr.item():.2f}dB'
        })
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/PSNR', psnr.item(), global_step)
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    
    return avg_loss, avg_psnr


def validate(model, dataloader, criterion, device, epoch, writer=None):
    """Validate the model"""
    model.eval()
    
    total_loss = 0.0
    total_psnr = 0.0
    
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc='Validation'):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward pass
            output = model(noisy)
            if isinstance(output, tuple):
                output = output[0]
            
            # Calculate loss
            loss = criterion(output, clean)
            psnr = calculate_psnr(output, clean)
            
            total_loss += loss.item()
            total_psnr += psnr.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    
    if writer is not None:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        
        # Log sample images
        if epoch % 5 == 0:
            writer.add_images('Val/Noisy', noisy[:4], epoch)
            writer.add_images('Val/Clean', clean[:4], epoch)
            writer.add_images('Val/Denoised', output[:4], epoch)
    
    return avg_loss, avg_psnr


def save_checkpoint(model, optimizer, epoch, loss, psnr, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'psnr': psnr,
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / f'best_{save_path.name}'
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model with PSNR: {psnr:.2f}dB")


def main():
    parser = argparse.ArgumentParser(description='Train Film Grain Denoising Models')
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['fgann', 'progressive_cnn'], required=True,
                       help='Model to train: fgann or progressive_cnn')
    
    # Dataset
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--mode', type=str, choices=['paired', 'self_supervised'], default='paired',
                       help='Training mode: paired (clean+noisy) or self_supervised (clean only)')
    parser.add_argument('--noise_level', type=float, default=0.05,
                       help='Noise level for self-supervised mode (0.0-1.0)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Training patch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Loss function
    parser.add_argument('--use_perceptual', action='store_true',
                       help='Use perceptual loss')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                       help='Weight for perceptual loss')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./training_output',
                       help='Directory to save checkpoints and logs')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.model / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize model
    print(f"\n{'='*60}")
    print(f"Initializing {args.model.upper()} model")
    print(f"{'='*60}")
    
    if args.model == 'fgann':
        model = FGANNModel(in_channels=3).to(device)
    else:  # progressive_cnn
        model = ProgressiveCNNModel(in_channels=3).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                      patience=10, verbose=True)
    
    # Initialize loss function
    criterion = CombinedLoss(use_perceptual=args.use_perceptual, 
                            perceptual_weight=args.perceptual_weight)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['psnr']
        print(f"✓ Resumed from epoch {start_epoch} with PSNR {best_psnr:.2f}dB")
    
    # Create dataset
    print(f"\n{'='*60}")
    print(f"Loading dataset from: {args.data_root}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}")
    
    full_dataset = FilmGrainDataset(
        root_dir=args.data_root,
        mode=args.mode,
        patch_size=args.patch_size,
        noise_level=args.noise_level,
        augment=True
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_psnr = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f}dB")
        
        # Validate
        val_loss, val_psnr = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        print(f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f}dB")
        
        # Learning rate scheduling
        scheduler.step(val_psnr)
        
        # Save checkpoint
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
        
        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, val_psnr, 
                          checkpoint_path, is_best)
    
    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Training complete! Final model saved to {final_path}")
    print(f"✓ Best validation PSNR: {best_psnr:.2f}dB")
    
    writer.close()


if __name__ == '__main__':
    main()
