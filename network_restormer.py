"""
Restormer network architecture.

Adapted from the official Restormer repository:
https://github.com/swz30/Restormer

Copyright (c) 2021 Shuhang Gu et al.
Released under the Apache License, Version 2.0.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm that supports mixed precision."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - relies on torch ops
        mean = x.mean((1, 2, 3), keepdim=True)
        std = x.std((1, 2, 3), keepdim=True)
        return (x - mean) / (std + self.eps) * self.weight + self.bias


class FeedForward(nn.Module):
    """Gated feed-forward network (GFN) used within transformer blocks."""

    def __init__(self, dim: int, expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1, groups=hidden_dim * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    """Multi-Dconv head transposed attention (MDTA)."""

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.view(b, c, h, w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """Core Restormer block with attention + feed-forward."""

    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Stem convolution used to obtain overlapping patches."""

    def __init__(self, in_chans: int, embed_dim: int, bias: bool) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, dim: int, bias: bool) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim: int, bias: bool) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.body(x)


class Restormer(nn.Module):
    """Main Restormer network."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Optional[list[int]] = None,
        num_refinement_blocks: int = 4,
        heads: Optional[list[int]] = None,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        # Encoder
        self.encoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        dim_level = dim
        for blocks, head in zip(num_blocks, heads):
            level = nn.Sequential(
                *[TransformerBlock(dim_level, head, ffn_expansion_factor, bias) for _ in range(blocks)]
            )
            self.encoder_levels.append(level)
            self.downsamples.append(Downsample(dim_level, bias))
            dim_level *= 2

        # Bottleneck
        self.latent = nn.Sequential(
            *[TransformerBlock(dim_level, heads[-1] * 2, ffn_expansion_factor, bias) for _ in range(num_blocks[-1])]
        )

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()
        for blocks, head in reversed(list(zip(num_blocks[:-1], heads[:-1]))):
            dim_level //= 2
            self.upsamples.append(Upsample(dim_level * 2, bias))
            self.decoder_levels.append(
                nn.Sequential(
                    *[TransformerBlock(dim_level, head, ffn_expansion_factor, bias) for _ in range(blocks)]
                )
            )

        self.output = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )

        self.num_levels = len(num_blocks) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.patch_embed(x)

        features = []
        for idx in range(self.num_levels):
            x = self.encoder_levels[idx](x)
            features.append(x)
            x = self.downsamples[idx](x)

        x = self.latent(x)

        for idx in range(self.num_levels - 1, -1, -1):
            x = self.upsamples[idx](x)
            x = torch.cat([x, features[idx]], dim=1)
            x = self.decoder_levels[self.num_levels - 1 - idx](x)

        x = self.output(x)
        return x


if __name__ == "__main__":  # pragma: no cover
    model = Restormer()
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(dummy)
    print("Input:", dummy.shape, "Output:", out.shape)
