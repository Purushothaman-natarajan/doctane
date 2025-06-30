import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

# ----------------------------- #
# Helper functions for reshaping
# ----------------------------- #

def to_3d(x):
    """Convert 4D tensor (B, C, H, W) to 3D (B, HW, C)."""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """Convert 3D tensor (B, HW, C) back to 4D (B, C, H, W)."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# ----------------------------- #
# Custom LayerNorm Variants
# ----------------------------- #

class BiasFree_LayerNorm(nn.Module):
    """LayerNorm without learnable bias term."""
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """LayerNorm with both weight and bias terms."""
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Switch between BiasFree and WithBias LayerNorm."""
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# ----------------------------- #
# Feed-forward Network with FFT
# ----------------------------- #

class DFFN(nn.Module):
    """FFT-augmented feed-forward network."""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, 1, 1, groups=hidden_dim * 2, bias=bias)
        self.fft = nn.Parameter(torch.ones((hidden_dim * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)

        # Patch-wise FFT
        x_patch = rearrange(x, 'b c (h ph) (w pw) -> b c h w ph pw', ph=self.patch_size, pw=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float()) * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w ph pw -> b c (h ph) (w pw)', ph=self.patch_size, pw=self.patch_size)

        # Channel-wise FFN
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

# ----------------------------- #
# Frequency Self-Attention with FFT
# ----------------------------- #

class FSAS(nn.Module):
    """Frequency self-attention mechanism."""
    def __init__(self, dim, bias):
        super().__init__()
        self.patch_size = 8
        self.to_hidden = nn.Conv2d(dim, dim * 6, 1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, 3, 1, 1, groups=dim * 6, bias=bias)
        self.norm = LayerNorm(dim * 2, 'WithBias')
        self.project_out = nn.Conv2d(dim * 2, dim, 1, bias=bias)

    def forward(self, x):
        hidden = self.to_hidden_dw(self.to_hidden(x))
        q, k, v = hidden.chunk(3, dim=1)

        # FFT-based attention computation
        q_patch = rearrange(q, 'b c (h ph) (w pw) -> b c h w ph pw', ph=self.patch_size, pw=self.patch_size)
        k_patch = rearrange(k, 'b c (h ph) (w pw) -> b c h w ph pw', ph=self.patch_size, pw=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out = torch.fft.irfft2(q_fft * k_fft, s=(self.patch_size, self.patch_size))

        out = rearrange(out, 'b c h w ph pw -> b c (h ph) (w pw)', ph=self.patch_size, pw=self.patch_size)
        out = self.norm(out)

        return self.project_out(v * out)

# ----------------------------- #
# Core Transformer Block
# ----------------------------- #

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers."""
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = FSAS(dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Residual connection after attention
        x = x + self.ffn(self.norm2(x))   # Residual connection after FFN
        return x

# ----------------------------- #
# Multi-Scale Feature Fusion
# ----------------------------- #

class Fuse(nn.Module):
    """Fuse decoder features with skip connections from encoder."""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        self.project = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.blocks = nn.Sequential(
            TransformerBlock(dim, ffn_expansion_factor, bias),
            TransformerBlock(dim, ffn_expansion_factor, bias)
        )

    def forward(self, x, res):
        x = self.project(torch.cat([x, res], dim=1))  # Concatenate and reduce channels
        return self.blocks(x)

# ----------------------------- #
# Patch Embedding (First Layer)
# ----------------------------- #

class OverlapPatchEmbed(nn.Module):
    """Initial patch embedding with overlapping convolution."""
    def __init__(self, in_channels, embed_dim, bias):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

# ----------------------------- #
# Downsample Block
# ----------------------------- #

class Downsample(nn.Module):
    """Downsampling with 3x3 stride-2 convolution."""
    def __init__(self, in_dim, out_dim, bias):
        super().__init__()
        self.body = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=bias)

    def forward(self, x):
        return self.body(x)

# ----------------------------- #
# Upsample Block
# ----------------------------- #

class Upsample(nn.Module):
    """Upsampling with nearest neighbor interpolation + 3x3 convolution."""
    def __init__(self, in_dim, out_dim, bias):
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        return self.body(x)

# ----------------------------- #
# FFTFormer Model Architecture
# ----------------------------- #

class FFTFormer(nn.Module):
    """Main FFTFormer network for image-to-image tasks."""
    def __init__(self, inp_channels=3, out_channels=3, dim=36, num_blocks=[2, 2, 4, 8],
                 ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        # Encoder
        self.encoder1 = nn.Sequential(*[
            TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        self.down1 = Downsample(dim, dim * 2, bias)

        self.encoder2 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        self.down2 = Downsample(dim * 2, dim * 4, bias)

        self.encoder3 = nn.Sequential(*[
            TransformerBlock(dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])
        self.down3 = Downsample(dim * 4, dim * 8, bias)

        # Latent (Bottleneck)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=dim * 8, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        # Decoder
        self.up3 = Upsample(dim * 8, dim * 4, bias)
        self.fuse3 = Fuse(dim * 4, ffn_expansion_factor, bias)

        self.up2 = Upsample(dim * 4, dim * 2, bias)
        self.fuse2 = Fuse(dim * 2, ffn_expansion_factor, bias)

        self.up1 = Upsample(dim * 2, dim, bias)
        self.fuse1 = Fuse(dim, ffn_expansion_factor, bias)

        self.output_proj = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        # Encoder
        x1 = self.patch_embed(x)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(self.down1(x1))
        x3 = self.encoder3(self.down2(x2))
        x4 = self.latent(self.down3(x3))

        # Decoder with skip connections
        x = self.fuse3(self.up3(x4), x3)
        x = self.fuse2(self.up2(x), x2)
        x = self.fuse1(self.up1(x), x1)

        # Final output
        return self.output_proj(x)
