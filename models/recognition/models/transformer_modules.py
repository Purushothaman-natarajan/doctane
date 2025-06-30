import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

__all__ = ["Decoder", "PositionalEncoding", "EncoderBlock", "MultiHeadAttention", "PositionwiseFeedForward", "PatchEmbedding"]


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # Scaled term
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        self.register_buffer("pe", pe.unsqueeze(0))  # Buffer to persist in model state but not trained ;;;

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encodings added
        """
        x = x + self.pe[:, :x.size(1)]  # Match sequence length
        return self.dropout(x)


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Returns:
        output: attended values
        attention: attention weights
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # Apply mask ;;;
    p_attn = torch.softmax(scores, dim=-1)  # Compute attention weights ;;;
    return torch.matmul(p_attn, value), p_attn  # Weighted sum and weights


class PositionwiseFeedForward(nn.Sequential):
    """Position-wise Feed-Forward Network (applied independently at each position)"""

    def __init__(
        self, d_model: int, ffd: int, dropout: float = 0.1, activation_fct: Callable[[Any], Any] = nn.ReLU()
    ) -> None:
        super().__init__(  # type: ignore[call-overload]
            nn.Linear(d_model, ffd),
            activation_fct,
            nn.Dropout(p=dropout),
            nn.Linear(ffd, d_model),
            nn.Dropout(p=dropout),
        )  # ;; standard FFN sequence


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""

    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # Linear projections for Q, K, V
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size = query.size(0)

        # Project Q, K, V and split into heads
        query, key, value = [
            linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linear_layers, (query, key, value))
        ]  # ;; (batch, heads, seq_len, d_k)

        x, attn = scaled_dot_product_attention(query, key, value, mask=mask)

        # Concatenate attention heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.output_linear(x)  # Final projection ;;;


class EncoderBlock(nn.Module):
    """Transformer Encoder Block (self-attention + feed-forward)"""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        dff: int,
        dropout: float,
        activation_fct: Callable[[Any], Any] = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.layer_norm_input = nn.LayerNorm(d_model)
        self.layer_norm_attention = nn.LayerNorm(d_model)
        self.layer_norm_output = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.ModuleList([
            MultiHeadAttention(num_heads, d_model, dropout) for _ in range(num_layers)
        ])
        self.position_feed_forward = nn.ModuleList([
            PositionwiseFeedForward(d_model, dff, dropout, activation_fct) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        output = x
        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output)
            output = output + self.dropout(self.attention[i](normed_output, normed_output, normed_output, mask))  # Self-attention
            normed_output = self.layer_norm_attention(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))  # Feed-forward
        return self.layer_norm_output(output)  # Final normalization ;;;


class Decoder(nn.Module):
    """Transformer Decoder (masked self-attn, encoder-decoder attn, FFN)"""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        vocab_size: int,
        dropout: float = 0.2,
        dff: int = 2048,
        maximum_position_encoding: int = 50,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.layer_norm_input = nn.LayerNorm(d_model)
        self.layer_norm_masked_attention = nn.LayerNorm(d_model)
        self.layer_norm_attention = nn.LayerNorm(d_model)
        self.layer_norm_output = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, maximum_position_encoding)

        self.attention = nn.ModuleList([
            MultiHeadAttention(num_heads, d_model, dropout) for _ in range(num_layers)
        ])
        self.source_attention = nn.ModuleList([
            MultiHeadAttention(num_heads, d_model, dropout) for _ in range(num_layers)
        ])
        self.position_feed_forward = nn.ModuleList([
            PositionwiseFeedForward(d_model, dff, dropout) for _ in range(num_layers)
        ])

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Embed tokens and add positional encodings
        tgt = self.embed(tgt) * math.sqrt(self.d_model)
        output = self.positional_encoding(tgt)

        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output)
            output = output + self.dropout(self.attention[i](normed_output, normed_output, normed_output, target_mask))  # Masked self-attention ;;;
            normed_output = self.layer_norm_masked_attention(output)
            output = output + self.dropout(self.source_attention[i](normed_output, memory, memory, source_mask))  # Encoder-decoder attention ;;;
            normed_output = self.layer_norm_attention(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))  # Feed-forward ;;;

        return self.layer_norm_output(output)  # Final output ;;;


class PatchEmbedding(nn.Module):
    """
    Compute 2D patch embeddings with an optional class token and positional encoding.
    This module is typically used in Vision Transformers (ViTs) to transform an image into a sequence of tokens.
    """

    def __init__(self, input_shape: tuple[int, int, int], embed_dim: int, patch_size: tuple[int, int]) -> None:
        """
        Args:
            input_shape: Tuple (channels, height, width) of the input image.
            embed_dim: Output embedding dimension per patch.
            patch_size: Size of each image patch (height, width).
        """
        super().__init__()
        channels, height, width = input_shape
        self.patch_size = patch_size

        # Determines whether to use interpolation for positional encoding (e.g., non-square patches)
        self.interpolate = patch_size[0] == patch_size[1]

        # Compute number of patches along each dimension
        self.grid_size = tuple(s // p for s, p in zip((height, width), patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Learnable positional embeddings for each patch + cls token
        self.positions = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # Linear projection of flattened patches (via conv2d with kernel_size = stride = patch_size)
        self.projection = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Interpolate positional encodings for non-square inputs or varying resolutions.
        Based on implementations from HuggingFace and Facebook's DINO project.

        Args:
            embeddings: Tensor of shape (batch, num_patches + 1, embed_dim)
            height: Input image height
            width: Input image width

        Returns:
            Interpolated positional encoding of shape (1, num_patches + 1, embed_dim)
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.positions.shape[1] - 1

        if num_patches == num_positions and height == width:
            return self.positions

        # Separate the class token from patch tokens
        class_pos_embed = self.positions[:, 0]
        patch_pos_embed = self.positions[:, 1:]
        dim = embeddings.shape[-1]

        h0, w0 = height // self.patch_size[0], width // self.patch_size[1]
        h0, w0 = h0 + 0.1, w0 + 0.1  # Prevent floating point interpolation issues

        # Reshape patch embeddings to 2D grid and interpolate
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, dim, h, w)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True,
        )

        # Ensure interpolated size matches expected
        assert int(h0) == patch_pos_embed.shape[-2], "Height of interpolated position embedding mismatch"
        assert int(w0) == patch_pos_embed.shape[-1], "Width of interpolated position embedding mismatch"

        # Flatten back to sequence
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)

        Returns:
            Patch embeddings with positional encodings: (batch_size, num_patches + 1, embed_dim)
        """
        B, C, H, W = x.shape

        # Ensure dimensions are divisible by patch size
        assert H % self.patch_size[0] == 0, "Image height must be divisible by patch height"
        assert W % self.patch_size[1] == 0, "Image width must be divisible by patch width"

        # Patchify the input using Conv2d and reshape to (B, num_patches, embed_dim)
        patches = self.projection(x).flatten(2).transpose(1, 2)

        # Expand and prepend the [CLS] token to the patch sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls_tokens, patches], dim=1)

        # Add positional embeddings
        if self.interpolate:
            embeddings += self.interpolate_pos_encoding(embeddings, H, W)
        else:
            embeddings += self.positions

        return embeddings
