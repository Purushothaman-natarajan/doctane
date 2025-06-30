
from .utils import BasicEncoder
from .utils import build_position_encoding

import argparse
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy
from typing import Optional

# ----------------------------- #
#     Custom LayerNorm Variants
# ----------------------------- #
class LayerNorm2d(nn.Module):
    """Applies LayerNorm over each channel in 2D image-like inputs (N, C, H, W)."""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        # Permute to (N, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        return x.permute(0, 3, 1, 2)

# ----------------------------- #
#        Utility Functions
# ----------------------------- #
def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    """Deep copy a module N times into a ModuleList."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def coords_grid(batch, ht, wd):
    """Generate coordinate grid of shape (B, 2, H, W)."""
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()  # (2, H, W)
    return coords[None].repeat(batch, 1, 1, 1)

def upflow8(flow, mode='bilinear'):
    """Upsample optical flow by factor of 8."""
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

# ----------------------------- #
#        Attention Block
# ----------------------------- #
class attnLayer(nn.Module):
    """Transformer Decoder-like attention layer with multiple memory inputs."""
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_list = nn.ModuleList([
            copy.deepcopy(nn.MultiheadAttention(d_model, nhead, dropout=dropout)) for _ in range(2)
        ])

        # Feedforward layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2_list = nn.ModuleList([copy.deepcopy(nn.LayerNorm(d_model)) for _ in range(2)])
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2_list = nn.ModuleList([copy.deepcopy(nn.Dropout(dropout)) for _ in range(2)])
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_list, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, memory_pos=None):
        # Self-attention
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention for each memory
        for memory, attn, norm2, dropout2, m_pos in zip(memory_list, self.multihead_attn_list, self.norm2_list, self.dropout2_list, memory_pos):
            tgt2 = attn(query=self.with_pos_embed(tgt, pos),
                        key=self.with_pos_embed(memory, m_pos),
                        value=memory, attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + dropout2(tgt2)
            tgt = norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            raise NotImplementedError("Pre-norm not implemented in this variant.")
        return self.forward_post(*args, **kwargs)

# ----------------------------- #
#        Transformer Blocks
# ----------------------------- #
class TransDecoder(nn.Module):
    """Stacked attention layers for decoding."""
    def __init__(self, num_attn_layers, hidden_dim=128):
        super().__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf, query_embed):
        pos = self.position_embedding(torch.ones_like(imgf[:, 0:1], dtype=torch.bool).cuda())
        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            query_embed = layer(query_embed, [imgf], pos=pos, memory_pos=[pos, pos])
        query_embed = query_embed.permute(1, 2, 0).reshape(bs, c, h, w)
        return query_embed

class TransEncoder(nn.Module):
    """Transformer encoder with self-attention."""
    def __init__(self, num_attn_layers, hidden_dim=128):
        super().__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf):
        pos = self.position_embedding(torch.ones_like(imgf[:, 0:1], dtype=torch.bool).cuda())
        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            imgf = layer(imgf, [imgf], pos=pos, memory_pos=[pos, pos])
        imgf = imgf.permute(1, 2, 0).reshape(bs, c, h, w)
        return imgf

# ----------------------------- #
#        Flow Estimation
# ----------------------------- #
class FlowHead(nn.Module):
    """Predicts optical flow from features."""
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class UpdateBlock(nn.Module):
    """Updates flow and generates mask for upsampling."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1))

    def forward(self, imgf, coords1):
        mask = 0.25 * self.mask(imgf)  # Scale for gradient stability
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow
        return mask, coords1

# ----------------------------- #
#          Main Model
# ----------------------------- #
class GeoTr(nn.Module):
    """GeoTr: Geometric Transformer for Optical Flow Estimation."""
    def __init__(self, num_attn_layers):
        super().__init__()
        self.hidden_dim = 256
        self.fnet = BasicEncoder(output_dim=self.hidden_dim, norm_fn='instance')
        self.TransEncoder = TransEncoder(num_attn_layers, hidden_dim=self.hidden_dim)
        self.TransDecoder = TransDecoder(num_attn_layers, hidden_dim=self.hidden_dim)
        self.query_embed = nn.Embedding(1296, self.hidden_dim)
        self.update_block = UpdateBlock(self.hidden_dim)

    def initialize_flow(self, img):
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1).view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).reshape(N, 2, 8 * H, 8 * W)
        return up_flow

    def forward(self, image1):
        fmap = self.fnet(image1)
        fmap_enc = self.TransEncoder(fmap)
        coords0, coords1 = self.initialize_flow(image1)

        corr = self.TransDecoder(fmap_enc, self.query_embed.weight)
        mask, coords1 = self.update_block(corr, coords1)

        flow_up = self.upsample_flow(coords1 - coords0, mask)
        return flow_up
