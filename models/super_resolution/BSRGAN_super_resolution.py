import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ----------------------------- #
# Weight Initialization Utility
# ----------------------------- #
def initialize_weights(net_l, scale=1):
    """Initializes Conv2d, Linear, and BatchNorm2d layers."""
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


# ----------------------------- #
# Utility to Make Sequential Layers
# ----------------------------- #
def make_layer(block, n_layers):
    """Creates a sequential container of n_layers of a block."""
    return nn.Sequential(*[block() for _ in range(n_layers)])


# ----------------------------- #
# Residual Dense Block (5 Convs)
# ----------------------------- #
class ResidualDenseBlock_5C(nn.Module):
    """Residual Dense Block with 5 convolutional layers."""
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + 0.2 * x5


# ----------------------------- #
# Residual in Residual Dense Block
# ----------------------------- #
class RRDB(nn.Module):
    """Residual in Residual Dense Block (stack of 3 ResidualDenseBlock_5C)."""
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb_blocks = nn.Sequential(
            ResidualDenseBlock_5C(nf, gc),
            ResidualDenseBlock_5C(nf, gc),
            ResidualDenseBlock_5C(nf, gc)
        )

    def forward(self, x):
        return x + 0.2 * self.rdb_blocks(x)


# ----------------------------- #
# RRDBNet Super-Resolution Network
# ----------------------------- #
class RRDBNet(nn.Module):
    """RRDBNet for image super-resolution."""
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super().__init__()
        self.sf = sf  # scale factor
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # Upsampling layers
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if sf == 4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf == 4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


# ----------------------------- #
# Custom LayerNorm Variants
# ----------------------------- #
class LayerNorm2d(nn.Module):
    """LayerNorm for 4D tensors (N, C, H, W)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        std = x.std(dim=[1, 2, 3], keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class BiasFreeLayerNorm(nn.Module):
    """Bias-Free LayerNorm for 4D tensors."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        variance = x.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        return self.weight * x / torch.sqrt(variance + self.eps)
