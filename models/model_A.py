"""
Model A: Time-Frequency Domain U-Net for Music Source Separation

Authors: Amit & Alon
Date: January 2026
"""
import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# Time-Frequency Domain U-Net Architecture
# =============================================================================

from tqdm.notebook import tqdm

class ConvLayer2D(nn.Module):
    """
    2D Convolutional layer with optional BatchNorm, Dropout, and ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer2D(in_channels, out_channels, 3, 1, 1, batchnorm, dropout), #defa
            ConvLayer2D(out_channels, out_channels, 3, 1, 1, batchnorm, dropout)
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.block(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True, dropout=0.0):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.block = nn.Sequential(
            ConvLayer2D(in_channels, out_channels, 3, 1, 1, batchnorm, dropout),
            ConvLayer2D(out_channels, out_channels, 3, 1, 1, batchnorm, dropout)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape != skip.shape:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

class TimeFrequencyDomainUNet(nn.Module):
    """
    U-Net architecture for time-frequency domain source separation.
    Supports optional batch normalization and dropout.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, num_layers=4, batchnorm=True, dropout=0.0):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            inc = in_channels if i == 0 else base_filters * (2 ** (i - 1))
            outc = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(inc, outc, batchnorm=batchnorm, dropout=dropout))
        bot_in = base_filters * (2 ** (num_layers - 1))
        bot_out = base_filters * (2 ** num_layers)
        self.bottleneck = ConvLayer2D(bot_in, bot_out, 3, 1, 1, batchnorm=batchnorm, dropout=dropout)
        for i in range(num_layers - 1, -1, -1):
            dec_in = bot_out if i == num_layers - 1 else base_filters * (2 ** (i + 1))
            dec_out = base_filters * (2 ** i)
            self.decoders.append(DecoderBlock(dec_in, dec_out, batchnorm=batchnorm, dropout=dropout))
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.shape
        pad_h = (16 - (h % 16)) % 16
        pad_w = (16 - (w % 16)) % 16
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        skips = []
        for enc in self.encoders:
            x, p = enc(x)
            skips.append(x)
            x = p
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.sigmoid(self.final_conv(x))
    
    