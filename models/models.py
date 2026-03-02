"""
Models
Here we define the architectures for our source separation models, including:
- Model: U-Net (2D Convolutional Architecture)
- Model: LSTM-Based Masking (Sequential Architecture)
- Model: Attention-Based U-Net (U-Net with Multi-Head Attention at Bottleneck)

Authors: Amit & Alon
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=True, dropout=0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer2D(in_channels, out_channels, 3, 1, 1, batchnorm, dropout),
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

    def __init__(self, in_channels=1, out_channels=1, base_filters=32, num_layers=4, batchnorm=True, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
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
        multiple = 2 ** self.num_layers
        pad_h = int((multiple - (h % multiple)) % multiple)
        pad_w = int((multiple - (w % multiple)) % multiple)
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        skips = []
        for enc in self.encoders:
            x, p = enc(x)
            skips.append(x)
            x = p
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        x = self.sigmoid(self.final_conv(x))
        return x[:, :, :h, :w]

class MultiHeadSelfAttention2D(nn.Module):

    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        assert in_channels % num_heads == 0, "Channels must be divisible by num_heads"

        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        self.norm = nn.GroupNorm(1, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        
        flattened = x.view(b, c, -1).permute(0, 2, 1)
        

        qkv = self.qkv(flattened)
        qkv = qkv.reshape(b, h * w, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, h * w, c)

        out = self.proj(out)
        out = self.dropout(out)

        out = out.permute(0, 2, 1).view(b, c, h, w)

        return self.norm(x + out)

class UNetAttention(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, base_filters=32, num_layers=4, num_heads=4, batchnorm=True, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(num_layers):
            inc = in_channels if i == 0 else base_filters * (2 ** (i - 1))
            outc = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(inc, outc, batchnorm=batchnorm, dropout=dropout))

        bot_in = base_filters * (2 ** (num_layers - 1))
        bot_out = base_filters * (2 ** num_layers)

        self.bottleneck_conv = ConvLayer2D(bot_in, bot_out, kernel_size=3, stride=1, padding=1, batchnorm=batchnorm, dropout=dropout)

        self.bottleneck_attn = MultiHeadSelfAttention2D(bot_out, num_heads=num_heads, dropout=dropout)

        for i in range(num_layers - 1, -1, -1):
            dec_in = bot_out if i == num_layers - 1 else base_filters * (2 ** (i + 1))
            dec_out = base_filters * (2 ** i)
            self.decoders.append(DecoderBlock(dec_in, dec_out, batchnorm=batchnorm, dropout=dropout))

        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.shape
        
        multiple = 2 ** self.num_layers
        pad_h = int((multiple - (h % multiple)) % multiple)
        pad_w = int((multiple - (w % multiple)) % multiple)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        skips = []
        for enc in self.encoders:
            x, p = enc(x)
            skips.append(x)
            x = p

        x = self.bottleneck_conv(x)
        x = self.bottleneck_attn(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        x = self.sigmoid(self.final_conv(x))
        return x[:, :, :h, :w]

class SpectrogramMaskingLSTM(nn.Module):

    def __init__(
        self, 
        freq_bins=1025,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()
        
        self.freq_bins = freq_bins
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.layer_norm = nn.LayerNorm(freq_bins)

        self.lstm = nn.LSTM(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.embedding = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, freq_bins),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")
        
        batch_size, num_channels, freq_bins, time_steps = x.shape
        
        if num_channels == 1:
            x = x.squeeze(1)
        else:
            x = x[:, 0, :, :]

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x_normalized = (x - mean) / std

        x_normalized = x_normalized.transpose(1, 2)

        lstm_out, _ = self.lstm(x_normalized)

        mask = self.embedding(lstm_out)

        mask = mask.transpose(1, 2)

        mask = mask.unsqueeze(1)
        
        return mask
