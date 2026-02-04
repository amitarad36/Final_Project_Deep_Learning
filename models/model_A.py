"""
Model A: Spectrogram-Based Music Source Separation Models

Two architectures for comparison:
- Model A (U-Net): 2D CNN encoder-decoder with skip connections
- Model A (LSTM): Sequential LSTM-based masking (paper's approach)

Authors: Amit & Alon
Date: January 2026
"""
import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# MODEL A (U-NET): 2D Convolutional Architecture
# =============================================================================

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


# =============================================================================
# MODEL A (LSTM): Sequential LSTM-Based Masking
# =============================================================================

class SpectrogramMaskingLSTM(nn.Module):
    """
    LSTM-based source separation model using spectrogram masking.
    Based on "Source Separation & Automatic Transcription for Music" paper.
    
    Algorithm:
    1. Input: Magnitude spectrogram S (already log-transformed)
    2. Batch normalization
    3. LSTM processing (temporal modeling)
    4. Embedding layer to generate mask M
    5. Output: mask in [0, 1]
    
    Architecture:
    - Input: (batch, 1, freq_bins, time_steps)
    - BatchNorm for normalization
    - Bidirectional LSTM for temporal modeling
    - Dense layers to generate mask
    - Sigmoid activation for mask in [0, 1]
    """
    
    def __init__(
        self, 
        freq_bins=1025,          # Number of frequency bins (n_fft // 2 + 1)
        hidden_size=512,         # LSTM hidden size
        num_layers=2,            # Number of LSTM layers
        dropout=0.3,             # Dropout rate
        bidirectional=True       # Use bidirectional LSTM
    ):
        super().__init__()
        
        self.freq_bins = freq_bins
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Batch normalization (applied to log-magnitude spectrogram)
        self.batch_norm = nn.BatchNorm1d(freq_bins)
        
        # LSTM for temporal modeling
        # Input: (batch, time, freq_bins)
        # Output: (batch, time, hidden_size * num_directions)
        self.lstm = nn.LSTM(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Embedding layers to map LSTM output to mask
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.embedding = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, freq_bins),
            nn.Sigmoid()  # Mask values in [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass following the paper's algorithm.
        
        Args:
            x: Input spectrogram of shape (batch, 1, freq_bins, time_steps)
               Already in log-magnitude representation
        
        Returns:
            mask: Predicted mask of shape (batch, 1, freq_bins, time_steps)
        """
        batch_size, _, freq_bins, time_steps = x.shape
        
        # Remove channel dimension: (batch, freq_bins, time_steps)
        x = x.squeeze(1)
        
        # Batch normalization
        x_normalized = self.batch_norm(x)
        
        # Transpose for LSTM: (batch, time_steps, freq_bins)
        x_transposed = x_normalized.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_transposed)
        
        # Embedding layer to generate mask
        mask = self.embedding(lstm_out)
        
        # Transpose back to spectrogram format
        mask = mask.transpose(1, 2)
        
        # Add channel dimension: (batch, 1, freq_bins, time_steps)
        mask = mask.unsqueeze(1)
        
        return mask


class CompactLSTMMasking(nn.Module):
    """
    Lightweight LSTM-based masking model for faster training.
    Simplified version with fewer parameters.
    """
    
    def __init__(
        self,
        freq_bins=1025,
        hidden_size=256,
        num_layers=1,
        dropout=0.2
    ):
        super().__init__()
        
        self.freq_bins = freq_bins
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(freq_bins)
        
        # Single-layer LSTM (faster training)
        self.lstm = nn.LSTM(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Simple embedding to mask
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size * 2, freq_bins),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass for compact model.
        
        Args:
            x: (batch, 1, freq_bins, time_steps)
        
        Returns:
            mask: (batch, 1, freq_bins, time_steps)
        """
        # Remove channel: (batch, freq_bins, time_steps)
        x = x.squeeze(1)
        
        # Batch norm
        x = self.batch_norm(x)
        
        # Transpose for LSTM: (batch, time_steps, freq_bins)
        x = x.transpose(1, 2)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Generate mask
        mask = self.embedding(x)
        
        # Transpose and add channel
        mask = mask.transpose(1, 2).unsqueeze(1)
        
        return mask


# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def get_unet_config():
    """
    Returns configuration for Model A (U-Net).
    """
    return {
        'model_type': 'unet',
        'in_channels': 1,
        'out_channels': 1,
        'base_filters': 48,
        'num_layers': 4,
        'batchnorm': True,
        'dropout': 0.1
    }


def get_lstm_config():
    """
    Returns configuration for Model A (LSTM) - full version.
    """
    return {
        'model_type': 'lstm',
        'freq_bins': 1025,  # For n_fft=2048: 2048//2 + 1
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True
    }
