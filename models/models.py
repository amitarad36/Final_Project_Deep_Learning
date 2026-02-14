"""
Models
Here we define the architectures for our source separation models, including:
- Model A: U-Net (2D Convolutional Architecture)
- Model A: LSTM-Based Masking (Sequential Architecture)
- Neural Linearizer: Invertible Source Separation Architecture

and several configuration functions to easily instantiate these models with predefined settings.

Authors: Amit & Alon
Date: January 2026
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# ATTENTION-BASED U-NET: U-Net with Self-Attention at Bottleneck
# =============================================================================

class MultiHeadSelfAttention2D(nn.Module):
    """
    Applies Multi-Head Self-Attention to 2D feature maps.
    Input: (Batch, Channels, Height, Width)
    Output: (Batch, Channels, Height, Width)
    """
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        assert in_channels % num_heads == 0, "Channels must be divisible by num_heads"

        # Projections for Q, K, V
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        self.norm = nn.GroupNorm(1, in_channels)  # LayerNorm equivalent for 2D inputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        # This treats every pixel (Freq, Time) as a token in the sequence.
        flattened = x.view(b, c, -1).permute(0, 2, 1)  # (Batch, Seq_Len, Channels)
        
        # 2. Compute Q, K, V
        # Shape: (B, Seq_Len, 3 * C)
        qkv = self.qkv(flattened)
        # Reshape to (B, Seq_Len, 3, Num_Heads, Head_Dim) -> (3, B, Heads, Seq_Len, Dim)
        qkv = qkv.reshape(b, h * w, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 3. Scaled Dot-Product Attention
        # (B, Heads, Seq_Len, Dim) @ (B, Heads, Dim, Seq_Len) -> (B, Heads, Seq_Len, Seq_Len)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 4. Combine Heads
        # (B, Heads, Seq_Len, Seq_Len) @ (B, Heads, Seq_Len, Dim) -> (B, Heads, Seq_Len, Dim)
        out = (attn @ v).transpose(1, 2).reshape(b, h * w, c)
        
        # 5. Output Projection
        out = self.proj(out)
        out = self.dropout(out)

        # 6. Reshape back to 2D feature map
        out = out.permute(0, 2, 1).view(b, c, h, w)
        
        # 7. Residual Connection + Norm
        return self.norm(x + out)


class UNetAttention(nn.Module):
    """
    U-Net with Self-Attention at the bottleneck.
    Improves feature representation by allowing the model to focus on important patterns.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, num_layers=4, num_heads=4, batchnorm=True, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # --- Encoder Path ---
        for i in range(num_layers):
            inc = in_channels if i == 0 else base_filters * (2 ** (i - 1))
            outc = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(inc, outc, batchnorm=batchnorm, dropout=dropout))

        # --- Bottleneck ---
        bot_in = base_filters * (2 ** (num_layers - 1))
        bot_out = base_filters * (2 ** num_layers)
        
        # 1. Standard Conv Bottleneck
        self.bottleneck_conv = ConvLayer2D(bot_in, bot_out, kernel_size=3, stride=1, padding=1, batchnorm=batchnorm, dropout=dropout)
        
        # 2. Attention Mechanism
        # We apply attention to the bottleneck features to improve feature representation
        self.bottleneck_attn = MultiHeadSelfAttention2D(bot_out, num_heads=num_heads, dropout=dropout)

        # --- Decoder Path ---
        for i in range(num_layers - 1, -1, -1):
            dec_in = bot_out if i == num_layers - 1 else base_filters * (2 ** (i + 1))
            dec_out = base_filters * (2 ** i)
            self.decoders.append(DecoderBlock(dec_in, dec_out, batchnorm=batchnorm, dropout=dropout))

        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.shape
        
        # Padding (Same as standard U-Net)
        multiple = 2 ** self.num_layers
        pad_h = int((multiple - (h % multiple)) % multiple)
        pad_w = int((multiple - (w % multiple)) % multiple)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        # Encoder
        skips = []
        for enc in self.encoders:
            x, p = enc(x)
            skips.append(x)
            x = p

        # Bottleneck (Conv -> Attention)
        x = self.bottleneck_conv(x)
        x = self.bottleneck_attn(x)  # <--- Attention mechanism applied

        # Decoder
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
        
        # Layer normalization (more robust across PyTorch versions than BatchNorm1d)
        self.layer_norm = nn.LayerNorm(freq_bins)
        
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
        # Handle input shape robustly - ensure it's 4D
        if x.ndim == 3:
            # If missing channel dimension, add it
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")
        
        batch_size, num_channels, freq_bins, time_steps = x.shape
        
        # Remove channel dimension: (batch, freq_bins, time_steps)
        if num_channels == 1:
            x = x.squeeze(1)
        else:
            # If multiple channels, take first one
            x = x[:, 0, :, :]
        
        # Manual normalization: (batch, freq_bins, time_steps)
        # Normalize across frequency dimension for each time step
        mean = x.mean(dim=1, keepdim=True)  # (batch, 1, time_steps)
        std = x.std(dim=1, keepdim=True) + 1e-8  # (batch, 1, time_steps)
        x_normalized = (x - mean) / std  # (batch, freq_bins, time_steps)
        
        # Transpose for LSTM: (batch, time_steps, freq_bins)
        x_normalized = x_normalized.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_normalized)
        
        # Embedding layer to generate mask
        mask = self.embedding(lstm_out)
        
        # Transpose back to spectrogram format: (batch, freq_bins, time_steps)
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
        
        # No normalization layer needed - we'll do manual normalization in forward
        
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
            x: (batch, 1, freq_bins, time_steps) or (batch, freq_bins, time_steps)
        
        Returns:
            mask: (batch, 1, freq_bins, time_steps)
        """
        # Handle input shape robustly - ensure it's 4D
        if x.ndim == 3:
            # If missing channel dimension, add it
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")
        
        batch_size, num_channels, freq_bins, time_steps = x.shape
        
        # Remove channel: (batch, freq_bins, time_steps)
        if num_channels == 1:
            x = x.squeeze(1)
        else:
            x = x[:, 0, :, :]
        
        # Manual normalization: normalize across frequency dimension
        mean = x.mean(dim=1, keepdim=True)  # (batch, 1, time_steps)
        std = x.std(dim=1, keepdim=True) + 1e-8  # (batch, 1, time_steps)
        x = (x - mean) / std  # (batch, freq_bins, time_steps)
        
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
# NEURAL LINEARIZER: Invertible Source Separation Architecture
# =============================================================================

class InvertibleBlock(nn.Module):
    """
    One block of the Invertible Encoder (g).
    Uses Affine Coupling to ensure x -> z is perfectly reversible.
    """
    def __init__(self, channels=4):
        super().__init__()
        self.split_len = channels // 2
        
        # Condition network (CNN) - predicts Scale (s) and Shift (t)
        # This part does NOT need to be invertible.
        self.cnn = nn.Sequential(
            nn.Conv2d(self.split_len, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, self.split_len * 2, kernel_size=3, padding=1)
        )
        
        # Learnable 1x1 Convolution for mixing channels
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # Initialize close to Identity for stability
        self.mix.weight.data = torch.eye(channels).unsqueeze(2).unsqueeze(3) + 0.01 * torch.randn_like(self.mix.weight.data)

    def forward(self, x, reverse=False):
        if not reverse:
            # --- Forward (Encode) ---
            x = self.mix(x)                     # 1. Mix channels
            x1, x2 = x.chunk(2, dim=1)          # 2. Split
            
            st = self.cnn(x1)                   # 3. Predict s, t
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)                   # Stability clamp
            
            y2 = x2 * torch.exp(s) + t          # 4. Affine Transform
            y1 = x1
            return torch.cat([y1, y2], dim=1)
            
        else:
            # --- Inverse (Decode) ---
            x1, x2 = x.chunk(2, dim=1)
            
            st = self.cnn(x1)                   # Predict s, t from x1 (same as forward!)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            
            y2 = (x2 - t) * torch.exp(-s)       # Inverse Affine
            y1 = x1
            y = torch.cat([y1, y2], dim=1)
            
            # Inverse 1x1 Conv
            inv_weight = torch.inverse(self.mix.weight.squeeze()).unsqueeze(2).unsqueeze(3)
            return F.conv2d(y, inv_weight)


class HyperNetwork(nn.Module):
    """
    Takes a Style Vector (w) and outputs the Matrix (A).
    """
    def __init__(self, input_dim=768, matrix_dim=4): # 768 for WavLM Base
        super().__init__()
        self.matrix_dim = matrix_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, matrix_dim * matrix_dim) # Output flattened matrix
        )

    def forward(self, style_vector):
        # Output shape: (Batch, 4, 4, 1, 1) for use in conv2d
        matrix_params = self.net(style_vector)
        return matrix_params.view(-1, self.matrix_dim, self.matrix_dim, 1, 1)


class NeuralLinearizer(nn.Module):
    """
    The Main Model wrapper.
    Updated to return the Matrix A for monitoring.
    """
    def __init__(self, num_blocks=6, input_dim=768):
        super().__init__()
        self.blocks = nn.ModuleList([InvertibleBlock(channels=4) for _ in range(num_blocks)])
        self.hypernet = HyperNetwork(input_dim=input_dim)

    def forward(self, x, style_vector):
        # 1. Encode (g)
        z = x
        for block in self.blocks:
            z = block(z, reverse=False)
            
        # 2. Predict Matrix A
        A = self.hypernet(style_vector) # Shape: (B, 4, 4, 1, 1)
        
        # 3. Apply Linear Transform (A * z)
        # Squeeze A for einsum: (B, 4, 4, 1, 1) -> (B, 4, 4)
        A_matrix = A.squeeze(-1).squeeze(-1)
        z_transformed = torch.einsum('bci,bihw->bchw', A_matrix, z)
        
        # 4. Decode (g inverse)
        out = z_transformed
        for block in reversed(self.blocks):
            out = block(out, reverse=True)
            
        # RETURN BOTH OUTPUT AND MATRIX A
        return out, A_matrix


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


def get_unet_attention_config():
    """
    Returns configuration for U-Net with Attention (UNetAttention).
    """
    return {
        'model_type': 'unet_attention',
        'in_channels': 1,
        'out_channels': 1,
        'base_filters': 48,
        'num_layers': 4,
        'num_heads': 4,
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
    
def get_linearizer_config():
    """
    Returns architecture configuration for the Neural Linearizer.
    Training hyperparameters are in utils.get_linearizer_training_config().
    """
    return {
        'input_dim': 768,       # WavLM Base embedding dimension
        'num_blocks': 6         # Depth of invertible encoder/decoder
    }
