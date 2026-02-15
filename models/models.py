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
# LINEARIZER FLOW: invertible g + UNet velocity v_theta(z,t,cond)
# =============================================================================

class ActNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.logs = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.initialized = False

    def forward(self, x, reverse=False):
        if (not self.initialized) and (x.numel() > 0):
            with torch.no_grad():
                flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
                mean = flatten.mean(1).view(1, self.channels, 1, 1)
                std = flatten.std(1).view(1, self.channels, 1, 1)
                self.bias.data.copy_(-mean)
                self.logs.data.copy_(-torch.log(std + 1e-6))
                self.initialized = True

        if not reverse:
            return (x + self.bias) * torch.exp(self.logs)
        else:
            return x * torch.exp(-self.logs) - self.bias


class InvertibleBlock(nn.Module):
    """
    ActNorm -> 1x1 Conv -> affine coupling
    """
    def __init__(self, channels=4, hidden=64):
        super().__init__()
        assert channels % 2 == 0
        self.channels = channels
        self.split_len = channels // 2

        self.actnorm = ActNorm(channels)

        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        with torch.no_grad():
            eye = torch.eye(channels).unsqueeze(2).unsqueeze(3)
            self.mix.weight.data = eye + 0.01 * torch.randn_like(eye)

        self.nn = nn.Sequential(
            nn.Conv2d(self.split_len, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, self.split_len * 2, 3, padding=1)
        )

    def forward(self, x, reverse=False):
        if not reverse:
            x = self.actnorm(x, reverse=False)
            x = self.mix(x)

            x1, x2 = x.chunk(2, dim=1)
            st = self.nn(x1)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            y2 = x2 * torch.exp(s) + t
            y = torch.cat([x1, y2], dim=1)
            return y

        else:
            x1, x2 = x.chunk(2, dim=1)
            st = self.nn(x1)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            y2 = (x2 - t) * torch.exp(-s)
            y = torch.cat([x1, y2], dim=1)

            inv_weight = torch.inverse(self.mix.weight.squeeze()).unsqueeze(2).unsqueeze(3)
            y = F.conv2d(y, inv_weight)
            y = self.actnorm(y, reverse=True)
            return y


def sinusoidal_t_embedding(t, dim):
    """
    t: (B,) in [0,1]
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000.0) * torch.arange(0, half, device=t.device).float() / (half - 1 + 1e-8)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class FiLMResBlock(nn.Module):
    """
    ResBlock with FiLM conditioning from a vector emb (time+style+content).
    """
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)  # gamma, beta
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))

        gamma_beta = self.emb_proj(emb)  # (B, 2*out_ch)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        h = self.norm2(h)
        h = h * (1.0 + gamma) + beta
        h = self.conv2(F.silu(h))

        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class VelocityUNet(nn.Module):
    """
    v_theta(z,t,cond) in latent space z.
    - z: (B, 4, H, W)
    - t: (B,)
    - cond: vector (B, cond_dim) [style + content pooled]
    returns velocity: (B, 4, H, W)
    """
    def __init__(self, z_ch=4, base=64, emb_dim=256, cond_dim=1536):
        super().__init__()
        self.z_ch = z_ch
        self.base = base
        self.emb_dim = emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.in_conv = nn.Conv2d(z_ch, base, 3, padding=1)

        self.rb1 = FiLMResBlock(base, base, emb_dim)
        self.down1 = Down(base)

        self.rb2 = FiLMResBlock(base, base * 2, emb_dim)
        self.down2 = Down(base * 2)

        self.mid1 = FiLMResBlock(base * 2, base * 2, emb_dim)
        self.mid2 = FiLMResBlock(base * 2, base * 2, emb_dim)

        self.up2 = Up(base * 2)
        self.rb_up2 = FiLMResBlock(base * 2 + base * 2, base, emb_dim)

        self.up1 = Up(base)
        self.rb_up1 = FiLMResBlock(base + base, base, emb_dim)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, z_ch, 3, padding=1)

    def forward(self, z, t, cond):
        t_emb = sinusoidal_t_embedding(t, self.emb_dim)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.cond_mlp(cond)
        emb = t_emb + c_emb

        x = self.in_conv(z)

        x1 = self.rb1(x, emb)
        d1 = self.down1(x1)

        x2 = self.rb2(d1, emb)
        d2 = self.down2(x2)

        m = self.mid1(d2, emb)
        m = self.mid2(m, emb)

        u2 = self.up2(m)
        if u2.shape[-2:] != x2.shape[-2:]:
            u2 = u2[:, :, :x2.shape[2], :x2.shape[3]]
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.rb_up2(u2, emb)

        u1 = self.up1(u2)
        if u1.shape[-2:] != x1.shape[-2:]:
            u1 = u1[:, :, :x1.shape[2], :x1.shape[3]]
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.rb_up1(u1, emb)

        v = self.out_conv(F.silu(self.out_norm(u1)))
        return v


class LinearizerFlow(nn.Module):
    """
    Full pipeline:
      x (squeezed spec) --g--> z
      integrate dz/dt = v_theta(z,t, cond)  t:0->1
      z1 --g^{-1}--> x_hat (squeezed spec)
    """
    def __init__(self, num_blocks=6, z_ch=4, flow_hidden=64, unet_base=64, emb_dim=256, cond_dim=1536):
        super().__init__()
        self.blocks = nn.ModuleList([InvertibleBlock(channels=z_ch, hidden=flow_hidden) for _ in range(num_blocks)])
        self.vnet = VelocityUNet(z_ch=z_ch, base=unet_base, emb_dim=emb_dim, cond_dim=cond_dim)

    def encode(self, x_squeezed):
        z = x_squeezed
        for b in self.blocks:
            z = b(z, reverse=False)
        return z

    def decode(self, z):
        x = z
        for b in reversed(self.blocks):
            x = b(x, reverse=True)
        return x

    def integrate(self, z0, cond, steps=20, solver="euler"):
        """
        Integrate from t=0 to t=1.
        z0: (B,4,H,W)
        cond: (B,cond_dim)
        """
        if steps < 1:
            return z0

        z = z0
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((z.shape[0],), float(i) / float(steps), device=z.device, dtype=z.dtype)

            if solver == "euler":
                v = self.vnet(z, t, cond)
                z = z + dt * v

            elif solver == "heun":
                v1 = self.vnet(z, t, cond)
                z_e = z + dt * v1
                t2 = torch.full((z.shape[0],), float(i + 1) / float(steps), device=z.device, dtype=z.dtype)
                v2 = self.vnet(z_e, t2, cond)
                z = z + 0.5 * dt * (v1 + v2)

            else:
                raise ValueError("solver must be 'euler' or 'heun'")

        return z

    def forward(self, x_squeezed, cond, steps=20, solver="euler"):
        z0 = self.encode(x_squeezed)
        z1 = self.integrate(z0, cond=cond, steps=steps, solver=solver)
        x_hat = self.decode(z1)
        return x_hat

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
