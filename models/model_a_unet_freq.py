"""
Model A: Time-Frequency Domain U-Net for Music Source Separation

Architecture:
- Input: Magnitude spectrogram of mixture
- Processing: 2D U-Net with skip connections
- Output: Soft mask [0, 1] indicating target source contribution
- Reconstruction: Learned mask x |Mixture| x e^(i·phase_mixture)

This model focuses on learning magnitude patterns while reusing phase from mixture
to avoid phase reconstruction artifacts.

Authors: Amit & Alon
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Tuple, List, Dict, Optional
from pathlib import Path


# =============================================================================
# STFT Processing for Frequency Domain
# =============================================================================

class STFTProcessor:
    """
    Handles Short-Time Fourier Transform (STFT) and inverse operations.
    
    Converts between time-domain waveforms and frequency-domain spectrograms,
    with separate handling of magnitude and phase.
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        window: str = 'hann'
    ):
        """
        Initialize STFT processor.
        
        Args:
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            window: Window function ('hann', 'hamming', etc.)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
    
    def waveform_to_magnitude_phase(
        self,
        waveform: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert waveform to magnitude and phase spectrograms.
        
        Args:
            waveform: Audio waveform of shape (n_samples,) or (channels, n_samples)
            
        Returns:
            (magnitude_spec, phase_spec)
            - magnitude_spec: Shape (freq_bins, time_frames) or (channels, freq_bins, time_frames)
            - phase_spec: Same shape as magnitude
        """
        if waveform.ndim == 1:
            # Mono: (n_samples,) -> (1, n_fft//2+1, n_frames)
            stft = librosa.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window
            )
            # FIX: Apply log compression to handle dynamic range
            # Squashes loud (100.0 -> 4.6) and preserves quiet (0.01 -> 0.01)
            magnitude = np.abs(stft)
            log_magnitude = np.log1p(magnitude)  # log(1 + x)
            phase = np.angle(stft)
            return log_magnitude, phase
        else:
            # Stereo/Multi-channel
            magnitudes = []
            phases = []
            for ch in range(waveform.shape[0]):
                stft = librosa.stft(
                    waveform[ch],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window
                )
                # FIX: Apply log compression to handle dynamic range
                magnitude = np.abs(stft)
                log_magnitude = np.log1p(magnitude)  # log(1 + x)
                magnitudes.append(log_magnitude)
                phases.append(np.angle(stft))
            return np.stack(magnitudes), np.stack(phases)
    
    def magnitude_phase_to_waveform(
        self,
        log_magnitude: np.ndarray,
        phase: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct waveform from log-magnitude and phase.
        
        Args:
            log_magnitude: Log-compressed magnitude spectrogram (output from waveform_to_magnitude_phase)
            phase: Phase spectrogram (same shape as magnitude)
            
        Returns:
            Reconstructed waveform of shape (n_samples,) or (channels, n_samples)
        """
        # FIX: Reverse log compression: exp(x) - 1
        linear_magnitude = np.expm1(log_magnitude)
        # Ensure non-negative (numerical stability)
        linear_magnitude = np.maximum(0, linear_magnitude)
        
        # Combine magnitude and phase
        complex_spec = linear_magnitude * np.exp(1j * phase)
        
        if log_magnitude.ndim == 2:
            # Mono
            waveform = librosa.istft(
                complex_spec,
                hop_length=self.hop_length,
                window=self.window
            )
            return waveform
        else:
            # Stereo/Multi-channel
            waveforms = []
            for ch in range(log_magnitude.shape[0]):
                wf = librosa.istft(
                    complex_spec[ch],
                    hop_length=self.hop_length,
                    window=self.window
                )
                waveforms.append(wf)
            return np.stack(waveforms)
    
    def normalize_magnitude(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Normalize magnitude spectrogram to [0, 1] range.
        
        Args:
            magnitude: Magnitude spectrogram
            
        Returns:
            Normalized magnitude
        """
        max_val = np.max(magnitude)
        if max_val > 0:
            return magnitude / max_val
        return magnitude


# =============================================================================
# Frequency Domain U-Net Architecture
# =============================================================================

class ConvBlock2D(nn.Module):
    """
    2D Convolution block with batch norm and activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        use_batch_norm: bool = True
    ):
        """Initialize 2D conv block."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        self.batch_norm = nn.BatchNorm2d(out_channels) if use_batch_norm else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv block."""
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block: Two conv layers + max pooling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True
    ):
        """Initialize encoder block."""
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBlock2D(in_channels, out_channels, use_batch_norm=use_batch_norm),
            ConvBlock2D(out_channels, out_channels, use_batch_norm=use_batch_norm)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (pooled_output, skip_connection)
        """
        skip = self.double_conv(x)
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsampling + two conv layers.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True
    ):
        """Initialize decoder block."""
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )
        self.double_conv = nn.Sequential(
            ConvBlock2D(in_channels, out_channels, use_batch_norm=use_batch_norm),
            ConvBlock2D(out_channels, out_channels, use_batch_norm=use_batch_norm)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with skip connection.
        
        Args:
            x: Upsampled features
            skip: Skip connection from encoder
            
        Returns:
            Decoded features
        """
        x = self.upsample(x)
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x


class FrequencyDomainUNet(nn.Module):
    """
    2D U-Net for music source separation in frequency domain.
    
    Architecture:
    - Encoder: Progressive downsampling with skip connections
    - Bottleneck: Deepest feature representation
    - Decoder: Progressive upsampling with skip connections
    - Output: Soft mask [0, 1]
    
    Input: (batch, 1, freq_bins, time_frames)
    Output: (batch, 1, freq_bins, time_frames)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        use_batch_norm: bool = True
    ):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Number of input channels (1 for magnitude)
            base_channels: Number of channels in first layer
            depth: Number of encoder/decoder levels (4 or 5)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.depth = depth
        self.base_channels = base_channels
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoder_blocks.append(
                EncoderBlock(in_ch, out_ch, use_batch_norm=use_batch_norm)
            )
            in_ch = out_ch
        
        # Bottleneck (deepest level)
        bottleneck_in = base_channels * (2 ** (depth - 1))
        bottleneck_out = base_channels * (2 ** depth)
        self.bottleneck = nn.Sequential(
            ConvBlock2D(bottleneck_in, bottleneck_out, use_batch_norm=use_batch_norm),
            ConvBlock2D(bottleneck_out, bottleneck_out, use_batch_norm=use_batch_norm)
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            decoder_in = base_channels * (2 ** (i + 1))
            decoder_out = base_channels * (2 ** i)
            self.decoder_blocks.append(
                DecoderBlock(decoder_in, decoder_out, use_batch_norm=use_batch_norm)
            )
        
        # Output layer: Produce mask [0, 1]
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights for better training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming/He initialization.
        
        Critical for training deep networks and preventing vanishing/exploding gradients.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Standard initialization for batch norm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: magnitude spectrogram -> soft mask.
        
        Args:
            x: Input magnitude spectrogram (batch, 1, freq, time)
            
        Returns:
            Soft mask (batch, 1, freq, time) with values in [0, 1]
        """
        # Encoder: save skip connections
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder: use skip connections in reverse
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = len(skip_connections) - 1 - i
            x = decoder_block(x, skip_connections[skip_idx])
        
        # Output: soft mask in [0, 1]
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x


# =============================================================================
# Training Components
# =============================================================================

class SourceSeparationDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for frequency domain source separation.
    """
    
    def __init__(
        self,
        mixture_paths: List[str],
        target_paths: List[str],
        stft_processor: STFTProcessor,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            mixture_paths: Paths to mixture audio files
            target_paths: Paths to target source audio files
            stft_processor: STFT processor instance
            normalize: Whether to normalize magnitude spectrograms
        """
        self.mixture_paths = mixture_paths
        self.target_paths = target_paths
        self.stft_processor = stft_processor
        self.normalize = normalize
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.mixture_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Args:
            idx: Index
            
        Returns:
            Dict with 'mixture_mag', 'target_mag', 'mixture_phase'
        """
        # Load audio (handles both .wav files and .npy arrays)
        mixture_path = self.mixture_paths[idx]
        target_path = self.target_paths[idx]
        
        # Check if loading from .npy files (synthetic data) or audio files
        if str(mixture_path).endswith('.npy'):
            mixture = np.load(mixture_path)
            target = np.load(target_path)
            sr = 22050  # Assume 22050 Hz for synthetic data
        else:
            mixture, sr = librosa.load(mixture_path, sr=22050)
            target, _ = librosa.load(target_path, sr=sr)
        
        # Ensure same length
        min_len = min(len(mixture), len(target))
        mixture = mixture[:min_len]
        target = target[:min_len]
        
        # Compute STFT
        mix_mag, mix_phase = self.stft_processor.waveform_to_magnitude_phase(mixture)
        tgt_mag, _ = self.stft_processor.waveform_to_magnitude_phase(target)
        
        # Pad spectrograms to compatible dimensions for U-Net
        # U-Net with depth=4 requires dimensions divisible by 16
        freq_dim, time_dim = mix_mag.shape
        padded_freq = ((freq_dim + 15) // 16) * 16  # Round up to nearest multiple of 16
        padded_time = ((time_dim + 15) // 16) * 16  # Round up to nearest multiple of 16
        
        # Pad frequency dimension
        if freq_dim < padded_freq:
            pad_freq = padded_freq - freq_dim
            mix_mag = np.pad(mix_mag, ((0, pad_freq), (0, 0)), mode='constant', constant_values=0)
            tgt_mag = np.pad(tgt_mag, ((0, pad_freq), (0, 0)), mode='constant', constant_values=0)
            mix_phase = np.pad(mix_phase, ((0, pad_freq), (0, 0)), mode='constant', constant_values=0)
        
        # Pad time dimension
        if time_dim < padded_time:
            pad_time = padded_time - time_dim
            mix_mag = np.pad(mix_mag, ((0, 0), (0, pad_time)), mode='constant', constant_values=0)
            tgt_mag = np.pad(tgt_mag, ((0, 0), (0, pad_time)), mode='constant', constant_values=0)
            mix_phase = np.pad(mix_phase, ((0, 0), (0, pad_time)), mode='constant', constant_values=0)
        
        # Normalize if needed
        if self.normalize:
            mix_mag = self.stft_processor.normalize_magnitude(mix_mag)
            tgt_mag = self.stft_processor.normalize_magnitude(tgt_mag)
        
        # Convert to torch tensors
        mixture_mag = torch.from_numpy(mix_mag[np.newaxis, :, :]).float()  # (1, freq, time)
        target_mag = torch.from_numpy(tgt_mag[np.newaxis, :, :]).float()
        mixture_phase = torch.from_numpy(mix_phase[np.newaxis, :, :]).float()
        
        return {
            'mixture_mag': mixture_mag,
            'target_mag': target_mag,
            'mixture_phase': mixture_phase
        }


class ModelATrainer:
    """
    Trainer for Model A (Frequency Domain U-Net).
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        learning_rate: float = 1e-4, #default rate is 1e-4
        device: str = 'cpu',
        use_energy_weighted_loss: bool = False,
        grad_clip_max_norm: Optional[float] = 10.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: U-Net model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            device: Device ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()  # MSE instead of L1 for better gradient flow
        self.use_energy_weighted_loss = use_energy_weighted_loss
        self.grad_clip_max_norm = grad_clip_max_norm
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, epoch: int = 0) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number (for diagnostics)
            
        Returns:
            Average training loss
        """
        from tqdm import tqdm
        
        self.model.train()
        total_loss = 0.0
        is_first_epoch = (epoch == 0)
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, batch in enumerate(pbar):
            mixture_mag = batch['mixture_mag'].to(self.device)
            target_mag = batch['target_mag'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_mask = self.model(mixture_mag)
            
            # Compute loss: L1 loss between predicted mask and ideal mask
            # mixture_mag/target_mag are log-compressed; build mask in linear domain
            mixture_lin = torch.expm1(mixture_mag)
            target_lin = torch.expm1(target_mag)
            target_mask = torch.clamp(
                target_lin / (mixture_lin + 1e-8),
                min=0.0,
                max=1.0
            )
            
            # Diagnostic output on first batch of first epoch
            if is_first_epoch and batch_idx == 0:
                print(f"\n[Epoch {epoch} Diagnostics]")
                print(f"  Mix mag range: [{mixture_mag.min():.4f}, {mixture_mag.max():.4f}]")
                print(f"  Tgt mag range: [{target_mag.min():.4f}, {target_mag.max():.4f}]")
                print(f"  Mix lin range: [{mixture_lin.min():.4f}, {mixture_lin.max():.4f}]")
                print(f"  Tgt lin range: [{target_lin.min():.4f}, {target_lin.max():.4f}]")
                print(f"  Target mask range: [{target_mask.min():.4f}, {target_mask.max():.4f}]")
                print(f"  Pred mask range: [{predicted_mask.min():.4f}, {predicted_mask.max():.4f}]")
            
            if self.use_energy_weighted_loss:
                # Weight per-bin by mixture energy in linear domain
                weights = mixture_lin.detach()
                weights = weights / (weights.mean() + 1e-8)
                l1 = torch.abs(predicted_mask - target_mask)
                loss = (weights * l1).sum() / (weights.sum() + 1e-8)
            else:
                loss = self.loss_fn(predicted_mask, target_mask)
            
            # Backward pass
            loss.backward()
            
            # Log gradient magnitude on first batch of first epoch
            if is_first_epoch and batch_idx == 0:
                grad_magnitudes = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_magnitudes.append(param.grad.abs().max().item())
                if grad_magnitudes:
                    print(f"  Grad max (pre-clip): {max(grad_magnitudes):.6f}")
            
            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
            
            # Log gradient magnitude on first batch of first epoch (post-clip)
            if is_first_epoch and batch_idx == 0 and self.grad_clip_max_norm is not None:
                grad_magnitudes = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_magnitudes.append(param.grad.abs().max().item())
                if grad_magnitudes:
                    print(f"  Grad max (post-clip): {max(grad_magnitudes):.6f}")
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate on validation set.
        
        Returns:
            Average validation loss
        """
        from tqdm import tqdm
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            for batch in pbar:
                mixture_mag = batch['mixture_mag'].to(self.device)
                target_mag = batch['target_mag'].to(self.device)
                
                # Forward pass
                predicted_mask = self.model(mixture_mag)
                
                # Compute loss: convert log-compressed magnitudes back to linear for mask
                mixture_lin = torch.expm1(mixture_mag)
                target_lin = torch.expm1(target_mag)
                target_mask = torch.clamp(
                    target_lin / (mixture_lin + 1e-8),
                    min=0.0,
                    max=1.0
                )
                loss = self.loss_fn(predicted_mask, target_mask)
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs: int, save_dir: Optional[str] = None) -> Dict:
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        from tqdm import tqdm
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
        for epoch in epoch_pbar:
            train_loss = self.train_epoch(epoch=epoch)
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Print epoch loss clearly
            print(f"  Epoch {epoch+1:02d}/{num_epochs}: Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}")
            
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}'
            })
            
            # Save checkpoint if best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_dir:
                    self.save_checkpoint(
                        str(Path(save_dir) / f'best_model.pth'),
                        epoch,
                        val_loss
                    )
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, path)
    
    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns epoch."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']


# =============================================================================
# Inference and Evaluation
# =============================================================================

class ModelAInference:
    """
    Inference pipeline for Model A.
    """
    
    def __init__(
        self,
        model: nn.Module,
        stft_processor: STFTProcessor,
        device: str = 'cpu'
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Trained U-Net model
            stft_processor: STFT processor instance
            device: Device to use
        """
        self.model = model.to(device)
        self.model.eval()
        self.stft_processor = stft_processor
        self.device = device
    
    def separate(self, mixture: np.ndarray) -> np.ndarray:
        """
        Separate a mixture into target source.
        
        Args:
            mixture: Mono or stereo waveform
            
        Returns:
            Separated target source waveform (same shape as input)
        """
        # Compute STFT
        mix_mag, mix_phase = self.stft_processor.waveform_to_magnitude_phase(mixture)
        
        # Store original dimensions for cropping later
        orig_freq, orig_time = mix_mag.shape
        
        # Pad spectrograms to compatible dimensions for U-Net (depth=4 requires divisible by 16)
        freq_dim, time_dim = mix_mag.shape
        padded_freq = ((freq_dim + 15) // 16) * 16
        padded_time = ((time_dim + 15) // 16) * 16
        
        # Pad frequency dimension
        if freq_dim < padded_freq:
            pad_freq = padded_freq - freq_dim
            mix_mag = np.pad(mix_mag, ((0, pad_freq), (0, 0)), mode='constant', constant_values=0)
            mix_phase = np.pad(mix_phase, ((0, pad_freq), (0, 0)), mode='constant', constant_values=0)
        
        # Pad time dimension
        if time_dim < padded_time:
            pad_time = padded_time - time_dim
            mix_mag = np.pad(mix_mag, ((0, 0), (0, pad_time)), mode='constant', constant_values=0)
            mix_phase = np.pad(mix_phase, ((0, 0), (0, pad_time)), mode='constant', constant_values=0)
        
        # Convert to tensor
        if mix_mag.ndim == 2:
            mix_mag_tensor = torch.from_numpy(mix_mag[np.newaxis, np.newaxis, :, :]).float()
        else:
            # Multi-channel: process each separately
            mix_mag_tensor = torch.from_numpy(mix_mag[:, np.newaxis, :, :]).float()
        
        # Predict mask
        with torch.no_grad():
            mix_mag_tensor = mix_mag_tensor.to(self.device)
            predicted_mask = self.model(mix_mag_tensor)
            predicted_mask = predicted_mask.cpu().numpy()
        
        # Apply mask to mixture magnitude
        if predicted_mask.ndim == 4 and predicted_mask.shape[0] > 1:
            # Multi-channel output
            estimated_mag = mix_mag * predicted_mask[:, 0]
        else:
            # Single-channel output
            estimated_mag = mix_mag * predicted_mask[0, 0]
        
        # Crop back to original dimensions
        estimated_mag = estimated_mag[:orig_freq, :orig_time]
        mix_phase_cropped = mix_phase[:orig_freq, :orig_time]
        
        # Reconstruct with mixture phase
        separated = self.stft_processor.magnitude_phase_to_waveform(
            estimated_mag,
            mix_phase_cropped
        )
        
        return separated
