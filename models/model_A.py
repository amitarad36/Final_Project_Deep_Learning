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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            ConvLayer2D(in_channels, out_channels, 3, 1, 1),
            ConvLayer2D(out_channels, out_channels, 3, 1, 1)
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.block(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.block = nn.Sequential(
            ConvLayer2D(in_channels, out_channels, 3, 1, 1),
            ConvLayer2D(out_channels, out_channels, 3, 1, 1)
        )
        
    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape != skip.shape:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

class TimeFrequencyDomainUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32, num_layers=4):
        super(TimeFrequencyDomainUNet, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            inc = in_channels if i == 0 else base_filters * (2 ** (i - 1))
            outc = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(inc, outc))
        
        bot_in = base_filters * (2 ** (num_layers - 1))
        bot_out = base_filters * (2 ** num_layers)
        self.bottleneck = ConvLayer2D(bot_in, bot_out, 3, 1, 1)
        
        for i in range(num_layers - 1, -1, -1):
            dec_in = bot_out if i == num_layers - 1 else base_filters * (2 ** (i + 1))
            dec_out = base_filters * (2 ** i)
            self.decoders.append(DecoderBlock(dec_in, dec_out))
        
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
    
# =============================================================================
# Training Components
# =============================================================================
  
class ModelATrainer:
    def __init__(self, model, train_loader, val_loader, processor, learning_rate=1e-4, device='cpu', patience=10):
        self.model = model.to(device)
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.L1Loss()
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch_idx} Training", leave=False)
        for batch in pbar:
            mix_wav = batch['mix'].to(self.device)
            tgt_wav = batch['tgt'].to(self.device)
            mix_mag, _ = self.processor.to_spectrogram(mix_wav)
            tgt_mag, _ = self.processor.to_spectrogram(tgt_wav)
            mix_mag = mix_mag.unsqueeze(1)
            tgt_mag = tgt_mag.unsqueeze(1)
            self.optimizer.zero_grad()
            mask = self.model(mix_mag)
            if mask.shape != mix_mag.shape:
                mask = mask[:, :, :mix_mag.shape[2], :mix_mag.shape[3]]
            est_mag = mask * torch.expm1(mix_mag)
            loss = self.loss_fn(est_mag, torch.expm1(tgt_mag))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                mix_wav = batch['mix'].to(self.device)
                tgt_wav = batch['tgt'].to(self.device)
                mix_mag, _ = self.processor.to_spectrogram(mix_wav)
                tgt_mag, _ = self.processor.to_spectrogram(tgt_wav)
                mix_mag = mix_mag.unsqueeze(1)
                tgt_mag = tgt_mag.unsqueeze(1)
                mask = self.model(mix_mag)
                if mask.shape != mix_mag.shape:
                    mask = mask[:, :, :mix_mag.shape[2], :mix_mag.shape[3]]
                est_mag = mask * torch.expm1(mix_mag)
                loss = self.loss_fn(est_mag, torch.expm1(tgt_mag))
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self, num_epochs, save_path=None):
        epochs_no_improve = 0
        global_pbar = tqdm(range(num_epochs), desc="Total Progress")
        for epoch in global_pbar:
            train_loss = self.train_epoch(epoch + 1)
            val_loss = self.validate()
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            global_pbar.set_postfix({'Train': f"{train_loss:.4f}", 'Val': f"{val_loss:.4f}"})
            print(f"Epoch {epoch+1}: Train {train_loss:.5f} | Val {val_loss:.5f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_no_improve = 0
                # Save both model and history
                if save_path is not None:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'history': self.history
                    }, save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        return self.history
    
# =============================================================================
# Inference and Evaluation
# =============================================================================

class ModelAInference:
    def __init__(self, model, processor, device='cpu'):
        self.model = model.to(device)
        self.processor = processor
        self.device = device

    def separate(self, mixture_waveform):
        self.model.eval()
        with torch.no_grad():
            mix_wav = torch.tensor(mixture_waveform).to(self.device)
            if mix_wav.ndim == 1: mix_wav = mix_wav.unsqueeze(0)
            mix_mag, mix_phase = self.processor.to_spectrogram(mix_wav)
            mix_mag_in = mix_mag.unsqueeze(1)
            mask = self.model(mix_mag_in)
            if mask.shape != mix_mag_in.shape:
                mask = mask[:, :, :mix_mag_in.shape[2], :mix_mag_in.shape[3]]
            est_mag = mask.squeeze(1) * mix_mag
            est_wav = self.processor.to_waveform(est_mag, mix_phase)
            return est_wav.squeeze().cpu().numpy()
    