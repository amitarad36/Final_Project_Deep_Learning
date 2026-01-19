import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
from IPython.display import Audio, display
from pathlib import Path
from torch.utils.data import Dataset

# ==============================================================================
# 1. AUDIO PROCESSOR
# ==============================================================================
class AudioProcessor:
    """
    Handles conversions between waveform and spectrogram representations
    """ 
    def __init__(self, n_fft=2048, hop_length=512, device='cpu'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        # CRITICAL FIX: Pre-allocate window to prevent PyTorch crash
        self.window = torch.hann_window(n_fft).to(device)

    def to_spectrogram(self, waveform):
        """Waveform -> Log-Magnitude + Phase"""
        # Ensure tensor and correct device
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) # Add channel dim
            
        waveform = waveform.to(self.device).float()
        
        # Pass the pre-allocated window
        complex_spec = torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window, 
            return_complex=True
        )
        
        mag = torch.abs(complex_spec)
        phase = torch.angle(complex_spec)
        log_mag = torch.log1p(mag) # Log compression
        
        return log_mag, phase

    def to_waveform(self, log_mag, phase):
        """Log-Magnitude + Phase -> Waveform"""
        if isinstance(log_mag, np.ndarray): log_mag = torch.from_numpy(log_mag)
        if isinstance(phase, np.ndarray): phase = torch.from_numpy(phase)
            
        log_mag = log_mag.to(self.device)
        phase = phase.to(self.device)
        
        lin_mag = torch.expm1(log_mag)
        complex_spec = lin_mag * torch.exp(1j * phase)
        
        # Pass window here too
        waveform = torch.istft(
            complex_spec, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window
        )
        return waveform.cpu().numpy()

# ==============================================================================
# 2. DATASET
# ==============================================================================
class SpectrogramDataset(Dataset):
    """
    Dataset for loading spectrogram pairs from cached .npy files
    """
    def __init__(self, mixture_files, target_files, limit=None):
        self.mixture_files = sorted(list(mixture_files))[:limit] if limit else sorted(list(mixture_files))
        self.target_files = sorted(list(target_files))[:limit] if limit else sorted(list(target_files))

    def __len__(self):
        return len(self.mixture_files)
    
    def __getitem__(self, idx):
        # Load Raw Audio
        mix = np.load(self.mixture_files[idx])
        tgt = np.load(self.target_files[idx])

        # Crop to matching length
        min_len = min(len(mix), len(tgt))
        mix, tgt = mix[:min_len], tgt[:min_len]
        
        return {
            'mix': torch.from_numpy(mix).float(),
            'tgt': torch.from_numpy(tgt).float()
        }

# ==============================================================================
# 3. HELPER FUNCTIONS (Audio & Viz)
# ==============================================================================
def play_audio(waveform, sr=22050, title="Audio"):
    if hasattr(waveform, 'cpu'): waveform = waveform.squeeze().cpu().numpy()
    print(f"🎵 {title}:")
    display(Audio(waveform, rate=sr))

def visualize_results(mix_mag, target_mag, pred_mag, title="Results"):
    # REMOVED: plt.style.use('default') <- This causes hangs in some environments
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    def show(ax, spec, name):
        # Handle Tensor vs Numpy
        if hasattr(spec, 'cpu'): 
            spec = spec.squeeze().cpu().numpy()
            
        img = ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(name)
        # Disable colorbar if it causes layout issues, or keep it simple
        plt.colorbar(img, ax=ax)

    show(axes[0], mix_mag, "Mixture")
    show(axes[1], target_mag, "Target")
    show(axes[2], pred_mag, "Prediction")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig) # Explicitly close to free memory

# ==============================================================================
# 4. CACHING LOGIC
# ==============================================================================
def prepare_curriculum_cache(mus, cache_dir="../data/curriculum", sr=22050):
    root = Path(cache_dir)
    if root.exists() and len(list(root.glob("**/*.npy"))) > 0:
        print(f"✓ Cache found at {root}. Skipping generation.")
        return

    print(f"⚙️ Generating Curriculum Cache at {root}...")
    for stage in ["stage1", "stage2"]:
        for type in ["mixture", "target"]:
            (root / stage / type).mkdir(parents=True, exist_ok=True)

    for i, track in enumerate(mus.tracks):
        print(f"   Processing: {track.title}...", end="\r")
        stems = {}
        for name, stem_obj in track.targets.items():
            audio = stem_obj.audio.T 
            resampled = librosa.resample(audio, orig_sr=track.rate, target_sr=sr)
            stems[name] = np.mean(resampled, axis=0).astype(np.float32)

        # Stage 1: Vocals + Other -> Other
        np.save(root / "stage1/mixture" / f"{i:03d}.npy", stems['vocals'] + stems['other'])
        np.save(root / "stage1/target" / f"{i:03d}.npy", stems['other'])

        # Stage 2: Full Mix -> Other
        s2_mix = stems['vocals'] + stems['other'] + stems['drums'] + stems['bass']
        np.save(root / "stage2/mixture" / f"{i:03d}.npy", s2_mix)
        np.save(root / "stage2/target" / f"{i:03d}.npy", stems['other'])
        
    print("\n✅ Cache generation complete!")