import numpy as np
import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import librosa

# Configure matplotlib for Jupyter/Colab environment
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

# ===============================================================================
# CHECKPOINT UTILITIES
# ===============================================================================
def check_checkpoint(checkpoint_path, checkpoint_name="Checkpoint"):
    """
    Check if a checkpoint exists and print status.
    Returns True if checkpoint exists, False otherwise.
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        print(f"✅ {checkpoint_name} found: {checkpoint_path}")
        print(f"   Loading existing checkpoint instead of retraining.")
        return True
    else:
        print(f"⏳ {checkpoint_name} not found: {checkpoint_path}")
        print(f"   Training will start...")
        return False

# ===============================================================================
# TRAIN/LOAD STAGE HELPER
# ===============================================================================
def train_stage(mix_files, tgt_files, model, processor, batch_size, num_epochs, patience, learning_rate, ckpt_path, device):
    """
    Helper to train or load a model stage with checkpointing.
    """
    from torch.utils.data import DataLoader
    split = int(len(mix_files) * 0.8)
    train_ds = StandardDataset(mix_files[:split], tgt_files[:split])
    val_ds = StandardDataset(mix_files[split:], tgt_files[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.L1Loss()
    trainer = UniversalTrainer(model, train_loader, val_loader, processor, optimizer, loss_fn, device, patience, input_type='spectrogram')
    if not os.path.exists(ckpt_path):
        history = trainer.train(num_epochs, ckpt_path)
    else:
        print(f"Found existing checkpoint: {ckpt_path}. Loading...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        history = checkpoint.get('history', {})
    return history
# ==============================================================================
# Universal Trainer
# ==============================================================================
class UniversalTrainer:
    """
    Generic trainer for spectrogram and waveform models.
    Handles both input types via config flag or input check.
    """
    def __init__(self, model, train_loader, val_loader, processor, optimizer, loss_fn, device='cpu', patience=10, input_type='spectrogram'):
        """
        Initializes UniversalTrainer with model, data loaders, processor, optimizer, loss function, device, patience, and input_type ('spectrogram' or 'waveform').
        """
        self.model = model.to(device)
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.input_type = input_type
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch_idx):
        """
        Trains for one epoch and returns average loss.
        Implements: mask is applied in linear domain, loss is computed in log domain (Option 3).
        """
        self.model.train()
        total_loss = 0
        # Use notebook tqdm if in notebook, else fallback to plain tqdm or print
        def _in_notebook():
            try:
                from IPython import get_ipython
                shell = get_ipython().__class__.__name__
                if shell == 'ZMQInteractiveShell':
                    return True  # Jupyter notebook or qtconsole
                else:
                    return False  # Other type (likely terminal)
            except Exception:
                return False

        if _in_notebook():
            from tqdm.notebook import tqdm as tqdm_bar
        else:
            from tqdm import tqdm as tqdm_bar

        pbar = tqdm_bar(self.train_loader, desc=f"Ep {epoch_idx} Training", leave=False)
        for batch in pbar:
            mix = batch['mix'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            if self.input_type == 'spectrogram':
                mix_log, _ = self.processor.to_spectrogram(mix)
                tgt_log, _ = self.processor.to_spectrogram(tgt)
                mix_log = mix_log.unsqueeze(1)
                tgt_log = tgt_log.unsqueeze(1)
                self.optimizer.zero_grad()
                mask = self.model(mix_log)
                if mask.shape != mix_log.shape:
                    mask = mask[:, :, :mix_log.shape[2], :mix_log.shape[3]]
                est_linear = mask * torch.expm1(mix_log)
                est_log = torch.log1p(est_linear)
                loss = self.loss_fn(est_log, tgt_log)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            else:
                self.optimizer.zero_grad()
                output = self.model(mix)
                loss = self.loss_fn(output, tgt)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
        return total_loss / len(self.train_loader)

    def validate(self):
        """
        Evaluates the model on the validation set and returns average loss
        Implements: mask is applied in linear domain, loss is computed in log domain
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                mix = batch['mix'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                if self.input_type == 'spectrogram':
                    mix_log, _ = self.processor.to_spectrogram(mix)
                    tgt_log, _ = self.processor.to_spectrogram(tgt)
                    mix_log = mix_log.unsqueeze(1)
                    tgt_log = tgt_log.unsqueeze(1)
                    mask = self.model(mix_log)
                    if mask.shape != mix_log.shape:
                        mask = mask[:, :, :mix_log.shape[2], :mix_log.shape[3]]
                    est_linear = mask * torch.expm1(mix_log)
                    est_log = torch.log1p(est_linear)
                    loss = self.loss_fn(est_log, tgt_log)
                    total_loss += loss.item()
                else:
                    output = self.model(mix)
                    loss = self.loss_fn(output, tgt)
                    total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self, num_epochs, save_path=None, log_file_path=None):
        """
        Trains the model for a given number of epochs and saves the best checkpoint.
        Returns training history.
        """
        def _in_notebook():
            try:
                from IPython import get_ipython
                shell = get_ipython().__class__.__name__
                if shell == 'ZMQInteractiveShell':
                    return True
                else:
                    return False
            except Exception:
                return False

        if _in_notebook():
            from tqdm.notebook import tqdm as tqdm_bar
        else:
            from tqdm import tqdm as tqdm_bar

        epochs_no_improve = 0
        global_pbar = tqdm_bar(range(num_epochs), desc="Total Progress")
        # Create a subfolder for this training run based on save_path
        import os
        epoch_dir = None
        if save_path is not None:
            base_dir = os.path.dirname(save_path)
            run_name = os.path.splitext(os.path.basename(save_path))[0]
            epoch_dir = os.path.join(base_dir, f"{run_name}_epochs")
            os.makedirs(epoch_dir, exist_ok=True)

        best_epoch = 0
        best_train_loss = None
        for epoch in global_pbar:
            train_loss = self.train_epoch(epoch + 1)
            val_loss = self.validate()
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            global_pbar.set_postfix({'Train': f"{train_loss:.4f}", 'Val': f"{val_loss:.4f}"})
            print(f"Epoch {epoch+1}: Train {train_loss:.5f} | Val {val_loss:.5f}")
            # Live logging to file
            if log_file_path:
                try:
                    with open(log_file_path, 'a') as f:
                        f.write(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")
                        f.flush()
                except Exception as e:
                    print(f"[WARN] Could not write to log file {log_file_path}: {e}")
            # Write a separate file for each epoch in the subfolder
            if epoch_dir is not None:
                try:
                    epoch_file = os.path.join(epoch_dir, f"epoch_{epoch+1:03d}.txt")
                    with open(epoch_file, 'w') as ef:
                        ef.write(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")
                except Exception as e:
                    print(f"[WARN] Could not write epoch file {epoch_file}: {e}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = epoch + 1
                best_train_loss = train_loss
                epochs_no_improve = 0
                if save_path is not None:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'history': self.history
                    }, save_path)
                # Write/update best_epoch.txt
                if epoch_dir is not None:
                    try:
                        best_file = os.path.join(epoch_dir, "best_epoch.txt")
                        with open(best_file, 'w') as bf:
                            bf.write(f"Best Epoch: {best_epoch}\nTrain Loss: {best_train_loss:.4f}\nVal Loss: {self.best_val_loss:.4f}\n")
                    except Exception as e:
                        print(f"[WARN] Could not write best_epoch.txt: {e}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        return self.history

# ==============================================================================
# Separator
# ==============================================================================
class Separator:
    """
    Generic inference class for source separation models.
    """
    def __init__(self, model, processor, device='cpu', input_type='spectrogram'):
        """
        Initializes Separator with model, processor, device, and input_type.
        """
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.input_type = input_type

    def separate(self, mixture):
        """
        Separates sources from mixture using trained model.
        Returns estimated output.
        """
        self.model.eval()
        with torch.no_grad():
            mix = torch.tensor(mixture).to(self.device)
            if mix.ndim == 1:
                mix = mix.unsqueeze(0)
            if self.input_type == 'spectrogram':
                mix, mix_phase = self.processor.to_spectrogram(mix)
                mix_in = mix.unsqueeze(1)
                mask = self.model(mix_in)
                if mask.shape != mix_in.shape:
                    mask = mask[:, :, :mix_in.shape[2], :mix_in.shape[3]]
                est_mag = mask.squeeze(1) * mix
                est = self.processor.to_waveform(est_mag, mix_phase)
                return est.squeeze().cpu().numpy()
            else:
                est = self.model(mix)
                return est.squeeze().cpu().numpy()

# ==============================================================================
# Metrics Calculation (use this in notebooks!!!)
# ==============================================================================
def calculate_metrics(reference, estimate, sr=22050):
    """
    Calculates SDR, SIR, SAR using museval.
    Returns a dict of metrics.
    """
    import museval
    import numpy as np
    # museval expects shape (sources, samples)
    reference = np.atleast_2d(reference)
    estimate = np.atleast_2d(estimate)
    scores = museval.evaluate(reference, estimate, win=1*sr)
    metrics = {
        'SDR': np.nanmean(scores['SDR']),
        'SIR': np.nanmean(scores['SIR']),
        'SAR': np.nanmean(scores['SAR'])
    }
    return metrics


# ==============================================================================
# 1. AUDIO PROCESSOR
# ==============================================================================
class AudioProcessor:
    """
    Handles conversions between waveform and spectrogram representations.
    """
    def __init__(self, n_fft=2048, hop_length=512, device='cpu'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        # Pre-allocate window for PyTorch
        self.window = torch.hann_window(n_fft).to(device)

    def to_spectrogram(self, waveform):
        """
        Converts waveform to log-magnitude spectrogram and phase.
        Returns (log_mag, phase).
        """
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
        """
        Converts log-magnitude and phase to waveform.
        Returns waveform as numpy array.
        """
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

# Dataset for loading spectrogram pairs from cached .npy files
class SpectrogramDataset(Dataset):
    """
    Loads spectrogram pairs from cached .npy files.
    Returns dicts with keys 'mix' and 'tgt'.
    """
    def __init__(self, mixture_files, target_files, limit=None):
        self.mixture_files = sorted(list(mixture_files))[:limit] if limit else sorted(list(mixture_files))
        self.target_files = sorted(list(target_files))[:limit] if limit else sorted(list(target_files))

    def __len__(self):
        """
        Returns number of samples.
        """
        return len(self.mixture_files)

    def __getitem__(self, idx):
        """
        Loads mixture and target spectrograms, returns as tensors in dict.
        """
        mix = np.load(self.mixture_files[idx])
        tgt = np.load(self.target_files[idx])
        min_len = min(len(mix), len(tgt))
        mix, tgt = mix[:min_len], tgt[:min_len]
        return {
            'mix': torch.from_numpy(mix).float(),
            'tgt': torch.from_numpy(tgt).float()
        }

# Robust waveform dataset for general use (moved from notebook)
class StandardDataset(Dataset):
    """
    Loads waveform pairs from cached .npy files for training/validation.
    Returns dicts with keys 'mix' and 'tgt'.
    """
    def __init__(self, mix_files, tgt_files):
        self.mix_files = list(mix_files)
        self.tgt_files = list(tgt_files)

    def __len__(self):
        """
        Returns number of samples.
        """
        return len(self.mix_files)

    def __getitem__(self, idx):
        """
        Loads mixture and target waveforms, returns as tensors in dict.
        """
        m = np.load(self.mix_files[idx])
        t = np.load(self.tgt_files[idx])
        return {
            'mix': torch.tensor(m, dtype=torch.float32),
            'tgt': torch.tensor(t, dtype=torch.float32)
        }

# Chunked dataset for fixed-length segments with overlap
class ChunkedDataset(Dataset):
    """
    Splits variable-length audio files into fixed-length chunks with overlap.
    Useful for training on consistent segment sizes (e.g., 1 second).
    """
    def __init__(self, mix_files, tgt_files, chunk_duration=1.0, overlap=0.3, sr=22050):
        """
        Args:
            mix_files: List of mixture file paths
            tgt_files: List of target file paths
            chunk_duration: Length of each chunk in seconds
            overlap: Overlap between chunks in seconds
            sr: Sample rate
        """
        self.mix_files = list(mix_files)
        self.tgt_files = list(tgt_files)
        self.chunk_size = int(chunk_duration * sr)
        self.hop_size = int((chunk_duration - overlap) * sr)
        self.sr = sr
        
        # Pre-compute chunk indices
        self.chunks = []
        for file_idx, (mix_path, tgt_path) in enumerate(zip(self.mix_files, self.tgt_files)):
            # Get file length
            file_length = np.load(mix_path, mmap_mode='r').shape[0]
            # Calculate chunk starts
            starts = list(range(0, file_length - self.chunk_size + 1, self.hop_size))
            for start in starts:
                self.chunks.append((file_idx, start))
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        """
        Returns a chunk of audio as tensors.
        """
        file_idx, start = self.chunks[idx]
        end = start + self.chunk_size
        
        mix_full = np.load(self.mix_files[file_idx])
        tgt_full = np.load(self.tgt_files[file_idx])
        
        mix_chunk = mix_full[start:end]
        tgt_chunk = tgt_full[start:end]
        
        return {
            'mix': torch.tensor(mix_chunk, dtype=torch.float32),
            'tgt': torch.tensor(tgt_chunk, dtype=torch.float32)
        }

# ==============================================================================
# CONFIGURATION FUNCTIONS
# ==============================================================================
def get_model_a_config():
    """
    Returns Model A (Time-Frequency Domain U-Net) architecture configuration.
    """
    return {
        'n_fft': 2048,
        'hop_length': 512,
        'encoder_channels': [1, 16, 32, 64, 128, 256],
        'decoder_channels': [256, 128, 64, 32, 16, 1],
        'use_batch_norm': True
    }

def get_training_config():
    """
    Returns general training configuration for Model A.
    """
    return {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'chunk_duration': 1.0,  # 1 second chunks
        'chunk_overlap': 0.5,   # 0.5 second overlap (50%)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

def get_overfit_config(chunk_duration=4.0, num_layers=4):
    """
    Overfit configuration with flexibility for different model sizes.
    
    Args:
        chunk_duration: Audio chunk length in seconds (default 4.0)
        num_layers: Number of U-Net layers (default 4)
    
    Returns:
        Config dict with all training parameters
    """
    return {
        'batch_size': 1,
        'learning_rate': 3e-4,
        'num_epochs': 100,
        'chunk_duration': chunk_duration,
        'chunk_overlap': 0.5,
        'patience': 200,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_layers': num_layers
    }

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def play_audio(waveform, sr=22050, title="Audio"):
    """
    Plays audio waveform in notebook.
    """
    if hasattr(waveform, 'cpu'):
        waveform = waveform.squeeze().cpu().numpy()
    print(f"{title}:")
    display(Audio(waveform, rate=sr))

def visualize_results(mix_mag, target_mag, pred_mag, title="Results"):
    """
    Visualizes mixture, target, and prediction spectrograms side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    def show(ax, spec, name):
        if hasattr(spec, 'cpu'):
            spec = spec.squeeze().cpu().numpy()
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(name)
    show(axes[0], mix_mag, "Mixture")
    show(axes[1], target_mag, "Target")
    show(axes[2], pred_mag, "Prediction")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ===============================================================================
# LOSS VISUALIZATION
# ===============================================================================
def plot_text_graph(history, title):
    """
    Print a simple text-based loss graph for quick inspection.
    """
    if 'train_loss' not in history or len(history['train_loss']) == 0:
        return
    print(f"\n {title} (Text Graph):")
    losses = history['train_loss']
    min_val, max_val = min(losses), max(losses)
    for i, loss in enumerate(losses):
        width = int(40 * (loss - min_val) / (max_val - min_val + 1e-9))
        print(f"Ep {i+1:02d}: {loss:.4f} | {'█' * width}")

# ==============================================================================
# NOTEBOOK COMPACT HELPERS
# ==============================================================================
def get_curriculum_file_lists(cache_dir="../data", split='train'):
    """
    Returns sorted file lists for stage1 and stage2 (mixture/target).
    
    Args:
        cache_dir: Root data directory
        split: 'train', 'val', or 'test' (default: 'train' for backward compatibility)
        
    Returns:
        Tuple of (mix_files_stage1, tgt_files_stage1, mix_files_stage2, tgt_files_stage2)
    """
    data_root = Path(cache_dir)
    
    # Try new structure first (stage/split/type)
    s1_mix_path = data_root / "stage1" / split / "mixture"
    s1_tgt_path = data_root / "stage1" / split / "target"
    s2_mix_path = data_root / "stage2" / split / "mixture"
    s2_tgt_path = data_root / "stage2" / split / "target"
    
    # Fallback to old structure (stage/type) if new doesn't exist
    if not s1_mix_path.exists():
        s1_mix_path = data_root / "stage1" / "mixture"
        s1_tgt_path = data_root / "stage1" / "target"
        s2_mix_path = data_root / "stage2" / "mixture"
        s2_tgt_path = data_root / "stage2" / "target"

    mix_files_stage1 = sorted(list(s1_mix_path.glob("*.npy"))) if s1_mix_path.exists() else []
    tgt_files_stage1 = sorted(list(s1_tgt_path.glob("*.npy"))) if s1_tgt_path.exists() else []
    mix_files_stage2 = sorted(list(s2_mix_path.glob("*.npy"))) if s2_mix_path.exists() else []
    tgt_files_stage2 = sorted(list(s2_tgt_path.glob("*.npy"))) if s2_tgt_path.exists() else []

    return mix_files_stage1, tgt_files_stage1, mix_files_stage2, tgt_files_stage2


def run_overfit_1song(
    overfit_model,
    overfit_processor,
    overfit_optimizer,
    overfit_loss_fn,
    overfit_config,
    cache_dir="../data",
    save_path="../checkpoints/debug_overfit_1song.pth",
    device="cpu",
):
    """
    Runs (or loads) an overfit sanity check on 1 random song.
    Uses preprocessed data from stage1/train.
    Returns training history.
    """
    import random
    from torch.utils.data import DataLoader

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load from preprocessed train split
    s1_train_root = Path(cache_dir) / "stage1" / "train"
    all_mix_files = sorted(list((s1_train_root / "mixture").glob("*.npy")))
    all_tgt_files = sorted(list((s1_train_root / "target").glob("*.npy")))

    if len(all_mix_files) < 1 or len(all_tgt_files) < 1:
        raise ValueError("Not enough data for overfit test. Please run preprocessing first.")

    # Select 1 random song
    idx = random.randint(0, len(all_mix_files) - 1)
    mix_files = [all_mix_files[idx]]
    tgt_files = [all_tgt_files[idx]]

    chunk_dur = overfit_config.get('chunk_duration', 4.0)
    tiny_ds = ChunkedDataset(
        mix_files, tgt_files,
        chunk_duration=chunk_dur,
        overlap=overfit_config.get('chunk_overlap', 0.3)
    )
    print(f"   Chunk file #{idx} (1 file): {len(tiny_ds)} sub-chunks ({chunk_dur}s each)")
    tiny_loader = DataLoader(tiny_ds, batch_size=overfit_config['batch_size'], shuffle=False)

    trainer_overfit = UniversalTrainer(
        model=overfit_model,
        train_loader=tiny_loader,
        val_loader=tiny_loader,
        processor=overfit_processor,
        optimizer=overfit_optimizer,
        loss_fn=overfit_loss_fn,
        device=device,
        patience=overfit_config.get('patience', 10)
    )

    history = {}
    if not os.path.exists(save_path):
        print(f"   Training from scratch on 1 random song (#{idx})...")
        history = trainer_overfit.train(num_epochs=overfit_config['num_epochs'], save_path=save_path)
    else:
        print(f"✓ Found Checkpoint: {save_path}")
        ckpt = torch.load(save_path, map_location=device)
        overfit_model.load_state_dict(ckpt['model_state_dict'])
        history = ckpt.get('history', {})

    return history


def run_full_training(
    model,
    processor,
    optimizer,
    loss_fn,
    train_config,
    cache_dir="../data",
    log_file_path=None,
    save_path_stage1="../checkpoints/full_stage1.pth",
    save_path_stage2="../checkpoints/full_stage2.pth",
    device="cpu",
):
    """
    Runs (or loads) full training for stage1 and stage2.
    Uses preprocessed train/val/test splits.
    Returns histories for both stages.
    """
    from torch.utils.data import DataLoader

    # Stage 1
    print("\n--- Stage 1: Weighted Mixture → Vocals ---")
    train_loader1 = get_data_loaders(cache_dir, stage='stage1', split='train', batch_size=train_config['batch_size'])
    val_loader1 = get_data_loaders(cache_dir, stage='stage1', split='val', batch_size=train_config['batch_size'])
    
    print(f"   Train: {len(train_loader1.dataset)} samples")
    print(f"   Val:   {len(val_loader1.dataset)} samples")

    trainer_s1 = UniversalTrainer(
        model=model,
        train_loader=train_loader1,
        val_loader=val_loader1,
        processor=processor,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        patience=train_config.get('patience', 10)
    )

    hist_s1 = {}
    if not os.path.exists(save_path_stage1):
        hist_s1 = trainer_s1.train(num_epochs=train_config['num_epochs'], save_path=save_path_stage1, log_file_path=log_file_path)
    else:
        print(f"✓ Found Checkpoint: {save_path_stage1}")
        ckpt = torch.load(save_path_stage1, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        hist_s1 = ckpt.get('history', {})

    # Stage 2
    print("\n--- Stage 2: Balanced Mixture → Vocals ---")
    train_loader2 = get_data_loaders(cache_dir, stage='stage2', split='train', batch_size=train_config['batch_size'])
    val_loader2 = get_data_loaders(cache_dir, stage='stage2', split='val', batch_size=train_config['batch_size'])
    
    print(f"   Train: {len(train_loader2.dataset)} samples")
    print(f"   Val:   {len(val_loader2.dataset)} samples")

    for param_group in optimizer.param_groups:
        param_group['lr'] = train_config['learning_rate'] * 0.1
    print(f"✓ Optimizer LR reduced to {train_config['learning_rate'] * 0.1}")

    trainer_s2 = UniversalTrainer(
        model=model,
        train_loader=train_loader2,
        val_loader=val_loader2,
        processor=processor,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        patience=train_config.get('patience', 10)
    )

    hist_s2 = {}
    if not os.path.exists(save_path_stage2):
        hist_s2 = trainer_s2.train(num_epochs=train_config['num_epochs'], save_path=save_path_stage2, log_file_path=log_file_path)
    else:
        print(f"✓ Found Checkpoint: {save_path_stage2}")
        ckpt = torch.load(save_path_stage2, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        hist_s2 = ckpt.get('history', {})

    return hist_s1, hist_s2


def plot_loss_from_checkpoint(ckpt_path, title="Loss Curves from Checkpoint"):
    """
    Loads a checkpoint and plots loss curves if history is present.
    Also checks for epoch folder to get complete training history.
    Works reliably on Colab with GPU.
    """
    try:
        ckpt_path = Path(ckpt_path)
        train_losses = []
        val_losses = []
        
        # First, try to read from epoch files (more complete history)
        epoch_folder = ckpt_path.parent / f"{ckpt_path.stem}_epochs"
        if epoch_folder.exists():
            print(f"📁 Reading from epoch folder: {epoch_folder.name}")
            epoch_files = sorted(epoch_folder.glob("epoch_*.txt"))
            
            for epoch_file in epoch_files:
                try:
                    with open(epoch_file, 'r') as f:
                        content = f.read()
                        # Parse loss values from file
                        import re
                        train_match = re.search(r'Train Loss[:\s=]+([\d.]+)', content)
                        val_match = re.search(r'Val Loss[:\s=]+([\d.]+)', content)
                        if train_match and val_match:
                            train_losses.append(float(train_match.group(1)))
                            val_losses.append(float(val_match.group(1)))
                except Exception as e:
                    print(f"⚠️  Error reading {epoch_file.name}: {e}")
                    continue
        
        # Fallback to checkpoint history if no epoch files found
        if not train_losses:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if 'history' in ckpt:
                history = ckpt['history']
                train_losses = history.get('train_loss', [])
                val_losses = history.get('val_loss', [])
        
        # Plot if we have data
        if train_losses:
            print(f"📊 Plotting {len(train_losses)} epochs")
            
            plt.figure(figsize=(12, 6), dpi=100)
            epochs = range(1, len(train_losses) + 1)
            
            plt.plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=6)
            if val_losses:
                plt.plot(epochs, val_losses, 's--', label='Val Loss', linewidth=2, markersize=6)
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(fontsize=11, loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.show()
            plt.close('all')
        else:
            print("⚠️  No training data found in checkpoint or epoch files.")
    except Exception as e:
        print(f"❌ Error plotting checkpoint: {e}")
        import traceback
        traceback.print_exc()


def demo_separation_sample(
    model,
    processor,
    cache_dir="../data",
    stage="stage1",
    split="train",
    song_num=12,
    duration=6,
    sr=22050,
    device="cpu",
    play_audio_output=True,
):
    """
    Visualizes and (optionally) plays mixture/target/predicted audio for one sample.
    
    Args:
        model: Trained model
        processor: AudioProcessor
        cache_dir: Root data directory
        stage: 'stage1' or 'stage2'
        split: 'train', 'val', or 'test'
        song_num: Index of sample to visualize
        duration: Duration in seconds
        sr: Sample rate
        device: 'cpu' or 'cuda'
        play_audio_output: Whether to play audio
    """
    data_root = Path(cache_dir)
    
    # Try new structure first (stage/split/type)
    mix_path = data_root / stage / split / "mixture"
    tgt_path = data_root / stage / split / "target"
    
    # Fallback to old structure (stage/type)
    if not mix_path.exists():
        mix_path = data_root / stage / "mixture"
        tgt_path = data_root / stage / "target"

    mix_files = sorted(list(mix_path.glob("*.npy")))
    tgt_files = sorted(list(tgt_path.glob("*.npy")))
    
    if len(mix_files) == 0:
        raise FileNotFoundError(f"No data found in {mix_path}")

    n_samples = sr * duration
    mix_wav = np.load(mix_files[song_num])[:n_samples]
    tgt_wav = np.load(tgt_files[song_num])[:n_samples]

    mix_mag, mix_phase = processor.to_spectrogram(torch.tensor(mix_wav))
    tgt_mag, tgt_phase = processor.to_spectrogram(torch.tensor(tgt_wav))
    show_spectrogram(mix_mag, title="Mixture Spectrogram (6 sec)")
    show_spectrogram(tgt_mag, title="Target Spectrogram (6 sec)")

    model.eval()
    with torch.no_grad():
        if mix_mag.dim() == 2:
            mix_mag_in = mix_mag.unsqueeze(0).unsqueeze(0).to(device)
        elif mix_mag.dim() == 3:
            mix_mag_in = mix_mag.unsqueeze(1).to(device)
        elif mix_mag.dim() == 4:
            mix_mag_in = mix_mag.to(device)
        else:
            raise ValueError(f"mix_mag must be 2D, 3D, or 4D, got shape {mix_mag.shape}")

        mask = model(mix_mag_in)
        if mask.shape != mix_mag_in.shape:
            mask = mask[:, :, :mix_mag_in.shape[2], :mix_mag_in.shape[3]]
        est_mag = mask.squeeze(0).squeeze(0) * mix_mag.to(device)
        est_wav = processor.to_waveform(est_mag.cpu(), mix_phase.cpu())

    show_spectrogram(est_mag.cpu(), title="Predicted Spectrogram (6 sec)")

    if play_audio_output:
        play_audio(mix_wav, sr=sr, title="Mixture Audio (6 sec)")
        play_audio(tgt_wav, sr=sr, title="Target Audio (6 sec)")
        play_audio(est_wav, sr=sr, title="Predicted Audio (6 sec)")

    return {
        "mix_wav": mix_wav,
        "tgt_wav": tgt_wav,
        "est_wav": est_wav,
        "mix_mag": mix_mag,
        "tgt_mag": tgt_mag,
        "est_mag": est_mag,
        "mix_phase": mix_phase,
        "tgt_phase": tgt_phase,
    }

# ==============================================================================
# MUSDB18 PREPROCESSING (Main Entry Point)
# ==============================================================================
def preprocess_musdb18(
    musdb18_path,
    output_dir,
    chunk_duration=8.0,
    overlap=0.5,
    sample_rate=22050,
    stage1_ratio=0.7,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
):
    """
    Complete preprocessing pipeline for MUSDB18 dataset.
    
    Steps:
    1. Load train + valid subsets from MUSDB18
    2. Chunk all full-length songs into training segments
    3. Create Stage 1 (70%) and Stage 2 (30%) curriculum splits
    4. Further split each stage into train/val/test
    5. Save organized data ready for DataLoaders
    
    Args:
        musdb18_path: Path to extracted musdb18 folder
        output_dir: Where to save processed data
        chunk_duration: Length of each chunk in seconds
        overlap: Overlap ratio (0.5 = 50% overlap)
        sample_rate: Target sample rate
        stage1_ratio: Ratio of chunks for stage1 (rest goes to stage2)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Dictionary with file counts for each split
    """
    import musdb
    from tqdm import tqdm
    
    print(f"\n{'='*70}")
    print("MUSDB18 PREPROCESSING PIPELINE")
    print(f"{'='*70}\n")
    
    # Load MUSDB18
    print(f"📂 Loading MUSDB18 from: {musdb18_path}")
    mus_train = musdb.DB(root=str(musdb18_path), is_wav=True, subsets='train')
    mus_valid = musdb.DB(root=str(musdb18_path), is_wav=True, subsets='valid')
    all_tracks = list(mus_train.tracks) + list(mus_valid.tracks)
    
    print(f"✅ Found {len(mus_train.tracks)} tracks in TRAIN")
    print(f"✅ Found {len(mus_valid.tracks)} tracks in VALID")
    print(f"📊 Total tracks to process: {len(all_tracks)}\n")
    
    # Configuration
    chunk_samples = int(chunk_duration * sample_rate)
    hop_samples = int(chunk_samples * (1 - overlap))
    
    print(f"⚙️  Settings:")
    print(f"   Chunk Duration: {chunk_duration}s ({chunk_samples} samples)")
    print(f"   Overlap: {overlap*100:.0f}%")
    print(f"   Sample Rate: {sample_rate} Hz")
    print(f"   Stage 1: {stage1_ratio*100:.0f}% | Stage 2: {(1-stage1_ratio)*100:.0f}%")
    print(f"   Train: {train_ratio*100:.0f}% | Val: {val_ratio*100:.0f}% | Test: {test_ratio*100:.0f}%\n")
    
    # Create output directories
    output_root = Path(output_dir)
    stage_dirs = {}
    for stage in ['stage1', 'stage2']:
        for split in ['train', 'val', 'test']:
            for data_type in ['mixture', 'target']:
                dir_path = output_root / stage / split / data_type
                dir_path.mkdir(parents=True, exist_ok=True)
                stage_dirs[f"{stage}_{split}_{data_type}"] = dir_path
    
    # Process tracks and collect chunks
    print("🔄 Processing tracks into chunks...\n")
    all_chunks = []  # Will store (stage, mixture, target) tuples
    
    for track_idx, track in enumerate(tqdm(all_tracks, desc="Processing tracks")):
        # Load all stems
        vocals = track.targets['vocals'].audio  # Shape: (samples, 2) stereo
        drums = track.targets['drums'].audio
        bass = track.targets['bass'].audio
        other = track.targets['other'].audio
        
        # Convert to mono and resample
        vocals_mono = librosa.to_mono(vocals.T)
        drums_mono = librosa.to_mono(drums.T)
        bass_mono = librosa.to_mono(bass.T)
        other_mono = librosa.to_mono(other.T)
        
        if track.rate != sample_rate:
            vocals_mono = librosa.resample(vocals_mono, orig_sr=track.rate, target_sr=sample_rate)
            drums_mono = librosa.resample(drums_mono, orig_sr=track.rate, target_sr=sample_rate)
            bass_mono = librosa.resample(bass_mono, orig_sr=track.rate, target_sr=sample_rate)
            other_mono = librosa.resample(other_mono, orig_sr=track.rate, target_sr=sample_rate)
        
        # Ensure all stems have same length
        min_len = min(len(vocals_mono), len(drums_mono), len(bass_mono), len(other_mono))
        vocals_mono = vocals_mono[:min_len]
        drums_mono = drums_mono[:min_len]
        bass_mono = bass_mono[:min_len]
        other_mono = other_mono[:min_len]
        
        # Chunk the audio
        num_chunks = (min_len - chunk_samples) // hop_samples + 1
        
        for i in range(num_chunks):
            start = i * hop_samples
            end = start + chunk_samples
            
            if end > min_len:
                break
            
            # Extract chunks for all stems (identical time segments)
            vocals_chunk = vocals_mono[start:end]
            drums_chunk = drums_mono[start:end]
            bass_chunk = bass_mono[start:end]
            other_chunk = other_mono[start:end]
            
            # Create accompaniment (drums + bass + other)
            accompaniment_chunk = drums_chunk + bass_chunk + other_chunk
            
            # Decide stage based on ratio
            stage = 'stage1' if np.random.rand() < stage1_ratio else 'stage2'
            
            if stage == 'stage1':
                # STAGE 1: Simple 2-source separation (vocals + other only)
                # Easier curriculum step: learn to separate just 2 sources
                mixture = 0.50 * vocals_chunk + 0.50 * other_chunk
                target = vocals_chunk
            else:
                # STAGE 2: Complex 4-source separation (vocals vs all accompaniment)
                # Harder task: vocals competing with drums, bass, and other
                mixture = 0.40 * vocals_chunk + 0.60 * accompaniment_chunk
                target = vocals_chunk
            
            # Normalize to prevent clipping
            max_val = max(np.abs(mixture).max(), np.abs(target).max())
            if max_val > 0:
                mixture = mixture / max_val
                target = target / max_val
            
            all_chunks.append((stage, mixture.astype(np.float32), target.astype(np.float32)))
    
    # Shuffle and split chunks
    print(f"\n📊 Total chunks created: {len(all_chunks)}")
    np.random.shuffle(all_chunks)
    
    # Separate by stage
    stage1_chunks = [c for c in all_chunks if c[0] == 'stage1']
    stage2_chunks = [c for c in all_chunks if c[0] == 'stage2']
    
    print(f"   Stage 1: {len(stage1_chunks)} chunks ({len(stage1_chunks)/len(all_chunks)*100:.1f}%)")
    print(f"   Stage 2: {len(stage2_chunks)} chunks ({len(stage2_chunks)/len(all_chunks)*100:.1f}%)\n")
    
    # Split each stage into train/val/test
    def split_and_save(chunks, stage_name):
        n = len(chunks)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = {
            'train': chunks[:train_end],
            'val': chunks[train_end:val_end],
            'test': chunks[val_end:]
        }
        
        counts = {}
        for split_name, split_chunks in splits.items():
            mix_dir = stage_dirs[f"{stage_name}_{split_name}_mixture"]
            tgt_dir = stage_dirs[f"{stage_name}_{split_name}_target"]
            
            for idx, (_, mixture, target) in enumerate(split_chunks):
                np.save(mix_dir / f"{idx:06d}.npy", mixture)
                np.save(tgt_dir / f"{idx:06d}.npy", target)
            
            counts[split_name] = len(split_chunks)
        
        return counts
    
    print("💾 Saving organized data...")
    stage1_counts = split_and_save(stage1_chunks, 'stage1')
    stage2_counts = split_and_save(stage2_chunks, 'stage2')
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ PREPROCESSING COMPLETE!")
    print(f"{'='*70}\n")
    print("📁 Data Organization:")
    print(f"\n   STAGE 1 (Weighted Mixture → Vocals):")
    print(f"      Train: {stage1_counts['train']:,} chunks")
    print(f"      Val:   {stage1_counts['val']:,} chunks")
    print(f"      Test:  {stage1_counts['test']:,} chunks")
    print(f"\n   STAGE 2 (Balanced Mixture → Vocals):")
    print(f"      Train: {stage2_counts['train']:,} chunks")
    print(f"      Val:   {stage2_counts['val']:,} chunks")
    print(f"      Test:  {stage2_counts['test']:,} chunks")
    print(f"\n   💾 Saved to: {output_root}\n")
    
    return {
        'stage1': stage1_counts,
        'stage2': stage2_counts,
        'total_chunks': len(all_chunks)
    }


# ==============================================================================
# DATA LOADING UTILITIES
# ==============================================================================
def get_data_loaders(data_dir, stage='stage1', split='train', batch_size=16):
    """
    Create DataLoader for a specific stage and split.
    
    Args:
        data_dir: Root data directory
        stage: 'stage1' or 'stage2'
        split: 'train', 'val', or 'test'
        batch_size: Batch size for DataLoader
        
    Returns:
        DataLoader ready for training/evaluation
    """
    from torch.utils.data import DataLoader
    
    data_root = Path(data_dir) / stage / split
    mix_files = sorted((data_root / 'mixture').glob("*.npy"))
    tgt_files = sorted((data_root / 'target').glob("*.npy"))
    
    if len(mix_files) == 0:
        raise FileNotFoundError(f"No data found in {data_root}")
    
    dataset = StandardDataset(mix_files, tgt_files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
    
    return loader


# ==============================================================================
# LEGACY CACHING LOGIC (Kept for backward compatibility)
# ==============================================================================
def prepare_curriculum_cache(mus, cache_dir="../data", sr=22050, force_rebuild=False):
    """
    DEPRECATED: Use preprocess_musdb18() instead.
    
    Generates and caches curriculum data from musdb tracks.
    Uses realistic mixture weights based on typical music production levels.
    Stage 1: Vocals + Other -> Other (simpler task)
    Stage 2: Full Mix -> Other (harder task with all stems)
    """
    root = Path(cache_dir)
    if not force_rebuild and root.exists() and len(list(root.glob("**/*.npy"))) > 0:
        print(f"Cache found at {root}. Skipping generation.")
        return
    print(f"Generating Curriculum Cache at {root}...")
    for stage in ["stage1", "stage2"]:
        for type in ["mixture", "target"]:
            (root / stage / type).mkdir(parents=True, exist_ok=True)
    
    # Realistic mixture weights (normalized to sum=1.0)
    # Based on typical music production levels
    weights = {
        'vocals': 0.35,   # Vocals usually prominent
        'drums': 0.30,    # Drums have strong presence
        'bass': 0.20,     # Bass provides foundation
        'other': 0.15     # Other instruments fill space
    }
    
    for i, track in enumerate(mus.tracks):
        print(f"Processing: {track.title}...", end="\r")
        stems = {}
        for name, stem_obj in track.targets.items():
            audio = stem_obj.audio.T
            resampled = librosa.resample(audio, orig_sr=track.rate, target_sr=sr)
            stems[name] = np.mean(resampled, axis=0).astype(np.float32)
        
        # Stage 1: Vocals + Other -> Other (simpler curriculum step)
        s1_mix = weights['vocals'] * stems['vocals'] + weights['other'] * stems['other']
        # Normalize to prevent clipping
        s1_mix = s1_mix / (weights['vocals'] + weights['other'])
        np.save(root / "stage1/mixture" / f"{i:03d}.npy", s1_mix)
        np.save(root / "stage1/target" / f"{i:03d}.npy", stems['other'])
        
        # Stage 2: Full Mix -> Other (realistic full mix)
        # Only use stems that exist in the weights dict
        s2_mix = sum(weights[stem] * stems[stem] for stem in stems if stem in weights)
        np.save(root / "stage2/mixture" / f"{i:03d}.npy", s2_mix)
        np.save(root / "stage2/target" / f"{i:03d}.npy", stems['other'])
    print("\nCache generation complete!")

def show_spectrogram(tensor, title="Spectrogram"):
    """
    Plots a tensor spectrogram (C, F, T) as a dB-scaled image.
    """
    if hasattr(tensor, 'cpu'):
        spec = tensor.squeeze().detach().cpu().numpy()
    else:
        spec = tensor
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec_db, sr=22050, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_loss_history(history, title="Training Loss"):
    """
    Safely plots training history without crashing the kernel.
    Works reliably on Colab with GPU.
    """
    # 1. Safety Check: Is there data?
    if not history or 'train_loss' not in history or len(history['train_loss']) == 0:
        print(f"⚠️  No training data found for {title}")
        return

    try:
        # 2. Create Figure explicitly
        plt.figure(figsize=(10, 5), dpi=100)
        
        # 3. Plot Data
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            plt.plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=4, linestyle='--')

        # 4. Styling
        plt.title(title, fontsize=13, fontweight='bold')
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        
        # 5. Render and Close
        plt.tight_layout()
        plt.show()
        plt.close('all')  # Close all figures to free memory
    except Exception as e:
        print(f"❌ Error plotting history: {e}")
        import traceback
        traceback.print_exc()

# ===============================================================================
# INFERENCE: SEPARATE FULL-LENGTH SONGS
# ===============================================================================
def separate_full_song(
    model,
    processor,
    audio_path,
    chunk_duration=8.0,
    overlap=0.5,
    sr=22050,
    device='cpu'
):
    """
    Separate vocals from a full-length song using overlapping chunks.
    
    Handles variable-length audio by:
    1. Chunking into fixed-size segments with overlap
    2. Processing each chunk through the model
    3. Reconstructing using Hann window for smooth overlaps
    4. Trimming back to original length
    
    Args:
        model: Trained separation model
        processor: AudioProcessor instance
        audio_path: Path to audio file (WAV/MP3)
        chunk_duration: Duration of each chunk in seconds (default 8.0)
        overlap: Overlap ratio 0-1 (default 0.5 = 50%)
        sr: Sample rate (default 22050)
        device: 'cpu' or 'cuda'
    
    Returns:
        numpy array: Separated vocals (same length as input)
    """
    # 1. Load full song
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    original_length = len(audio)
    
    print(f"\n{'='*70}")
    print(f"SEPARATING: {Path(audio_path).name}")
    print(f"{'='*70}")
    print(f"Duration: {original_length / sr:.1f}s | Sample Rate: {sr} Hz")
    
    # 2. Define chunk parameters
    chunk_samples = int(chunk_duration * sr)  # e.g., 8s * 22050 = 176,400
    hop_samples = int(chunk_samples * (1 - overlap))  # e.g., 50% overlap = 88,200
    
    # 3. Pad audio to fit chunks evenly
    num_chunks = int(np.ceil((original_length - chunk_samples) / hop_samples)) + 1
    padded_length = (num_chunks - 1) * hop_samples + chunk_samples
    audio_padded = np.pad(audio, (0, padded_length - original_length), mode='constant')
    
    print(f"Chunk Size: {chunk_duration}s ({chunk_samples} samples)")
    print(f"Hop Size: {chunk_duration * (1-overlap):.1f}s ({hop_samples} samples)")
    print(f"Num Chunks: {num_chunks}")
    print()
    
    # 4. Create Hann window for overlap-add reconstruction
    window = np.hanning(chunk_samples + 1)[:-1]
    
    # 5. Process each chunk
    model.eval()
    reconstructed = np.zeros(padded_length)
    window_sum = np.zeros(padded_length)  # For normalization
    
    print(f"Processing chunks...")
    with torch.no_grad():
        for i in range(num_chunks):
            if (i + 1) % max(1, num_chunks // 5) == 0 or i == 0:
                print(f"  [{i+1}/{num_chunks}] chunks processed")
            
            start = i * hop_samples
            end = start + chunk_samples
            
            # Extract chunk
            chunk = audio_padded[start:end]
            
            # Convert to spectrogram
            chunk_mag, chunk_phase = processor.to_spectrogram(torch.tensor(chunk))
            
            # Prepare input for model
            if chunk_mag.dim() == 2:
                chunk_mag_in = chunk_mag.unsqueeze(0).unsqueeze(0).to(device)
            elif chunk_mag.dim() == 3:
                chunk_mag_in = chunk_mag.unsqueeze(1).to(device)
            else:
                chunk_mag_in = chunk_mag.unsqueeze(0).unsqueeze(0).to(device)
            
            # Run through model
            with torch.no_grad():
                mask = model(chunk_mag_in)
                
                # Handle shape mismatch
                if mask.shape != chunk_mag_in.shape:
                    mask = mask[:, :, :chunk_mag_in.shape[2], :chunk_mag_in.shape[3]]
                
                # Apply mask to mixture
                est_mag = mask.squeeze(0).squeeze(0) * chunk_mag.to(device)
                
                # Convert back to waveform
                est_wav = processor.to_waveform(est_mag.cpu(), chunk_phase.cpu())
            
            # Overlap-add with windowing
            weighted_chunk = est_wav * window
            reconstructed[start:end] += weighted_chunk
            window_sum[start:end] += window
    
    # 6. Normalize by window overlap
    reconstructed = np.divide(reconstructed, window_sum, where=window_sum > 0, out=reconstructed)
    
    # 7. Trim back to original length
    output = reconstructed[:original_length]
    
    print(f"\n✅ Separation complete!")
    print(f"Output shape: {output.shape} | Duration: {len(output)/sr:.1f}s")
    print(f"{'='*70}\n")
    
    return output


def save_separated_audio(audio, output_path, sr=22050):
    """
    Save separated audio to disk.
    
    Args:
        audio: numpy array of audio samples
        output_path: Path to save WAV file
        sr: Sample rate (default 22050)
    """
    import soundfile as sf
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize to prevent clipping
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    sf.write(str(output_path), audio, sr)
    print(f"✅ Saved: {output_path}")


# ==============================================================================
# VISUALIZATION AND EVALUATION UTILITIES
# ==============================================================================
def plot_spectrograms_and_play_audio(mixture, prediction, ground_truth, sr=22050, 
                                     title="Evaluation", show_audio=True):
    """
    Plot spectrograms and provide audio playback for mixture, prediction, and ground truth.
    
    Args:
        mixture: Input mixture audio (1-D array)
        prediction: Model prediction (1-D array)
        ground_truth: Ground truth target (1-D array)
        sr: Sample rate
        title: Title for the plots
        show_audio: Whether to display audio playback widgets
    """
    import matplotlib.pyplot as plt
    from IPython.display import display, HTML, Audio
    import librosa
    
    # Compute spectrograms
    S_mix = librosa.stft(mixture)
    S_pred = librosa.stft(prediction)
    S_truth = librosa.stft(ground_truth)
    
    mag_mix = np.abs(S_mix)
    mag_pred = np.abs(S_pred)
    mag_truth = np.abs(S_truth)
    
    # Convert to dB scale
    S_db_mix = librosa.power_to_db(mag_mix**2, ref=np.max(mag_mix**2))
    S_db_pred = librosa.power_to_db(mag_pred**2, ref=np.max(mag_pred**2))
    S_db_truth = librosa.power_to_db(mag_truth**2, ref=np.max(mag_truth**2))
    
    # Plot spectrograms
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    im1 = axes[0].imshow(S_db_mix, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Input Mixture", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Frequency Bin")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(S_db_pred, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title("Model Prediction (Separated Vocals)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Frequency Bin")
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(S_db_truth, aspect='auto', origin='lower', cmap='magma')
    axes[2].set_title("Ground Truth (Target Vocals)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Frequency Bin")
    axes[2].set_xlabel("Time Frame")
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    # Audio playback
    if show_audio:
        print(f"\n{'='*70}")
        print("🔊 AUDIO PLAYBACK")
        print(f"{'='*70}\n")
        
        # Normalize for playback
        mix_norm = mixture / (np.max(np.abs(mixture)) + 1e-8)
        pred_norm = prediction / (np.max(np.abs(prediction)) + 1e-8)
        truth_norm = ground_truth / (np.max(np.abs(ground_truth)) + 1e-8)
        
        print("🎵 Input Mixture:")
        display(Audio(mix_norm, rate=sr))
        
        print("\n🎵 Model Prediction (Separated Vocals):")
        display(Audio(pred_norm, rate=sr))
        
        print("\n🎵 Ground Truth (Target Vocals):")
        display(Audio(truth_norm, rate=sr))


def evaluate_with_song_selector(model, processor, data_dir, sr=22050, device='cpu'):
    """
    Create interactive song selector for full-length song evaluation.
    Shows spectrograms and audio for selected test songs.
    
    Args:
        model: Trained model
        processor: AudioProcessor instance
        data_dir: Path to data root directory
        sr: Sample rate
        device: CPU or GPU
    """
    from IPython.display import display, HTML, Audio
    import ipywidgets as widgets
    from pathlib import Path
    
    data_dir = Path(data_dir)
    
    # Find all test songs
    test_dir = data_dir / "stage2" / "test" / "mixture"
    if not test_dir.exists():
        print(f"⚠️  Test data not found at {test_dir}")
        return
    
    test_files = sorted(test_dir.glob("*.npy"))
    num_songs = len(test_files)
    
    print(f"\n{'='*70}")
    print(f"INTERACTIVE SONG EVALUATION - {num_songs} test songs available")
    print(f"{'='*70}\n")
    
    if num_songs == 0:
        print("No test songs found")
        return
    
    # Create dropdown widget
    song_dropdown = widgets.Dropdown(
        options=[(f"Song {i:04d}", i) for i in range(num_songs)],
        description='Select Song:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='300px')
    )
    
    output_widget = widgets.Output()
    
    def on_song_selected(change):
        output_widget.clear_output(wait=True)
        with output_widget:
            song_idx = change['new']
            
            # Load mixture and ground truth
            mix_file = data_dir / "stage2" / "test" / "mixture" / f"{song_idx:06d}.npy"
            tgt_file = data_dir / "stage2" / "test" / "target" / f"{song_idx:06d}.npy"
            
            if mix_file.exists() and tgt_file.exists():
                mixture = np.load(mix_file)
                ground_truth = np.load(tgt_file)
                
                # Process with model
                mix_tensor = torch.FloatTensor(mixture).unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    model.eval()
                    pred_tensor = model(mix_tensor)
                
                prediction = pred_tensor.squeeze().cpu().numpy()
                
                # Plot and play
                title = f"Test Song {song_idx:04d} Evaluation"
                plot_spectrograms_and_play_audio(
                    mixture, prediction, ground_truth,
                    sr=sr, title=title, show_audio=True
                )
    
    song_dropdown.observe(on_song_selected, names='value')
    
    # Display initial song
    display(song_dropdown)
    display(output_widget)
    
    # Auto-display first song
    on_song_selected({'new': 0})