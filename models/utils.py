import numpy as np
import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import librosa

# Configure CUDA memory to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
            else:
                self.optimizer.zero_grad()
                output = self.model(mix)
                loss = self.loss_fn(output, tgt)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
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
            
            # Clear CUDA cache periodically to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            global_pbar.set_postfix({'Train': f"{train_loss:.4f}", 'Val': f"{val_loss:.4f}"})
            # Only print every 5 epochs to reduce I/O overhead in Colab
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}: Train {train_loss:.5f} | Val {val_loss:.5f}")
            # Live logging to file (only every 5 epochs)
            if log_file_path and ((epoch + 1) % 5 == 0 or epoch == 0):
                try:
                    with open(log_file_path, 'a') as f:
                        f.write(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")
                        f.flush()
                except Exception as e:
                    print(f"[WARN] Could not write to log file {log_file_path}: {e}")
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
        # Always save a final checkpoint (even if best-save never triggered)
        if save_path is not None:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'history': self.history
                }, save_path)
            except Exception as e:
                print(f"[WARN] Could not save final checkpoint to {save_path}: {e}")
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
        'batch_size': 4,  # Reduced from 16 to fit in VRAM (was causing OOM with batch_size=16)
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


    def compare_models_on_audio_file(
        file_path,
        model_lstm,
        model_unet,
        processor_lstm,
        processor_unet,
        device="cpu",
        sr=22050,
        duration=None,
    ):
        """
        Loads an audio file and runs inference with both models.
        Plots spectrograms and plays audio for mixture and predictions.
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Load audio (mono)
        mix_wav, _ = librosa.load(file_path, sr=sr, mono=True)
        if duration is not None:
            mix_wav = mix_wav[: int(sr * duration)]

        # LSTM inference
        mix_mag_lstm, mix_phase_lstm = processor_lstm.to_spectrogram(torch.tensor(mix_wav))
        mix_mag_in = mix_mag_lstm.unsqueeze(0).unsqueeze(0).to(device)
        model_lstm.eval()
        with torch.no_grad():
            mask_lstm = model_lstm(mix_mag_in)
            if mask_lstm.shape != mix_mag_in.shape:
                mask_lstm = mask_lstm[:, :, :mix_mag_in.shape[2], :mix_mag_in.shape[3]]
            est_mag_lstm = mask_lstm.squeeze(0).squeeze(0) * mix_mag_lstm.to(device)
            est_wav_lstm = processor_lstm.to_waveform(est_mag_lstm.cpu(), mix_phase_lstm.cpu())

        # U-Net inference
        mix_mag_unet, mix_phase_unet = processor_unet.to_spectrogram(torch.tensor(mix_wav))
        mix_mag_in = mix_mag_unet.unsqueeze(0).unsqueeze(0).to(device)
        model_unet.eval()
        with torch.no_grad():
            mask_unet = model_unet(mix_mag_in)
            if mask_unet.shape != mix_mag_in.shape:
                mask_unet = mask_unet[:, :, :mix_mag_in.shape[2], :mix_mag_in.shape[3]]
            est_mag_unet = mask_unet.squeeze(0).squeeze(0) * mix_mag_unet.to(device)
            est_wav_unet = processor_unet.to_waveform(est_mag_unet.cpu(), mix_phase_unet.cpu())

        # Spectrograms
        print("\n=== Spectrograms ===")
        show_spectrogram(mix_mag_lstm, title="Mixture Spectrogram")
        show_spectrogram(est_mag_lstm.cpu(), title="LSTM Predicted Spectrogram")
        show_spectrogram(est_mag_unet.cpu(), title="U-Net Predicted Spectrogram")

        # Audio playback
        print("\n=== Audio Playback ===")
        play_audio(mix_wav, sr=sr, title="Input Mixture")
        play_audio(est_wav_lstm, sr=sr, title="LSTM Prediction")
        play_audio(est_wav_unet, sr=sr, title="U-Net Prediction")

        return {
            "mix_wav": mix_wav,
            "est_wav_lstm": est_wav_lstm,
            "est_wav_unet": est_wav_unet,
            "mix_mag": mix_mag_lstm,
            "est_mag_lstm": est_mag_lstm,
            "est_mag_unet": est_mag_unet,
        }
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


# ===============================================================================
# MODEL A COMPARISON UTILITIES
# ===============================================================================

def initialize_model_a_lstm(device='cuda'):
    """Initialize Model A (LSTM) with default configuration."""
    from . import model_A as ma
    import torch.optim as optim
    import torch.nn as nn
    
    processor = AudioProcessor(device=device)
    model = ma.CompactLSTMMasking(
        freq_bins=1025,
        hidden_size=256,
        num_layers=1,
        dropout=0.2
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=get_training_config()['learning_rate'])
    loss_fn = nn.MSELoss()
    
    return model, processor, optimizer, loss_fn


def initialize_model_a_unet(device='cuda'):
    """Initialize Model A (U-Net) with default configuration."""
    from . import model_A as ma
    import torch.optim as optim
    import torch.nn as nn
    
    processor = AudioProcessor(device=device)
    model = ma.TimeFrequencyDomainUNet(
        in_channels=1,
        out_channels=1,
        base_filters=64,
        num_layers=4,
        batchnorm=True,
        dropout=0.1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=get_training_config()['learning_rate'])
    loss_fn = nn.MSELoss()
    
    return model, processor, optimizer, loss_fn


def train_model_stage(
    model,
    processor,
    optimizer,
    loss_fn,
    training_data_dir,
    stage,
    ckpt_path,
    device,
    train_config,
    skip_training=False,
):
    """
    Train or load a single stage for a given model.
    """
    ckpt_path = Path(ckpt_path)
    hist = {}

    if ckpt_path.exists():
        print(f"✅ Found checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        hist = checkpoint.get('history', {})
        if hist:
            print(f"   Best val loss: {min(hist.get('val_loss', [float('inf')])):.6f}")
        return hist

    if skip_training:
        print("⏭️  Training skipped")
        return hist

    batch_size = train_config.get('batch_size', 8)
    num_epochs = train_config.get('num_epochs', 50)
    patience = train_config.get('patience', 10)

    train_loader = get_data_loaders(training_data_dir, stage=stage, split='train',
                                    batch_size=batch_size)
    val_loader = get_data_loaders(training_data_dir, stage=stage, split='val',
                                  batch_size=batch_size)

    trainer = UniversalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        processor=processor,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        patience=patience,
        input_type='spectrogram'
    )

    hist = trainer.train(num_epochs=num_epochs, save_path=str(ckpt_path))
    print(f"✅ Training complete! Best val loss: {min(hist['val_loss']):.6f}")
    return hist


def train_model_a_comparison(data_dir, checkpoint_dir, chunk_duration=8.0, device='cuda', skip_training=False, train_stage2=True):
    """
    Train both Model A architectures (LSTM and U-Net) sequentially.
    
    Args:
        data_dir: Path to preprocessed data
        checkpoint_dir: Path to save checkpoints
        chunk_duration: Chunk duration in seconds
        device: 'cuda' or 'cpu'
        skip_training: If True, only load existing checkpoints
        
    Returns:
        Dictionary with models, processors, and training histories
    """
    from pathlib import Path
    
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    training_data_dir = data_dir / f"chunks_{chunk_duration:.0f}s"
    train_config = get_training_config()
    
    results = {}
    
    print(f"\n{'='*70}")
    print(f"MODEL A COMPARISON TRAINING")
    print(f"{'='*70}\n")
    
    # ===== MODEL A (LSTM) - STAGE 1 =====
    print("1️⃣ Model A (LSTM) - Stage 1")
    print("-" * 70)
    
    model_lstm, processor_lstm, optimizer_lstm, loss_fn_lstm = initialize_model_a_lstm(device)
    lstm_params = sum(p.numel() for p in model_lstm.parameters())
    print(f"   Parameters: {lstm_params:,}")
    
    ckpt_lstm_s1 = checkpoint_dir / f"model_a_lstm_stage1_{chunk_duration:.0f}s.pth"
    
    if ckpt_lstm_s1.exists():
        print(f"✅ Found checkpoint: {ckpt_lstm_s1.name}")
        checkpoint = torch.load(ckpt_lstm_s1, map_location=device)
        model_lstm.load_state_dict(checkpoint['model_state_dict'])
        hist_lstm_s1 = checkpoint.get('history', {})
        if hist_lstm_s1:
            print(f"   Best val loss: {min(hist_lstm_s1.get('val_loss', [float('inf')])):.6f}")
    elif not skip_training:
        print("🚀 Starting training...")
        batch_size = train_config.get('batch_size', 8)
        num_epochs = train_config.get('num_epochs', 50)
        patience = train_config.get('patience', 10)

        train_loader = get_data_loaders(training_data_dir, stage='stage1', split='train', 
                           batch_size=batch_size)
        val_loader = get_data_loaders(training_data_dir, stage='stage1', split='val', 
                         batch_size=batch_size)
        
        trainer = UniversalTrainer(
            model=model_lstm, train_loader=train_loader, val_loader=val_loader,
            processor=processor_lstm, optimizer=optimizer_lstm, loss_fn=loss_fn_lstm,
            device=device, patience=patience, input_type='spectrogram'
        )
        hist_lstm_s1 = trainer.train(num_epochs=num_epochs, 
                                     save_path=str(ckpt_lstm_s1))
        print(f"✅ Training complete! Best val loss: {min(hist_lstm_s1['val_loss']):.6f}")
    else:
        hist_lstm_s1 = {}
    
    # ===== MODEL A (LSTM) - STAGE 2 =====
    ckpt_lstm_s2 = checkpoint_dir / f"model_a_lstm_stage2_{chunk_duration:.0f}s.pth"
    hist_lstm_s2 = {}

    if train_stage2:
        print(f"\n1️⃣ Model A (LSTM) - Stage 2")
        print("-" * 70)
        if ckpt_lstm_s2.exists():
            print(f"✅ Found checkpoint: {ckpt_lstm_s2.name}")
            checkpoint = torch.load(ckpt_lstm_s2, map_location=device)
            model_lstm.load_state_dict(checkpoint['model_state_dict'])
            hist_lstm_s2 = checkpoint.get('history', {})
            if hist_lstm_s2:
                print(f"   Best val loss: {min(hist_lstm_s2.get('val_loss', [float('inf')])):.6f}")
        elif not skip_training:
            print("🚀 Starting training...")
            batch_size = train_config.get('batch_size', 8)
            num_epochs = train_config.get('num_epochs', 50)
            patience = train_config.get('patience', 10)

            train_loader = get_data_loaders(training_data_dir, stage='stage2', split='train',
                                           batch_size=batch_size)
            val_loader = get_data_loaders(training_data_dir, stage='stage2', split='val',
                                         batch_size=batch_size)

            trainer = UniversalTrainer(
                model=model_lstm, train_loader=train_loader, val_loader=val_loader,
                processor=processor_lstm, optimizer=optimizer_lstm, loss_fn=loss_fn_lstm,
                device=device, patience=patience, input_type='spectrogram'
            )
            hist_lstm_s2 = trainer.train(num_epochs=num_epochs,
                                         save_path=str(ckpt_lstm_s2))
            print(f"✅ Training complete! Best val loss: {min(hist_lstm_s2['val_loss']):.6f}")

    results['lstm'] = {
        'model': model_lstm,
        'processor': processor_lstm,
        'optimizer': optimizer_lstm,
        'loss_fn': loss_fn_lstm,
        'history': {
            'stage1': hist_lstm_s1,
            'stage2': hist_lstm_s2
        },
        'checkpoint': {
            'stage1': ckpt_lstm_s1,
            'stage2': ckpt_lstm_s2
        }
    }
    
    # ===== MODEL A (U-NET) - STAGE 1 =====
    print(f"\n2️⃣ Model A (U-Net) - Stage 1")
    print("-" * 70)
    
    model_unet, processor_unet, optimizer_unet, loss_fn_unet = initialize_model_a_unet(device)
    unet_params = sum(p.numel() for p in model_unet.parameters())
    print(f"   Parameters: {unet_params:,}")
    
    ckpt_unet_s1 = checkpoint_dir / f"model_a_unet_stage1_{chunk_duration:.0f}s.pth"
    
    if ckpt_unet_s1.exists():
        print(f"✅ Found checkpoint: {ckpt_unet_s1.name}")
        checkpoint = torch.load(ckpt_unet_s1, map_location=device)
        model_unet.load_state_dict(checkpoint['model_state_dict'])
        hist_unet_s1 = checkpoint.get('history', {})
        if hist_unet_s1:
            print(f"   Best val loss: {min(hist_unet_s1.get('val_loss', [float('inf')])):.6f}")
    elif not skip_training:
        print("🚀 Starting training...")
        batch_size = train_config.get('batch_size', 8)
        num_epochs = train_config.get('num_epochs', 50)
        patience = train_config.get('patience', 10)

        train_loader = get_data_loaders(training_data_dir, stage='stage1', split='train', 
                           batch_size=batch_size)
        val_loader = get_data_loaders(training_data_dir, stage='stage1', split='val', 
                         batch_size=batch_size)
        
        trainer = UniversalTrainer(
            model=model_unet, train_loader=train_loader, val_loader=val_loader,
            processor=processor_unet, optimizer=optimizer_unet, loss_fn=loss_fn_unet,
            device=device, patience=patience, input_type='spectrogram'
        )
        hist_unet_s1 = trainer.train(num_epochs=num_epochs, 
                                     save_path=str(ckpt_unet_s1))
        print(f"✅ Training complete! Best val loss: {min(hist_unet_s1['val_loss']):.6f}")
    else:
        hist_unet_s1 = {}
    
    # ===== MODEL A (U-NET) - STAGE 2 =====
    ckpt_unet_s2 = checkpoint_dir / f"model_a_unet_stage2_{chunk_duration:.0f}s.pth"
    hist_unet_s2 = {}

    if train_stage2:
        print(f"\n2️⃣ Model A (U-Net) - Stage 2")
        print("-" * 70)
        if ckpt_unet_s2.exists():
            print(f"✅ Found checkpoint: {ckpt_unet_s2.name}")
            checkpoint = torch.load(ckpt_unet_s2, map_location=device)
            model_unet.load_state_dict(checkpoint['model_state_dict'])
            hist_unet_s2 = checkpoint.get('history', {})
            if hist_unet_s2:
                print(f"   Best val loss: {min(hist_unet_s2.get('val_loss', [float('inf')])):.6f}")
        elif not skip_training:
            print("🚀 Starting training...")
            batch_size = train_config.get('batch_size', 8)
            num_epochs = train_config.get('num_epochs', 50)
            patience = train_config.get('patience', 10)

            train_loader = get_data_loaders(training_data_dir, stage='stage2', split='train',
                                           batch_size=batch_size)
            val_loader = get_data_loaders(training_data_dir, stage='stage2', split='val',
                                         batch_size=batch_size)

            trainer = UniversalTrainer(
                model=model_unet, train_loader=train_loader, val_loader=val_loader,
                processor=processor_unet, optimizer=optimizer_unet, loss_fn=loss_fn_unet,
                device=device, patience=patience, input_type='spectrogram'
            )
            hist_unet_s2 = trainer.train(num_epochs=num_epochs,
                                         save_path=str(ckpt_unet_s2))
            print(f"✅ Training complete! Best val loss: {min(hist_unet_s2['val_loss']):.6f}")

    results['unet'] = {
        'model': model_unet,
        'processor': processor_unet,
        'optimizer': optimizer_unet,
        'loss_fn': loss_fn_unet,
        'history': {
            'stage1': hist_unet_s1,
            'stage2': hist_unet_s2
        },
        'checkpoint': {
            'stage1': ckpt_unet_s1,
            'stage2': ckpt_unet_s2
        }
    }
    
    return results


def load_training_history_from_checkpoint(ckpt_path):
    """Load training history from checkpoint or epoch files."""
    import re
    
    ckpt_path = Path(ckpt_path)
    
    # Try epoch folder first
    epoch_folder = ckpt_path.parent / f"{ckpt_path.stem}_epochs"
    if epoch_folder.exists():
        epoch_files = sorted(epoch_folder.glob("epoch_*.txt"))
        train_losses = []
        val_losses = []
        
        for epoch_file in epoch_files:
            with open(epoch_file, 'r') as f:
                content = f.read()
                train_match = re.search(r'Train Loss[:\s=]+([\d.]+)', content)
                val_match = re.search(r'Val Loss[:\s=]+([\d.]+)', content)
                if train_match and val_match:
                    train_losses.append(float(train_match.group(1)))
                    val_losses.append(float(val_match.group(1)))
        
        if train_losses:
            return {'train_loss': train_losses, 'val_loss': val_losses}
    
    # Fallback to checkpoint
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        return ckpt.get('history', {})
    
    return {}


def plot_model_comparison(hist_lstm, hist_unet, title="Model A Comparison: LSTM vs U-Net"):
    """Plot side-by-side training curves for LSTM and U-Net."""
    if not hist_lstm or not hist_unet:
        print("⚠️  Training histories not available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Model A (LSTM)
    epochs_lstm = range(1, len(hist_lstm['train_loss']) + 1)
    ax1.plot(epochs_lstm, hist_lstm['train_loss'], 'o-', label='Train', linewidth=2, markersize=5)
    ax1.plot(epochs_lstm, hist_lstm['val_loss'], 's--', label='Val', linewidth=2, markersize=5)
    ax1.set_title('Model A (LSTM)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Model A (U-Net)
    epochs_unet = range(1, len(hist_unet['train_loss']) + 1)
    ax2.plot(epochs_unet, hist_unet['train_loss'], 'o-', label='Train', linewidth=2, markersize=5)
    ax2.plot(epochs_unet, hist_unet['val_loss'], 's--', label='Val', linewidth=2, markersize=5)
    ax2.set_title('Model A (U-Net)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    print(f"\nModel A (LSTM):")
    print(f"   Epochs: {len(hist_lstm['train_loss'])}")
    print(f"   Best Val Loss: {min(hist_lstm['val_loss']):.6f} (epoch {hist_lstm['val_loss'].index(min(hist_lstm['val_loss']))+1})")
    print(f"   Final Train Loss: {hist_lstm['train_loss'][-1]:.6f}")
    
    print(f"\nModel A (U-Net):")
    print(f"   Epochs: {len(hist_unet['train_loss'])}")
    print(f"   Best Val Loss: {min(hist_unet['val_loss']):.6f} (epoch {hist_unet['val_loss'].index(min(hist_unet['val_loss']))+1})")
    print(f"   Final Train Loss: {hist_unet['train_loss'][-1]:.6f}")
    
    # Winner
    winner = "LSTM" if min(hist_lstm['val_loss']) < min(hist_unet['val_loss']) else "U-Net"
    print(f"\n🏆 Best Performance: {winner}")


def evaluate_separation_quality(model_lstm, model_unet, processor_lstm, processor_unet, 
                                test_data_dir, stage='stage1', num_samples=10, sr=22050, device='cuda'):
    """
    Evaluate both models on test set using BSS metrics (SDR/SIR/SAR).
    
    Returns:
        Dictionary with metrics for both models
    """
    try:
        import museval
    except ImportError:
        print("Installing museval...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'museval'])
        import museval
    
    test_mix_dir = Path(test_data_dir) / stage / "test" / "mixture"
    test_tgt_dir = Path(test_data_dir) / stage / "test" / "target"
    
    if not test_mix_dir.exists():
        print(f"⚠️  Test data not found at {test_mix_dir}")
        return None
    
    mix_files = sorted(test_mix_dir.glob("*.npy"))[:num_samples]
    tgt_files = sorted(test_tgt_dir.glob("*.npy"))[:num_samples]
    
    print("="*70)
    print(f"QUANTITATIVE EVALUATION - SDR/SIR/SAR METRICS ({stage})")
    print("="*70)
    print(f"\nEvaluating on {len(mix_files)} test samples...\n")
    
    lstm_metrics = {'SDR': [], 'SIR': [], 'SAR': []}
    unet_metrics = {'SDR': [], 'SIR': [], 'SAR': []}
    
    model_lstm.eval()
    model_unet.eval()
    
    for idx, (mix_file, tgt_file) in enumerate(zip(mix_files, tgt_files)):
        # Load audio
        mix_wav = np.load(mix_file)
        tgt_wav = np.load(tgt_file)
        
        min_len = min(len(mix_wav), len(tgt_wav))
        mix_wav = mix_wav[:min_len]
        tgt_wav = tgt_wav[:min_len]
        
        # Convert to spectrograms
        mix_mag, mix_phase = processor_lstm.to_spectrogram(torch.tensor(mix_wav))
        mix_mag_in = mix_mag.unsqueeze(0).unsqueeze(0).to(device)
        
        # LSTM prediction
        with torch.no_grad():
            mask_lstm = model_lstm(mix_mag_in)
            est_mag_lstm = mask_lstm.squeeze(0).squeeze(0) * mix_mag.to(device)
            est_wav_lstm = processor_lstm.to_waveform(est_mag_lstm.cpu(), mix_phase.cpu()).numpy()
        
        # U-Net prediction
        with torch.no_grad():
            mask_unet = model_unet(mix_mag_in)
            est_mag_unet = mask_unet.squeeze(0).squeeze(0) * mix_mag.to(device)
            est_wav_unet = processor_unet.to_waveform(est_mag_unet.cpu(), mix_phase.cpu()).numpy()
        
        # Ensure same length for evaluation
        min_eval_len = min(len(tgt_wav), len(est_wav_lstm), len(est_wav_unet))
        tgt_wav = tgt_wav[:min_eval_len]
        est_wav_lstm = est_wav_lstm[:min_eval_len]
        est_wav_unet = est_wav_unet[:min_eval_len]
        
        # Compute metrics using museval (BSS eval)
        sdr_lstm, sir_lstm, sar_lstm, _ = museval.evaluate(
            tgt_wav.reshape(1, -1), est_wav_lstm.reshape(1, -1),
            win=sr, hop=sr
        )
        lstm_metrics['SDR'].append(np.nanmedian(sdr_lstm))
        lstm_metrics['SIR'].append(np.nanmedian(sir_lstm))
        lstm_metrics['SAR'].append(np.nanmedian(sar_lstm))
        
        sdr_unet, sir_unet, sar_unet, _ = museval.evaluate(
            tgt_wav.reshape(1, -1), est_wav_unet.reshape(1, -1),
            win=sr, hop=sr
        )
        unet_metrics['SDR'].append(np.nanmedian(sdr_unet))
        unet_metrics['SIR'].append(np.nanmedian(sir_unet))
        unet_metrics['SAR'].append(np.nanmedian(sar_unet))
        
        print(f"  Sample {idx+1}/{len(mix_files)} - LSTM: SDR={lstm_metrics['SDR'][-1]:.2f} | U-Net: SDR={unet_metrics['SDR'][-1]:.2f}")
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nModel A (LSTM):")
    print(f"   SDR: {np.mean(lstm_metrics['SDR']):.2f} ± {np.std(lstm_metrics['SDR']):.2f} dB")
    print(f"   SIR: {np.mean(lstm_metrics['SIR']):.2f} ± {np.std(lstm_metrics['SIR']):.2f} dB")
    print(f"   SAR: {np.mean(lstm_metrics['SAR']):.2f} ± {np.std(lstm_metrics['SAR']):.2f} dB")
    
    print(f"\nModel A (U-Net):")
    print(f"   SDR: {np.mean(unet_metrics['SDR']):.2f} ± {np.std(unet_metrics['SDR']):.2f} dB")
    print(f"   SIR: {np.mean(unet_metrics['SIR']):.2f} ± {np.std(unet_metrics['SIR']):.2f} dB")
    print(f"   SAR: {np.mean(unet_metrics['SAR']):.2f} ± {np.std(unet_metrics['SAR']):.2f} dB")
    
    print(f"\n📊 Reference (Paper): SI-SDRi ~6.93 dB for vocals")
    
    winner = "LSTM" if np.mean(lstm_metrics['SDR']) > np.mean(unet_metrics['SDR']) else "U-Net"
    improvement = abs(np.mean(lstm_metrics['SDR']) - np.mean(unet_metrics['SDR']))
    print(f"\n🏆 Best SDR: {winner} (+{improvement:.2f} dB)")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_list = ['SDR', 'SIR', 'SAR']
    colors = ['#1f77b4', '#ff7f0e']
    
    for idx, metric in enumerate(metrics_list):
        data = [lstm_metrics[metric], unet_metrics[metric]]
        bp = axes[idx].boxplot(data, positions=[1, 2], widths=0.6,
                               patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[idx].set_title(f'{metric} (dB)', fontsize=14, fontweight='bold')
        axes[idx].set_xticks([1, 2])
        axes[idx].set_xticklabels(['LSTM', 'U-Net'], fontsize=11)
        axes[idx].set_ylabel('dB', fontsize=11)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        for pos, vals, color in zip([1, 2], data, colors):
            mean_val = np.mean(vals)
            axes[idx].text(pos, mean_val, f'{mean_val:.2f}', 
                          ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.suptitle('Model A Evaluation: LSTM vs U-Net - BSS Metrics', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    return {'lstm': lstm_metrics, 'unet': unet_metrics}