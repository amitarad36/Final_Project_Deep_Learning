import numpy as np
import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import librosa

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
            print("[DEBUG] Using tqdm.notebook for progress bars.")
            from tqdm.notebook import tqdm as tqdm_bar
        else:
            print("[DEBUG] Using plain tqdm for progress bars.")
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
            print("[DEBUG] Using tqdm.notebook for progress bars.")
            from tqdm.notebook import tqdm as tqdm_bar
        else:
            print("[DEBUG] Using plain tqdm for progress bars.")
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
# CACHING LOGIC
# ==============================================================================
def prepare_curriculum_cache(mus, cache_dir="../data/curriculum", sr=22050):
    """
    Generates and caches curriculum data from musdb tracks.
    """
    root = Path(cache_dir)
    if root.exists() and len(list(root.glob("**/*.npy"))) > 0:
        print(f"Cache found at {root}. Skipping generation.")
        return
    print(f"Generating Curriculum Cache at {root}...")
    for stage in ["stage1", "stage2"]:
        for type in ["mixture", "target"]:
            (root / stage / type).mkdir(parents=True, exist_ok=True)
    for i, track in enumerate(mus.tracks):
        print(f"Processing: {track.title}...", end="\r")
        stems = {}
        for name, stem_obj in track.targets.items():
            audio = stem_obj.audio.T
            resampled = librosa.resample(audio, orig_sr=track.rate, target_sr=sr)
            stems[name] = np.mean(resampled, axis=0).astype(np.float32)
        # Stage 1: Vocals + Other -> Other
        np.save(root / "stage1/mixture" / f"{i:03d}.npy", 0.7*stems['vocals'] + 0.3*stems['other']) # Weighted mix
        np.save(root / "stage1/target" / f"{i:03d}.npy", stems['other'])
        # Stage 2: Full Mix -> Other
        s2_mix = sum(stems[stem] for stem in stems)
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
    """
    # 1. Safety Check: Is there data?
    if not history or 'train_loss' not in history or len(history['train_loss']) == 0:
        print(f"No training data found for {title}")
        return

    # 2. Create Figure explicitly (avoids interference with previous plots)
    plt.figure(figsize=(10, 5))
    
    # 3. Plot Data
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')

    # 4. Styling
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Render and Close
    plt.tight_layout()
    plt.show()  # Renders the image in the notebook
    plt.close() # Frees memory immediately