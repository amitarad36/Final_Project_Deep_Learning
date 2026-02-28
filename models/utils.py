import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import librosa
import torch.nn as nn

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

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

    def train_epoch(self, epoch_idx, num_epochs):
        """
        Trains for one epoch and returns average loss.
        Implements: mask is applied in linear domain, loss is computed in log domain (Option 3).
        """
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        def _in_notebook() -> bool:
            try:
                from IPython import get_ipython
                shell: str = get_ipython().__class__.__name__
                return shell == 'ZMQInteractiveShell'
            except:
                return False
        
        if _in_notebook():
            from tqdm.notebook import tqdm as tqdm_bar
        else:
            from tqdm import tqdm as tqdm_bar
        
        pbar = tqdm_bar(self.train_loader, desc=f"Epoch {epoch_idx}/{num_epochs}", leave=True)
        
        for batch in pbar:
            mix = batch['mix']
            tgt = batch['tgt']
            
            if isinstance(mix, torch.Tensor):
                mix: torch.Tensor = mix.to(self.device)
            if isinstance(tgt, torch.Tensor):
                tgt: torch.Tensor = tgt.to(self.device)
            
            if self.input_type == 'spectrogram':
                if isinstance(mix, tuple):
                    mix_mag = mix[0].to(self.device)
                    tgt_mag = tgt[0].to(self.device)
                    
                    if mix_mag.dim() == 3:
                        mix_mag = mix_mag.unsqueeze(1)
                    elif mix_mag.dim() == 2:
                        mix_mag = mix_mag.unsqueeze(0).unsqueeze(0)
                    
                    if tgt_mag.dim() == 3:
                        tgt_mag = tgt_mag.unsqueeze(1)
                    elif tgt_mag.dim() == 2:
                        tgt_mag = tgt_mag.unsqueeze(0).unsqueeze(0)
                    
                    mix_log = mix_mag
                    tgt_log = tgt_mag
                else:
                    mix_log, _ = self.processor.to_spectrogram(mix)
                    tgt_log, _ = self.processor.to_spectrogram(tgt)
                    
                    if mix_log.dim() == 3:
                        mix_log = mix_log.unsqueeze(1)
                    if tgt_log.dim() == 3:
                        tgt_log = tgt_log.unsqueeze(1)
                
                self.optimizer.zero_grad()
                mask = self.model(mix_log)
                if mask.shape != mix_log.shape:
                    mask = mask[:, :, :mix_log.shape[2], :mix_log.shape[3]]
                est_linear = mask * torch.expm1(mix_log)
                est_log: torch.Tensor = torch.log1p(est_linear)
                loss = self.loss_fn(est_log, tgt_log)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                if batch_count % 10 == 0:
                    avg_loss = total_loss / batch_count
                    pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
            else:
                self.optimizer.zero_grad()
                output = self.model(mix)
                loss = self.loss_fn(output, tgt)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                if batch_count % 10 == 0:
                    avg_loss = total_loss / batch_count
                    pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
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
                mix = batch['mix'].to(self.device) if not isinstance(batch['mix'], tuple) else batch['mix']
                tgt = batch['tgt'].to(self.device) if not isinstance(batch['tgt'], tuple) else batch['tgt']
                
                if self.input_type == 'spectrogram':
                    if isinstance(mix, tuple):
                        mix_mag = mix[0].to(self.device)
                        tgt_mag = tgt[0].to(self.device)
                        
                        if mix_mag.dim() == 3:
                            mix_mag = mix_mag.unsqueeze(1)
                        elif mix_mag.dim() == 2:
                            mix_mag = mix_mag.unsqueeze(0).unsqueeze(0)
                        
                        if tgt_mag.dim() == 3:
                            tgt_mag = tgt_mag.unsqueeze(1)
                        elif tgt_mag.dim() == 2:
                            tgt_mag = tgt_mag.unsqueeze(0).unsqueeze(0)
                        
                        mix_log = mix_mag
                        tgt_log = tgt_mag
                    else:
                        mix_log, _ = self.processor.to_spectrogram(mix)
                        tgt_log, _ = self.processor.to_spectrogram(tgt)
                        
                        if mix_log.dim() == 3:
                            mix_log = mix_log.unsqueeze(1)
                        if tgt_log.dim() == 3:
                            tgt_log = tgt_log.unsqueeze(1)
                    
                    mask = self.model(mix_log)
                    if mask.shape != mix_log.shape:
                        mask = mask[:, :, :mix_log.shape[2], :mix_log.shape[3]]
                    est_linear = mask * torch.expm1(mix_log)
                    est_log: torch.Tensor = torch.log1p(est_linear)
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
        epochs_no_improve = 0
        epoch_dir = None
        if save_path is not None:
            base_dir = os.path.dirname(save_path)
            run_name = os.path.splitext(os.path.basename(save_path))[0]
            epoch_dir = os.path.join(base_dir, f"{run_name}_epochs")
            os.makedirs(epoch_dir, exist_ok=True)

        best_epoch = 0
        best_train_loss = None
        
        print(f"\n{'='*60}")
        print(f"Training: {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch + 1, num_epochs)
            val_loss = self.validate()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} Complete → Train: {train_loss:.5f} | Val: {val_loss:.5f}")
            
            if log_file_path:
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
        if save_path is not None:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'history': self.history
                }, save_path)
            except Exception as e:
                print(f"[WARN] Could not save final checkpoint to {save_path}: {e}")
        return self.history

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
        
        FIXED: Applies mask in LINEAR domain to match training pipeline.
        """
        self.model.eval()
        with torch.no_grad():
            mix: torch.Tensor = torch.tensor(mixture).to(self.device).float()
            if mix.ndim == 1:
                mix: torch.Tensor = mix.unsqueeze(0)
            if self.input_type == 'spectrogram':
                mix_log, mix_phase = self.processor.to_spectrogram(mix)
                mix_in = mix_log.unsqueeze(1)
                mask = self.model(mix_in)
                if mask.shape != mix_in.shape:
                    mask = mask[:, :, :mix_in.shape[2], :mix_in.shape[3]]
                est_linear = mask.squeeze(1) * torch.expm1(mix_log)
                est = self.processor.to_waveform(torch.log1p(est_linear), mix_phase)
                return est.squeeze().cpu().numpy()
            else:
                est = self.model(mix)
                return est.squeeze().cpu().numpy()

def calculate_metrics(reference, estimate, sr=22050):
    """
    Calculates SDR, SIR, SAR using museval.
    Returns a dict of metrics.
    """
    import museval
    reference = np.atleast_2d(reference)
    estimate = np.atleast_2d(estimate)
    scores = museval.evaluate(reference, estimate, win=1*sr)
    metrics = {
        'SDR': np.nanmean(scores['SDR']),
        'SIR': np.nanmean(scores['SIR']),
        'SAR': np.nanmean(scores['SAR'])
    }
    return metrics

class AudioProcessor:
    """
    Handles conversions between waveform and spectrogram representations.
    """
    def __init__(self, n_fft=2048, hop_length=512, device='cpu'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.window = torch.hann_window(n_fft).to(device)

    def to_spectrogram(self, waveform):
        """
        Converts waveform to log-magnitude spectrogram and phase.
        Returns (log_mag, phase).
        """
        if isinstance(waveform, np.ndarray):
            waveform: torch.Tensor = torch.from_numpy(waveform)
        elif isinstance(waveform, (list, tuple)):
            if len(waveform) == 0:
                waveform: torch.Tensor = torch.empty(0)
            elif isinstance(waveform[0], torch.Tensor):
                waveform: torch.Tensor = torch.stack(waveform)
            elif isinstance(waveform[0], np.ndarray):
                waveform: torch.Tensor = torch.from_numpy(np.stack(waveform))
            else:
                waveform: torch.Tensor = torch.tensor(waveform)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) # Add channel dim
            
        waveform = waveform.to(self.device).float()
        
        complex_spec: torch.Tensor = torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window, 
            return_complex=True
        )
        
        mag: torch.Tensor = torch.abs(complex_spec)
        phase: torch.Tensor = torch.angle(complex_spec)
        log_mag: torch.Tensor = torch.log1p(mag) # Log compression
        
        return log_mag, phase

    def to_waveform(self, log_mag, phase):
        """
        Converts log-magnitude and phase to waveform.
        Returns waveform as numpy array.
        """
        if isinstance(log_mag, np.ndarray): log_mag: torch.Tensor = torch.from_numpy(log_mag)
        if isinstance(phase, np.ndarray): phase: torch.Tensor = torch.from_numpy(phase)
            
        log_mag = log_mag.to(self.device)
        phase = phase.to(self.device)
        
        lin_mag: torch.Tensor = torch.expm1(log_mag)
        complex_spec: torch.Tensor = lin_mag * torch.exp(1j * phase)
        
        waveform = torch.istft(
            complex_spec, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window
        )
        return waveform.cpu().numpy()

class StandardDataset(Dataset):
    """
    Loads pairs from cached files for training/validation.
    Supports both waveforms (.npy) and spectrograms (.npz).
    Returns dicts with keys 'mix' and 'tgt'.
    """
    def __init__(self, mix_files, tgt_files):
        self.mix_files = list(mix_files)
        self.tgt_files = list(tgt_files)
        
        if len(self.mix_files) > 0:
            self.is_spectrogram = str(self.mix_files[0]).endswith('.npz')
        else:
            self.is_spectrogram = False

    def __len__(self):
        """
        Returns number of samples.
        """
        return len(self.mix_files)

    def __getitem__(self, idx):
        """
        Loads mixture and target, returns as tensors in dict.
        For spectrograms: returns (magnitude, phase) tuple
        For waveforms: returns single tensor
        """
        if self.is_spectrogram:
            mix_data = np.load(self.mix_files[idx])
            tgt_data = np.load(self.tgt_files[idx])
            
            return {
                'mix': (
                    torch.tensor(mix_data['magnitude'], dtype=torch.float32),
                    torch.tensor(mix_data['phase'], dtype=torch.float32)
                ),
                'tgt': (
                    torch.tensor(tgt_data['magnitude'], dtype=torch.float32),
                    torch.tensor(tgt_data['phase'], dtype=torch.float32)
                )
            }
        else:
            m = np.load(self.mix_files[idx])
            t = np.load(self.tgt_files[idx])
            return {
                'mix': torch.tensor(m, dtype=torch.float32),
                'tgt': torch.tensor(t, dtype=torch.float32)
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

def get_training_config_lstm():
    """
    Returns training configuration for the full LSTM model.
    """
    config = get_training_config().copy()
    config['batch_size'] = 32
    return config

def get_training_config_unet():
    """
    Returns training configuration for the U-Net model (smaller batch for VRAM).
    """
    config = get_training_config().copy()
    config['batch_size'] = 8
    return config

def play_audio(waveform, sr=22050, title="Audio"):
    """
    Plays audio waveform in notebook.
    """
    if hasattr(waveform, 'cpu'):
        waveform = waveform.squeeze().cpu().numpy()
    print(f"{title}:")
    display(Audio(waveform, rate=sr))

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
    
    s1_mix_path = data_root / "stage1" / split / "mixture"
    s1_tgt_path = data_root / "stage1" / split / "target"
    s2_mix_path = data_root / "stage2" / split / "mixture"
    s2_tgt_path = data_root / "stage2" / split / "target"
    
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
        
        epoch_folder = ckpt_path.parent / f"{ckpt_path.stem}_epochs"
        if epoch_folder.exists():
            print(f"Reading from epoch folder: {epoch_folder.name}")
            epoch_files = sorted(epoch_folder.glob("epoch_*.txt"))
            
            for epoch_file in epoch_files:
                try:
                    with open(epoch_file, 'r') as f:
                        content = f.read()
                        import re
                        train_match = re.search(r'Train Loss[:\s=]+([\d.]+)', content)
                        val_match = re.search(r'Val Loss[:\s=]+([\d.]+)', content)
                        if train_match and val_match:
                            train_losses.append(float(train_match.group(1)))
                            val_losses.append(float(val_match.group(1)))
                except Exception as e:
                    print(f"Error reading {epoch_file.name}: {e}")
                    continue
        
        if not train_losses:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if 'history' in ckpt:
                history = ckpt['history']
                train_losses = history.get('train_loss', [])
                val_losses = history.get('val_loss', [])
        
        if train_losses:
            print(f"Plotting {len(train_losses)} epochs")
            
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
            print("No training data found in checkpoint or epoch files.")
    except Exception as e:
        print(f"Error plotting checkpoint: {e}")
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
    
    mix_path = data_root / stage / split / "mixture"
    tgt_path = data_root / stage / split / "target"
    
    if not mix_path.exists():
        mix_path = data_root / stage / "mixture"
        tgt_path = data_root / stage / "target"

    mix_files = sorted(list(mix_path.glob("*.npy")))
    tgt_files = sorted(list(tgt_path.glob("*.npy")))
    
    if len(mix_files) == 0:
        raise FileNotFoundError(f"No data found in {mix_path}")

    n_samples = sr * duration
    mix_wav = np.load(mix_files[song_num])[:n_samples]
    tgt_instrumental_wav = np.load(tgt_files[song_num])[:n_samples]  # Ground truth instruments (target)

    mix_mag, mix_phase = processor.to_spectrogram(torch.tensor(mix_wav))
    tgt_instrumental_mag, _ = processor.to_spectrogram(torch.tensor(tgt_instrumental_wav))

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
        
        pred_instrumental_mag = mask.squeeze(0).squeeze(0) * mix_mag.to(device)
        pred_instrumental_wav = processor.to_waveform(pred_instrumental_mag.cpu(), mix_phase.cpu())

    show_spectrogram(mix_mag, title="1. Mixture Spectrogram (Vocals + Instruments)")
    show_spectrogram(tgt_instrumental_mag, title="2. Ground Truth Instruments (Target)")
    show_spectrogram(pred_instrumental_mag.cpu(), title="3. Predicted Instruments (Model Output - Karaoke)")

    if play_audio_output:
        play_audio(mix_wav, sr=sr, title="1. Mixture Audio (Vocals + Instruments)")
        play_audio(tgt_instrumental_wav, sr=sr, title="2. Ground Truth Instruments (Target)")
        play_audio(pred_instrumental_wav, sr=sr, title="3. Predicted Instruments (Model Output - Karaoke)")

    return {
        "mix_wav": mix_wav,
        "tgt_instrumental_wav": tgt_instrumental_wav,
        "pred_instrumental_wav": pred_instrumental_wav,
        "mix_mag": mix_mag,
        "tgt_instrumental_mag": tgt_instrumental_mag,
        "pred_instrumental_mag": pred_instrumental_mag,
        "mix_phase": mix_phase,
    }

def preprocess_musdb18(
    musdb18_path,
    output_dir,
    chunk_duration=8.0,
    overlap=0.5,
    sample_rate=22050,
    stage1_ratio=0.7,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    save_spectrograms=True,  # NEW: Save spectrograms instead of waveforms
    stream_save=True  # NEW: Save chunks on-the-fly to reduce RAM usage
):
    """
    Complete preprocessing pipeline for MUSDB18 dataset.
    
    Steps:
    1. Load train + valid subsets from MUSDB18
    2. Chunk all full-length songs into training segments
    3. Create Stage 1 (70%) and Stage 2 (30%) curriculum splits
    4. Further split each stage into train/val/test
    5. Save organized data ready for DataLoaders (as spectrograms or waveforms)
    
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
        save_spectrograms: If True, save spectrograms instead of waveforms (10x faster training)
        stream_save: If True, save chunks as they are generated to avoid RAM overflow
        
    Returns:
        Dictionary with file counts for each split
    """
    import musdb
    from tqdm import tqdm
    
    if save_spectrograms:
        processor = AudioProcessor(device='cpu')
    
    print(f"\n{'='*70}")
    print("MUSDB18 PREPROCESSING PIPELINE")
    print(f"{'='*70}\n")
    
    data_format = "spectrograms" if save_spectrograms else "waveforms"
    print(f"Output format: {data_format}")
    
    print(f"Loading MUSDB18 from: {musdb18_path}")
    mus_train = musdb.DB(root=str(musdb18_path), is_wav=True, subsets='train')
    mus_valid = musdb.DB(root=str(musdb18_path), is_wav=True, subsets='valid')
    all_tracks = list(mus_train.tracks) + list(mus_valid.tracks)
    
    print(f"Found {len(mus_train.tracks)} tracks in TRAIN")
    print(f"Found {len(mus_valid.tracks)} tracks in VALID")
    print(f"Total tracks to process: {len(all_tracks)}\n")
    
    chunk_samples = int(chunk_duration * sample_rate)
    hop_samples = int(chunk_samples * (1 - overlap))
    
    print(f"Settings:")
    print(f"   Chunk Duration: {chunk_duration}s ({chunk_samples} samples)")
    print(f"   Overlap: {overlap*100:.0f}%")
    print(f"   Sample Rate: {sample_rate} Hz")
    print(f"   Stage 1: {stage1_ratio*100:.0f}% | Stage 2: {(1-stage1_ratio)*100:.0f}%")
    print(f"   Train: {train_ratio*100:.0f}% | Val: {val_ratio*100:.0f}% | Test: {test_ratio*100:.0f}%\n")
    
    output_root = Path(output_dir)
    stage_dirs = {}
    for stage in ['stage1', 'stage2']:
        for split in ['train', 'val', 'test']:
            for data_type in ['mixture', 'target']:
                dir_path = output_root / stage / split / data_type
                dir_path.mkdir(parents=True, exist_ok=True)
                stage_dirs[f"{stage}_{split}_{data_type}"] = dir_path
    
    print("Processing tracks into chunks...\n")
    all_chunks = []
    
    import gc
    
    if stream_save:
        counters = {
            'stage1': {'train': 0, 'val': 0, 'test': 0},
            'stage2': {'train': 0, 'val': 0, 'test': 0}
        }

    for track_idx, track in enumerate(tqdm(all_tracks, desc="Processing tracks")):
        vocals = track.targets['vocals'].audio
        drums = track.targets['drums'].audio
        bass = track.targets['bass'].audio
        other = track.targets['other'].audio
        
        vocals_mono = librosa.to_mono(vocals.T)
        drums_mono = librosa.to_mono(drums.T)
        bass_mono = librosa.to_mono(bass.T)
        other_mono = librosa.to_mono(other.T)
        
        if track.rate != sample_rate:
            vocals_mono = librosa.resample(vocals_mono, orig_sr=track.rate, target_sr=sample_rate)
            drums_mono = librosa.resample(drums_mono, orig_sr=track.rate, target_sr=sample_rate)
            bass_mono = librosa.resample(bass_mono, orig_sr=track.rate, target_sr=sample_rate)
            other_mono = librosa.resample(other_mono, orig_sr=track.rate, target_sr=sample_rate)
        
        min_len = min(len(vocals_mono), len(drums_mono), len(bass_mono), len(other_mono))
        vocals_mono = vocals_mono[:min_len]
        drums_mono = drums_mono[:min_len]
        bass_mono = bass_mono[:min_len]
        other_mono = other_mono[:min_len]
        
        num_chunks = (min_len - chunk_samples) // hop_samples + 1
        
        for i in range(num_chunks):
            start = i * hop_samples
            end = start + chunk_samples
            
            if end > min_len:
                break
            
            vocals_chunk = vocals_mono[start:end]
            drums_chunk = drums_mono[start:end]
            bass_chunk = bass_mono[start:end]
            other_chunk = other_mono[start:end]
            
            accompaniment_chunk = drums_chunk + bass_chunk + other_chunk
            
            stage = 'stage1' if np.random.rand() < stage1_ratio else 'stage2'
            
            if stage == 'stage1':
                mixture = 0.50 * vocals_chunk + 0.50 * other_chunk
                target = other_chunk
            else:
                mixture = 0.40 * vocals_chunk + 0.60 * accompaniment_chunk
                target = accompaniment_chunk
            
            max_val = max(np.abs(mixture).max(), np.abs(target).max())
            if max_val > 0:
                mixture = mixture / max_val
                target = target / max_val
            
            if save_spectrograms:
                mix_tensor = torch.from_numpy(mixture.astype(np.float32))
                tgt_tensor = torch.from_numpy(target.astype(np.float32))
                
                mix_mag, mix_phase = processor.to_spectrogram(mix_tensor)
                tgt_mag, tgt_phase = processor.to_spectrogram(tgt_tensor)

                if stream_save:
                    r = np.random.rand()
                    if r < train_ratio:
                        split_name = 'train'
                    elif r < train_ratio + val_ratio:
                        split_name = 'val'
                    else:
                        split_name = 'test'

                    mix_dir = stage_dirs[f"{stage}_{split_name}_mixture"]
                    tgt_dir = stage_dirs[f"{stage}_{split_name}_target"]
                    idx = counters[stage][split_name]
                    np.savez(mix_dir / f"{idx:06d}.npz", magnitude=mix_mag.numpy(), phase=mix_phase.numpy())
                    np.savez(tgt_dir / f"{idx:06d}.npz", magnitude=tgt_mag.numpy(), phase=tgt_phase.numpy())
                    counters[stage][split_name] += 1
                else:
                    all_chunks.append((
                        stage,
                        mix_mag.numpy(), mix_phase.numpy(),
                        tgt_mag.numpy(), tgt_phase.numpy()
                    ))
            else:
                if stream_save:
                    r = np.random.rand()
                    if r < train_ratio:
                        split_name = 'train'
                    elif r < train_ratio + val_ratio:
                        split_name = 'val'
                    else:
                        split_name = 'test'

                    mix_dir = stage_dirs[f"{stage}_{split_name}_mixture"]
                    tgt_dir = stage_dirs[f"{stage}_{split_name}_target"]
                    idx = counters[stage][split_name]
                    np.save(mix_dir / f"{idx:06d}.npy", mixture.astype(np.float32))
                    np.save(tgt_dir / f"{idx:06d}.npy", target.astype(np.float32))
                    counters[stage][split_name] += 1
                else:
                    all_chunks.append((stage, mixture.astype(np.float32), target.astype(np.float32)))
        
        if (track_idx + 1) % 10 == 0:
            gc.collect()
    
    if stream_save:
        total_chunks = sum(counters[s][sp] for s in counters for sp in counters[s])
        stage1_total = sum(counters['stage1'].values())
        stage2_total = sum(counters['stage2'].values())

        print(f"\nTotal chunks created: {total_chunks}")
        print(f"   Stage 1: {stage1_total} chunks ({(stage1_total / max(1, total_chunks)) * 100:.1f}%)")
        print(f"   Stage 2: {stage2_total} chunks ({(stage2_total / max(1, total_chunks)) * 100:.1f}%)\n")
    else:
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        gc.collect()
        
        np.random.shuffle(all_chunks)
        
        stage1_chunks = [c for c in all_chunks if c[0] == 'stage1']
        stage2_chunks = [c for c in all_chunks if c[0] == 'stage2']
        
        print(f"   Stage 1: {len(stage1_chunks)} chunks ({len(stage1_chunks)/len(all_chunks)*100:.1f}%)")
        print(f"   Stage 2: {len(stage2_chunks)} chunks ({len(stage2_chunks)/len(all_chunks)*100:.1f}%)\n")
    
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
            
            if save_spectrograms:
                for idx, (_, mix_mag, mix_phase, tgt_mag, tgt_phase) in enumerate(split_chunks):
                    np.savez(mix_dir / f"{idx:06d}.npz", magnitude=mix_mag, phase=mix_phase)
                    np.savez(tgt_dir / f"{idx:06d}.npz", magnitude=tgt_mag, phase=tgt_phase)
            else:
                for idx, (_, mixture, target) in enumerate(split_chunks):
                    np.save(mix_dir / f"{idx:06d}.npy", mixture)
                    np.save(tgt_dir / f"{idx:06d}.npy", target)
            
            counts[split_name] = len(split_chunks)
        
        return counts
    
    print("Saving organized data...")
    if stream_save:
        stage1_counts = counters['stage1']
        stage2_counts = counters['stage2']
    else:
        stage1_counts = split_and_save(stage1_chunks, 'stage1')
        
        del stage1_chunks
        gc.collect()
        
        stage2_counts = split_and_save(stage2_chunks, 'stage2')
        
        del stage2_chunks, all_chunks
        gc.collect()
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*70}\n")
    print("Data Organization:")
    print(f"\n   STAGE 1 (Weighted Mixture → Vocals):")
    print(f"      Train: {stage1_counts['train']:,} chunks")
    print(f"      Val:   {stage1_counts['val']:,} chunks")
    print(f"      Test:  {stage1_counts['test']:,} chunks")
    print(f"\n   STAGE 2 (Balanced Mixture → Vocals):")
    print(f"      Train: {stage2_counts['train']:,} chunks")
    print(f"      Val:   {stage2_counts['val']:,} chunks")
    print(f"      Test:  {stage2_counts['test']:,} chunks")
    print(f"\n   Saved to: {output_root}\n")
    
    total_chunks = sum(stage1_counts.values()) + sum(stage2_counts.values())
    return {
        'stage1': stage1_counts,
        'stage2': stage2_counts,
        'total_chunks': total_chunks
    }

def _collate_dict_batch(batch):
    """
    Module-level collate function to batch dictionary-based samples.
    Must be at module level to be picklable for multiprocessing.
    """
    mix_batch = [item['mix'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    
    if isinstance(mix_batch[0], tuple):
        mix_mag_list = []
        for m in mix_batch:
            mag = torch.tensor(m[0]) if not isinstance(m[0], torch.Tensor) else m[0]
            mag = mag.squeeze(0) if mag.dim() == 3 else mag
            mix_mag_list.append(mag)
        
        mix_ph_list = []
        for m in mix_batch:
            ph = torch.tensor(m[1]) if not isinstance(m[1], torch.Tensor) else m[1]
            ph = ph.squeeze(0) if ph.dim() == 3 else ph
            mix_ph_list.append(ph)
        
        tgt_mag_list = []
        for t in tgt_batch:
            mag = torch.tensor(t[0]) if not isinstance(t[0], torch.Tensor) else t[0]
            mag = mag.squeeze(0) if mag.dim() == 3 else mag
            tgt_mag_list.append(mag)
        
        tgt_ph_list = []
        for t in tgt_batch:
            ph = torch.tensor(t[1]) if not isinstance(t[1], torch.Tensor) else t[1]
            ph = ph.squeeze(0) if ph.dim() == 3 else ph
            tgt_ph_list.append(ph)
        
        mix_mag = torch.stack(mix_mag_list)
        mix_ph = torch.stack(mix_ph_list)
        tgt_mag = torch.stack(tgt_mag_list)
        tgt_ph = torch.stack(tgt_ph_list)
        return {
            'mix': (mix_mag, mix_ph),
            'tgt': (tgt_mag, tgt_ph)
        }
    else:
        return {
            'mix': torch.stack(mix_batch),
            'tgt': torch.stack(tgt_batch)
        }

def get_data_loaders(data_dir, stage='stage1', split='train', batch_size=16, num_workers=None):
    """
    Create DataLoader for a specific stage and split.
    Updated for speed: Uses multiprocessing workers on GPU.
    """
    data_root = Path(data_dir) / stage / split
    
    mix_files = sorted((data_root / 'mixture').glob("*.npz"))
    tgt_files = sorted((data_root / 'target').glob("*.npz"))
    
    if len(mix_files) == 0:
        mix_files = sorted((data_root / 'mixture').glob("*.npy"))
        tgt_files = sorted((data_root / 'target').glob("*.npy"))
    
    if len(mix_files) == 0:
        raise FileNotFoundError(f"No data found in {data_root}")
    
    dataset = StandardDataset(mix_files, tgt_files)
    
    if num_workers is None:
        num_workers = 2 if torch.cuda.is_available() else 0
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split=='train'),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        collate_fn=_collate_dict_batch
    )
    
    return loader

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

def load_musdb_stems(track_folder, sr=22050):
    """
    Loads the four MUSDB18 stems (vocals, drums, bass, other) from a track folder.
    Returns a dict: {'vocals': ..., 'drums': ..., 'bass': ..., 'other': ...} (all mono, resampled to sr)
    """
    import librosa
    from pathlib import Path
    stems = {}
    for stem in ['vocals', 'drums', 'bass', 'other']:
        wav_path = Path(track_folder) / f"{stem}.wav"
        mp4_path = Path(track_folder) / f"{stem}.mp4"
        if wav_path.exists():
            audio, file_sr = librosa.load(wav_path, sr=None, mono=True)
        elif mp4_path.exists():
            audio, file_sr = librosa.load(mp4_path, sr=None, mono=True)
        else:
            raise FileNotFoundError(f"Stem file not found for {stem} in {track_folder}")
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        stems[stem] = audio
    return stems

def process_stage1() -> None:
    print(f"\n{'='*70}\nSTAGE 1: vocals+other → other (single stem)\n{'='*70}")
    for split, folder in MUSDB_SPLITS.items():
        out_dir = DATA_DIR / 'stage1' / split
        mix_dir = out_dir / 'mixture'
        tgt_dir = out_dir / 'target'
        if mix_dir.exists() and tgt_dir.exists() and any(mix_dir.glob('*.npy')) and any(tgt_dir.glob('*.npy')):
            print(f"{split}: already exists, skipping...")
            continue
        print(f"Processing {split} ({folder})...")
        mix_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)
        for track_folder in folder.iterdir():
            stems = load_musdb_stems(track_folder, sr=SAMPLE_RATE)
            vocals = stems['vocals']
            other = stems['other']
            mix = vocals + other
            target = other
            total_len: int = min(len(mix), len(target))
            step = int((CHUNK_DURATION - CHUNK_OVERLAP) * SAMPLE_RATE)
            chunk_len = int(CHUNK_DURATION * SAMPLE_RATE)
            for i, start in enumerate(range(0, total_len - chunk_len + 1, step)):
                mix_chunk = mix[start:start+chunk_len]
                tgt_chunk = target[start:start+chunk_len]
                np.save(mix_dir / f"{track_folder.name}_chunk{i}.npy", mix_chunk)
                np.save(tgt_dir / f"{track_folder.name}_chunk{i}.npy", tgt_chunk)
        print(f"{split} complete: {len(list(mix_dir.glob('*.npy')))} chunks")

def process_stage2() -> None:
    print(f"\n{'='*70}\nSTAGE 2: all 4 stems → accompaniment (3 stems)\n{'='*70}")
    for split, folder in MUSDB_SPLITS.items():
        out_dir = DATA_DIR / 'stage2' / split
        mix_dir = out_dir / 'mixture'
        tgt_dir = out_dir / 'target'
        if mix_dir.exists() and tgt_dir.exists() and any(mix_dir.glob('*.npy')) and any(tgt_dir.glob('*.npy')):
            print(f"{split}: already exists, skipping...")
            continue
        print(f"Processing {split} ({folder})...")
        mix_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)
        for track_folder in folder.iterdir():
            stems = load_musdb_stems(track_folder, sr=SAMPLE_RATE)
            vocals = stems['vocals']
            drums = stems['drums']
            bass = stems['bass']
            other = stems['other']
            mix = vocals + drums + bass + other
            target = drums + bass + other
            total_len: int = min(len(mix), len(target))
            step = int((CHUNK_DURATION - CHUNK_OVERLAP) * SAMPLE_RATE)
            chunk_len = int(CHUNK_DURATION * SAMPLE_RATE)
            for i, start in enumerate(range(0, total_len - chunk_len + 1, step)):
                mix_chunk = mix[start:start+chunk_len]
                tgt_chunk = target[start:start+chunk_len]
                np.save(mix_dir / f"{track_folder.name}_chunk{i}.npy", mix_chunk)
                np.save(tgt_dir / f"{track_folder.name}_chunk{i}.npy", tgt_chunk)
        print(f"{split} complete: {len(list(mix_dir.glob('*.npy')))} chunks")

def process_vocals() -> None:
    """
    Extracts clean vocal stems from MUSDB18 for source separation training.
    Used as clean target vocals when training models to separate vocals from mixtures.
    
    Output structure:
        data/vocals/train/*.npy  (chunks named: SingerName_chunk0.npy, etc.)
        data/vocals/val/*.npy
        data/vocals/test/*.npy
    
    Each chunk is named with the singer/track name for organization.
    """
    print(f"\n{'='*70}\nVOCALS PREPROCESSING: Clean vocal stems for source separation\n{'='*70}")
    
    for split, folder in MUSDB_SPLITS.items():
        out_dir = DATA_DIR / 'vocals' / split
        
        if out_dir.exists() and any(out_dir.glob('*.npy')):
            print(f"{split}: already exists, skipping...")
            continue
        
        print(f"Processing {split} ({folder})...")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for track_folder in folder.iterdir():
            stems = load_musdb_stems(track_folder, sr=SAMPLE_RATE)
            vocals = stems['vocals']
            
            step = int((CHUNK_DURATION - CHUNK_OVERLAP) * SAMPLE_RATE)
            chunk_len = int(CHUNK_DURATION * SAMPLE_RATE)
            
            for i, start in enumerate(range(0, len(vocals) - chunk_len + 1, step)):
                vocal_chunk = vocals[start:start+chunk_len]
                np.save(out_dir / f"{track_folder.name}_chunk{i}.npy", vocal_chunk)
        
        num_chunks = len(list(out_dir.glob('*.npy')))
        num_singers = len(set(f.stem.split('_chunk')[0] for f in out_dir.glob('*.npy')))
        print(f"{split} complete: {num_chunks} chunks from {num_singers} singers")

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
    model = TimeFrequencyDomainUNet(
        input_channels=1,
        output_channels=1,
        base_filters=64,  # Increased base filters for more capacity
        depth=4,
        kernel_size=5,
        stride=2,
        padding=2,
        norm_type='batch',
        activation='relu',
        final_activation=None
    ).to(device)
        chunk_duration: Duration of each chunk in seconds (default 8.0)
        overlap: Overlap ratio 0-1 (default 0.5 = 50%)
        sr: Sample rate (default 22050)
        device: 'cpu' or 'cuda'
    
    Returns:
        numpy array: Separated vocals (same length as input)
    """
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    original_length = len(audio)
    
    print(f"\n{'='*70}")
    print(f"SEPARATING: {Path(audio_path).name}")
    print(f"{'='*70}")
    print(f"Duration: {original_length / sr:.1f}s | Sample Rate: {sr} Hz")
    
    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(chunk_samples * (1 - overlap))
    
    num_chunks = int(np.ceil((original_length - chunk_samples) / hop_samples)) + 1
    padded_length = (num_chunks - 1) * hop_samples + chunk_samples
    audio_padded = np.pad(audio, (0, padded_length - original_length), mode='constant')
    
    print(f"Chunk Size: {chunk_duration}s ({chunk_samples} samples)")
    print(f"Hop Size: {chunk_duration * (1-overlap):.1f}s ({hop_samples} samples)")
    print(f"Num Chunks: {num_chunks}")
    print()
    
    window = np.hanning(chunk_samples + 1)[:-1]
    
    model.eval()
    reconstructed = np.zeros(padded_length)
    window_sum = np.zeros(padded_length)
    
    print(f"Processing chunks...")
    with torch.no_grad():
        for i in range(num_chunks):
            if (i + 1) % max(1, num_chunks // 5) == 0 or i == 0:
                print(f"  [{i+1}/{num_chunks}] chunks processed")
            
            start = i * hop_samples
            end = start + chunk_samples
            
            chunk = audio_padded[start:end]
            
            chunk_mag, chunk_phase = processor.to_spectrogram(torch.tensor(chunk))
            
            if chunk_mag.dim() == 2:
                chunk_mag_in = chunk_mag.unsqueeze(0).unsqueeze(0).to(device)
            elif chunk_mag.dim() == 3:
                chunk_mag_in = chunk_mag.unsqueeze(1).to(device)
            else:
                chunk_mag_in = chunk_mag.unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                mask = model(chunk_mag_in)
                
                if mask.shape != chunk_mag_in.shape:
                    mask = mask[:, :, :chunk_mag_in.shape[2], :chunk_mag_in.shape[3]]
                
                est_mag = mask.squeeze(0).squeeze(0) * chunk_mag.to(device)
                
                est_wav = processor.to_waveform(est_mag.cpu(), chunk_phase.cpu())
            
            weighted_chunk = est_wav * window
            reconstructed[start:end] += weighted_chunk
            window_sum[start:end] += window
    
    reconstructed = np.divide(reconstructed, window_sum, where=window_sum > 0, out=reconstructed)
    
    output = reconstructed[:original_length]
    
    print(f"\nSeparation complete!")
    print(f"Output shape: {output.shape} | Duration: {len(output)/sr:.1f}s")
    print(f"{'='*70}\n")
    
    return output

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
    
    S_mix = librosa.stft(mixture)
    S_pred = librosa.stft(prediction)
    S_truth = librosa.stft(ground_truth)
    
    mag_mix = np.abs(S_mix)
    mag_pred = np.abs(S_pred)
    mag_truth = np.abs(S_truth)
    
    S_db_mix = librosa.power_to_db(mag_mix**2, ref=np.max(mag_mix**2))
    S_db_pred = librosa.power_to_db(mag_pred**2, ref=np.max(mag_pred**2))
    S_db_truth = librosa.power_to_db(mag_truth**2, ref=np.max(mag_truth**2))
    
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
    
    if show_audio:
        print(f"\n{'='*70}")
        print("AUDIO PLAYBACK")
        print(f"{'='*70}\n")
        
        mix_norm = mixture / (np.max(np.abs(mixture)) + 1e-8)
        pred_norm = prediction / (np.max(np.abs(prediction)) + 1e-8)
        truth_norm = ground_truth / (np.max(np.abs(ground_truth)) + 1e-8)
        
        print("Input Mixture:")
        display(Audio(mix_norm, rate=sr))
        
        print("\nModel Prediction (Separated Vocals):")
        display(Audio(pred_norm, rate=sr))
        
        print("\nGround Truth (Target Vocals):")
        display(Audio(truth_norm, rate=sr))

def initialize_model_a_lstm(device='cuda'):
    """Initialize Model A (LSTM) with full configuration."""
    from models import models
    
    processor = AudioProcessor(device=device)
    model = models.SpectrogramMaskingLSTM(
        freq_bins=1025,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=get_training_config_lstm()['learning_rate'])
    loss_fn = nn.MSELoss()
    
    return model, processor, optimizer, loss_fn

def initialize_model_a_unet(device='cuda'):
    """Initialize Model A (U-Net) with default configuration."""
    from models import models
    
    processor = AudioProcessor(device=device)
    model = models.TimeFrequencyDomainUNet(
        in_channels=1,
        out_channels=1,
        base_filters=32,
        num_layers=5,
        batchnorm=True,
        dropout=0.1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=get_training_config_unet()['learning_rate'])
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
        print(f"Found checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        hist = checkpoint.get('history', {})
        if hist:
            print(f"   Best val loss: {min(hist.get('val_loss', [float('inf')])):.6f}")
        return hist

    if skip_training:
        print("Training skipped")
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
    print(f"Training complete! Best val loss: {min(hist['val_loss']):.6f}")
    return hist

def load_training_history_from_checkpoint(ckpt_path):
    """Load training history from checkpoint or epoch files."""
    import re
    
    ckpt_path = Path(ckpt_path)
    
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
    
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        return ckpt.get('history', {})
    
    return {}

def plot_model_comparison(hist_lstm, hist_unet, title="Model A Comparison: LSTM vs U-Net"):
    """Plot side-by-side training curves for LSTM and U-Net."""
    if not hist_lstm or not hist_unet:
        print("Training histories not available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    epochs_lstm = range(1, len(hist_lstm['train_loss']) + 1)
    ax1.plot(epochs_lstm, hist_lstm['train_loss'], 'o-', label='Train', linewidth=2, markersize=5)
    ax1.plot(epochs_lstm, hist_lstm['val_loss'], 's--', label='Val', linewidth=2, markersize=5)
    ax1.set_title('Model A (LSTM)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
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
    
    winner = "LSTM" if min(hist_lstm['val_loss']) < min(hist_unet['val_loss']) else "U-Net"
    print(f"\nBest Performance: {winner}")

def evaluate_separation_quality(model_1=None, model_2=None, processor_1=None, processor_2=None,
                                test_data_dir=None, stage='stage1', num_samples=10, sr=22050, device='cuda',
                                save_path=None, load_if_exists=True,
                                random_sampling=False, random_seed=42, **kwargs):
    """
    Evaluate both models on test set using BSS metrics (SDR/SIR/SAR).

    Args:
        save_path: Optional path to persist metrics (pickle).
        load_if_exists: If True and save_path exists, returns cached metrics.
        random_sampling: If True, randomly sample chunks from the test set.
        random_seed: Seed used when random_sampling=True.
    
    Returns:
        Dictionary with metrics for both models
    """
    def _safe_median(metric_array):
        arr = np.asarray(metric_array, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.median(arr))

    def _finite(arr_like):
        arr = np.asarray(arr_like, dtype=np.float64).ravel()
        return arr[np.isfinite(arr)]

    def _summary_stats(values):
        arr = _finite(values)
        if arr.size == 0:
            return np.nan, np.nan
        return float(np.mean(arr)), float(np.std(arr))

    model_1 = model_1 if model_1 is not None else kwargs.pop('model_lstm', None)
    model_2 = model_2 if model_2 is not None else kwargs.pop('model_unet', None)
    processor_1 = processor_1 if processor_1 is not None else kwargs.pop('processor_lstm', None)
    processor_2 = processor_2 if processor_2 is not None else kwargs.pop('processor_unet', None)

    if model_1 is None or model_2 is None or processor_1 is None or processor_2 is None:
        raise ValueError(
            "evaluate_separation_quality requires model_1/model_2 and processor_1/processor_2 "
            "(or legacy model_lstm/model_unet and processor_lstm/processor_unet)."
        )
    if test_data_dir is None:
        raise ValueError("evaluate_separation_quality requires test_data_dir.")

    def _print_and_plot_metrics(metrics_1, metrics_2, heading="EVALUATION RESULTS"):
        print("\n" + "="*70)
        print(heading)
        print("="*70)

        m1_sdr_mean, m1_sdr_std = _summary_stats(metrics_1['SDR'])
        m1_sir_mean, m1_sir_std = _summary_stats(metrics_1['SIR'])
        m1_sar_mean, m1_sar_std = _summary_stats(metrics_1['SAR'])
        m2_sdr_mean, m2_sdr_std = _summary_stats(metrics_2['SDR'])
        m2_sir_mean, m2_sir_std = _summary_stats(metrics_2['SIR'])
        m2_sar_mean, m2_sar_std = _summary_stats(metrics_2['SAR'])

        print(f"\nModel 1:")
        print(f"   SDR: {m1_sdr_mean:.2f} ± {m1_sdr_std:.2f} dB")
        print(f"   SIR: {m1_sir_mean:.2f} ± {m1_sir_std:.2f} dB")
        print(f"   SAR: {m1_sar_mean:.2f} ± {m1_sar_std:.2f} dB")

        print(f"\nModel 2:")
        print(f"   SDR: {m2_sdr_mean:.2f} ± {m2_sdr_std:.2f} dB")
        print(f"   SIR: {m2_sir_mean:.2f} ± {m2_sir_std:.2f} dB")
        print(f"   SAR: {m2_sar_mean:.2f} ± {m2_sar_std:.2f} dB")

        if np.isfinite(m1_sdr_mean) and np.isfinite(m2_sdr_mean):
            winner = "Model 1" if m1_sdr_mean > m2_sdr_mean else "Model 2"
            improvement = abs(m1_sdr_mean - m2_sdr_mean)
            print(f"\nBest SDR: {winner} (+{improvement:.2f} dB)")
        else:
            print("\nBest SDR: unavailable (non-finite values)")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics_list = ['SDR', 'SIR', 'SAR']
        colors = ['#1f77b4', '#ff7f0e']

        for idx, metric in enumerate(metrics_list):
            m1_vals = _finite(metrics_1[metric])
            m2_vals = _finite(metrics_2[metric])

            if m1_vals.size == 0 and m2_vals.size == 0:
                axes[idx].set_title(f'{metric} (dB)', fontsize=14, fontweight='bold')
                axes[idx].text(0.5, 0.5, 'No finite values', ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
                continue

            plot_m1 = m1_vals if m1_vals.size > 0 else np.array([np.nan])
            plot_m2 = m2_vals if m2_vals.size > 0 else np.array([np.nan])
            data = [plot_m1, plot_m2]
            bp = axes[idx].boxplot(data, positions=[1, 2], widths=0.6,
                                   patch_artist=True, showmeans=True)

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            axes[idx].set_title(f'{metric} (dB)', fontsize=14, fontweight='bold')
            axes[idx].set_xticks([1, 2])
            axes[idx].set_xticklabels(['Model 1', 'Model 2'], fontsize=11)
            axes[idx].set_ylabel('dB', fontsize=11)
            axes[idx].grid(True, alpha=0.3, axis='y')

            for pos, vals in zip([1, 2], [m1_vals, m2_vals]):
                if vals.size > 0:
                    mean_val = float(np.mean(vals))
                    axes[idx].text(pos, mean_val, f'{mean_val:.2f}',
                                   ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.suptitle('Quantitative Evaluation: Model 1 vs Model 2 - BSS Metrics',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    def _sar_fallback(reference_wav, estimate_wav, eps=1e-12):
        reference_wav = np.asarray(reference_wav, dtype=np.float64)
        estimate_wav = np.asarray(estimate_wav, dtype=np.float64)
        num = float(np.sum(estimate_wav ** 2))
        den = float(np.sum((estimate_wav - reference_wav) ** 2))
        return 10.0 * np.log10((num + eps) / (den + eps))

    if save_path is not None:
        save_path = Path(save_path)
        if load_if_exists and save_path.exists():
            cached = load_test_results(save_path)
            if cached is not None:
                print(f"Loaded cached quantitative metrics: {save_path.name}")
                m1_cached = cached.get('model_1', cached.get('lstm', {})) if isinstance(cached, dict) else {}
                m2_cached = cached.get('model_2', cached.get('unet', {})) if isinstance(cached, dict) else {}
                if isinstance(m1_cached, dict) and isinstance(m2_cached, dict):
                    _print_and_plot_metrics(m1_cached, m2_cached, heading="CACHED EVALUATION RESULTS")
                return cached

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
        print(f"Test data not found at {test_mix_dir}")
        return None
    
    mix_files_all = sorted(test_mix_dir.glob("*.npy"))
    if len(mix_files_all) == 0:
        print(f"No test mixture chunks found in {test_mix_dir}")
        return None

    if random_sampling:
        rng = np.random.default_rng(random_seed)
        sample_count = min(num_samples, len(mix_files_all))
        chosen_indices = rng.choice(len(mix_files_all), size=sample_count, replace=False)
        mix_files = [mix_files_all[i] for i in chosen_indices]
    else:
        mix_files = mix_files_all[:num_samples]

    file_pairs = []
    for mix_file in mix_files:
        tgt_name = mix_file.name.replace('mix_', 'tgt_')
        tgt_file = test_tgt_dir / tgt_name
        if not tgt_file.exists():
            alt_tgt_file = test_tgt_dir / mix_file.name
            if alt_tgt_file.exists():
                tgt_file = alt_tgt_file
            else:
                continue
        file_pairs.append((mix_file, tgt_file))

    if len(file_pairs) == 0:
        print(f"No valid mix/target pairs found in {test_mix_dir} and {test_tgt_dir}")
        return None
    
    print("="*70)
    print(f"QUANTITATIVE EVALUATION - SDR/SIR/SAR METRICS ({stage})")
    print("="*70)
    sampling_mode = "random" if random_sampling else "sequential"
    print(f"\nEvaluating on {len(file_pairs)} test samples ({sampling_mode} sampling)...\n")
    
    metrics_1 = {'SDR': [], 'SIR': [], 'SAR': []}
    metrics_2 = {'SDR': [], 'SIR': [], 'SAR': []}
    evaluated_count = 0
    skipped_silent = 0
    skipped_errors = 0
    
    model_1.eval()
    model_2.eval()
    
    for idx, (mix_file, tgt_file) in enumerate(file_pairs):
        mix_wav = np.load(mix_file)
        tgt_wav = np.load(tgt_file)
        
        min_len: int = min(len(mix_wav), len(tgt_wav))
        mix_wav = mix_wav[:min_len]
        tgt_wav = tgt_wav[:min_len]
        
        mix_mag, mix_phase = processor_1.to_spectrogram(torch.tensor(mix_wav))
        mix_mag_in = mix_mag.unsqueeze(0).to(device)
        
        with torch.no_grad():
            mask_1 = model_1(mix_mag_in)
            est_linear_1 = mask_1.squeeze(0) * torch.expm1(mix_mag.to(device))
            est_mag_1 = torch.log1p(est_linear_1)
            est_wav_1 = processor_1.to_waveform(est_mag_1.squeeze(0).cpu(), mix_phase.squeeze(0).cpu())
        
        with torch.no_grad():
            mask_2 = model_2(mix_mag_in)
            est_linear_2 = mask_2.squeeze(0) * torch.expm1(mix_mag.to(device))
            est_mag_2 = torch.log1p(est_linear_2)
            est_wav_2 = processor_2.to_waveform(est_mag_2.squeeze(0).cpu(), mix_phase.squeeze(0).cpu())
        
        min_eval_len = min(len(tgt_wav), len(est_wav_1), len(est_wav_2))
        tgt_wav = tgt_wav[:min_eval_len]
        est_wav_1 = est_wav_1[:min_eval_len]
        est_wav_2 = est_wav_2[:min_eval_len]

        if min_eval_len == 0 or np.max(np.abs(tgt_wav)) < 1e-8:
            skipped_silent += 1
            print(f"  Sample {idx+1}/{len(file_pairs)} - skipped (silent target)")
            continue
        
        try:
            sdr_1, sir_1, sar_1, _ = museval.evaluate(
                tgt_wav.reshape(1, -1), est_wav_1.reshape(1, -1),
                win=sr, hop=sr
            )
            metrics_1['SDR'].append(_safe_median(sdr_1))
            metrics_1['SIR'].append(_safe_median(sir_1))
            sar_1_val = _safe_median(sar_1)
            if not np.isfinite(sar_1_val):
                sar_1_val = _sar_fallback(tgt_wav, est_wav_1)
            metrics_1['SAR'].append(sar_1_val)

            sdr_2, sir_2, sar_2, _ = museval.evaluate(
                tgt_wav.reshape(1, -1), est_wav_2.reshape(1, -1),
                win=sr, hop=sr
            )
            metrics_2['SDR'].append(_safe_median(sdr_2))
            metrics_2['SIR'].append(_safe_median(sir_2))
            sar_2_val = _safe_median(sar_2)
            if not np.isfinite(sar_2_val):
                sar_2_val = _sar_fallback(tgt_wav, est_wav_2)
            metrics_2['SAR'].append(sar_2_val)
        except ValueError as e:
            skipped_errors += 1
            print(f"  Sample {idx+1}/{len(file_pairs)} - skipped (museval error: {e})")
            continue

        evaluated_count += 1
        
        print(f"  Sample {idx+1}/{len(file_pairs)} - Model 1: SDR={metrics_1['SDR'][-1]:.2f} | Model 2: SDR={metrics_2['SDR'][-1]:.2f}")
    
    print(f"Valid evaluated chunks: {evaluated_count}")
    print(f"Skipped silent targets: {skipped_silent}")
    print(f"Skipped metric errors: {skipped_errors}")
    _print_and_plot_metrics(metrics_1, metrics_2, heading="EVALUATION RESULTS")
    
    results = {
        'model_1': metrics_1,
        'model_2': metrics_2,
        'lstm': metrics_1,
        'unet': metrics_2,
    }

    if save_path is not None:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_test_results(results, save_path)
        except Exception as e:
            print(f"[WARN] Could not save quantitative metrics to {save_path}: {e}")

    return results

def load_test_results(ckpt_path):
    """Load test results from checkpoint if it exists"""
    import pickle
    if ckpt_path.exists():
        with open(ckpt_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_test_results(results, ckpt_path):
    """Save test results to checkpoint"""
    import pickle
    with open(ckpt_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved test results to: {ckpt_path.name}")

def evaluate_test_set(model, processor, test_data_dir, stage, loss_fn, device, sr=22050):
    """
    Compute loss on test set (forward pass only, no gradients).
    Assess generalization to unseen data.
    """
    test_loader = get_data_loaders(test_data_dir, stage=stage, split='test', batch_size=32)

    model.eval()
    test_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            mix = batch['mix']
            target = batch['tgt']

            if isinstance(mix, tuple):
                mix_mag = mix[0].to(device)
                tgt_mag = target[0].to(device)
                
                if mix_mag.dim() == 3:
                    mix_mag = mix_mag.unsqueeze(1)
                if tgt_mag.dim() == 3:
                    tgt_mag = tgt_mag.unsqueeze(1)
                
                mix_processed = mix_mag
                target_processed = tgt_mag
            else:
                mix = mix.to(device)
                target = target.to(device)
                
                mix_spec = processor.to_spectrogram(mix)
                target_spec = processor.to_spectrogram(target)
                
                mix_processed = mix_spec[0].unsqueeze(1) if mix_spec[0].dim() == 3 else mix_spec[0]
                target_processed = target_spec[0].unsqueeze(1) if target_spec[0].dim() == 3 else target_spec[0]

            mask = model(mix_processed)
            
            if mask.shape != mix_processed.shape:
                mask = mask[:, :, :mix_processed.shape[2], :mix_processed.shape[3]]
            
            est_linear = mask * torch.expm1(mix_processed)
            est_log = torch.log1p(est_linear)
            
            loss = loss_fn(est_log, target_processed)
            test_losses.append(loss.item())

            if (batch_idx + 1) % max(1, len(test_loader) // 5) == 0 or batch_idx == 0:
                print(f"  Batch {batch_idx+1}/{len(test_loader)} | Loss: {loss.item():.6f}")

    avg_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)

    print(f"\n{'='*70}")
    print(f"TEST RESULTS:")
    print(f"  Mean Loss: {avg_test_loss:.6f}")
    print(f"  Std Loss:  {std_test_loss:.6f}")
    print(f"  Min Loss:  {min(test_losses):.6f}")
    print(f"  Max Loss:  {max(test_losses):.6f}")
    print(f"{'='*70}\n")

    return {'test_losses': test_losses, 'mean': avg_test_loss, 'std': std_test_loss}

def sliding_window_inference(model, processor, audio, chunk_len=8.0, sr=22050, device='cuda'):
    """Apply model with overlapping windows for long audio"""
    chunk_samples = int(chunk_len * sr)
    stride = chunk_samples // 2  # 50% overlap
    output = np.zeros_like(audio)
    weights = np.zeros_like(audio)
    
    print("   Processing: ", end="", flush=True)
    for pos in range(0, len(audio), stride):
        if pos % (stride * 5) == 0:
            print(".", end="", flush=True)
        
        segment = audio[pos:pos + chunk_samples]
        if len(segment) < chunk_samples:
            segment = np.pad(segment, (0, chunk_samples - len(segment)))
        
        with torch.no_grad():
            spec = processor.to_spectrogram(segment)
            mag, phase = spec[0].squeeze(), spec[1].squeeze()
            if mag.dim() > 1 and mag.shape[0] < mag.shape[1]:
                mag = mag.t()
            
            inp = mag.unsqueeze(0).unsqueeze(0).to(device)
            mask = model(inp).squeeze()
            
            if mask.shape != mag.shape:
                mask = mask[:mag.shape[0], :mag.shape[1]]
            
            est_mag = torch.log1p(mask * torch.expm1(mag))
            est_seg = processor.to_waveform(est_mag.cpu().numpy(), phase.cpu().numpy())
        
        valid_len = min(len(segment), len(audio) - pos)
        if len(est_seg) > valid_len:
            est_seg = est_seg[:valid_len]
        elif len(est_seg) < valid_len:
            est_seg = np.pad(est_seg, (0, valid_len - len(est_seg)))
        
        window = np.hanning(chunk_samples)[:valid_len]
        output[pos:pos + valid_len] += est_seg * window
        weights[pos:pos + valid_len] += window
    
    print(" Done!")
    return np.divide(output, weights, where=weights > 0, out=output.copy())

def to_spec(wav, processor):
    """
    Convert waveform to displayable spectrogram.
    Returns shape (freq, time) for imshow: freq on y-axis (vertical), time on x-axis (horizontal).
    Frequency bins ~1025, time frames >> 1025, so if shape[0] < shape[1], already (freq, time).
    """
    s = processor.to_spectrogram(wav)[0].squeeze().cpu().numpy()
    return s if s.shape[0] < s.shape[1] else s.T

def unet_inference_accelerator(model, processor, audio, chunk_len=8.0, sr=22050, device='cuda', batch_size=16):
    """
    Accelerated inference for U-Net by batching chunks through the forward pass in parallel.
    Unlike training, inference typically processes one chunk at a time (batch_size=1),
    but the model's forward pass supports batching - this function exploits that!
    
    Args:
        model: U-Net model
        processor: AudioProcessor for spectrogram conversion
        audio: Input audio waveform
        chunk_len: Length of each chunk in seconds (default: 8.0)
        sr: Sample rate (default: 22050)
        device: Device to run inference on (default: 'cuda')
        batch_size: Number of chunks to process in parallel (default: 16)
                    Increase for more speed (if GPU memory allows), decrease if OOM
    
    Returns:
        Separated audio waveform
    """
    chunk_samples = int(chunk_len * sr)
    stride = chunk_samples // 2
    output = np.zeros_like(audio)
    weights = np.zeros_like(audio)
    
    chunks_data = []
    for pos in range(0, len(audio), stride):
        segment = audio[pos:pos + chunk_samples]
        if len(segment) < chunk_samples:
            segment = np.pad(segment, (0, chunk_samples - len(segment)))
        
        spec = processor.to_spectrogram(segment)
        mag, phase = spec[0].squeeze(), spec[1].squeeze()
        if mag.dim() > 1 and mag.shape[0] < mag.shape[1]:
            mag = mag.t()
        
        valid_len = min(len(segment), len(audio) - pos)
        chunks_data.append({
            'pos': pos,
            'mag': mag,
            'phase': phase,
            'valid_len': valid_len
        })
    
    total_chunks = len(chunks_data)
    print(f"   Processing: ", end="", flush=True)
    
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, total_chunks, batch_size):
            if batch_start % (batch_size * 5) == 0:
                print(".", end="", flush=True)
            
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks_data[batch_start:batch_end]
            
            batch_mags = torch.stack([chunk['mag'].unsqueeze(0) for chunk in batch_chunks]).to(device)
            
            batch_masks = model(batch_mags)
            
            for i, chunk in enumerate(batch_chunks):
                mask = batch_masks[i].squeeze()
                mag = chunk['mag']
                phase = chunk['phase']
                pos = chunk['pos']
                valid_len = chunk['valid_len']
                
                if mask.shape != mag.shape:
                    mask = mask[:mag.shape[0], :mag.shape[1]]
                
                est_mag = torch.log1p(mask * torch.expm1(mag))
                est_seg = processor.to_waveform(est_mag.cpu().numpy(), phase.cpu().numpy())
                
                if len(est_seg) > valid_len:
                    est_seg = est_seg[:valid_len]
                elif len(est_seg) < valid_len:
                    est_seg = np.pad(est_seg, (0, valid_len - len(est_seg)))
                
                window = np.hanning(chunk_samples)[:valid_len]
                output[pos:pos + valid_len] += est_seg * window
                weights[pos:pos + valid_len] += window
    
    print(" Done!")
    return np.divide(output, weights, where=weights > 0, out=output.copy())

def compare_models_on_audio_file(file_path, model_lstm, model_unet, processor_lstm, 
                                  processor_unet, device, sr=22050, duration=None, unet_batch_size=16):
    """
    Load a custom audio file, run inference with both LSTM and U-Net models,
    and display spectrograms + audio playback for comparison.
    
    Args:
        file_path: Path to audio file
        model_lstm: Trained LSTM model
        model_unet: Trained U-Net model
        processor_lstm: AudioProcessor for LSTM
        processor_unet: AudioProcessor for U-Net
        device: torch device ('cuda' or 'cpu')
        sr: Sample rate (default: 22050)
        duration: Duration to process in seconds (None for full file)
        unet_batch_size: Batch size for U-Net accelerated inference (default: 16)
                         Increase for more speed (if GPU memory allows), decrease if OOM
    """
    import librosa
    import matplotlib.pyplot as plt
    from IPython.display import Audio, display
    
    print("="*70)
    print("CUSTOM AUDIO INFERENCE")
    print("="*70)
    print(f"\nLoading: {file_path}")
    
    try:
        audio, file_sr = librosa.load(file_path, sr=None, mono=True)
        print(f"   Original SR: {file_sr} Hz, Duration: {len(audio)/file_sr:.2f}s")
        
        if file_sr != sr:
            print(f"   Resampling to {sr} Hz...")
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        
        if duration is not None:
            target_samples = int(duration * sr)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
                print(f"   Trimmed to {duration}s")
        
        print(f"Loaded: {len(audio)/sr:.2f}s @ {sr} Hz")
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    model_lstm.eval()
    model_unet.eval()
    
    print("\nRunning inference...")
    print("LSTM (sequential, batch_size=1):")
    est_lstm = sliding_window_inference(model_lstm, processor_lstm, audio, 
                                       chunk_len=8.0, sr=sr, device=device)
    print(f"U-Net (accelerated batched forward pass, batch_size={unet_batch_size}):")
    est_unet = unet_inference_accelerator(model_unet, processor_unet, audio,
                                         chunk_len=8.0, sr=sr, device=device, batch_size=unet_batch_size)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    axes[0].imshow(to_spec(audio, processor_lstm), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("Input: Original Audio", fontweight='bold', fontsize=12)
    axes[0].set_ylabel("Frequency")
    
    axes[1].imshow(to_spec(est_lstm, processor_lstm), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title("LSTM Separation", fontweight='bold', fontsize=12)
    axes[1].set_ylabel("Frequency")
    
    axes[2].imshow(to_spec(est_unet, processor_unet), aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title("U-Net Separation", fontweight='bold', fontsize=12)
    axes[2].set_ylabel("Frequency")
    axes[2].set_xlabel("Time")
    
    plt.suptitle(f"Custom Audio Inference: {file_path.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nAudio Playback:\n")
    print("Original Audio:")
    display(Audio(audio, rate=sr))
    
    print("\nLSTM Separated Vocals:")
    display(Audio(est_lstm, rate=sr))
    
    print("\nU-Net Separated Vocals:")
    display(Audio(est_unet, rate=sr))
    
    print(f"\n{'='*70}")
    print("INFERENCE COMPLETE")
    print(f"{'='*70}")
