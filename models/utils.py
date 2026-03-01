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

def verify_and_install_packages(packages):
    print("Verifying and installing missing packages if necessary...")
    
    for package in packages:
        try:
            __import__(package)
            print(f"  {package} is installed.")
        except ImportError:
            print(f"  {package} is NOT installed. Attempting to install...")
            try:
                import sys
                import subprocess
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                __import__(package)
                print(f"  {package} is now installed.")
            except Exception as e:
                print(f"  Failed to install {package}: {e}")

def check_pytorch_cuda_status():
    print("\n--- PyTorch CUDA status ---")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  PyTorch with CUDA (version {torch.version.cuda}) is available.")
            print(f"     CUDA Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("  PyTorch is installed, but CUDA is NOT available.")
    except ImportError:
        print("  PyTorch is NOT installed.")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_project_environment():
    import sys
    
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        PROJECT_ROOT = Path('/content/drive/MyDrive/Colab Notebooks/Final_Project_Deep_Learning')
        print(f"Colab Project Root: {PROJECT_ROOT}")
    else:
        PROJECT_ROOT = Path.cwd()
        if not (PROJECT_ROOT / 'mainNB.ipynb').exists():
            for p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
                if (p / 'mainNB.ipynb').exists():
                    PROJECT_ROOT = p
                    break
    
    os.chdir(PROJECT_ROOT)
    
    DATA_DIR = PROJECT_ROOT / "data"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration Complete:")
    print(f"   - Device: {device}")
    print(f"   - Working Directory: {os.getcwd()}")
    print(f"   - Data Directory: {DATA_DIR}")
    print(f"   - Checkpoints Directory: {CHECKPOINT_DIR}")
    
    return PROJECT_ROOT, DATA_DIR, CHECKPOINT_DIR, device, IN_COLAB

def verify_musdb18_dataset(project_root):
    musdb18_path = project_root / "musdb18"
    
    if not musdb18_path.exists():
        print("MUSDB18 folder not found!")
        print(f"   Expected location: {musdb18_path}")
        print("")
        print("Download MUSDB18:")
        print("   1. Register at https://zenodo.org/record/1438122")
        print("   2. Download MUSDB18-HQ.zip (~22GB)")
        print("   3. Extract to project folder as 'musdb18'")
        print("")
        return None
    
    train_dir = musdb18_path / "train"
    valid_dir = musdb18_path / "valid"
    test_dir = musdb18_path / "test"
    
    if train_dir.exists() and test_dir.exists():
        num_train = len(list(train_dir.iterdir()))
        num_valid = len(list(valid_dir.iterdir())) if valid_dir.exists() else 0
        num_test = len(list(test_dir.iterdir()))
        print(f"MUSDB18 dataset found: {musdb18_path}")
        print(f"   Train: {num_train} tracks")
        print(f"   Valid: {num_valid} tracks")
        print(f"   Test: {num_test} tracks")
        return musdb18_path
    else:
        print(f"Found musdb18 folder but missing train/test subfolders")
        print(f"   Path: {musdb18_path}")
        return None

def extract_and_process_data(project_root, data_dir, musdb18_path, in_colab, folders_to_extract=['stage1', 'stage2', 'vocals'], sample_rate=22050, chunk_duration=8.0, chunk_overlap=4.0):
    import shutil
    import time
    import zipfile
    from tqdm import tqdm
    
    drive_zip_path = project_root / "data.zip"
    local_extract_root = Path("/content/local_data")
    local_data_dir = local_extract_root / "data"
    
    if in_colab:
        print(f"Checking for local data at: {local_data_dir}")
        
        if not local_data_dir.exists():
            print("Data not found locally. Starting extraction...")
            print(f"Source: {drive_zip_path}")
            print(f"Destination: {local_extract_root}")
            if folders_to_extract:
                print(f"Extracting only: {', '.join(folders_to_extract)}")
            
            local_extract_root.mkdir(parents=True, exist_ok=True)
            
            if drive_zip_path.exists():
                t0 = time.time()
                print("Unzipping from Drive with progress tracking...")
                
                try:
                    with zipfile.ZipFile(drive_zip_path, 'r') as zip_ref:
                        all_files = zip_ref.namelist()
                        
                        if folders_to_extract:
                            folders_normalized = [f.strip('/') for f in folders_to_extract]
                            file_list = [
                                f for f in all_files 
                                if any(
                                    f.startswith(f'data/{folder}/') or f.startswith(f'{folder}/')
                                    for folder in folders_normalized
                                )
                            ]
                            print(f"Filtering: {len(file_list):,} / {len(all_files):,} files selected")
                        else:
                            file_list = all_files
                            print(f"Extracting all {len(file_list):,} files")
                        
                        for file in tqdm(file_list, desc="Extracting", unit="file"):
                            zip_ref.extract(file, local_extract_root)
                    
                    t_final = time.time() - t0
                    print(f"Unzip complete in {t_final/60:.1f} minutes!")
                    
                    if local_data_dir.exists():
                        extracted_folders = [d.name for d in local_data_dir.iterdir() if d.is_dir()]
                        print(f"Extracted folders: {', '.join(extracted_folders)}")
                    
                    data_dir = local_data_dir
                except Exception as e:
                    print(f"Unzip failed: {e}")
                    print("   Falling back to Drive path (slow)")
                    data_dir = project_root / "data"
            else:
                print(f"Error: Could not find {drive_zip_path}")
                print("   Please ensure 'data.zip' is uploaded to your Drive project folder.")
                data_dir = project_root / "data"
        else:
            print("Fast local data already exists! Skipping unzip.")
            data_dir = local_data_dir
        
        print(f"DATA_DIR set to: {data_dir}")
        
        total, used, free = shutil.disk_usage("/")
        print(f"Disk Space: {free // (2**30)} GB free / {total // (2**30)} GB total")
    else:
        data_dir = project_root / "data"
        print(f"Running locally. Using repo data: {data_dir}")
    
    if musdb18_path:
        global MUSDB_SPLITS, DATA_DIR, SAMPLE_RATE, CHUNK_DURATION, CHUNK_OVERLAP
        MUSDB_SPLITS = {
            'train': musdb18_path / 'train',
            'val': musdb18_path / 'valid',
            'test': musdb18_path / 'test',
        }
        DATA_DIR = data_dir
        SAMPLE_RATE = sample_rate
        CHUNK_DURATION = chunk_duration
        CHUNK_OVERLAP = chunk_overlap
        
        print(f"\n{'='*70}")
        print("CHECKING DATA FOLDERS")
        print(f"{'='*70}")
        
        has_stage1 = (data_dir / 'stage1').exists()
        has_stage2 = (data_dir / 'stage2').exists()
        has_vocals = (data_dir / 'vocals').exists()
        
        if has_stage1:
            print("Stage1 data found - processing...")
            process_stage1()
        else:
            print("Stage1 not extracted - skipping")
        
        if has_stage2:
            print("Stage2 data found - processing...")
            process_stage2()
        else:
            print("Stage2 not extracted - skipping")
        
        if has_vocals:
            print("Vocals data found - processing...")
            process_vocals()
        else:
            print("Vocals not extracted - skipping")
        
        print(f"\n{'='*70}")
        print("SYSTEM READY")
        print(f"{'='*70}")
    else:
        print("MUSDB18_PATH not set - skipping preprocessing")
    
    return data_dir

class UniversalTrainer:
    def __init__(self, model, train_loader, val_loader, processor, optimizer, loss_fn, device='cpu', patience=10, input_type='spectrogram'):
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
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        def _in_notebook():
            try:
                from IPython import get_ipython
                shell = get_ipython().__class__.__name__
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
                mix = mix.to(self.device)
            if isinstance(tgt, torch.Tensor):
                tgt = tgt.to(self.device)
            
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
    def __init__(self, model, processor, device='cpu', input_type='spectrogram'):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.input_type = input_type

    def separate(self, mixture):
        self.model.eval()
        with torch.no_grad():
            mix = torch.tensor(mixture).to(self.device).float()
            if mix.ndim == 1:
                mix = mix.unsqueeze(0)
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
    def __init__(self, n_fft=2048, hop_length=512, device='cpu'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.window = torch.hann_window(n_fft).to(device)

    def to_spectrogram(self, waveform):
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        elif isinstance(waveform, (list, tuple)):
            if len(waveform) == 0:
                waveform = torch.empty(0)
            elif isinstance(waveform[0], torch.Tensor):
                waveform = torch.stack(waveform)
            elif isinstance(waveform[0], np.ndarray):
                waveform = torch.from_numpy(np.stack(waveform))
            else:
                waveform = torch.tensor(waveform)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
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
        log_mag: torch.Tensor = torch.log1p(mag)
        
        return log_mag, phase

    def to_waveform(self, log_mag, phase):

        if isinstance(log_mag, np.ndarray): log_mag: torch.Tensor = torch.from_numpy(log_mag)
        if isinstance(phase, np.ndarray): phase: torch.Tensor = torch.from_numpy(phase)
            
        log_mag = log_mag.to(self.device)
        phase = phase.to(self.device)
        
        lin_mag = torch.expm1(log_mag)
        complex_spec = lin_mag * torch.exp(1j * phase)
        
        waveform = torch.istft(
            complex_spec, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window
        )
        return waveform.cpu().numpy()

class StandardDataset(Dataset):
    def __init__(self, mix_files, tgt_files):
        self.mix_files = list(mix_files)
        self.tgt_files = list(tgt_files)
        
        if len(self.mix_files) > 0:
            self.is_spectrogram = str(self.mix_files[0]).endswith('.npz')
        else:
            self.is_spectrogram = False

    def __len__(self):

        return len(self.mix_files)

    def __getitem__(self, idx):

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

def get_training_config(model_type='default'):
    config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'chunk_duration': 1.0,
        'chunk_overlap': 0.5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if model_type == 'lstm':
        config['batch_size'] = 32
    elif model_type == 'unet':
        config['batch_size'] = 8
    
    return config

def play_audio(waveform, sr=22050, title="Audio"):

    if hasattr(waveform, 'cpu'):
        waveform = waveform.squeeze().cpu().numpy()
    print(f"{title}:")
    display(Audio(waveform, rate=sr))

def get_curriculum_file_lists(cache_dir="../data", split='train'):
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
    tgt_instrumental_wav = np.load(tgt_files[song_num])[:n_samples]

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
    save_spectrograms=True,
    stream_save=True
):

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

def _process_stage_helper(stage_name, stage_desc, stage_dir, mix_fn, target_fn):
    print(f"\n{'='*70}\n{stage_desc}\n{'='*70}")
    for split, folder in MUSDB_SPLITS.items():
        out_dir = DATA_DIR / stage_dir / split
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
            mix = mix_fn(stems)
            target = target_fn(stems)
            total_len = min(len(mix), len(target))
            step = int((CHUNK_DURATION - CHUNK_OVERLAP) * SAMPLE_RATE)
            chunk_len = int(CHUNK_DURATION * SAMPLE_RATE)
            for i, start in enumerate(range(0, total_len - chunk_len + 1, step)):
                mix_chunk = mix[start:start+chunk_len]
                tgt_chunk = target[start:start+chunk_len]
                np.save(mix_dir / f"{track_folder.name}_chunk{i}.npy", mix_chunk)
                np.save(tgt_dir / f"{track_folder.name}_chunk{i}.npy", tgt_chunk)
        print(f"{split} complete: {len(list(mix_dir.glob('*.npy')))} chunks")

def process_stage1():
    _process_stage_helper(
        'stage1',
        'STAGE 1: vocals+other → other (single stem)',
        'stage1',
        lambda s: s['vocals'] + s['other'],
        lambda s: s['other']
    )

def process_stage2():
    _process_stage_helper(
        'stage2',
        'STAGE 2: all 4 stems → accompaniment (3 stems)',
        'stage2',
        lambda s: s['vocals'] + s['drums'] + s['bass'] + s['other'],
        lambda s: s['drums'] + s['bass'] + s['other']
    )

def process_vocals():
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
    from models import models
    
    processor = AudioProcessor(device=device)
    model = models.SpectrogramMaskingLSTM(
        freq_bins=1025,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=get_training_config('lstm')['learning_rate'])
    loss_fn = nn.MSELoss()
    
    return model, processor, optimizer, loss_fn

def initialize_model_a_unet(device='cuda'):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=get_training_config('unet')['learning_rate'])
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

def train_stage1_models(data_dir, checkpoint_dir, device, chunk_duration=8.0, skip_training=False, lstm_batch_size=128, unet_batch_size=32):

    import gc
    
    print(f"\n{'='*70}")
    print('STAGE 1 TRAINING: 2 → 1')
    print(f"{'='*70}\n")
    
    ckpt_lstm_s1 = checkpoint_dir / f'model_a_lstm_stage1_{chunk_duration:.0f}s.pth'
    ckpt_unet_s1 = checkpoint_dir / f'model_a_unet_stage1_{chunk_duration:.0f}s.pth'
    
    skip_lstm = skip_training or ckpt_lstm_s1.exists()
    skip_unet = skip_training or ckpt_unet_s1.exists()
    
    print('Model (LSTM) - Stage 1')
    print('-' * 70)
    if ckpt_lstm_s1.exists():
        print(f"LSTM Stage 1 checkpoint found: {ckpt_lstm_s1.name}")
    
    lstm_config = get_training_config('lstm')
    lstm_config['batch_size'] = lstm_batch_size
    
    model_lstm, processor_lstm, optimizer_lstm, loss_fn_lstm = initialize_model_a_lstm(device)
    hist_lstm_s1 = train_model_stage(
        model=model_lstm,
        processor=processor_lstm,
        optimizer=optimizer_lstm,
        loss_fn=loss_fn_lstm,
        training_data_dir=data_dir,
        stage='stage1',
        ckpt_path=ckpt_lstm_s1,
        device=device,
        train_config=lstm_config,
        skip_training=skip_lstm
    )
    
    model_lstm.to('cpu')
    del optimizer_lstm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print('\nModel (U-Net) - Stage 1')
    print('-' * 70)
    if ckpt_unet_s1.exists():
        print(f"U-Net Stage 1 checkpoint found: {ckpt_unet_s1.name}")
    
    unet_config = get_training_config('unet')
    unet_config['batch_size'] = unet_batch_size
    
    model_unet, processor_unet, optimizer_unet, loss_fn_unet = initialize_model_a_unet(device)
    hist_unet_s1 = train_model_stage(
        model=model_unet,
        processor=processor_unet,
        optimizer=optimizer_unet,
        loss_fn=loss_fn_unet,
        training_data_dir=data_dir,
        stage='stage1',
        ckpt_path=ckpt_unet_s1,
        device=device,
        train_config=unet_config,
        skip_training=skip_unet
    )
    
    return {
        'model_lstm': model_lstm,
        'processor_lstm': processor_lstm,
        'loss_fn_lstm': loss_fn_lstm,
        'model_unet': model_unet,
        'processor_unet': processor_unet,
        'loss_fn_unet': loss_fn_unet,
        'hist_lstm_s1': hist_lstm_s1,
        'hist_unet_s1': hist_unet_s1,
        'ckpt_lstm_s1': ckpt_lstm_s1,
        'ckpt_unet_s1': ckpt_unet_s1
    }

def train_stage2_models(data_dir, checkpoint_dir, device, chunk_duration=8.0, skip_training=False, 
                        lstm_batch_size=128, unet_batch_size=32, stage2_enabled=True):

    import gc
    
    ckpt_lstm_s1 = checkpoint_dir / f'model_a_lstm_stage1_{chunk_duration:.0f}s.pth'
    ckpt_unet_s1 = checkpoint_dir / f'model_a_unet_stage1_{chunk_duration:.0f}s.pth'
    ckpt_lstm_s2 = checkpoint_dir / f'model_a_lstm_stage2_{chunk_duration:.0f}s.pth'
    ckpt_unet_s2 = checkpoint_dir / f'model_a_unet_stage2_{chunk_duration:.0f}s.pth'
    
    hist_lstm_s2 = {}
    hist_unet_s2 = {}
    
    if not stage2_enabled:
        print("Stage 2 training disabled (STAGE2_ENABLED = False)")
        return {
            'hist_lstm_s2': hist_lstm_s2,
            'hist_unet_s2': hist_unet_s2
        }
    
    print(f"\n{'='*70}")
    print('STAGE 2 TRAINING: 4 → 1')
    print(f"{'='*70}\n")
    
    print('1. Model A (LSTM) - Stage 2')
    print('-' * 70)
    
    lstm_config = get_training_config('lstm')
    lstm_config['batch_size'] = lstm_batch_size
    
    model_lstm, processor_lstm, optimizer_lstm, loss_fn_lstm = initialize_model_a_lstm(device)
    
    if ckpt_lstm_s1.exists():
        print(f"Loading Stage 2 weights from: {ckpt_lstm_s1.name}")
        checkpoint = torch.load(ckpt_lstm_s1, map_location=device)
        model_lstm.load_state_dict(checkpoint['model_state_dict'])
        print(f"LSTM Stage 2 weights loaded successfully.")
    else:
        print("LSTM Stage 2 checkpoint NOT found. Starting from scratch (not ideal).")
    
    hist_lstm_s2 = train_model_stage(
        model=model_lstm,
        processor=processor_lstm,
        optimizer=optimizer_lstm,
        loss_fn=loss_fn_lstm,
        training_data_dir=data_dir,
        stage='stage2',
        ckpt_path=ckpt_lstm_s2,
        device=device,
        train_config=lstm_config,
        skip_training=skip_training
    )
    
    model_lstm.to('cpu')
    del optimizer_lstm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print('\n2. Model A (U-Net) - Stage 2')
    print('-' * 70)
    
    unet_config = get_training_config('unet')
    unet_config['batch_size'] = unet_batch_size
    
    model_unet, processor_unet, optimizer_unet, loss_fn_unet = initialize_model_a_unet(device)
    
    if ckpt_unet_s1.exists():
        print(f"Loading Stage 2 weights from: {ckpt_unet_s1.name}")
        checkpoint = torch.load(ckpt_unet_s1, map_location=device)
        try:
            model_unet.load_state_dict(checkpoint['model_state_dict'])
            print(f"U-Net Stage 2 weights loaded successfully.")
        except RuntimeError as e:
            print(f"CRITICAL ERROR loading weights: {e}")
            print("   (This usually means the model structure in Stage 2 doesn't match Stage 1)")
            raise e
    else:
        print("U-Net Stage 2 checkpoint NOT found. Starting from scratch.")
    
    hist_unet_s2 = train_model_stage(
        model=model_unet,
        processor=processor_unet,
        optimizer=optimizer_unet,
        loss_fn=loss_fn_unet,
        training_data_dir=data_dir,
        stage='stage2',
        ckpt_path=ckpt_unet_s2,
        device=device,
        train_config=unet_config,
        skip_training=skip_training
    )
    
    print(f"\n{'='*70}")
    print("STAGE 2 COMPLETE!")
    print(f"{'='*70}")
    
    return {
        'model_lstm': model_lstm,
        'processor_lstm': processor_lstm,
        'loss_fn_lstm': loss_fn_lstm,
        'model_unet': model_unet,
        'processor_unet': processor_unet,
        'loss_fn_unet': loss_fn_unet,
        'hist_lstm_s2': hist_lstm_s2,
        'hist_unet_s2': hist_unet_s2,
        'ckpt_lstm_s2': ckpt_lstm_s2,
        'ckpt_unet_s2': ckpt_unet_s2
    }

def train_unetattention_stage1(data_dir, checkpoint_dir, device, chunk_duration=8.0, 
                                skip_training=False, batch_size=32, base_filters=32, 
                                num_layers=4, num_heads=4, learning_rate=None):

    from models import models as ma
    
    print(f"\n{'='*70}")
    print('STAGE 1 TRAINING: UNetAttention (2 → 1)')
    print(f"{'='*70}\n")
    
    ckpt_attn_s1 = checkpoint_dir / f'model_unetattention_stage1_{chunk_duration:.0f}s.pth'
    
    if ckpt_attn_s1.exists():
        print(f"UNetAttention Stage 1 checkpoint found: {ckpt_attn_s1.name}")
    
    if skip_training and ckpt_attn_s1.exists():
        print("Skipping training (skip_training=True and checkpoint exists)")
        
        print(f"Initializing UNetAttention ({base_filters} filters, {num_layers} layers, {num_heads} heads)...")
        model_attn_s1 = ma.UNetAttention(
            in_channels=1,
            out_channels=1,
            base_filters=base_filters,
            num_layers=num_layers,
            num_heads=num_heads,
            batchnorm=True,
            dropout=0.1
        ).to(device)
        
        checkpoint = torch.load(ckpt_attn_s1, map_location=device)
        model_attn_s1.load_state_dict(checkpoint['model_state_dict'])
        hist_attn_s1 = checkpoint.get('history', {})
        
        processor_attn = AudioProcessor(device=device)
        loss_fn_attn = nn.MSELoss()
        
        print("Model loaded from checkpoint")
    else:
        attn_config = get_training_config('unet')
        attn_config['batch_size'] = batch_size
        
        if learning_rate is not None:
            attn_config['learning_rate'] = learning_rate
        
        print(f"Initializing UNetAttention ({base_filters} filters, {num_layers} layers, {num_heads} heads)...")
        model_attn_s1 = ma.UNetAttention(
            in_channels=1,
            out_channels=1,
            base_filters=base_filters,
            num_layers=num_layers,
            num_heads=num_heads,
            batchnorm=True,
            dropout=0.1
        ).to(device)
        
        processor_attn = AudioProcessor(device=device)
        optimizer_attn = torch.optim.Adam(model_attn_s1.parameters(), lr=attn_config['learning_rate'])
        loss_fn_attn = nn.MSELoss()
        
        hist_attn_s1 = train_model_stage(
            model=model_attn_s1,
            processor=processor_attn,
            optimizer=optimizer_attn,
            loss_fn=loss_fn_attn,
            training_data_dir=data_dir,
            stage='stage1',
            ckpt_path=ckpt_attn_s1,
            device=device,
            train_config=attn_config,
            skip_training=False
        )
    
    return {
        'model_attn_s1': model_attn_s1,
        'processor_attn': processor_attn,
        'loss_fn_attn': loss_fn_attn,
        'hist_attn_s1': hist_attn_s1,
        'ckpt_attn_s1': ckpt_attn_s1
    }

def compare_unet_vs_unetattention_stage1(checkpoint_dir, chunk_duration, hist_unet_s1=None, hist_attn_s1=None):

    if not hist_unet_s1 or 'train_loss' not in hist_unet_s1:
        ckpt_unet_s1 = checkpoint_dir / f"model_a_unet_stage1_{chunk_duration:.0f}s.pth"
        hist_unet_s1 = load_training_history_from_checkpoint(ckpt_unet_s1)
    
    if not hist_attn_s1 or 'train_loss' not in hist_attn_s1:
        ckpt_attn_s1 = checkpoint_dir / f"model_unetattention_stage1_{chunk_duration:.0f}s.pth"
        hist_attn_s1 = load_training_history_from_checkpoint(ckpt_attn_s1)
    
    test_results_ckpt_unet_s1 = checkpoint_dir / f"test_results_unet_stage1_{chunk_duration:.0f}s.pkl"
    test_results_ckpt_attn_s1 = checkpoint_dir / f"test_results_unetattention_stage1_{chunk_duration:.0f}s.pkl"
    
    test_unet_s1 = load_test_results(test_results_ckpt_unet_s1)
    test_attn_s1 = load_test_results(test_results_ckpt_attn_s1)
    
    if not hist_unet_s1 or not hist_attn_s1:
        print("Could not plot comparison - missing training history for one or both models.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'train_loss' in hist_unet_s1:
        epochs_unet = range(1, len(hist_unet_s1['train_loss']) + 1)
        ax1.plot(epochs_unet, hist_unet_s1['train_loss'], 'o-', label='Train', linewidth=2)
        ax1.plot(epochs_unet, hist_unet_s1['val_loss'], 's--', label='Val', linewidth=2)
        
        if test_unet_s1 and 'mean' in test_unet_s1 and 'std' in test_unet_s1:
            ax1.fill_between(
                epochs_unet,
                test_unet_s1['mean'] - test_unet_s1['std'],
                test_unet_s1['mean'] + test_unet_s1['std'],
                alpha=0.2, color='red'
            )
            ax1.axhline(
                test_unet_s1['mean'],
                color='red', linestyle=':', linewidth=2,
                label=f"Test (μ={test_unet_s1['mean']:.4f})"
            )
    
    ax1.set_title('Standard U-Net - Stage 1', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if 'train_loss' in hist_attn_s1:
        epochs_attn = range(1, len(hist_attn_s1['train_loss']) + 1)
        ax2.plot(epochs_attn, hist_attn_s1['train_loss'], 'o-', label='Train', linewidth=2)
        ax2.plot(epochs_attn, hist_attn_s1['val_loss'], 's--', label='Val', linewidth=2)
        
        if test_attn_s1 and 'mean' in test_attn_s1 and 'std' in test_attn_s1:
            ax2.fill_between(
                epochs_attn,
                test_attn_s1['mean'] - test_attn_s1['std'],
                test_attn_s1['mean'] + test_attn_s1['std'],
                alpha=0.2, color='red'
            )
            ax2.axhline(
                test_attn_s1['mean'],
                color='red', linestyle=':', linewidth=2,
                label=f"Test (μ={test_attn_s1['mean']:.4f})"
            )
    
    ax2.set_title('UNetAttention - Stage 1', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Train/Val/Test Comparison - Stage 1 (U-Net vs UNetAttention)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def compare_unet_vs_unetattention_stage2(checkpoint_dir, chunk_duration, hist_unet_s2=None, hist_attn_s2=None):

    if not hist_unet_s2 or 'train_loss' not in hist_unet_s2:
        ckpt_unet_s2 = checkpoint_dir / f"model_a_unet_stage2_{chunk_duration:.0f}s.pth"
        hist_unet_s2 = load_training_history_from_checkpoint(ckpt_unet_s2)
    
    if not hist_attn_s2 or 'train_loss' not in hist_attn_s2:
        ckpt_attn_s2 = checkpoint_dir / f"model_unetattention_stage2_{chunk_duration:.0f}s.pth"
        hist_attn_s2 = load_training_history_from_checkpoint(ckpt_attn_s2)
    
    test_results_ckpt_unet_s2 = checkpoint_dir / f"test_results_unet_stage2_{chunk_duration:.0f}s.pkl"
    test_results_ckpt_attn_s2 = checkpoint_dir / f"test_results_unetattention_stage2_{chunk_duration:.0f}s.pkl"
    
    test_unet_s2 = load_test_results(test_results_ckpt_unet_s2)
    test_attn_s2 = load_test_results(test_results_ckpt_attn_s2)
    
    if not hist_unet_s2 or not hist_attn_s2:
        print("Could not plot comparison - missing training history for one or both models.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'train_loss' in hist_unet_s2:
        epochs_unet = range(1, len(hist_unet_s2['train_loss']) + 1)
        ax1.plot(epochs_unet, hist_unet_s2['train_loss'], 'o-', label='Train', linewidth=2)
        ax1.plot(epochs_unet, hist_unet_s2['val_loss'], 's--', label='Val', linewidth=2)
        
        if test_unet_s2 and 'mean' in test_unet_s2 and 'std' in test_unet_s2:
            ax1.fill_between(
                epochs_unet,
                test_unet_s2['mean'] - test_unet_s2['std'],
                test_unet_s2['mean'] + test_unet_s2['std'],
                alpha=0.2, color='red'
            )
            ax1.axhline(
                test_unet_s2['mean'],
                color='red', linestyle=':', linewidth=2,
                label=f"Test (μ={test_unet_s2['mean']:.4f})"
            )
    
    ax1.set_title('Standard U-Net - Stage 2', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if 'train_loss' in hist_attn_s2:
        epochs_attn = range(1, len(hist_attn_s2['train_loss']) + 1)
        ax2.plot(epochs_attn, hist_attn_s2['train_loss'], 'o-', label='Train', linewidth=2)
        ax2.plot(epochs_attn, hist_attn_s2['val_loss'], 's--', label='Val', linewidth=2)
        
        if test_attn_s2 and 'mean' in test_attn_s2 and 'std' in test_attn_s2:
            ax2.fill_between(
                epochs_attn,
                test_attn_s2['mean'] - test_attn_s2['std'],
                test_attn_s2['mean'] + test_attn_s2['std'],
                alpha=0.2, color='red'
            )
            ax2.axhline(
                test_attn_s2['mean'],
                color='red', linestyle=':', linewidth=2,
                label=f"Test (μ={test_attn_s2['mean']:.4f})"
            )
    
    ax2.set_title('UNetAttention - Stage 2', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Train/Val/Test Comparison - Stage 2 (U-Net vs UNetAttention)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def load_training_history_from_checkpoint(ckpt_path):
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

def evaluate_stage1_models(model_lstm, processor_lstm, loss_fn_lstm, model_unet, processor_unet, loss_fn_unet, 
                           ckpt_lstm_s1, ckpt_unet_s1, data_dir, checkpoint_dir, chunk_duration, device):

    test_results_ckpt_lstm = checkpoint_dir / f'test_results_lstm_stage1_{chunk_duration:.0f}s.pkl'
    test_results_ckpt_unet = checkpoint_dir / f'test_results_unet_stage1_{chunk_duration:.0f}s.pkl'
    
    test_lstm_s1 = load_test_results(test_results_ckpt_lstm)
    test_unet_s1 = load_test_results(test_results_ckpt_unet)
    
    if test_lstm_s1 and test_unet_s1:
        print(f"Loaded cached test results from checkpoints")
        print(f"   LSTM: {test_results_ckpt_lstm.name}")
        print(f"   U-Net: {test_results_ckpt_unet.name}")
    else:
        print("Running Stage 1 test evaluation (no cached results found)...\n")
        print("STAGE 1 TEST GENERALIZATION\n")
        
        model_lstm.to(device)
        model_unet.to(device)
        
        if ckpt_lstm_s1.exists():
            print(f"Loading LSTM Stage 1 weights for evaluation from: {ckpt_lstm_s1.name}")
            checkpoint = torch.load(ckpt_lstm_s1, map_location=device, weights_only=False)
            model_lstm.load_state_dict(checkpoint['model_state_dict'])
            print(f"LSTM weights loaded (trained for {checkpoint.get('epoch', '?')} epochs)")
        else:
            print("No LSTM checkpoint found - evaluating untrained model!")
        
        if ckpt_unet_s1.exists():
            print(f"Loading U-Net Stage 1 weights for evaluation from: {ckpt_unet_s1.name}")
            checkpoint = torch.load(ckpt_unet_s1, map_location=device, weights_only=False)
            model_unet.load_state_dict(checkpoint['model_state_dict'])
            print(f"U-Net weights loaded (trained for {checkpoint.get('epoch', '?')} epochs)")
        else:
            print("No U-Net checkpoint found - evaluating untrained model!")
        
        print("\n" + "="*70)
        print("EVALUATING LSTM MODEL")
        print("="*70)
        test_lstm_s1 = evaluate_test_set(model_lstm, processor_lstm, data_dir,
                                        'stage1', loss_fn_lstm, device)
        
        print("\n" + "="*70)
        print("EVALUATING U-NET MODEL")
        print("="*70)
        test_unet_s1 = evaluate_test_set(model_unet, processor_unet, data_dir,
                                        'stage1', loss_fn_unet, device)
        
        save_test_results(test_lstm_s1, test_results_ckpt_lstm)
        save_test_results(test_unet_s1, test_results_ckpt_unet)
    
    return test_lstm_s1, test_unet_s1

def plot_stage1_training_curves(hist_lstm_s1, hist_unet_s1, test_lstm_s1, test_unet_s1, checkpoint_dir, chunk_duration):

    if not hist_lstm_s1:
        ckpt_lstm = checkpoint_dir / f"model_a_lstm_stage1_{chunk_duration:.0f}s.pth"
        hist_lstm_s1 = load_training_history_from_checkpoint(ckpt_lstm)
    
    if not hist_unet_s1:
        ckpt_unet = checkpoint_dir / f"model_a_unet_stage1_{chunk_duration:.0f}s.pth"
        hist_unet_s1 = load_training_history_from_checkpoint(ckpt_unet)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if hist_lstm_s1 and 'train_loss' in hist_lstm_s1:
        epochs_lstm = range(1, len(hist_lstm_s1['train_loss']) + 1)
        ax1.plot(epochs_lstm, hist_lstm_s1['train_loss'], 'o-', label='Train', linewidth=2)
        ax1.plot(epochs_lstm, hist_lstm_s1['val_loss'], 's--', label='Val', linewidth=2)
        
        ax1.fill_between(epochs_lstm,
                         test_lstm_s1['mean'] - test_lstm_s1['std'],
                         test_lstm_s1['mean'] + test_lstm_s1['std'],
                         alpha=0.2, color='red')
    
    ax1.axhline(test_lstm_s1['mean'], color='red', linestyle=':', linewidth=2, 
                label=f"Test (μ={test_lstm_s1['mean']:.4f})")
    ax1.set_title('Model A (LSTM) - Stage 1', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if hist_unet_s1 and 'train_loss' in hist_unet_s1:
        epochs_unet = range(1, len(hist_unet_s1['train_loss']) + 1)
        ax2.plot(epochs_unet, hist_unet_s1['train_loss'], 'o-', label='Train', linewidth=2)
        ax2.plot(epochs_unet, hist_unet_s1['val_loss'], 's--', label='Val', linewidth=2)
        
        ax2.fill_between(epochs_unet,
                         test_unet_s1['mean'] - test_unet_s1['std'],
                         test_unet_s1['mean'] + test_unet_s1['std'],
                         alpha=0.2, color='red')
    
    ax2.axhline(test_unet_s1['mean'], color='red', linestyle=':', linewidth=2, 
                label=f"Test (μ={test_unet_s1['mean']:.4f})")
    ax2.set_title('Model A (U-Net) - Stage 1', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Train/Val/Test Comparison - Stage 1', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def evaluate_stage2_models(ckpt_lstm_s2, ckpt_unet_s2, data_dir, checkpoint_dir, chunk_duration, device):

    from models import models as ma
    
    test_results_ckpt_lstm = checkpoint_dir / f'test_results_lstm_stage2_{chunk_duration:.0f}s.pkl'
    test_results_ckpt_unet = checkpoint_dir / f'test_results_unet_stage2_{chunk_duration:.0f}s.pkl'
    
    test_lstm_s2 = load_test_results(test_results_ckpt_lstm)
    test_unet_s2 = load_test_results(test_results_ckpt_unet)
    
    if test_lstm_s2 and test_unet_s2:
        print(f"Loaded cached test results from checkpoints")
        print(f"   LSTM: {test_results_ckpt_lstm.name}")
        print(f"   U-Net: {test_results_ckpt_unet.name}")
    else:
        print("Running Stage 2 test evaluation (no cached results found)...\n")
        print("STAGE 2 TEST GENERALIZATION\n")
        
        model_lstm_s2, processor_lstm_s2, _, loss_fn_lstm_s2 = initialize_model_a_lstm(device)
        
        model_unet_s2 = ma.TimeFrequencyDomainUNet(
            in_channels=1,
            out_channels=1,
            base_filters=32,
            num_layers=5,
            batchnorm=True,
            dropout=0.1
        ).to(device)
        processor_unet_s2 = AudioProcessor(device=device)
        loss_fn_unet_s2 = torch.nn.MSELoss()
        
        if ckpt_lstm_s2.exists():
            print(f"Loading LSTM Stage 2 weights from: {ckpt_lstm_s2.name}")
            checkpoint = torch.load(ckpt_lstm_s2, map_location=device, weights_only=False)
            model_lstm_s2.load_state_dict(checkpoint['model_state_dict'])
            print(f"LSTM weights loaded (trained for {checkpoint.get('epoch', '?')} epochs)")
        else:
            print("No LSTM Stage 2 checkpoint found - evaluating untrained model!")
        
        if ckpt_unet_s2.exists():
            print(f"Loading U-Net Stage 2 weights from: {ckpt_unet_s2.name}")
            checkpoint = torch.load(ckpt_unet_s2, map_location=device, weights_only=False)
            model_unet_s2.load_state_dict(checkpoint['model_state_dict'])
            print(f"U-Net weights loaded (trained for {checkpoint.get('epoch', '?')} epochs)")
        else:
            print("No U-Net Stage 2 checkpoint found - evaluating untrained model!")
        
        print("\n" + "="*70)
        print("EVALUATING LSTM MODEL (STAGE 2)")
        print("="*70)
        test_lstm_s2 = evaluate_test_set(model_lstm_s2, processor_lstm_s2, data_dir,
                                        'stage2', loss_fn_lstm_s2, device)
        
        print("\n" + "="*70)
        print("EVALUATING U-NET MODEL (STAGE 2)")
        print("="*70)
        test_unet_s2 = evaluate_test_set(model_unet_s2, processor_unet_s2, data_dir,
                                        'stage2', loss_fn_unet_s2, device)
        
        save_test_results(test_lstm_s2, test_results_ckpt_lstm)
        save_test_results(test_unet_s2, test_results_ckpt_unet)
    
    return test_lstm_s2, test_unet_s2

def plot_stage2_training_curves(hist_lstm_s2, hist_unet_s2, test_lstm_s2, test_unet_s2, checkpoint_dir, chunk_duration, stage2_enabled=True):

    if not stage2_enabled:
        print("Stage 2 plotting skipped (STAGE2_ENABLED = False)")
        return
    
    if not hist_lstm_s2:
        ckpt_lstm = checkpoint_dir / f"model_a_lstm_stage2_{chunk_duration:.0f}s.pth"
        hist_lstm_s2 = load_training_history_from_checkpoint(ckpt_lstm)
    
    if not hist_unet_s2:
        ckpt_unet = checkpoint_dir / f"model_a_unet_stage2_{chunk_duration:.0f}s.pth"
        hist_unet_s2 = load_training_history_from_checkpoint(ckpt_unet)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if hist_lstm_s2 and 'train_loss' in hist_lstm_s2:
        epochs_lstm = range(1, len(hist_lstm_s2['train_loss']) + 1)
        ax1.plot(epochs_lstm, hist_lstm_s2['train_loss'], 'o-', label='Train', linewidth=2)
        ax1.plot(epochs_lstm, hist_lstm_s2['val_loss'], 's--', label='Val', linewidth=2)
        
        ax1.fill_between(epochs_lstm,
                         test_lstm_s2['mean'] - test_lstm_s2['std'],
                         test_lstm_s2['mean'] + test_lstm_s2['std'],
                         alpha=0.2, color='red')
    
    ax1.axhline(test_lstm_s2['mean'], color='red', linestyle=':', linewidth=2, 
                label=f"Test (μ={test_lstm_s2['mean']:.4f})")
    ax1.set_title('Model A (LSTM) - Stage 2', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if hist_unet_s2 and 'train_loss' in hist_unet_s2:
        epochs_unet = range(1, len(hist_unet_s2['train_loss']) + 1)
        ax2.plot(epochs_unet, hist_unet_s2['train_loss'], 'o-', label='Train', linewidth=2)
        ax2.plot(epochs_unet, hist_unet_s2['val_loss'], 's--', label='Val', linewidth=2)
        
        ax2.fill_between(epochs_unet,
                         test_unet_s2['mean'] - test_unet_s2['std'],
                         test_unet_s2['mean'] + test_unet_s2['std'],
                         alpha=0.2, color='red')
    
    ax2.axhline(test_unet_s2['mean'], color='red', linestyle=':', linewidth=2, 
                label=f"Test (μ={test_unet_s2['mean']:.4f})")
    ax2.set_title('Model A (U-Net) - Stage 2', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Train/Val/Test Comparison - Stage 2 (Full Mix -> Vocals)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def load_and_stitch_test_chunks(data_dir, stage='stage1', sr=22050, duration=120.0, hop_length=4.0, auto_select=False):

    test_base = data_dir / stage / 'test'
    print(f"\nSearching for {stage.upper()} test data at: {test_base}")
    
    if (test_base / 'mixture').exists():
        mix_dir = test_base / 'mixture'
        print(f"Found subdirectories: mixture/ and target/")
    else:
        print(f"ERROR: {stage.upper()} Test directory not found!")
        raise FileNotFoundError(f"Test data directory not found: {test_base}")
    
    all_mix_files = sorted(list(mix_dir.glob('*.npy')))
    if len(all_mix_files) == 0:
        raise FileNotFoundError("No .npy files found")
    
    songs = {}
    for f in all_mix_files:
        name = f.stem
        if '_chunk' in name:
            parts = name.rsplit('_chunk', 1)
            if len(parts) == 2 and parts[1].isdigit():
                song_name = parts[0]
                idx = int(parts[1])
                if song_name not in songs: songs[song_name] = {}
                songs[song_name][idx] = f
        else:
            parts = name.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                idx = int(parts[-1])
                song_name = '_'.join(parts[:-1])
                if song_name not in songs: songs[song_name] = {}
                songs[song_name][idx] = f
    
    song_list = sorted(songs.keys())
    
    if len(song_list) == 1 or auto_select:
        selected = song_list[0]
        print(f"\nAuto-selected: {selected}")
    else:
        print(f"\nSelect a song [0-{len(song_list)-1}] or press Enter for default (0):")
        try:
            choice = input("Choice: ").strip()
            idx = int(choice) if choice else 0
        except:
            idx = 0
        selected = song_list[idx]
        print(f"Selected: {selected}")
    
    chunks = songs[selected]
    
    print(f"Stitching chunks...")
    hop_samples = int(hop_length * sr)
    target_samples = int(duration * sr)
    mix_wav = np.zeros(target_samples, dtype=np.float32)
    tgt_wav = np.zeros(target_samples, dtype=np.float32)
    weights = np.zeros(target_samples, dtype=np.float32)
    has_target = True
    
    for i in sorted(chunks.keys()):
        pos = i * hop_samples
        if pos >= target_samples: break
        
        mix_chunk = np.load(chunks[i])
        tgt_file = chunks[i].name.replace('mix_', 'tgt_')
        tgt_path = test_base / 'target' / tgt_file
        
        if tgt_path.exists():
            tgt_chunk = np.load(tgt_path)
        else:
            tgt_chunk = np.zeros_like(mix_chunk)
            has_target = False
        
        valid_len = min(len(mix_chunk), target_samples - pos)
        window = np.hanning(len(mix_chunk))[:valid_len]
        
        mix_wav[pos:pos+valid_len] += mix_chunk[:valid_len] * window
        tgt_wav[pos:pos+valid_len] += tgt_chunk[:valid_len] * window
        weights[pos:pos+valid_len] += window
    
    mix_wav = np.divide(mix_wav, weights, where=weights > 0)
    tgt_wav = np.divide(tgt_wav, weights, where=weights > 0)
    
    return mix_wav, tgt_wav, has_target, selected

def evaluate_and_visualize_stage1(model_lstm, processor_lstm, model_unet, processor_unet, 
                                   mix_wav, tgt_wav, has_target, selected_song, 
                                   sr=22050, chunk_len=8.0, device='cuda'):

    from IPython.display import Audio, display
    
    model_lstm = model_lstm.to(device)
    model_unet = model_unet.to(device)
    model_lstm.eval()
    model_unet.eval()
    
    print("\nRunning Stage 1 inference...")
    est_lstm_s1 = sliding_window_inference(model_lstm, processor_lstm, mix_wav, chunk_len=chunk_len, sr=sr, device=device)
    est_unet_s1 = sliding_window_inference(model_unet, processor_unet, mix_wav, chunk_len=chunk_len, sr=sr, device=device)
    
    print(f"\n{'='*70}\nSTAGE 1 RESULTS\n{'='*70}")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    axes[0,0].imshow(to_spec(mix_wav, processor_lstm), aspect='auto', origin='lower', cmap='viridis')
    axes[0,0].set_title("Input: Simplified Mixture (Vocals+Other)", fontweight='bold')
    
    if has_target:
        axes[0,1].imshow(to_spec(tgt_wav, processor_lstm), aspect='auto', origin='lower', cmap='viridis')
        axes[0,1].set_title("Ground Truth: Other", fontweight='bold')
    else:
        axes[0,1].text(0.5, 0.5, "Target Not Available", ha='center', va='center', transform=axes[0,1].transAxes)
    
    spec_tgt_lstm = to_spec(tgt_wav, processor_lstm)
    spec_pred_lstm = to_spec(est_lstm_s1, processor_lstm)
    min_len = min(spec_tgt_lstm.shape[1], spec_pred_lstm.shape[1])
    err_lstm = np.abs(spec_pred_lstm[:, :min_len] - spec_tgt_lstm[:, :min_len])
    
    axes[1,0].imshow(err_lstm, aspect='auto', origin='lower', cmap='magma')
    axes[1,0].set_title("LSTM Error Map (|Pred - Tgt|)", fontweight='bold')
    
    axes[1,1].imshow(spec_pred_lstm, aspect='auto', origin='lower', cmap='viridis')
    axes[1,1].set_title("LSTM Prediction (Stage 1)", fontweight='bold')
    
    spec_tgt_unet = to_spec(tgt_wav, processor_unet)
    spec_pred_unet = to_spec(est_unet_s1, processor_unet)
    min_len_u = min(spec_tgt_unet.shape[1], spec_pred_unet.shape[1])
    err_unet = np.abs(spec_pred_unet[:, :min_len_u] - spec_tgt_unet[:, :min_len_u])
    
    axes[2,0].imshow(err_unet, aspect='auto', origin='lower', cmap='magma')
    axes[2,0].set_title("U-Net Error Map (|Pred - Tgt|)", fontweight='bold')
    
    axes[2,1].imshow(spec_pred_unet, aspect='auto', origin='lower', cmap='viridis')
    axes[2,1].set_title("U-Net Prediction (Stage 1)", fontweight='bold')
    
    for ax in axes.flatten():
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
    
    plt.suptitle(f"Stage 1 Evaluation: {selected_song}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nAudio Playback:")
    print("Input (Mix):")
    display(Audio(mix_wav, rate=sr))
    
    print("\nGround Truth (Target):")
    display(Audio(tgt_wav, rate=sr))
    
    print("\nLSTM Prediction:")
    display(Audio(est_lstm_s1, rate=sr))
    
    print("\nU-Net Prediction:")
    display(Audio(est_unet_s1, rate=sr))
    
    return est_lstm_s1, est_unet_s1

def evaluate_and_visualize_stage2(model_lstm, processor_lstm, model_unet, processor_unet, 
                                   mix_wav, tgt_wav, has_target, selected_song, 
                                   sr=22050, chunk_len=8.0, device='cuda'):

    from IPython.display import Audio, display
    
    model_lstm = model_lstm.to(device)
    model_unet = model_unet.to(device)
    model_lstm.eval()
    model_unet.eval()
    
    print("\nRunning Stage 2 inference...")
    est_lstm_s2 = sliding_window_inference(model_lstm, processor_lstm, mix_wav, chunk_len=chunk_len, sr=sr, device=device)
    est_unet_s2 = sliding_window_inference(model_unet, processor_unet, mix_wav, chunk_len=chunk_len, sr=sr, device=device)
    
    print(f"\n{'='*70}\nSTAGE 2 RESULTS\n{'='*70}")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    axes[0,0].imshow(to_spec(mix_wav, processor_lstm), aspect='auto', origin='lower', cmap='viridis')
    axes[0,0].set_title("Input: Full Band Mixture", fontweight='bold')
    
    if has_target:
        axes[0,1].imshow(to_spec(tgt_wav, processor_lstm), aspect='auto', origin='lower', cmap='viridis')
        axes[0,1].set_title("Ground Truth: Vocals", fontweight='bold')
    else:
        axes[0,1].text(0.5, 0.5, "Target Not Available", ha='center', va='center', transform=axes[0,1].transAxes)
    
    spec_tgt_lstm = to_spec(tgt_wav, processor_lstm)
    spec_pred_lstm = to_spec(est_lstm_s2, processor_lstm)
    min_len = min(spec_tgt_lstm.shape[1], spec_pred_lstm.shape[1])
    err_lstm = np.abs(spec_pred_lstm[:, :min_len] - spec_tgt_lstm[:, :min_len])
    
    axes[1,0].imshow(err_lstm, aspect='auto', origin='lower', cmap='magma')
    axes[1,0].set_title("LSTM Error Map (|Pred - Tgt|)", fontweight='bold')
    
    axes[1,1].imshow(spec_pred_lstm, aspect='auto', origin='lower', cmap='viridis')
    axes[1,1].set_title("LSTM Prediction (Stage 2)", fontweight='bold')
    
    spec_tgt_unet = to_spec(tgt_wav, processor_unet)
    spec_pred_unet = to_spec(est_unet_s2, processor_unet)
    min_len_u = min(spec_tgt_unet.shape[1], spec_pred_unet.shape[1])
    err_unet = np.abs(spec_pred_unet[:, :min_len_u] - spec_tgt_unet[:, :min_len_u])
    
    axes[2,0].imshow(err_unet, aspect='auto', origin='lower', cmap='magma')
    axes[2,0].set_title("U-Net Error Map (|Pred - Tgt|)", fontweight='bold')
    
    axes[2,1].imshow(spec_pred_unet, aspect='auto', origin='lower', cmap='viridis')
    axes[2,1].set_title("U-Net Prediction (Stage 2)", fontweight='bold')
    
    for ax in axes.flatten():
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
    
    plt.suptitle(f"Stage 2 Evaluation: {selected_song}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nAudio Playback:")
    print("Input (Mix):")
    display(Audio(mix_wav, rate=sr))
    
    print("\nGround Truth (Target):")
    display(Audio(tgt_wav, rate=sr))
    
    print("\nLSTM Prediction:")
    display(Audio(est_lstm_s2, rate=sr))
    
    print("\nU-Net Prediction:")
    display(Audio(est_unet_s2, rate=sr))
    
    return est_lstm_s2, est_unet_s2

def plot_model_comparison(hist_lstm, hist_unet, title="Model Comparison: LSTM vs U-Net"):

    if not hist_lstm or not hist_unet:
        print("Training histories not available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    epochs_lstm = range(1, len(hist_lstm['train_loss']) + 1)
    ax1.plot(epochs_lstm, hist_lstm['train_loss'], 'o-', label='Train', linewidth=2, markersize=5)
    ax1.plot(epochs_lstm, hist_lstm['val_loss'], 's--', label='Val', linewidth=2, markersize=5)
    ax1.set_title('Model (LSTM)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    epochs_unet = range(1, len(hist_unet['train_loss']) + 1)
    ax2.plot(epochs_unet, hist_unet['train_loss'], 'o-', label='Train', linewidth=2, markersize=5)
    ax2.plot(epochs_unet, hist_unet['val_loss'], 's--', label='Val', linewidth=2, markersize=5)
    ax2.set_title('Model (U-Net)', fontsize=14, fontweight='bold')
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
    
    print(f"\nModel (LSTM):")
    print(f"   Epochs: {len(hist_lstm['train_loss'])}")
    print(f"   Best Val Loss: {min(hist_lstm['val_loss']):.6f} (epoch {hist_lstm['val_loss'].index(min(hist_lstm['val_loss']))+1})")
    print(f"   Final Train Loss: {hist_lstm['train_loss'][-1]:.6f}")
    
    print(f"\nModel (U-Net):")
    print(f"   Epochs: {len(hist_unet['train_loss'])}")
    print(f"   Best Val Loss: {min(hist_unet['val_loss']):.6f} (epoch {hist_unet['val_loss'].index(min(hist_unet['val_loss']))+1})")
    print(f"   Final Train Loss: {hist_unet['train_loss'][-1]:.6f}")
    
    winner = "LSTM" if min(hist_lstm['val_loss']) < min(hist_unet['val_loss']) else "U-Net"
    print(f"\nBest Performance: {winner}")

def evaluate_separation_quality(model_1=None, model_2=None, processor_1=None, processor_2=None,
                                test_data_dir=None, stage='stage1', num_samples=10, sr=22050, device='cuda',
                                save_path=None, load_if_exists=True,
                                random_sampling=False, random_seed=42, **kwargs):

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
                    _print_and_plot_metrics(m1_cached, m2_cached, heading="EVALUATION RESULTS")
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
        
        min_len = min(len(mix_wav), len(tgt_wav))
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

    import pickle
    if ckpt_path.exists():
        with open(ckpt_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_test_results(results, ckpt_path):

    import pickle
    with open(ckpt_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved test results to: {ckpt_path.name}")

def evaluate_test_set(model, processor, test_data_dir, stage, loss_fn, device, sr=22050):

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

    chunk_samples = int(chunk_len * sr)
    stride = chunk_samples // 2
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

    s = processor.to_spectrogram(wav)[0].squeeze().cpu().numpy()
    return s if s.shape[0] < s.shape[1] else s.T

def unet_inference_accelerator(model, processor, audio, chunk_len=8.0, sr=22050, device='cuda', batch_size=16):

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
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks_data[batch_start:batch_end]
            
            batch_mags = torch.stack([chunk['mag'].unsqueeze(0) for chunk in batch_chunks]).to(device)
            
            batch_masks = model(batch_mags)
            
            for i, chunk in enumerate(batch_chunks):
                chunk_idx = batch_start + i
                if chunk_idx % 5 == 0:
                    print(".", end="", flush=True)
                
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
    print("LSTM:")
    est_lstm = sliding_window_inference(model_lstm, processor_lstm, audio, 
                                       chunk_len=8.0, sr=sr, device=device)
    print(f"U-Net:")
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

def handle_user_upload_and_inference(data_dir, model_lstm, model_unet, processor_lstm, 
                                      processor_unet, device, in_colab=False, sr=22050, 
                                      duration=None, unet_batch_size=16):

    from pathlib import Path
    
    upload_dir = Path(data_dir) / "user_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    if in_colab:
        try:
            from google.colab import files
            uploaded = files.upload()
            for name, data in uploaded.items():
                out_path = upload_dir / name
                with open(out_path, "wb") as f:
                    f.write(data)
                print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Colab upload failed: {e}")
            print(f"Place a file manually in: {upload_dir}")
    else:
        print(f"Place your audio file in: {upload_dir}")
    
    exts = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]
    audio_files = []
    for ext in exts:
        audio_files += list(upload_dir.glob(ext))
    
    audio_files = sorted(audio_files, key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not audio_files:
        print("No audio files found in upload folder.")
        return
    
    audio_path = audio_files[0]
    print(f"Using file: {audio_path.name}")
    
    compare_models_on_audio_file(
        file_path=audio_path,
        model_lstm=model_lstm,
        model_unet=model_unet,
        processor_lstm=processor_lstm,
        processor_unet=processor_unet,
        device=device,
        sr=sr,
        duration=duration,
        unet_batch_size=unet_batch_size
    )

def compare_unet_vs_unetattention_on_audio_file(
    file_path, model_unet, model_attn, processor_unet, processor_attn, 
    device, sr=22050, duration=None, chunk_len=8.0):

    import librosa
    import matplotlib.pyplot as plt
    from IPython.display import Audio, display
    
    print("="*70)
    print("CUSTOM AUDIO INFERENCE: U-Net vs UNetAttention (Stage 2)")
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
    
    model_unet.eval()
    model_attn.eval()
    
    print("\nRunning inference...")
    print("   - Standard U-Net (Stage 2)")
    est_unet = sliding_window_inference(
        model_unet, processor_unet, audio, 
        chunk_len=chunk_len, sr=sr, device=device
    )
    
    print("   - UNetAttention (Stage 2)")
    est_attn = sliding_window_inference(
        model_attn, processor_attn, audio,
        chunk_len=chunk_len, sr=sr, device=device
    )
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON: U-Net vs UNetAttention (Stage 2)")
    print(f"{'='*70}\n")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    axes[0].imshow(to_spec(audio, processor_unet), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("Original Mix", fontweight='bold')
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlabel("Time")
    
    axes[1].imshow(to_spec(est_unet, processor_unet), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title("U-Net Output (Stage 2)", fontweight='bold')
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlabel("Time")
    
    axes[2].imshow(to_spec(est_attn, processor_attn), aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title("UNetAttention Output (Stage 2)", fontweight='bold')
    axes[2].set_ylabel("Frequency")
    axes[2].set_xlabel("Time")
    
    plt.tight_layout()
    plt.show()
    
    print("Audio results:\n")
    print("Original Mix:")
    display(Audio(audio, rate=sr))
    
    print("\nU-Net Output:")
    display(Audio(est_unet, rate=sr))
    
    print("\nUNetAttention Output:")
    display(Audio(est_attn, rate=sr))
    
    print(f"\n{'='*70}")
    print("INFERENCE COMPLETE")
    print(f"{'='*70}")

def handle_user_upload_unetattention_inference(
    data_dir, model_unet, model_attn, processor_unet, processor_attn,
    device, in_colab=False, sr=22050, duration=None, chunk_len=8.0):

    from pathlib import Path
    
    upload_dir = Path(data_dir) / "user_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    if in_colab:
        try:
            from google.colab import files
            uploaded = files.upload()
            for name, data in uploaded.items():
                out_path = upload_dir / name
                with open(out_path, "wb") as f:
                    f.write(data)
                print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Colab upload failed: {e}")
            print(f"Place a file manually in: {upload_dir}")
    else:
        print(f"Place your audio file in: {upload_dir}")
    
    exts = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]
    audio_files = []
    for ext in exts:
        audio_files += list(upload_dir.glob(ext))
    
    audio_files = sorted(audio_files, key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not audio_files:
        print("No audio files found in upload folder.")
        return
    
    audio_path = audio_files[0]
    print(f"Using file: {audio_path.name}")
    
    compare_unet_vs_unetattention_on_audio_file(
        file_path=audio_path,
        model_unet=model_unet,
        model_attn=model_attn,
        processor_unet=processor_unet,
        processor_attn=processor_attn,
        device=device,
        sr=sr,
        duration=duration,
        chunk_len=chunk_len
    )

def evaluate_and_visualize_unet_vs_unetattention_stage1(
    model_unet, processor_unet, model_attn, processor_attn,
    mix_wav, tgt_wav, has_target, selected_song,
    sr=22050, chunk_len=8.0, device='cuda'):
    
    from IPython.display import Audio, display
    
    print("="*70)
    print("STAGE 1 EVALUATION: U-Net vs UNetAttention (Simplified Mixture → Other)")
    print("="*70)
    
    model_unet = model_unet.to(device)
    model_attn = model_attn.to(device)
    model_unet.eval()
    model_attn.eval()
    
    print("\nRunning Stage 1 inference...")
    est_unet = sliding_window_inference(
        model_unet, processor_unet, mix_wav, chunk_len=chunk_len, sr=sr, device=device
    )
    est_attn = sliding_window_inference(
        model_attn, processor_attn, mix_wav, chunk_len=chunk_len, sr=sr, device=device
    )
    
    print(f"\n{'='*70}\nSTAGE 1 RESULTS: U-Net vs UNetAttention\n{'='*70}")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    axes[0,0].imshow(to_spec(mix_wav, processor_unet), aspect='auto', origin='lower', cmap='viridis')
    axes[0,0].set_title("Input: Simplified Mixture", fontweight='bold')
    
    if has_target:
        axes[0,1].imshow(to_spec(tgt_wav, processor_unet), aspect='auto', origin='lower', cmap='viridis')
        axes[0,1].set_title("Ground Truth: Other", fontweight='bold')
    else:
        axes[0,1].text(0.5, 0.5, "Target Not Available", ha='center', va='center', transform=axes[0,1].transAxes)
    
    spec_tgt_unet = to_spec(tgt_wav, processor_unet)
    spec_pred_unet = to_spec(est_unet, processor_unet)
    min_len_u = min(spec_tgt_unet.shape[1], spec_pred_unet.shape[1])
    err_unet = np.abs(spec_pred_unet[:, :min_len_u] - spec_tgt_unet[:, :min_len_u])
    
    axes[1,0].imshow(err_unet, aspect='auto', origin='lower', cmap='magma')
    axes[1,0].set_title("U-Net Error Map (|Pred - Tgt|)", fontweight='bold')
    
    axes[1,1].imshow(spec_pred_unet, aspect='auto', origin='lower', cmap='viridis')
    axes[1,1].set_title("U-Net Prediction (Stage 1)", fontweight='bold')
    
    spec_tgt_attn = to_spec(tgt_wav, processor_attn)
    spec_pred_attn = to_spec(est_attn, processor_attn)
    min_len_a = min(spec_tgt_attn.shape[1], spec_pred_attn.shape[1])
    err_attn = np.abs(spec_pred_attn[:, :min_len_a] - spec_tgt_attn[:, :min_len_a])
    
    axes[2,0].imshow(err_attn, aspect='auto', origin='lower', cmap='magma')
    axes[2,0].set_title("UNetAttention Error Map (|Pred - Tgt|)", fontweight='bold')
    
    axes[2,1].imshow(spec_pred_attn, aspect='auto', origin='lower', cmap='viridis')
    axes[2,1].set_title("UNetAttention Prediction (Stage 1)", fontweight='bold')
    
    for ax in axes.flatten():
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
    
    plt.suptitle(f"Stage 1 Evaluation: {selected_song}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nAudio Playback:")
    print("Input (Mix):")
    display(Audio(mix_wav, rate=sr))
    
    if has_target:
        print("\nGround Truth (Target):")
        display(Audio(tgt_wav, rate=sr))
    
    print("\nU-Net Prediction:")
    display(Audio(est_unet, rate=sr))
    
    print("\nUNetAttention Prediction:")
    display(Audio(est_attn, rate=sr))
    
    return est_unet, est_attn

def evaluate_and_visualize_unet_vs_unetattention_stage2(
    model_unet, processor_unet, model_attn, processor_attn,
    mix_wav, tgt_wav, has_target, selected_song,
    sr=22050, chunk_len=8.0, device='cuda'):
    
    from IPython.display import Audio, display
    
    print("="*70)
    print("STAGE 2 EVALUATION: U-Net vs UNetAttention (Full Band Mixture → Vocals)")
    print("="*70)
    
    model_unet = model_unet.to(device)
    model_attn = model_attn.to(device)
    model_unet.eval()
    model_attn.eval()
    
    print("\nRunning Stage 2 inference...")
    est_unet = sliding_window_inference(
        model_unet, processor_unet, mix_wav, chunk_len=chunk_len, sr=sr, device=device
    )
    est_attn = sliding_window_inference(
        model_attn, processor_attn, mix_wav, chunk_len=chunk_len, sr=sr, device=device
    )
    
    print(f"\n{'='*70}\nSTAGE 2 RESULTS: U-Net vs UNetAttention\n{'='*70}")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    axes[0,0].imshow(to_spec(mix_wav, processor_unet), aspect='auto', origin='lower', cmap='viridis')
    axes[0,0].set_title("Input: Full Band Mixture", fontweight='bold')
    
    if has_target:
        axes[0,1].imshow(to_spec(tgt_wav, processor_unet), aspect='auto', origin='lower', cmap='viridis')
        axes[0,1].set_title("Ground Truth: Vocals", fontweight='bold')
    else:
        axes[0,1].text(0.5, 0.5, "Target Not Available", ha='center', va='center', transform=axes[0,1].transAxes)
    
    spec_tgt_unet = to_spec(tgt_wav, processor_unet)
    spec_pred_unet = to_spec(est_unet, processor_unet)
    min_len_u = min(spec_tgt_unet.shape[1], spec_pred_unet.shape[1])
    err_unet = np.abs(spec_pred_unet[:, :min_len_u] - spec_tgt_unet[:, :min_len_u])
    
    axes[1,0].imshow(err_unet, aspect='auto', origin='lower', cmap='magma')
    axes[1,0].set_title("U-Net Error Map (|Pred - Tgt|)", fontweight='bold')
    
    axes[1,1].imshow(spec_pred_unet, aspect='auto', origin='lower', cmap='viridis')
    axes[1,1].set_title("U-Net Prediction (Stage 2)", fontweight='bold')
    
    spec_tgt_attn = to_spec(tgt_wav, processor_attn)
    spec_pred_attn = to_spec(est_attn, processor_attn)
    min_len_a = min(spec_tgt_attn.shape[1], spec_pred_attn.shape[1])
    err_attn = np.abs(spec_pred_attn[:, :min_len_a] - spec_tgt_attn[:, :min_len_a])
    
    axes[2,0].imshow(err_attn, aspect='auto', origin='lower', cmap='magma')
    axes[2,0].set_title("UNetAttention Error Map (|Pred - Tgt|)", fontweight='bold')
    
    axes[2,1].imshow(spec_pred_attn, aspect='auto', origin='lower', cmap='viridis')
    axes[2,1].set_title("UNetAttention Prediction (Stage 2)", fontweight='bold')
    
    for ax in axes.flatten():
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
    
    plt.suptitle(f"Stage 2 Evaluation: {selected_song}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nAudio Playback:")
    print("Input (Mix):")
    display(Audio(mix_wav, rate=sr))
    
    if has_target:
        print("\nGround Truth (Target):")
        display(Audio(tgt_wav, rate=sr))
    
    print("\nU-Net Prediction:")
    display(Audio(est_unet, rate=sr))
    
    print("\nUNetAttention Prediction:")
    display(Audio(est_attn, rate=sr))
    
    return est_unet, est_attn

def train_unetattention_stage2(data_dir, checkpoint_dir, device, chunk_duration=8.0,
                                skip_training=False, batch_size=32, base_filters=32,
                                num_layers=4, num_heads=4, learning_rate=None,
                                ckpt_attn_s1=None):
    
    from models import models as ma
    from pathlib import Path
    
    print(f"\n{'='*70}")
    print('STAGE 2 TRAINING: UNetAttention (Full Mix -> Target)')
    print(f"{'='*70}\n")
    
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_attn_s2 = checkpoint_dir / f'model_unetattention_stage2_{chunk_duration:.0f}s.pth'
    
    if ckpt_attn_s1 is None:
        ckpt_attn_s1 = checkpoint_dir / f'model_unetattention_stage1_{chunk_duration:.0f}s.pth'
    
    if ckpt_attn_s2.exists():
        print(f"UNetAttention Stage 2 checkpoint found: {ckpt_attn_s2.name}")
    
    if skip_training and ckpt_attn_s2.exists():
        print("Skipping training (skip_training=True and checkpoint exists)")
        
        print(f"Initializing UNetAttention ({base_filters} filters, {num_layers} layers, {num_heads} heads)...")
        model_attn_s2 = ma.UNetAttention(
            in_channels=1,
            out_channels=1,
            base_filters=base_filters,
            num_layers=num_layers,
            num_heads=num_heads,
            batchnorm=True,
            dropout=0.1
        ).to(device)
        
        checkpoint = torch.load(ckpt_attn_s2, map_location=device)
        model_attn_s2.load_state_dict(checkpoint['model_state_dict'])
        hist_attn_s2 = checkpoint.get('history', {})
        
        processor_attn_s2 = AudioProcessor(device=device)
        loss_fn_attn_s2 = nn.MSELoss()
        
        print("Model loaded from checkpoint")
    else:
        attn_config = get_training_config('unet')
        attn_config['batch_size'] = batch_size
        
        if learning_rate is not None:
            attn_config['learning_rate'] = learning_rate
        
        print(f"Initializing UNetAttention ({base_filters} filters, {num_layers} layers, {num_heads} heads)...")
        model_attn_s2 = ma.UNetAttention(
            in_channels=1,
            out_channels=1,
            base_filters=base_filters,
            num_layers=num_layers,
            num_heads=num_heads,
            batchnorm=True,
            dropout=0.1
        ).to(device)
        
        if Path(ckpt_attn_s1).exists():
            print(f"Loading Stage 1 weights from: {Path(ckpt_attn_s1).name}")
            checkpoint = torch.load(ckpt_attn_s1, map_location=device)
            model_attn_s2.load_state_dict(checkpoint['model_state_dict'])
            print("Curriculum Learning: Stage 1 weights loaded.")
        else:
            print("Stage 1 checkpoint not found! Training from scratch (harder convergence).")
        
        processor_attn_s2 = AudioProcessor(device=device)
        optimizer_attn_s2 = torch.optim.Adam(model_attn_s2.parameters(), lr=attn_config['learning_rate'])
        loss_fn_attn_s2 = nn.MSELoss()
        
        hist_attn_s2 = train_model_stage(
            model=model_attn_s2,
            processor=processor_attn_s2,
            optimizer=optimizer_attn_s2,
            loss_fn=loss_fn_attn_s2,
            training_data_dir=data_dir,
            stage='stage2',
            ckpt_path=ckpt_attn_s2,
            device=device,
            train_config=attn_config,
            skip_training=False
        )
    
    return {
        'model_attn_s2': model_attn_s2,
        'processor_attn_s2': processor_attn_s2,
        'loss_fn_attn_s2': loss_fn_attn_s2,
        'hist_attn_s2': hist_attn_s2,
        'ckpt_attn_s2': ckpt_attn_s2
    }

def train_unetattention_vocals_only(data_dir, checkpoint_dir, device, 
                                     skip_training=False, batch_size=4, 
                                     base_filters=32, num_layers=5, num_heads=4,
                                     learning_rate=1e-4, num_epochs=20):
    
    from models import models as ma
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    print(f"\n{'='*70}")
    print('TRAINING: UNetAttention (Stage 2 Mix -> Clean Vocals)')
    print(f"{'='*70}\n")
    
    train_mix_dir = data_dir / 'stage2' / 'train' / 'mixture'
    train_voc_dir = data_dir / 'vocals' / 'train'
    val_mix_dir = data_dir / 'stage2' / 'val' / 'mixture'
    val_voc_dir = data_dir / 'vocals' / 'val'
    
    ckpt_voc_attn = checkpoint_dir / 'model_unetattention_vocals_only.pth'
    
    def get_paired_files(mix_dir, voc_dir, split_name):
        mix_files = sorted(list(mix_dir.glob('*.npy')))
        voc_files_dict = {f.name: f for f in voc_dir.glob('*.npy')}
        valid_pairs = [(mf, voc_files_dict[mf.name]) for mf in mix_files if mf.name in voc_files_dict]
        print(f"   {split_name}: Found {len(valid_pairs)} pairs")
        if not valid_pairs:
            return [], []
        mix_list = [p[0] for p in valid_pairs]
        tgt_list = [p[1] for p in valid_pairs]
        return mix_list, tgt_list
    
    if ckpt_voc_attn.exists():
        print(f"Checkpoint found: {ckpt_voc_attn.name}")
        if skip_training:
            print("Skipping training (skip_training=True and checkpoint exists)")
            
            model_voc_attn = ma.UNetAttention(
                in_channels=1,
                out_channels=1,
                base_filters=base_filters,
                num_layers=num_layers,
                num_heads=num_heads,
                batchnorm=True,
                dropout=0.1
            ).to(device)
            
            checkpoint = torch.load(ckpt_voc_attn, map_location=device, weights_only=False)
            model_voc_attn.load_state_dict(checkpoint['model_state_dict'])
            voc_history = checkpoint.get('history', {})
            
            processor_voc = AudioProcessor(device=device)
            loss_fn_voc = nn.MSELoss()
            
            print("Model loaded from checkpoint")
            
            return {
                'model_voc_attn': model_voc_attn,
                'processor_voc': processor_voc,
                'loss_fn_voc': loss_fn_voc,
                'voc_history': voc_history,
                'ckpt_voc_attn': ckpt_voc_attn
            }
    
    if not train_mix_dir.exists() or not train_voc_dir.exists():
        print("Training folders not found (data_sub mode detected).")
        if ckpt_voc_attn.exists():
            print(f"Using existing checkpoint: {ckpt_voc_attn.name}")
            
            model_voc_attn = ma.UNetAttention(
                in_channels=1,
                out_channels=1,
                base_filters=base_filters,
                num_layers=num_layers,
                num_heads=num_heads,
                batchnorm=True,
                dropout=0.1
            ).to(device)
            
            checkpoint = torch.load(ckpt_voc_attn, map_location=device, weights_only=False)
            model_voc_attn.load_state_dict(checkpoint['model_state_dict'])
            voc_history = checkpoint.get('history', {})
            
            processor_voc = AudioProcessor(device=device)
            loss_fn_voc = nn.MSELoss()
            
            return {
                'model_voc_attn': model_voc_attn,
                'processor_voc': processor_voc,
                'loss_fn_voc': loss_fn_voc,
                'voc_history': voc_history,
                'ckpt_voc_attn': ckpt_voc_attn
            }
        else:
            print("No checkpoint found, and training data is unavailable.")
            print("Provide full data (train/val) or place a pretrained checkpoint in checkpoints/.")
            raise FileNotFoundError("Cannot train: no training data and no existing checkpoint.")
    
    print("Matching files...")
    train_mix, train_tgt = get_paired_files(train_mix_dir, train_voc_dir, "Train")
    
    if val_mix_dir.exists() and val_voc_dir.exists():
        val_mix, val_tgt = get_paired_files(val_mix_dir, val_voc_dir, "Validation")
    else:
        print("Validation folders not found. Reusing train pairs for validation.")
        val_mix, val_tgt = train_mix, train_tgt
    
    if not train_mix:
        print("No training pairs found.")
        if ckpt_voc_attn.exists():
            print(f"Using existing checkpoint: {ckpt_voc_attn.name}")
            
            model_voc_attn = ma.UNetAttention(
                in_channels=1,
                out_channels=1,
                base_filters=base_filters,
                num_layers=num_layers,
                num_heads=num_heads,
                batchnorm=True,
                dropout=0.1
            ).to(device)
            
            checkpoint = torch.load(ckpt_voc_attn, map_location=device, weights_only=False)
            model_voc_attn.load_state_dict(checkpoint['model_state_dict'])
            voc_history = checkpoint.get('history', {})
            
            processor_voc = AudioProcessor(device=device)
            loss_fn_voc = nn.MSELoss()
            
            return {
                'model_voc_attn': model_voc_attn,
                'processor_voc': processor_voc,
                'loss_fn_voc': loss_fn_voc,
                'voc_history': voc_history,
                'ckpt_voc_attn': ckpt_voc_attn
            }
        else:
            raise RuntimeError("Cannot train: no paired training files and no existing checkpoint.")
    
    train_dataset = StandardDataset(train_mix, train_tgt)
    val_dataset = StandardDataset(val_mix, val_tgt)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Initializing UNetAttention ({base_filters} filters, {num_layers} layers, {num_heads} heads)...")
    model_voc_attn = ma.UNetAttention(
        in_channels=1,
        out_channels=1,
        base_filters=base_filters,
        num_layers=num_layers,
        num_heads=num_heads,
        batchnorm=True,
        dropout=0.1
    ).to(device)
    
    processor_voc = AudioProcessor(device=device)
    optimizer_voc = torch.optim.Adam(model_voc_attn.parameters(), lr=learning_rate)
    loss_fn_voc = nn.MSELoss()
    
    if ckpt_voc_attn.exists():
        print(f"Checkpoint found: {ckpt_voc_attn.name}")
        print("Loading existing model instead of training...")
        
        checkpoint = torch.load(ckpt_voc_attn, map_location=device, weights_only=False)
        model_voc_attn.load_state_dict(checkpoint['model_state_dict'])
        voc_history = checkpoint.get('history', {})
        
        print("Model loaded successfully from checkpoint")
    else:
        voc_trainer = UniversalTrainer(
            model=model_voc_attn,
            train_loader=train_loader,
            val_loader=val_loader,
            processor=processor_voc,
            optimizer=optimizer_voc,
            loss_fn=loss_fn_voc,
            device=device,
            input_type='spectrogram'
        )
        
        print("Starting Vocal Extraction Training...")
        voc_history = voc_trainer.train(num_epochs=num_epochs, save_path=ckpt_voc_attn)
        
        plt.figure(figsize=(10, 5))
        plt.plot(voc_history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(voc_history['val_loss'], label='Val Loss', linewidth=2)
        plt.title("UNetAttention: Vocal Extraction Training", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return {
        'model_voc_attn': model_voc_attn,
        'processor_voc': processor_voc,
        'loss_fn_voc': loss_fn_voc,
        'voc_history': voc_history,
        'ckpt_voc_attn': ckpt_voc_attn
    }

def evaluate_vocals_extraction_unetattention(data_dir, checkpoint_dir, device, 
                                              model_voc_attn=None, processor_voc=None,
                                              sr=22050, duration=120.0, chunk_len=8.0, 
                                              hop_length=4.0, base_filters=32, 
                                              num_layers=5, num_heads=4, auto_select=False):
    
    from models import models as ma
    import matplotlib.pyplot as plt
    from IPython.display import Audio, display
    
    print("="*70)
    print("VOCAL EXTRACTION FROM TEST SET (Stage 2 - UNetAttention)")
    print("="*70)
    
    if model_voc_attn is None:
        model_voc_attn = ma.UNetAttention(
            in_channels=1,
            out_channels=1,
            base_filters=base_filters,
            num_layers=num_layers,
            num_heads=num_heads,
            batchnorm=True,
            dropout=0.1
        ).to(device)
    
    if processor_voc is None:
        processor_voc = AudioProcessor(device=device)
    
    ckpt_voc_attn = checkpoint_dir / 'model_unetattention_vocals_only.pth'
    if ckpt_voc_attn.exists():
        checkpoint = torch.load(ckpt_voc_attn, map_location=device, weights_only=False)
        model_voc_attn.load_state_dict(checkpoint['model_state_dict'])
        model_voc_attn.eval()
        print(f"Model loaded: {ckpt_voc_attn.name}")
    else:
        print(f"Model not found: {ckpt_voc_attn}")
        raise FileNotFoundError(f"Checkpoint {ckpt_voc_attn} does not exist")
    
    mix_base = data_dir / 'stage2' / 'test'
    vocal_base = data_dir / 'vocals' / 'test'
    mix_dir = mix_base / 'mixture'
    vocal_dir = vocal_base
    
    if not mix_dir.exists() or not vocal_dir.exists():
        print("ERROR: Test directories not found")
        print(f"   Mixture dir: {mix_dir.exists()}")
        print(f"   Vocals dir: {vocal_dir.exists()}")
        raise FileNotFoundError("Test data directories not found")
    
    print("Found test directories")
    
    def parse_song_and_idx(stem):
        if '_chunk' in stem:
            parts = stem.rsplit('_chunk', 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0], int(parts[1])
        parts = stem.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            return '_'.join(parts[:-1]), int(parts[-1])
        return stem, 0
    
    all_mix_files = sorted(list(mix_dir.glob('*.npy')))
    all_vocal_files = sorted(list(vocal_dir.glob('*.npy')))
    
    if len(all_mix_files) == 0:
        raise FileNotFoundError("No mixture files found")
    if len(all_vocal_files) == 0:
        raise FileNotFoundError("No vocal files found")
    
    songs = {}
    for f in all_mix_files:
        song_name, idx = parse_song_and_idx(f.stem)
        if song_name not in songs:
            songs[song_name] = {}
        songs[song_name][idx] = f
    
    vocal_song_names = set(parse_song_and_idx(f.stem)[0] for f in all_vocal_files)
    song_list = sorted([s for s in songs.keys() if s in vocal_song_names])
    
    if not song_list:
        song_list = sorted(songs.keys())
    
    if len(song_list) == 1 or auto_select:
        selected_song = song_list[0]
        print(f"\nAuto-selected: {selected_song}")
    else:
        print(f"\nFound {len(song_list)} songs:")
        for i, song in enumerate(song_list):
            print(f"   {i}: {song}")
        print(f"\nSelect a song [0-{len(song_list)-1}] or press Enter for default (0):")
        try:
            choice = input("Choice: ").strip()
            idx = int(choice) if choice else 0
            if idx < 0 or idx >= len(song_list):
                idx = 0
        except Exception:
            idx = 0
        selected_song = song_list[idx]
    
    print(f"\nSelected: {selected_song}")
    
    chunks = songs[selected_song]
    print(f"   Found {len(chunks)} chunks")
    
    print("\nStitching chunks...")
    hop_samples = int(hop_length * sr)
    target_samples = int(duration * sr)
    mix_wav = np.zeros(target_samples, dtype=np.float32)
    vocal_wav_gt = np.zeros(target_samples, dtype=np.float32)
    weights = np.zeros(target_samples, dtype=np.float32)
    has_target = True
    
    for chunk_idx in sorted(chunks.keys()):
        pos = chunk_idx * hop_samples
        if pos >= target_samples:
            break
        
        mix_chunk = np.load(chunks[chunk_idx])
        vocal_filename = chunks[chunk_idx].name
        vocal_path = vocal_dir / vocal_filename
        
        if vocal_path.exists():
            vocal_chunk = np.load(vocal_path)
        else:
            vocal_chunk = np.zeros_like(mix_chunk)
            has_target = False
        
        valid_len = min(len(mix_chunk), target_samples - pos)
        window = np.hanning(len(mix_chunk))[:valid_len]
        
        mix_wav[pos:pos+valid_len] += mix_chunk[:valid_len] * window
        vocal_wav_gt[pos:pos+valid_len] += vocal_chunk[:valid_len] * window
        weights[pos:pos+valid_len] += window
    
    mask = weights > 0
    mix_wav[mask] = mix_wav[mask] / weights[mask]
    vocal_wav_gt[mask] = vocal_wav_gt[mask] / weights[mask]
    
    print("Stitching complete:")
    print(f"   Mix shape: {mix_wav.shape}, range=[{mix_wav.min():.4f}, {mix_wav.max():.4f}]")
    print(f"   Vocal GT shape: {vocal_wav_gt.shape}, range=[{vocal_wav_gt.min():.4f}, {vocal_wav_gt.max():.4f}]")
    
    print("\nRunning UNetAttention inference...")
    model_voc_attn.to(device)
    model_voc_attn.eval()
    extracted_vocals = sliding_window_inference(
        model_voc_attn, processor_voc, mix_wav,
        chunk_len=chunk_len, sr=sr, device=device
    )
    
    print("Inference complete:")
    print(f"   Extracted shape: {extracted_vocals.shape}")
    
    print(f"\n{'='*70}\nVISUALIZATION\n{'='*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    spec_mix = to_spec(mix_wav, processor_voc)
    im0 = axes[0, 0].imshow(spec_mix, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title("Input: Full Band Mixture", fontweight='bold', fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0])
    
    if has_target:
        spec_vocal_gt = to_spec(vocal_wav_gt, processor_voc)
        im1 = axes[0, 1].imshow(spec_vocal_gt, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title("Ground Truth: Vocals", fontweight='bold', fontsize=12)
        plt.colorbar(im1, ax=axes[0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, "Target Not Available", ha='center', fontsize=14)
        axes[0, 1].set_title("Ground Truth: Vocals", fontweight='bold', fontsize=12)
    
    spec_pred = to_spec(extracted_vocals, processor_voc)
    im2 = axes[1, 0].imshow(spec_pred, aspect='auto', origin='lower', cmap='viridis')
    axes[1, 0].set_title("Extracted Vocals (UNetAttention)", fontweight='bold', fontsize=12)
    plt.colorbar(im2, ax=axes[1, 0])
    
    if has_target:
        min_len = min(spec_vocal_gt.shape[1], spec_pred.shape[1])
        err = np.abs(spec_pred[:, :min_len] - spec_vocal_gt[:, :min_len])
        im3 = axes[1, 1].imshow(err, aspect='auto', origin='lower', cmap='hot')
        axes[1, 1].set_title("Error Map |Pred - GT|", fontweight='bold', fontsize=12)
        plt.colorbar(im3, ax=axes[1, 1])
        mse = np.mean(err**2)
        print(f"   MSE Loss: {mse:.6f}")
    else:
        axes[1, 1].text(0.5, 0.5, "No Target Available\nfor Error Computation", 
                       ha='center', fontsize=12)
        axes[1, 1].set_title("Error Map", fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    print("\nAudio Comparison:")
    print("   1) Input Mixture:")
    display(Audio(mix_wav, rate=sr))
    
    if has_target:
        print("\n   2) Ground Truth Vocals:")
        display(Audio(vocal_wav_gt, rate=sr))
    
    print("\n   3) Extracted Vocals (UNetAttention):")
    display(Audio(extracted_vocals, rate=sr))
    
    print("\nExtraction complete!")
    
    return {
        'mix_wav': mix_wav,
        'vocal_wav_gt': vocal_wav_gt,
        'extracted_vocals': extracted_vocals,
        'has_target': has_target,
        'selected_song': selected_song
    }

