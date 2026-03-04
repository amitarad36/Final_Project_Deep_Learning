import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import torch.nn as nn
import gc
import random
import museval
from . import models
from .config import GENERAL_CONFIG
SR = GENERAL_CONFIG['sr']
SEED = GENERAL_CONFIG['seed']
N_FFT = GENERAL_CONFIG['n_fft']
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from .trainer import UniversalTrainer

from IPython.display import display, Audio
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass


def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_project_environment():

    PROJECT_ROOT = Path.cwd()
    if not (PROJECT_ROOT / 'mainNB.ipynb').exists():
        for p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
            if (p / 'mainNB.ipynb').exists():
                PROJECT_ROOT = p
    
    os.chdir(PROJECT_ROOT)
    
    DATA_DIR = PROJECT_ROOT / "data"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")

    print(f"--------------")
    print(f"Device: {device}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"--------------")
    
    return {'project_root':PROJECT_ROOT ,'data_dir': DATA_DIR, 'checkpoint_dir':CHECKPOINT_DIR,'device':device}




class AudioProcessor:
    def __init__(self, n_fft=N_FFT, hop_length=512, device='cpu'):
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
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        collate_fn=_collate_dict_batch
    )
    
    return loader


def load_musdb_stems(track_folder):

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
        if file_sr != SR:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
        stems[stem] = audio
    return stems


def _process_stage_helper(stage_desc, stage_dir_name, splits, data_root, chunk_dur, overlap, sr, mix_fn, target_fn):
    print(f"\n{'=' * 70}\n{stage_desc}\n{'=' * 70}")

    for split_name, folder_path in splits.items():
        out_dir = data_root / stage_dir_name / split_name
        mix_dir = out_dir / 'mixture'
        tgt_dir = out_dir / 'target'

        # Prevent re-processing if data exists
        if mix_dir.exists() and any(mix_dir.glob('*.npy')):
            print(f"{split_name}: already exists, skipping...")
            continue

        mix_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)

        for track_folder in folder_path.iterdir():
            stems = load_musdb_stems(track_folder)
            mix = mix_fn(stems)
            target = target_fn(stems)

            # Math for chunking
            step = int((chunk_dur - overlap) * sr)
            chunk_len = int(chunk_dur * sr)

            for i, start in enumerate(range(0, len(mix) - chunk_len + 1, step)):
                np.save(mix_dir / f"{track_folder.name}_chunk{i}.npy", mix[start:start + chunk_len])
                np.save(tgt_dir / f"{track_folder.name}_chunk{i}.npy", target[start:start + chunk_len])


def process_stage1(splits, data_dir, chunk_dur, overlap, sr):
    _process_stage_helper(
        stage_desc='STAGE 1: vocals+other → other (separation training)',
        stage_dir_name='stage1',
        splits=splits,
        data_root=data_dir,
        chunk_dur=chunk_dur,
        overlap=overlap,
        sr=sr,
        mix_fn=lambda s: s['vocals'] + s['other'],
        target_fn=lambda s: s['other']
    )


def process_stage2(splits, data_dir, chunk_dur, overlap, sr):
    _process_stage_helper(
        stage_desc='STAGE 2: Full Mix → Accompaniment',
        stage_dir_name='stage2',
        splits=splits,
        data_root=data_dir,
        chunk_dur=chunk_dur,
        overlap=overlap,
        sr=sr,
        mix_fn=lambda s: s['vocals'] + s['drums'] + s['bass'] + s['other'],
        target_fn=lambda s: s['drums'] + s['bass'] + s['other']
    )


################################################################################
def initialize_model_unetattention(device='cuda'):

    processor = AudioProcessor(device=device)
    loss_fn = nn.MSELoss()
    model = models.UNetAttention(
        in_channels=1, out_channels=1,
        # base_filters=attn_kwargs['base_filters'],
        # num_layers=attn_kwargs['num_layers'],
        # num_heads=attn_kwargs['num_heads'],
        batchnorm=True, dropout=0.1
    ).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])


################################################################################

def train_pipeline_stage(stage_key, data_dir, checkpoint_dir, device, batch_sizes, chunk_duration=8.0, skip_training=False):

    print(f"\n{'=' * 70}\n TRAINING: {stage_key.upper()} \n{'=' * 70}")

    results = {}
    model_types = ['lstm', 'unet']

    for m_type in model_types:
        print(f'\nModel ({m_type.upper()}) - {stage_key}')
        print('-' * 70)

        ckpt_path = checkpoint_dir / f'model_a_{m_type}_{stage_key}_{chunk_duration:.0f}s.pth'
        skip_this_model = skip_training or ckpt_path.exists()

        if ckpt_path.exists():
            print(f"Checkpoint found: {ckpt_path.name}")

        # 2. THE LOCAL LOOKUP FIX:

        init_fn_name = f'initialize_model_a_{m_type}'
        init_fn = globals()[init_fn_name]
        model, processor, optimizer, loss_fn = init_fn(device)

        config = get_training_config(m_type)
        config['batch_size'] = batch_sizes.get(m_type, 32)

        history = train_model(model=model, processor=processor, optimizer=optimizer, loss_fn=loss_fn,
                              training_data_dir=data_dir, stage=stage_key, ckpt_path=ckpt_path, device=device,
                              train_config=config)

        results.update({
            f'model_{m_type}': model,
            f'processor_{m_type}': processor,
            f'loss_fn_{m_type}': loss_fn,
            f'hist_{m_type}_{stage_key}': history,
            f'ckpt_{m_type}_{stage_key}': ckpt_path
        })

        model.to('cpu')
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


def train_unetattention_pipeline(stage_key, data_dir, checkpoint_dir, device, chunk_duration=8.0, skip_training=False, batch_size=32, base_filters=32, num_layers=4, num_heads=4, learning_rate=None):
    print(f"\n{'=' * 70}\nUNETATTENTION TRAINING: {stage_key.upper()}\n{'=' * 70}")

    checkpoint_dir = Path(checkpoint_dir)
    ckpt_path = checkpoint_dir / f'model_unetattention_{stage_key}_{chunk_duration:.0f}s.pth'

    # 1. Initialize the Model Architecture
    print(f"Initializing UNetAttention ({base_filters} filters, {num_layers} layers, {num_heads} heads)...")
    model = models.UNetAttention(
        in_channels=1, out_channels=1,
        base_filters=base_filters, num_layers=num_layers,
        num_heads=num_heads, batchnorm=True, dropout=0.1
    ).to(device)

    processor = AudioProcessor(device=device)
    loss_fn = nn.MSELoss()

    # 2. Logic for Loading Existing Checkpoints (Skip Training)
    if skip_training and ckpt_path.exists():
        print(f"Skipping training. Loading {stage_key} checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        history = checkpoint.get('history', {})

    else:
        # 3. Setup Training Config
        config = get_training_config('unet')  # Base on unet defaults
        config['batch_size'] = batch_size
        if learning_rate:
            config['learning_rate'] = learning_rate

        # 4. SPECIAL CURRICULUM LEARNING (Stage 2 only)
        if stage_key == 'stage2':
            ckpt_s1 = checkpoint_dir / f'model_unetattention_stage1_{chunk_duration:.0f}s.pth'
            if ckpt_s1.exists():
                print(f"Curriculum Learning: Loading Stage 1 weights from {ckpt_s1.name}")
                checkpoint_s1 = torch.load(ckpt_s1, map_location=device)
                model.load_state_dict(checkpoint_s1['model_state_dict'])
            else:
                print("Stage 1 weights not found! Training from scratch.")

        # 5. Run Training
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        history = train_model(model=model, processor=processor, optimizer=optimizer, loss_fn=loss_fn,
                              training_data_dir=data_dir, stage=stage_key, ckpt_path=ckpt_path, device=device,
                              train_config=config)

    return {
        f'model_attn_{stage_key}': model,
        f'processor_attn_{stage_key}': processor,
        f'loss_fn_{stage_key}': loss_fn,
        f'hist_attn_{stage_key}': history,
        f'ckpt_attn_{stage_key}': ckpt_path
    }


def train_model(model, processor, optimizer, loss_fn, training_data_dir, stage, ckpt_path, device, train_config):

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

    if train_config['skip_training']:
        print("Training skipped")
        return hist

    train_loader = get_data_loaders(training_data_dir, stage=stage, split='train',
                                    batch_size=train_config['batch_size'])
    val_loader = get_data_loaders(training_data_dir, stage=stage, split='val',
                                  batch_size=train_config['batch_size'])

    trainer = UniversalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        processor=processor,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=train_config['device'],
        early_stopping=train_config['early_stopping'],
        input_type='spectrogram'
    )

    hist = trainer.train(num_epochs=train_config['epochs'], save_path=str(ckpt_path))
    print(f"Training complete! Best val loss: {min(hist['val_loss']):.6f}")
    return hist


def train_models(model_names, stage_list, hp_list, shared_config):
    """
    wrapper to train_model
    """
    data_dir = shared_config["data_dir"]
    checkpoint_dir = shared_config["checkpoint_dir"]
    device = shared_config["device"]
    chunk_duration = shared_config["chunk_duration"]
    skip_training = shared_config.get('skip_training')
    print(f"\n{'=' * 70}\n TRAINING MODELS: {model_names} \n{'=' * 70}")

    checkpoint_dir = Path(checkpoint_dir)
    results = {}

    processor = AudioProcessor(device=device)
    loss_fn = nn.MSELoss()
    for name, stage, hp in zip(model_names, stage_list, hp_list):
        config = shared_config | hp
        stage_key = "stage"+str(stage)
        print(f'\nModel ({name.upper()}) - {stage}')
        print('-' * 70)

        ckpt_path = checkpoint_dir / f'model_{name}_{stage_key}_{chunk_duration:.0f}s.pth'
        ckpt_s1 = checkpoint_dir / f'model_{name}_stage1_{chunk_duration:.0f}s.pth'

        skip_this_model = skip_training or ckpt_path.exists()

        # 2. Setup Configuration


        # 3. Initialize Model Architecture

        if name == 'lstm':
            model = models.SpectrogramMaskingLSTM(
                freq_bins=config['freq_bins'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                bidirectional=config['bidirectional']
            ).to(device)

        elif name == 'unet':
            model = models.TimeFrequencyDomainUNet(
                in_channels=hp['in_channels'],
                out_channels=hp['out_channels'],
                base_filters=hp['base_filters'],
                num_layers=hp['num_layers'],
                batchnorm=hp['batchnorm'],
                dropout=hp['dropout']
            ).to(device)

        # elif m == 'unetattention':
        #     print(f"Initializing UNetAttention ({attn_kwargs['base_filters']} filters, {attn_kwargs['num_layers']} layers, {attn_kwargs['num_heads']} heads)...")
        #     processor = AudioProcessor(device=device)
        #     loss_fn = nn.MSELoss()
        #     model = models.UNetAttention(
        #         in_channels=1, out_channels=1,
        #         base_filters=attn_kwargs['base_filters'],
        #         num_layers=attn_kwargs['num_layers'],
        #         num_heads=attn_kwargs['num_heads'],
        #         batchnorm=True, dropout=0.1
        #     ).to(device)
        #     optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        else:
            raise ValueError(f"Unknown model type: {name}")
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        # 4. UNIVERSAL CURRICULUM LEARNING (Stage 2 only)
        if stage_key == 'stage2' and not skip_this_model:
            if ckpt_s1.exists():
                print(f"Curriculum Learning: Loading Stage 1 weights from {ckpt_s1.name}")
                checkpoint_s1 = torch.load(ckpt_s1, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint_s1['model_state_dict'])
            else:
                print(f"Stage 1 weights not found for {name.upper()}! Training from scratch.")

        # 5. Run Training
        history = train_model(model=model, processor=processor, optimizer=optimizer, loss_fn=loss_fn,
                              training_data_dir=data_dir, stage=stage_key, ckpt_path=ckpt_path, device=device,
                              train_config=config)

        # 6. Store Results
        dict_key = 'attn' if name == 'unetattention' else name
        if dict_key not in results:
            results[dict_key] = {}

        results[dict_key].update({
            'name': name,
            'model': model,
            'processor': processor,
            'loss_fn': loss_fn,
            f'hist': history,
            f'ckpt': ckpt_path
            # f'hist_{stage_key}': history,
            # f'ckpt_{stage_key}': ckpt_path
        })

        # 7. GPU Memory Cleanup
        model.to('cpu')
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


def visualize_results(predictions, model_list, audio_data, general_config, stage=None):
    num_models = len(model_list)
    mix_wav = audio_data['mix_wav']
    tgt_wav = audio_data.get('tgt_wav')
    selected_song = audio_data['selected_song']
    sr = general_config['sr']
    ref_proc = model_list[0]['processor']

    # --- Step 1: Determine Layout Strategy ---
    has_target = tgt_wav is not None
    num_cols = 2 if has_target else 1
    num_rows = num_models + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 4 * num_rows))

    # Ensure axes is always a 2D array for consistent indexing [row, col]
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.reshape(num_rows, num_cols)

    # --- Step 2: Plot Row 0 (Inputs) ---
    axes[0, 0].imshow(to_spec(mix_wav, ref_proc), aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title("Input Mixture", fontweight='bold')

    if has_target:
        axes[0, 1].imshow(to_spec(tgt_wav, ref_proc), aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title("Ground Truth", fontweight='bold')

    # --- Step 3: Plot Subsequent Rows (Models) ---
    for i, (model, est) in enumerate(zip(model_list, predictions)):
        proc = model['processor']
        name = model['name']
        row = i + 1
        spec_pred = to_spec(est, proc)

        if has_target:
            # Mode A: Error Map + Prediction
            spec_tgt = to_spec(tgt_wav, proc)
            min_len = min(spec_tgt.shape[1], spec_pred.shape[1])
            err = np.abs(spec_pred[:, :min_len] - spec_tgt[:, :min_len])

            axes[row, 0].imshow(err, aspect='auto', origin='lower', cmap='magma')
            axes[row, 0].set_title(f"{name}: Error Map", fontweight='bold')

            axes[row, 1].imshow(spec_pred, aspect='auto', origin='lower', cmap='viridis')
            axes[row, 1].set_title(f"{name}: Prediction", fontweight='bold')
        else:
            # Mode B: Just Prediction (Single Column)
            axes[row, 0].imshow(spec_pred, aspect='auto', origin='lower', cmap='viridis')
            axes[row, 0].set_title(f"{name}: Prediction", fontweight='bold')

    # --- Step 4: Cleanup & Audio ---
    for ax in axes.flatten():
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

    title_text = f"{stage if stage else 'User Song'} Evaluation: {selected_song}"
    plt.suptitle(title_text, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\nAudio Playback: {selected_song} ---")
    display(Audio(mix_wav, rate=sr))

    if has_target:
        print("Ground Truth:")
        display(Audio(tgt_wav, rate=sr))

    for model, est in zip(model_list, predictions):
        print(f"{model['name']} Prediction:")
        display(Audio(est, rate=sr))


def plot_learning_comparison(model_list, test_list, title="Training Comparison"):

    num_models = len(model_list)
    if num_models == 0:
        return

    fig, axes = plt.subplots(1, num_models, figsize=(7 * num_models, 5), squeeze=False)

    for i, model in enumerate(model_list):
        hist = model['hist']
        name = model['name']
        ax = axes[0, i]

        if hist and 'train_loss' in hist:
            epochs = range(1, len(hist['train_loss']) + 1)
            ax.plot(epochs, hist['train_loss'], 'o-', label='Train', linewidth=2)
            ax.plot(epochs, hist['val_loss'], 's--', label='Val', linewidth=2)

            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No data for {name}", ha='center')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nTest Results - Mean ± Std:")
    print("=" * 40)

    for model, t in zip(model_list, test_list):
        name = model['name']
        if t and 'mean' in t:
            mu = t['mean']
            std = t.get('std', 0.0)  # Uses 0.0 if 'std' is missing
            print(f"{name:<15} : {mu:.5f} ± {std:.5f}")
        else:
            print(f"{name:<15} : [No test data available]")

    print("=" * 40)


def test_models(model_list, stage_list, config):
    results = {}
    device = config['device']
    cached_paths = {}
    chunk_duration = str(int(config['chunk_duration']))
    for model, stage in zip(model_list, stage_list):
        name = model['name']
        stage_key = "stage" + str(stage)
        cached_paths[name] = config['checkpoint_dir'] / f'test_results_{name}_{stage_key}_{chunk_duration}s.pkl'

    all_cached = True
    for name, path in cached_paths.items():
        res = load_test_results(path)
        if res:
            results[name] = res
        else:
            all_cached = False

    if all_cached:
        print(f"Loaded all cached results.")
        return results

    # 2. If not cached, run evaluation
    print(f"Running test evaluation...\n")

    for model_dict, stage in zip(model_list, stage_list):
        name = model_dict['name']
        model = model_dict['model'].to(device)
        ckpt_path = model_dict['ckpt']
        print("not those checkpoints")
        # Load weights
        if ckpt_path.exists():
            print(f"Loading {name.upper()} weights...")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"No {name} checkpoint found - evaluating untrained!")

        # Evaluate
        print(f"\n{'=' * 70}\nEVALUATING {name.upper()} MODEL (STAGE {stage})\n{'=' * 70}")
        results[name] = evaluate_test_set(
            model, model_dict['processor'], config['data_dir'], f'stage{stage}', model_dict['loss_fn'], device
        )

        # Save cache
        print(cached_paths)
        save_test_results(results[name], cached_paths[name])

    return results


def load_and_stitch_test_chunks(general_config, stage):
    data_dir = general_config['data_dir']
    sr = general_config['sr']
    duration = general_config['duration']
    hop_length = general_config['hop_length']
    stage = "stage" + str(stage)
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
    
    if len(song_list) == 1:
        selected_song = song_list[0]
        print(f"\nAuto-selected: {selected_song}")
    else:
        print(f"\nSelect a song [0-{len(song_list)-1}] or press Enter for default (0):")
        try:
            choice = input("Choice: ").strip()
            idx = int(choice) if choice else 0
        except:
            idx = 0
        selected_song = song_list[idx]
        print(f"Selected: {selected_song}")
    
    chunks = songs[selected_song]
    
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
    
    return {'mix_wav': mix_wav, 'tgt_wav': tgt_wav, 'selected_song': selected_song}


def plot_saved_results(results_path):
    results_path = Path(results_path)

    if results_path.suffix == '.pkl':
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = np.load(results_path, allow_pickle=True)
        if hasattr(data, 'item'):
            data = data.item()

    m1_metrics = data.get('model_1', data.get('lstm'))
    m2_metrics = data.get('model_2', data.get('unet'))

    if not m1_metrics or not m2_metrics:
        print("Error: Could not find model metrics in the file.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    metrics = ['SDR', 'SIR', 'SAR']
    colors = ['#3498db', '#e67e22']

    for i, metric in enumerate(metrics):
        m1_vals = [v for v in m1_metrics[metric] if np.isfinite(v)]
        m2_vals = [v for v in m2_metrics[metric] if np.isfinite(v)]

        bp = axes[i].boxplot([m1_vals, m2_vals], patch_artist=True,
                             labels=['Model 1', 'Model 2'], widths=0.5)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[i].set_title(f'{metric} (dB)', fontweight='bold', fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)

        for j, vals in enumerate([m1_vals, m2_vals]):
            mean_val = np.mean(vals)
            axes[i].text(j + 1, mean_val, f'{mean_val:.2f}',
                         ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Performance Comparison: Model 1 vs Model 2', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()


def evaluate_separation_quality(model_1=None, model_2=None, processor_1=None, processor_2=None, test_data_dir=None, stage='stage1', num_samples=10, device='cuda', save_path=None, load_if_exists=True, random_sampling=False, random_seed=SEED, **kwargs):

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
                win=SR, hop=SR
            )
            metrics_1['SDR'].append(_safe_median(sdr_1))
            metrics_1['SIR'].append(_safe_median(sir_1))
            sar_1_val = _safe_median(sar_1)
            if not np.isfinite(sar_1_val):
                sar_1_val = _sar_fallback(tgt_wav, est_wav_1)
            metrics_1['SAR'].append(sar_1_val)

            sdr_2, sir_2, sar_2, _ = museval.evaluate(
                tgt_wav.reshape(1, -1), est_wav_2.reshape(1, -1),
                win=SR, hop=SR
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

    if ckpt_path.exists():
        with open(ckpt_path, 'rb') as f:
            return pickle.load(f)
    return None


def save_test_results(results, ckpt_path):

    with open(ckpt_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved test results to: {ckpt_path.name}")


def evaluate_test_set(model, processor, test_data_dir, stage, loss_fn, device):

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


def eval_models(model_list, audio_data, shared_config):
    device = shared_config['device']
    predictions = []
    for model in model_list:
        model['model'].to(device).eval()
        with torch.no_grad():
            pred = sliding_window_inference(model, audio_data, shared_config)
            predictions.append(pred)

    return predictions


def sliding_window_inference(model_dict, audio_data, shared_config):
    audio = audio_data['mix_wav']
    model = model_dict['model']
    processor = model_dict['processor']
    device = shared_config['device']
    chunk_duration = shared_config['chunk_duration']
    chunk_samples = int(chunk_duration * SR)
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


def unet_inference_accelerator(model, processor, audio, chunk_len=8.0, device='cuda', batch_size=16):

    chunk_samples = int(chunk_len * SR)
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


def evaluate_models_on_audio_file(file_path, model_list, shared_config, general_config):
    
    print("="*70)
    print("CUSTOM AUDIO INFERENCE")
    print("="*70)
    print(f"\nLoading: {file_path}")
    try:
        audio, file_sr = librosa.load(file_path, sr=None, mono=True)
        print(f"   Original SR: {file_sr} Hz, Duration: {len(audio)/file_sr:.2f}s")
        
        if file_sr != SR:
            print(f"   Resampling to {SR} Hz...")
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
        # duration = general_config.get('duration')
        # if duration is not None:
        #     target_samples = int(duration * SR)
        #     if len(audio) > target_samples:
        #         audio = audio[:target_samples]
        #         print(f"   Trimmed to {duration}s")
        
        print(f"Loaded: {len(audio)/SR:.2f}s @ {SR} Hz")
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    # audio_data = load_and_stitch_test_chunks(general_config=general_config, stage=stage)

    # model_lstm.eval()
    # model_unet.eval()
    audio_data = {'mix_wav': audio, 'selected_song': file_path.name}
    predictions = eval_models(model_list, audio_data, shared_config)
    visualize_results(predictions, model_list, audio_data, general_config)


def handle_user_upload_and_inference(data_dir, model_list, shared_config, general_config):

    upload_dir = Path(data_dir) / "user_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

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

    evaluate_models_on_audio_file(audio_path, model_list, shared_config, general_config)


def compare_unet_vs_unetattention_on_audio_file(file_path, model_unet, model_attn, processor_unet, processor_attn, device, duration=None, chunk_len=8.0):

    print("="*70)
    print("CUSTOM AUDIO INFERENCE: U-Net vs UNetAttention (Stage 2)")
    print("="*70)
    print(f"\nLoading: {file_path}")
    
    try:
        audio, file_sr = librosa.load(file_path, sr=None, mono=True)
        print(f"   Original SR: {file_sr} Hz, Duration: {len(audio)/file_sr:.2f}s")
        
        if file_sr != SR:
            print(f"   Resampling to {SR} Hz...")
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
        
        if duration is not None:
            target_samples = int(duration * SR)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
                print(f"   Trimmed to {duration}s")
        
        print(f"Loaded: {len(audio)/SR:.2f}s @ {SR} Hz")
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    model_unet.eval()
    model_attn.eval()
    
    print("\nRunning inference...")
    print("   - Standard U-Net (Stage 2)")
    est_unet = sliding_window_inference(model_unet, processor_unet, audio, chunk_len=chunk_len, device=device)
    
    print("   - UNetAttention (Stage 2)")
    est_attn = sliding_window_inference(model_attn, processor_attn, audio, chunk_len=chunk_len, device=device)
    
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
    display(Audio(audio, rate=SR))
    
    print("\nU-Net Output:")
    display(Audio(est_unet, rate=SR))
    
    print("\nUNetAttention Output:")
    display(Audio(est_attn, rate=SR))
    
    print(f"\n{'='*70}")
    print("INFERENCE COMPLETE")
    print(f"{'='*70}")


def handle_user_upload_unetattention_inference(data_dir, model_unet, model_attn, processor_unet, processor_attn, device,sr=SR, duration=None, chunk_len=8.0):

    upload_dir = Path(data_dir) / "user_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

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

    compare_unet_vs_unetattention_on_audio_file(file_path=audio_path, model_unet=model_unet, model_attn=model_attn,
                                                processor_unet=processor_unet, processor_attn=processor_attn,
                                                device=device, duration=duration, chunk_len=chunk_len)


def evaluate_vocals_extraction_unetattention(data_dir, checkpoint_dir, device, model_voc_attn=None, processor_voc=None,sr=SR, duration=120.0, chunk_len=8.0, hop_length=4.0, base_filters=32, num_layers=5, num_heads=4, auto_select=False):

    print("="*70)
    print("VOCAL EXTRACTION FROM TEST SET (Stage 2 - UNetAttention)")
    print("="*70)
    
    if model_voc_attn is None:
        model_voc_attn = models.UNetAttention(
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
    extracted_vocals = sliding_window_inference(model_voc_attn, processor_voc, mix_wav, chunk_len=chunk_len,
                                                device=device)
    
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

