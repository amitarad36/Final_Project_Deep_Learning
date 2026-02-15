import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import librosa
import torch.nn as nn
import torch.optim as optim
import torchaudio
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

# Configure CUDA memory to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configure matplotlib for Jupyter/Colab environment
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

# ==============================================================================
# Neural Linearizer Utilities
# ==============================================================================

def squeeze(x):
    """
    Reshapes spectrogram to be deeper and smaller spatially.
    (Batch, 1, F, T) -> (Batch, 4, F/2, T/2)
    """
    b, c, h, w = x.size()
    # Handle odd dimensions by trimming 1 pixel
    if h % 2 != 0: x = x[:, :, :-1, :]
    if w % 2 != 0: x = x[:, :, :, :-1]
    
    b, c, h, w = x.size()
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(b, c * 4, h // 2, w // 2)
    return x

def unsqueeze(x):
    """
    Inverse of squeeze.
    (Batch, 4, F/2, T/2) -> (Batch, 1, F, T)
    """
    b, c, h, w = x.size()
    out_c = c // 4
    x = x.view(b, out_c, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(b, out_c, h * 2, w * 2)
    return x

class StyleEncoderWrapper(nn.Module):
    """
    Wrapper to load and run WavLM for style extraction.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print("Loading WavLM...")
        # Force use of safetensors to avoid torch.load security vulnerability
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus-sv", use_safetensors=True).to(device)
        self.model.eval() # Always freeze
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        # Resampler: 22050 -> 16000 (WavLM requires 16k)
        self.resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000).to(device)

    def get_style(self, audio_waveform):
        """
        Input: Audio waveform (Batch, Time) at 22050Hz
        Output: Style Vector (Batch, 768)
        """
        # 1. Resample to 16k (Gradients Preserved)
        audio_16k = self.resampler(audio_waveform)
        
        # 2. Manual Normalization (PyTorch Native)
        # WavLM expects zero mean and unit variance per sample
        # We do this manually so we don't have to leave the GPU or detach
        mean = audio_16k.mean(dim=-1, keepdim=True)
        var = audio_16k.var(dim=-1, keepdim=True)
        audio_norm = (audio_16k - mean) / torch.sqrt(var + 1e-5)
        
        # 3. Run Model Directly (Skip feature_extractor wrapper)
        # We feed 'input_values' directly.
        with torch.no_grad():
            outputs = self.model(input_values=audio_norm)
            # Mean pooling over time to get one vector per song
            style_emb = outputs.last_hidden_state.mean(dim=1)
            
        return style_emb

class ContentEncoderWrapper(nn.Module):
    """
    Wrapper for Wav2Vec2 to extract content features.
    Used for content loss (L_content).
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print("Loading Wav2Vec2 for Content...")
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True).to(device)
        self.model.eval()  # Always frozen
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000).to(device)
    
    def get_content(self, audio_waveform):
        """
        Input: Audio waveform (Batch, Time) at 22050Hz
        Output: Content Features (Batch, Time_frames, 768)
        """
        # 1. Resample to 16k (Gradients Preserved)
        audio_16k = self.resampler(audio_waveform)
        
        # 2. Manual Normalization (PyTorch Native)
        # Wav2Vec2 expects zero mean and unit variance per sample
        # We do this manually so we don't have to leave the GPU or detach
        mean = audio_16k.mean(dim=-1, keepdim=True)
        var = audio_16k.var(dim=-1, keepdim=True)
        audio_norm = (audio_16k - mean) / torch.sqrt(var + 1e-5)
        
        # 3. Run Model Directly (Skip feature_extractor wrapper)
        # We feed 'input_values' directly.
        outputs = self.model(input_values=audio_norm)
        
        return outputs.last_hidden_state

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

    def train_epoch(self, epoch_idx, num_epochs):
        """
        Trains for one epoch and returns average loss.
        Implements: mask is applied in linear domain, loss is computed in log domain (Option 3).
        """
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        # Detect notebook environment for appropriate tqdm
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
        
        # Progress bar for this epoch
        pbar = tqdm_bar(self.train_loader, desc=f"Epoch {epoch_idx}/{num_epochs}", leave=True)
        
        for batch in pbar:
            mix = batch['mix']
            tgt = batch['tgt']
            
            # Move to device if tensor, leave tuples as-is
            if isinstance(mix, torch.Tensor):
                mix: torch.Tensor = mix.to(self.device)
            if isinstance(tgt, torch.Tensor):
                tgt: torch.Tensor = tgt.to(self.device)
            
            if self.input_type == 'spectrogram':
                # Check if data is already spectrograms (tuple of magnitude and phase)
                if isinstance(mix, tuple):
                    # Robust loading: Ensure exactly 4D [Batch, 1, Freq, Time]
                    mix_mag = mix[0].to(self.device)
                    tgt_mag = tgt[0].to(self.device)
                    
                    # Handle 3D tensors (batch, freq, time) -> add channel dimension
                    if mix_mag.dim() == 3:
                        mix_mag = mix_mag.unsqueeze(1)
                    elif mix_mag.dim() == 2:
                        # Handle 2D tensors (freq, time) -> add batch and channel
                        mix_mag = mix_mag.unsqueeze(0).unsqueeze(0)
                    
                    if tgt_mag.dim() == 3:
                        tgt_mag = tgt_mag.unsqueeze(1)
                    elif tgt_mag.dim() == 2:
                        tgt_mag = tgt_mag.unsqueeze(0).unsqueeze(0)
                    
                    mix_log = mix_mag  # Already log-magnitude from dataset
                    tgt_log = tgt_mag
                else:
                    # Waveforms - need to convert to spectrograms
                    mix_log, _ = self.processor.to_spectrogram(mix)
                    tgt_log, _ = self.processor.to_spectrogram(tgt)
                    
                    # Ensure 4D
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
                # Update progress bar with running average loss every 10 batches
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
                # Update progress bar with running average loss every 10 batches
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
                    # Check if data is already spectrograms (tuple of magnitude and phase)
                    if isinstance(mix, tuple):
                        # Robust loading: Ensure exactly 4D [Batch, 1, Freq, Time]
                        mix_mag = mix[0].to(self.device)
                        tgt_mag = tgt[0].to(self.device)
                        
                        # Handle 3D tensors (batch, freq, time) -> add channel dimension
                        if mix_mag.dim() == 3:
                            mix_mag = mix_mag.unsqueeze(1)
                        elif mix_mag.dim() == 2:
                            mix_mag = mix_mag.unsqueeze(0).unsqueeze(0)
                        
                        if tgt_mag.dim() == 3:
                            tgt_mag = tgt_mag.unsqueeze(1)
                        elif tgt_mag.dim() == 2:
                            tgt_mag = tgt_mag.unsqueeze(0).unsqueeze(0)
                        
                        mix_log = mix_mag  # Already log-magnitude from dataset
                        tgt_log = tgt_mag
                    else:
                        # Waveforms - need to convert to spectrograms
                        mix_log, _ = self.processor.to_spectrogram(mix)
                        tgt_log, _ = self.processor.to_spectrogram(tgt)
                        
                        # Ensure 4D
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
        # Create a subfolder for this training run based on save_path
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
            
            # Clear CUDA cache periodically to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Print summary for this epoch
            print(f"Epoch {epoch+1}/{num_epochs} Complete → Train: {train_loss:.5f} | Val: {val_loss:.5f}")
            
            # Live logging to file (every epoch)
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
        
        FIXED: Applies mask in LINEAR domain to match training pipeline.
        """
        self.model.eval()
        with torch.no_grad():
            mix: torch.Tensor = torch.tensor(mixture).to(self.device)
            if mix.ndim == 1:
                mix: torch.Tensor = mix.unsqueeze(0)
            if self.input_type == 'spectrogram':
                mix_log, mix_phase = self.processor.to_spectrogram(mix)
                mix_in = mix_log.unsqueeze(1)
                mask = self.model(mix_in)
                if mask.shape != mix_in.shape:
                    mask = mask[:, :, :mix_in.shape[2], :mix_in.shape[3]]
                # FIXED: Apply mask in LINEAR domain (convert log→linear, apply mask, convert back)
                est_linear = mask.squeeze(1) * torch.expm1(mix_log)
                est = self.processor.to_waveform(torch.log1p(est_linear), mix_phase)
                return est.squeeze().cpu().numpy()
            else:
                est = self.model(mix)
                return est.squeeze().cpu().numpy()

# ==============================================================================
# Linearizer Trainer (Multi-Objective: Reconstruction + Content + Style)
# ==============================================================================

def get_linearizer_training_config():
    """
    Returns training hyperparameters for the Neural Linearizer.
    Architecture params are in models.get_linearizer_config().
    """
    return {
        'lr': 1e-4,             # Learning rate
        'batch_size': 16,        # Batch size (keep small for spectrograms)
        'num_epochs': 5,        # Number of training epochs
        'chunk_duration': 8.0   # Audio chunk duration in seconds
    }


class LinearizerTrainer:
    """
    Multi-objective trainer for Neural Linearizer.
    Supports 3 loss terms:
    - Reconstruction (MSE on spectrograms)
    - Content (L1 on Wav2Vec2 features)
    - Style (Cosine similarity on WavLM embeddings)
    """
    def __init__(self, 
                 model, 
                 style_encoder, 
                 content_encoder,
                 processor, 
                 train_loader, 
                 val_loader, 
                 optimizer, 
                 device='cuda',
                 lambda_rec=1.0,
                 lambda_content=0.0,
                 lambda_style=0.0):
        
        self.model = model.to(device)
        self.style_encoder = style_encoder.to(device)
        self.content_encoder = content_encoder.to(device)
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        
        # Hyperparameters
        self.lambda_rec = lambda_rec
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        
        # Loss Functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        # Identity Matrix for monitoring
        self.I = torch.eye(4).to(device)
        
        self.history = {
            'total_loss': [], 
            'rec_loss': [], 
            'content_loss': [], 
            'style_loss': [],
            'identity_error': []
        }

    def to_waveform_with_gradients(self, mag, phase):
        """
        Converts Spectrogram -> Waveform while preserving gradients.
        Unlike processor.to_waveform() which detaches and converts to numpy,
        this keeps tensors intact so gradients can backprop to the model.
        """
        # mag, phase: (B, 1, F, T) -> squeeze channel -> (B, F, T)
        mag = mag.squeeze(1)
        phase = phase.squeeze(1)
        
        # Pad back to 1025 bins if cropped by squeeze (n_fft=2048 needs 1025)
        if mag.shape[1] == 1024:
            # Add zero bin at Nyquist frequency
            mag = torch.nn.functional.pad(mag, (0, 0, 0, 1), value=0)  # (B, 1025, T)
            phase = torch.nn.functional.pad(phase, (0, 0, 0, 1), value=0)
        
        complex_spec = torch.polar(mag, phase)
        
        # Inverse STFT
        waveform = torch.istft(
            complex_spec, 
            n_fft=self.processor.n_fft, 
            hop_length=self.processor.hop_length, 
            window=self.processor.window,
            return_complex=False
        )
        return waveform

    def process_batch(self, batch):
        """
        Processes batch and returns all necessary tensors for loss computation.
        Supports both old format (mix/tgt) and new format (input/target).
        """
        # Handle both old and new data formats
        if 'target' in batch:
            target_data = batch['target']  # New vocals dataset format
        elif 'tgt' in batch:
            target_data = batch['tgt']     # Old stage1/stage2 format
        else:
            raise KeyError("Batch must contain either 'target' or 'tgt' key")
        
        # 1. Prepare Target Spectrogram & Waveform
        if isinstance(target_data, tuple) or isinstance(target_data, list):
            mag, phase = target_data
            mag = mag.to(self.device)
            phase = phase.to(self.device)
            if mag.dim() == 3: mag = mag.unsqueeze(1)
            if phase.dim() == 3: phase = phase.unsqueeze(1)
            
            # Convert to waveform while preserving gradients
            target_wav = self.to_waveform_with_gradients(mag, phase)
            spec_input = mag
        else:
            target_wav = target_data.to(self.device)
            mag, phase = self.processor.to_spectrogram(target_wav)
            if mag.dim() == 3: mag = mag.unsqueeze(1)
            if phase.dim() == 3: phase = phase.unsqueeze(1)
            spec_input = mag

        # 2. Extract Target Style & Content (frozen encoders)
        with torch.no_grad():
            target_style_vec = self.style_encoder.get_style(target_wav)
            
            target_content_feat = None
            if self.lambda_content > 0:
                target_content_feat = self.content_encoder.get_content(target_wav)

        # 3. Forward Pass (Linearizer)
        spec_squeezed = squeeze(spec_input)
        pred_squeezed, A_matrix = self.model(spec_squeezed, target_style_vec)
        
        # 4. Process Output
        pred_spec = unsqueeze(pred_squeezed)
        
        # Crop target to match output (1025 -> 1024)
        target_spec_cropped = spec_input[:, :, :pred_spec.shape[2], :pred_spec.shape[3]]
        
        # 5. Convert Pred Spectrogram -> Pred Waveform (preserving gradients)
        # Reuse phase from input (standard practice)
        phase_cropped = phase[:, :, :pred_spec.shape[2], :pred_spec.shape[3]]
        pred_wav = self.to_waveform_with_gradients(pred_spec, phase_cropped)

        return {
            'pred_spec': pred_spec,
            'target_spec': target_spec_cropped,
            'pred_wav': pred_wav,
            'target_wav': target_wav,
            'target_style': target_style_vec,
            'target_content': target_content_feat,
            'A_matrix': A_matrix
        }

    def train_epoch(self, epoch_idx):
        self.model.train()
        logs = {'total': 0, 'rec': 0, 'cont': 0, 'sty': 0, 'id': 0}
        batch_count = 0
        
        try: 
            from tqdm.notebook import tqdm
        except ImportError: 
            from tqdm import tqdm
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch_idx}")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Run Pipeline
            data = self.process_batch(batch)
            
            # --- LOSS 1: RECONSTRUCTION (MSE on Spectrograms) ---
            L_rec = self.mse_loss(data['pred_spec'], data['target_spec'])
            
            # --- LOSS 2: CONTENT (L1 on Wav2Vec2 Features) ---
            L_content = torch.tensor(0.0, device=self.device)
            if self.lambda_content > 0:
                pred_content = self.content_encoder.get_content(data['pred_wav'])
                # Match lengths (ISTFT can cause 1-sample difference)
                min_len = min(pred_content.size(1), data['target_content'].size(1))
                L_content = self.l1_loss(
                    pred_content[:, :min_len, :], 
                    data['target_content'][:, :min_len, :]
                )

            # --- LOSS 3: STYLE (Cosine Distance on WavLM Features) ---
            L_style = torch.tensor(0.0, device=self.device)
            if self.lambda_style > 0:
                pred_style = self.style_encoder.get_style(data['pred_wav'])
                # Cosine embedding loss expects label 1 (similar)
                target_ones = torch.ones(pred_style.size(0), device=self.device)
                L_style = self.cosine_loss(pred_style, data['target_style'], target_ones)

            # --- TOTAL LOSS --- 
            loss = (self.lambda_rec * L_rec) + \
                   (self.lambda_content * L_content) + \
                   (self.lambda_style * L_style)
            
            # Monitor Identity Error
            I_batch = self.I.expand_as(data['A_matrix'])
            id_error = torch.mean((data['A_matrix'] - I_batch) ** 2)

            loss.backward()
            self.optimizer.step()
            
            # Logging
            logs['total'] += loss.item()
            logs['rec'] += L_rec.item()
            logs['cont'] += L_content.item()
            logs['sty'] += L_style.item()
            logs['id'] += id_error.item()
            batch_count += 1
            
            pbar.set_postfix({
                'L_tot': f"{loss.item():.4f}", 
                'L_rec': f"{L_rec.item():.4f}",
                'A-Err': f"{id_error.item():.4f}"
            })

        # Average logs
        for k in logs: 
            logs[k] /= max(1, batch_count)
        return logs

    def train(self, num_epochs, save_path=None):
        print(f"\n{'='*60}")
        print(f"Training Linearizer - {num_epochs} Epochs")
        print(f"Config: λ_rec={self.lambda_rec} | λ_content={self.lambda_content} | λ_style={self.lambda_style}")
        print(f"{'='*60}\n")
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            metrics = self.train_epoch(epoch + 1)
            
            self.history['total_loss'].append(metrics['total'])
            self.history['rec_loss'].append(metrics['rec'])
            self.history['content_loss'].append(metrics['cont'])
            self.history['style_loss'].append(metrics['sty'])
            self.history['identity_error'].append(metrics['id'])
            
            print(f"Epoch {epoch+1}: Total {metrics['total']:.5f} | Rec {metrics['rec']:.5f} | "
                  f"Content {metrics['cont']:.5f} | Style {metrics['sty']:.5f} | A-Err {metrics['id']:.5f}")
            
            if save_path and metrics['total'] < best_loss:
                best_loss = metrics['total']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'history': self.history
                }, save_path)
                print(f"✅ Saved Best Model")
                
        return self.history


# ==============================================================================
# Flow-Based Linearizer Trainer (UNet velocity in latent + integration)
# ==============================================================================
class FlowLinearizerTrainer:
    """
    Trainer for Flow-based linearization models (LinearizerFlow).
    
    Uses unpaired data with frozen WavLM + HuBERT encoders to extract:
      - Style: WavLM embeddings (speaker/singer characteristics)
      - Content: HuBERT features (lyrics, melody, timing)
    
    The model learns a velocity field in squeezed spectrogram space,
    integrated via ODE solvers to convert content spectrograms to target style.
    """
    
    def __init__(
        self,
        model,                 # models.LinearizerFlow
        processor,             # utils.AudioProcessor
        style_encoder,         # StyleEncoderWrapper (frozen)
        content_encoder,       # ContentEncoderWrapper (frozen)
        train_loader,
        val_loader,
        optimizer,
        device="cuda",
        lambda_content=1.0,
        lambda_style=1.0,
        lambda_id=0.1,         # identity/reconstruction weight
        p_identity=0.2,        # fraction of batches forced to use style=content
        steps=20,
        solver="euler",
    ):
        """
        Initialize the Flow Linearizer Trainer.
        
        Args:
            model: LinearizerFlow model instance
            processor: AudioProcessor for STFT/iSTFT
            style_encoder: Frozen StyleEncoderWrapper (WavLM)
            content_encoder: Frozen ContentEncoderWrapper (HuBERT)
            train_loader: DataLoader for training (UnpairedVocalChunksDataset)
            val_loader: DataLoader for validation
            optimizer: torch.optim optimizer
            device: 'cuda' or 'cpu'
            lambda_content: Weight for content loss
            lambda_style: Weight for style loss
            lambda_id: Weight for identity/reconstruction loss
            p_identity: Probability of forcing style=content each batch
            steps: Number of ODE integration steps
            solver: ODE solver ('euler', 'rk4', etc.)
        """
        self.model = model.to(device)
        self.processor = processor
        self.style_encoder = style_encoder.to(device)
        self.content_encoder = content_encoder.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.lambda_id = lambda_id
        self.p_identity = p_identity

        self.steps = steps
        self.solver = solver

        # Loss functions
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()

        # Training history
        self.history = {"train_total": [], "val_total": []}

        # Freeze encoders (they provide ground truth features, don't train)
        self.style_encoder.eval()
        self.content_encoder.eval()
        for p in self.style_encoder.parameters():
            p.requires_grad = False
        for p in self.content_encoder.parameters():
            p.requires_grad = False

    def _make_cond(self, style_wav, content_wav):
        """
        Extract conditioning vectors from audio waveforms.
        
        Args:
            style_wav: (B, time) - reference waveform for style
            content_wav: (B, time) - reference waveform for content/lyrics
        
        Returns:
            cond: (B, 1536) - concatenated style+content representation
            style_vec: (B, 768) - WavLM style embedding
            content_feat: (B, T', 768) - HuBERT time-series features
        """
        # Extract style: WavLM -> (B, 768)
        style_vec = self.style_encoder.get_style(style_wav)

        # Extract content: HuBERT -> (B, T', 768), pool to (B, 768)
        content_feat = self.content_encoder.get_content(content_wav)
        content_vec = content_feat.mean(dim=1)  # Pool over time

        # Concatenate for conditioning: (B, 1536)
        cond = torch.cat([style_vec, content_vec], dim=1)
        
        return cond, style_vec, content_feat

    def _forward_once(self, content_wav, style_wav):
        """
        Single forward pass: content STFT -> flow -> target spectrogram.
        
        Args:
            content_wav: (B, time) - waveform carrying lyrical content
            style_wav: (B, time) - waveform carrying target style/speaker
        
        Returns:
            Dictionary with:
              - xhat_log: output log-magnitude spectrogram
              - x_log: input log-magnitude (for reconstruction loss)
              - xhat_wav: reconstructed waveform (gradients preserved)
              - style_vec: target style embedding
              - content_feat: content features for loss computation
        """
        # (1) Content waveform -> STFT (log-magnitude + phase)
        x_log, x_phase = self.processor.to_spectrogram(content_wav)  # log1p magnitude
        if x_log.dim() == 3:
            x_log = x_log.unsqueeze(1)    # (B, F, T) -> (B, 1, F, T)
            x_phase = x_phase.unsqueeze(1)

        # (2) Extract conditioning from waveforms
        cond, style_vec, content_feat = self._make_cond(style_wav, content_wav)

        # (3) Run flow model in squeezed latent space
        x_sq = squeeze(x_log)              # (B, 1, F, T) -> (B, 4, F/2, T/2)
        xhat_sq = self.model(x_sq, cond=cond, steps=self.steps, solver=self.solver)
        xhat_log = unsqueeze(xhat_sq)      # (B, 4, ...) -> (B, 1, F, T)

        # (4) Crop phase/log to match output shape (squeeze may trim odd dimensions)
        x_log_c = x_log[:, :, :xhat_log.shape[2], :xhat_log.shape[3]]
        x_phase_c = x_phase[:, :, :xhat_log.shape[2], :xhat_log.shape[3]]

        # (5) Inverse STFT (keep gradients for backprop)
        xhat_wav = logmag_to_waveform_with_gradients(self.processor, xhat_log, x_phase_c)

        return {
            "xhat_log": xhat_log,
            "x_log": x_log_c,
            "xhat_wav": xhat_wav,
            "style_vec": style_vec,
            "content_feat": content_feat,
        }

    def train_epoch(self, epoch_idx, num_epochs):
        """
        Train for one epoch.
        
        Args:
            epoch_idx: Current epoch (1-indexed)
            num_epochs: Total epochs
        
        Returns:
            Average training loss
        """
        self.model.train()
        total = 0.0
        n = 0

        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm

        pbar = tqdm(self.train_loader, desc=f"Flow Epoch {epoch_idx}/{num_epochs}", leave=True)

        for batch in pbar:
            content_wav = batch["content_wav"].to(self.device)
            
            # Check if we have pre-computed embeddings or raw style waveforms
            if "style_embedding" in batch:
                # Pre-computed WavLM embeddings (NEW: much faster!)
                style_vec = batch["style_embedding"].to(self.device)
                
                # Extract content features
                content_feat = self.content_encoder.get_content(content_wav)
                content_vec = content_feat.mean(dim=1)
                
                # Build conditioning
                cond = torch.cat([style_vec, content_vec], dim=1)
                
            else:
                # Raw waveforms (OLD: compute embeddings on-the-fly)
                style_wav = batch["style_wav"].to(self.device)
                
                # Occasionally force identity pairs
                if np.random.rand() < self.p_identity:
                    style_wav = content_wav
                
                # Extract both style and content
                style_vec = self.style_encoder.get_style(style_wav)
                content_feat = self.content_encoder.get_content(content_wav)
                content_vec = content_feat.mean(dim=1)
                
                # Build conditioning
                cond = torch.cat([style_vec, content_vec], dim=1)

            self.optimizer.zero_grad()

            # Forward pass (content_wav -> spectrogram -> flow)
            x_log, x_phase = self.processor.to_spectrogram(content_wav)
            if x_log.dim() == 3:
                x_log = x_log.unsqueeze(1)
                x_phase = x_phase.unsqueeze(1)
            
            x_sq = squeeze(x_log)
            xhat_sq = self.model(x_sq, cond=cond, steps=self.steps, solver=self.solver)
            xhat_log = unsqueeze(xhat_sq)
            
            x_log_c = x_log[:, :, :xhat_log.shape[2], :xhat_log.shape[3]]
            x_phase_c = x_phase[:, :, :xhat_log.shape[2], :xhat_log.shape[3]]
            
            xhat_wav = logmag_to_waveform_with_gradients(self.processor, xhat_log, x_phase_c)

            # --- Loss Computation ---
            
            # Content Loss
            pred_content = self.content_encoder.get_content(xhat_wav)
            min_len = min(pred_content.size(1), content_feat.size(1))
            Lc = self.l1(pred_content[:, :min_len, :], content_feat[:, :min_len, :])

            # Style Loss (using pre-computed or extracted embedding)
            pred_style = self.style_encoder.get_style(xhat_wav)
            ones = torch.ones(pred_style.size(0), device=self.device)
            Ls = self.cos(pred_style, style_vec, ones)

            # Identity Loss
            Lid = self.mse(xhat_log, x_log_c)

            # Total loss
            loss = self.lambda_content * Lc + self.lambda_style * Ls + self.lambda_id * Lid
            
            loss.backward()
            self.optimizer.step()

            total += loss.item()
            n += 1
            
            if n % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{total/n:.4f}",
                    "Lc": f"{Lc.item():.3f}",
                    "Ls": f"{Ls.item():.3f}"
                })

        return total / max(1, n)

    @torch.no_grad()
    def validate(self):
        """
        Validation loop (no gradients).
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total = 0.0
        n = 0
        
        for batch in self.val_loader:
            content_wav = batch["content_wav"].to(self.device)
            
            # Check if we have pre-computed embeddings or raw style waveforms
            if "style_embedding" in batch:
                # Pre-computed WavLM embeddings
                style_vec = batch["style_embedding"].to(self.device)
                content_feat = self.content_encoder.get_content(content_wav)
                content_vec = content_feat.mean(dim=1)
                cond = torch.cat([style_vec, content_vec], dim=1)
            else:
                # Raw waveforms
                style_wav = batch["style_wav"].to(self.device)
                style_vec = self.style_encoder.get_style(style_wav)
                content_feat = self.content_encoder.get_content(content_wav)
                content_vec = content_feat.mean(dim=1)
                cond = torch.cat([style_vec, content_vec], dim=1)

            # Forward pass
            x_log, x_phase = self.processor.to_spectrogram(content_wav)
            if x_log.dim() == 3:
                x_log = x_log.unsqueeze(1)
                x_phase = x_phase.unsqueeze(1)
            
            x_sq = squeeze(x_log)
            xhat_sq = self.model(x_sq, cond=cond, steps=self.steps, solver=self.solver)
            xhat_log = unsqueeze(xhat_sq)
            
            x_log_c = x_log[:, :, :xhat_log.shape[2], :xhat_log.shape[3]]
            x_phase_c = x_phase[:, :, :xhat_log.shape[2], :xhat_log.shape[3]]
            
            xhat_wav = logmag_to_waveform_with_gradients(self.processor, xhat_log, x_phase_c)

            # Content loss
            pred_content = self.content_encoder.get_content(xhat_wav)
            min_len = min(pred_content.size(1), content_feat.size(1))
            Lc = self.l1(pred_content[:, :min_len, :], content_feat[:, :min_len, :])

            # Style loss
            pred_style = self.style_encoder.get_style(xhat_wav)
            ones = torch.ones(pred_style.size(0), device=self.device)
            Ls = self.cos(pred_style, style_vec, ones)

            # Identity loss
            Lid = self.mse(xhat_log, x_log_c)

            loss = self.lambda_content * Lc + self.lambda_style * Ls + self.lambda_id * Lid
            total += loss.item()
            n += 1

        return total / max(1, n)

    def train(self, num_epochs, save_path=None):
        """
        Full training loop for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Optional path to save best model checkpoint
        
        Returns:
            Training history dictionary
        """
        for ep in range(1, num_epochs + 1):
            tr = self.train_epoch(ep, num_epochs)
            va = self.validate()
            
            self.history["train_total"].append(tr)
            self.history["val_total"].append(va)
            
            print(f"🔄 Epoch {ep}: train={tr:.5f} | val={va:.5f}")

            # Save checkpoint
            if save_path is not None:
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "history": self.history
                }, save_path)

        return self.history


# ==============================================================================
# Metrics Calculation (use this in notebooks!!!)
# ==============================================================================
def calculate_metrics(reference, estimate, sr=22050):
    """
    Calculates SDR, SIR, SAR using museval.
    Returns a dict of metrics.
    """
    import museval
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
        
        # Pass the pre-allocated window
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
        
        # Pass window here too
        waveform = torch.istft(
            complex_spec, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window
        )
        return waveform.cpu().numpy()

# ==============================================================================
# Log-Magnitude to Waveform (Gradient-Preserving)
# ==============================================================================
def logmag_to_waveform_with_gradients(processor, log_mag, phase):
    """
    Converts log-magnitude and phase spectrograms back to waveform.
    Preserves gradients for backpropagation (unlike AudioProcessor.to_waveform).
    
    This function is designed for use in model forward passes where you need
    gradient flow through the inverse STFT operation.
    
    Args:
        processor: AudioProcessor instance (for n_fft, hop_length, window)
        log_mag: Log-magnitude spectrogram, shape (B, 1, F, T) or (B, F, T)
        phase: Phase spectrogram, shape (B, 1, F, T) or (B, F, T)
    
    Returns:
        waveform: Tensor of shape (B, time) with gradients enabled
    """
    # Remove channel dimension if present (squeezed from 4D to 3D)
    if log_mag.dim() == 4:
        log_mag = log_mag.squeeze(1)
    if phase.dim() == 4:
        phase = phase.squeeze(1)

    # Handle frequency dimension mismatch
    # If squeeze() was applied during preprocessing, freq bins may be 1024 instead of 1025
    if log_mag.shape[1] == 1024:
        log_mag = torch.nn.functional.pad(log_mag, (0, 0, 0, 1), value=0.0)
        phase   = torch.nn.functional.pad(phase,   (0, 0, 0, 1), value=0.0)

    # Reconstruct magnitude from log scale
    mag = torch.expm1(log_mag).clamp_min(0.0)
    
    # Create complex spectrogram using polar coordinates
    complex_spec = torch.polar(mag, phase)

    # Inverse STFT (preserves gradients)
    wav = torch.istft(
        complex_spec,
        n_fft=processor.n_fft,
        hop_length=processor.hop_length,
        window=processor.window,
        return_complex=False,
    )
    return wav

# ==============================================================================
# 2. DATASET
# ==============================================================================

# Robust waveform dataset for general use
class StandardDataset(Dataset):
    """
    Loads pairs from cached files for training/validation.
    Supports both waveforms (.npy) and spectrograms (.npz).
    Returns dicts with keys 'mix' and 'tgt'.
    """
    def __init__(self, mix_files, tgt_files):
        self.mix_files = list(mix_files)
        self.tgt_files = list(tgt_files)
        
        # Auto-detect file format from first file
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
            # Load spectrograms from .npz files
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
            # Load waveforms from .npy files
            m = np.load(self.mix_files[idx])
            t = np.load(self.tgt_files[idx])
            return {
                'mix': torch.tensor(m, dtype=torch.float32),
                'tgt': torch.tensor(t, dtype=torch.float32)
            }

# Chunked dataset for fixed-length segments with overlap
# ==============================================================================
# CONFIGURATION FUNCTIONS
# ==============================================================================
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


def plot_linearizer_losses(history, title="Linearizer Training"):
    """
    Plots all loss components from Linearizer training.
    Supports: total_loss, rec_loss, content_loss, style_loss, identity_error
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(history['total_loss']) + 1)
        
        # Plot 1: Total Loss
        axes[0, 0].plot(epochs, history['total_loss'], 'o-', color='navy', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Reconstruction Loss
        axes[0, 1].plot(epochs, history['rec_loss'], 'o-', color='green', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss (MSE)', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Content + Style Loss
        axes[1, 0].plot(epochs, history['content_loss'], 'o-', color='orange', label='Content (L1)', linewidth=2)
        axes[1, 0].plot(epochs, history['style_loss'], 's-', color='purple', label='Style (Cosine)', linewidth=2)
        axes[1, 0].set_title('Content & Style Loss', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Identity Error (A matrix convergence)
        axes[1, 1].plot(epochs, history['identity_error'], 'o-', color='red', linewidth=2)
        axes[1, 1].set_title('A Matrix Distance from Identity', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('||A - I||²')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error plotting losses: {e}")
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
    tgt_instrumental_wav = np.load(tgt_files[song_num])[:n_samples]  # Ground truth instruments (target)

    # Convert to spectrograms
    mix_mag, mix_phase = processor.to_spectrogram(torch.tensor(mix_wav))
    tgt_instrumental_mag, _ = processor.to_spectrogram(torch.tensor(tgt_instrumental_wav))

    # Model prediction
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
        
        # Predicted instruments (what the model outputs directly now)
        pred_instrumental_mag = mask.squeeze(0).squeeze(0) * mix_mag.to(device)
        pred_instrumental_wav = processor.to_waveform(pred_instrumental_mag.cpu(), mix_phase.cpu())

    # Display spectrograms
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
    
    # Initialize processor if saving spectrograms
    if save_spectrograms:
        processor = AudioProcessor(device='cpu')  # Use CPU for preprocessing
    
    print(f"\n{'='*70}")
    print("MUSDB18 PREPROCESSING PIPELINE")
    print(f"{'='*70}\n")
    
    data_format = "spectrograms" if save_spectrograms else "waveforms"
    print(f"💾 Output format: {data_format}")
    
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
    all_chunks = []  # Will store (stage, mixture, target) tuples when stream_save=False
    
    import gc
    
    # Pre-allocate counters for streaming saves
    if stream_save:
        counters = {
            'stage1': {'train': 0, 'val': 0, 'test': 0},
            'stage2': {'train': 0, 'val': 0, 'test': 0}
        }

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
                # Model learns to OUTPUT the INSTRUMENTS (other) directly
                mixture = 0.50 * vocals_chunk + 0.50 * other_chunk
                target = other_chunk  # Target: instruments/other
            else:
                # STAGE 2: Complex 4-source separation (vocals vs all accompaniment)
                # Model learns to OUTPUT the ACCOMPANIMENT (drums+bass+other) directly
                mixture = 0.40 * vocals_chunk + 0.60 * accompaniment_chunk
                target = accompaniment_chunk  # Target: all instruments
            
            # Normalize to prevent clipping
            max_val = max(np.abs(mixture).max(), np.abs(target).max())
            if max_val > 0:
                mixture = mixture / max_val
                target = target / max_val
            
            # Convert to spectrograms if requested
            if save_spectrograms:
                # Convert waveforms to spectrograms
                mix_tensor = torch.from_numpy(mixture.astype(np.float32))
                tgt_tensor = torch.from_numpy(target.astype(np.float32))
                
                mix_mag, mix_phase = processor.to_spectrogram(mix_tensor)
                tgt_mag, tgt_phase = processor.to_spectrogram(tgt_tensor)

                if stream_save:
                    # Decide split on-the-fly
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
                    # Store spectrograms as numpy arrays
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
        
        # Clean up memory every 10 tracks to prevent RAM overflow
        if (track_idx + 1) % 10 == 0:
            gc.collect()
    
    if stream_save:
        total_chunks = sum(counters[s][sp] for s in counters for sp in counters[s])
        stage1_total = sum(counters['stage1'].values())
        stage2_total = sum(counters['stage2'].values())

        print(f"\n📊 Total chunks created: {total_chunks}")
        print(f"   Stage 1: {stage1_total} chunks ({(stage1_total / max(1, total_chunks)) * 100:.1f}%)")
        print(f"   Stage 2: {stage2_total} chunks ({(stage2_total / max(1, total_chunks)) * 100:.1f}%)\n")
    else:
        # Shuffle and split chunks
        print(f"\n📊 Total chunks created: {len(all_chunks)}")
        
        # Clean memory before shuffling
        gc.collect()
        
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
            
            if save_spectrograms:
                # Save spectrograms as .npz files (magnitude + phase)
                for idx, (_, mix_mag, mix_phase, tgt_mag, tgt_phase) in enumerate(split_chunks):
                    np.savez(mix_dir / f"{idx:06d}.npz", magnitude=mix_mag, phase=mix_phase)
                    np.savez(tgt_dir / f"{idx:06d}.npz", magnitude=tgt_mag, phase=tgt_phase)
            else:
                # Save waveforms as .npy files
                for idx, (_, mixture, target) in enumerate(split_chunks):
                    np.save(mix_dir / f"{idx:06d}.npy", mixture)
                    np.save(tgt_dir / f"{idx:06d}.npy", target)
            
            counts[split_name] = len(split_chunks)
        
        return counts
    
    print("💾 Saving organized data...")
    if stream_save:
        stage1_counts = counters['stage1']
        stage2_counts = counters['stage2']
    else:
        stage1_counts = split_and_save(stage1_chunks, 'stage1')
        
        # Free memory after Stage 1 save
        del stage1_chunks
        gc.collect()
        
        stage2_counts = split_and_save(stage2_chunks, 'stage2')
        
        # Free memory after Stage 2 save
        del stage2_chunks, all_chunks
        gc.collect()
    
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
    
    total_chunks = sum(stage1_counts.values()) + sum(stage2_counts.values())
    return {
        'stage1': stage1_counts,
        'stage2': stage2_counts,
        'total_chunks': total_chunks
    }

# ==============================================================================
# DATA LOADING UTILITIES
# ==============================================================================

def _collate_dict_batch(batch):
    """
    Module-level collate function to batch dictionary-based samples.
    Must be at module level to be picklable for multiprocessing.
    """
    mix_batch = [item['mix'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    
    # Check if spectrograms (tuples) or waveforms (tensors)
    if isinstance(mix_batch[0], tuple):
        # Spectrograms: stack magnitude and phase separately
        # Ensure data is tensor and remove extra channel dimension if present
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
        
        # Stack to (batch, freq_bins, time_steps) - NO channel dimension yet
        # The trainer will add it via unsqueeze(1)
        mix_mag = torch.stack(mix_mag_list)
        mix_ph = torch.stack(mix_ph_list)
        tgt_mag = torch.stack(tgt_mag_list)
        tgt_ph = torch.stack(tgt_ph_list)
        return {
            'mix': (mix_mag, mix_ph),
            'tgt': (tgt_mag, tgt_ph)
        }
    else:
        # Waveforms: simple stack
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
    
    # Try both spectrogram (.npz) and waveform (.npy) files
    mix_files = sorted((data_root / 'mixture').glob("*.npz"))
    tgt_files = sorted((data_root / 'target').glob("*.npz"))
    
    if len(mix_files) == 0:
        # Fallback to waveform files
        mix_files = sorted((data_root / 'mixture').glob("*.npy"))
        tgt_files = sorted((data_root / 'target').glob("*.npy"))
    
    if len(mix_files) == 0:
        raise FileNotFoundError(f"No data found in {data_root}")
    
    dataset = StandardDataset(mix_files, tgt_files)
    
    # SPEED FIX: Use 2 workers if GPU is present, otherwise 0
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

# ==============================================================================
# LEGACY CACHING LOGIC (Kept for backward compatibility)
# ==============================================================================
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


# ===============================================================================
# LINEARIZER DATASETS (VOCALS ONLY)
# ===============================================================================

class VocalsDatasetPhase1(Dataset):
    """
    Phase 1 Dataset: Identity Reconstruction (Same Singer).
    Returns the same vocal chunk as both input and target.
    Used to train the Linearizer to learn identity mapping (A ≈ I).
    """
    def __init__(self, vocals_dir):
        self.vocals_dir = Path(vocals_dir)
        self.chunks = sorted(list(self.vocals_dir.glob('*.npy')))
        
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = np.load(self.chunks[idx])
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
        
        # Phase 1: Same chunk as input and target
        return {
            'input': chunk_tensor,
            'target': chunk_tensor,
            'singer': self.chunks[idx].stem.split('_chunk')[0]
        }


class VocalsDatasetPhase2(Dataset):
    """
    Phase 2 Dataset: Cross-Singer Style Transfer.
    Returns different vocal chunks from different singers.
    Used to train the Linearizer for voice conversion.
    """
    def __init__(self, vocals_dir):
        self.vocals_dir = Path(vocals_dir)
        
        # Organize chunks by singer
        self.singers = {}
        for chunk_file in self.vocals_dir.glob('*.npy'):
            singer = chunk_file.stem.split('_chunk')[0]
            if singer not in self.singers:
                self.singers[singer] = []
            self.singers[singer].append(chunk_file)
        
        # Create list of all singers with multiple chunks
        self.singer_list = [s for s, chunks in self.singers.items() if len(chunks) >= 2]
        
        if len(self.singer_list) < 2:
            raise ValueError(f"Need at least 2 singers with multiple chunks. Found {len(self.singer_list)}")
        
        # Create flat list of chunks for indexing
        self.all_chunks = []
        for singer in self.singer_list:
            self.all_chunks.extend(self.singers[singer])
    
    def __len__(self):
        return len(self.all_chunks)
    
    def __getitem__(self, idx):
        # Input chunk
        input_file = self.all_chunks[idx]
        input_singer = input_file.stem.split('_chunk')[0]
        input_chunk = np.load(input_file)
        
        # Target chunk: Different singer, different chunk
        target_singers = [s for s in self.singer_list if s != input_singer]
        target_singer = np.random.choice(target_singers)
        target_file = np.random.choice(self.singers[target_singer])
        target_chunk = np.load(target_file)
        
        return {
            'input': torch.tensor(input_chunk, dtype=torch.float32),
            'target': torch.tensor(target_chunk, dtype=torch.float32),
            'input_singer': input_singer,
            'target_singer': target_singer
        }


def precompute_style_embeddings_from_musdb(musdb_dir, cache_dir, style_encoder, device="cuda"):
    """
    Pre-compute WavLM style embeddings using MUSDB18 full vocals (instead of chunked data).
    
    This is much cleaner than concatenating chunks:
    - Full vocals from MUSDB18 (1-5 min of continuous audio)
    - One WavLM pass per song
    - Save embeddings as .pt files in data/style_embeddings/
    
    Args:
        musdb_dir: Path to MUSDB18 directory (train/valid/test)
        cache_dir: Directory to save embeddings (e.g., data/style_embeddings/train)
        style_encoder: StyleEncoderWrapper (WavLM model)
        device: Device to use
    """
    from tqdm import tqdm
    import librosa
    
    musdb_dir = Path(musdb_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    style_encoder = style_encoder.to(device)
    style_encoder.eval()
    
    # Find all song folders and their vocals.wav files
    all_song_dirs = sorted([d for d in musdb_dir.iterdir() if d.is_dir()])
    
    pbar = tqdm(all_song_dirs, desc="Pre-computing WavLM embeddings from MUSDB18")
    
    with torch.no_grad():
        for song_dir in pbar:
            vocals_file = song_dir / "vocals.wav"
            
            if not vocals_file.exists():
                continue
            
            # Load full vocals (handles variable lengths)
            vocals_wav, sr = librosa.load(str(vocals_file), sr=22050, mono=True)
            vocals_tensor = torch.tensor(vocals_wav, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Compute WavLM embedding
            style_embedding = style_encoder.get_style(vocals_tensor)  # (1, 768)
            
            # Save to cache directory with song name as filename
            song_name = song_dir.name  # e.g., "A Classic Education - NightOwl"
            cache_file = cache_dir / f"{song_name}.pt"
            torch.save(style_embedding.cpu(), cache_file)
            
            pbar.update(1)
    
    pbar.close()
    print(f"✅ Pre-computed {len(all_song_dirs)} embeddings to: {cache_dir}")


class UnpairedVocalChunksDataset(Dataset):
    """
    Unpaired Dataset for Singer Conversion (Content-Style Separation).
    
    Returns:
      - content_wav: SINGLE chunk (carries lyrics + melody) → HuBERT
      - style_embedding: PRE-COMPUTED WavLM embedding from full song (cached)
    
    Embeddings are pre-computed once before training to avoid redundant computation.
    
    Folder structure:
      root/
        singer_A/
          song_name_chunk0.npy
          song_name_chunk1.npy
          song_name_style_embedding.pt    ← Pre-computed, loaded instead of computing
        singer_B/
          ...
    
    Args:
        root_dir: Path to root directory containing singer subdirectories
        embedding_cache_dir: Directory containing cached embeddings (e.g., data/style_embeddings/train)
        enforce_different_singer: If True, content and style come from different singers
    """
    def __init__(self, root_dir, embedding_cache_dir, enforce_different_singer=True):
        self.root_dir = Path(root_dir)
        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.enforce_different_singer = enforce_different_singer

        # Find all .npy files (flat directory structure)
        self.files = sorted(list(self.root_dir.glob("*.npy")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"❌ No .npy files found under {self.root_dir}")

        # Filter out embedding files
        self.files = [f for f in self.files if "_style_embedding" not in f.name]
        
        # Extract artist_id and song_name from flat files
        self.artist_ids = []
        self.song_names = []
        
        for f in self.files:
            filename = f.stem
            
            # Parse: "artist - songname_chunkN"
            if "_chunk" not in filename or " - " not in filename:
                continue
            
            # Split by last _chunk
            parts = filename.rsplit("_chunk", 1)
            if len(parts) != 2:
                continue
            
            song_id_part = parts[0]  # e.g., "artist - songname"
            try:
                chunk_id = int(parts[1])
            except ValueError:
                continue
            
            # Split by " - " to extract artist
            if " - " not in song_id_part:
                continue
            
            artist, song_name = song_id_part.split(" - ", 1)
            
            self.artist_ids.append(artist)
            self.song_names.append(song_name)

        # Build index: artist -> [song_names]
        self.by_artist_song = {}
        for i, (artist, song_name) in enumerate(zip(self.artist_ids, self.song_names)):
            if artist not in self.by_artist_song:
                self.by_artist_song[artist] = set()
            self.by_artist_song[artist].add(song_name)
        
        # Convert to dict of lists
        for artist in self.by_artist_song:
            self.by_artist_song[artist] = sorted(list(self.by_artist_song[artist]))
        
        self.artists = list(self.by_artist_song.keys())
        self.has_artist_structure = len(self.artists) > 1

        total_songs = sum(len(self.by_artist_song[a]) for a in self.artists)
        print(f"✅ {len(self.files):5d} chunks | {len(self.artists):3d} artists | {total_songs:3d} total songs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load CONTENT: Single 8s chunk
        content_wav = np.load(self.files[idx]).astype(np.float32)
        content_wav = torch.tensor(content_wav, dtype=torch.float32)
        content_artist = self.artist_ids[idx]

        # Select STYLE artist (different from content for unpaired learning)
        if self.enforce_different_singer and self.has_artist_structure:
            other_artists = [a for a in self.artists if a != content_artist]
            if len(other_artists) == 0:
                style_artist = content_artist
            else:
                style_artist = other_artists[np.random.randint(0, len(other_artists))]
        else:
            style_artist = self.artists[np.random.randint(0, len(self.artists))]

        # Pick a random song from style_artist
        available_songs = self.by_artist_song[style_artist]
        random_song_name = available_songs[np.random.randint(0, len(available_songs))]

        # Load PRE-COMPUTED WavLM embedding from cache directory
        # Format: cache_dir/Artist - Song.pt
        full_name = f"{style_artist} - {random_song_name}"
        embedding_file = self.embedding_cache_dir / f"{full_name}.pt"
        
        if embedding_file.exists():
            style_embedding = torch.load(embedding_file, map_location="cpu")  # (1, 768)
            style_embedding = style_embedding.squeeze(0)  # (768,)
        else:
            raise FileNotFoundError(
                f"❌ Style embedding not found: {embedding_file}\n"
                f"   Run precompute_style_embeddings_from_musdb() first!\n"
                f"   Cache directory: {self.embedding_cache_dir}"
            )

        return {
            "content_wav": content_wav,
            "style_embedding": style_embedding,  # From MUSDB18 full vocals
            "content_artist": content_artist,
            "style_artist": style_artist,
            "style_song_name": random_song_name,
        }


def get_vocals_loaders(data_dir, phase=1, split='train', batch_size=8, num_workers=None):
    """
    Creates DataLoader for Linearizer training.
    
    Args:
        data_dir: Root data directory
        phase: 1 for identity (same singer), 2 for style transfer (different singers)
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of workers (default: 0 for Windows, 4 otherwise)
    
    Returns:
        DataLoader
    """
    vocals_dir = Path(data_dir) / 'vocals' / split
    
    if not vocals_dir.exists():
        raise FileNotFoundError(f"Vocals directory not found: {vocals_dir}")
    
    # Select dataset based on phase
    if phase == 1:
        dataset = VocalsDatasetPhase1(vocals_dir)
    elif phase == 2:
        dataset = VocalsDatasetPhase2(vocals_dir)
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 1 or 2.")
    
    # Auto-detect num_workers
    if num_workers is None:
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 4
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    
    print(f"✅ Loaded {split} data: {len(dataset)} chunks, Phase {phase}")
    return loader


# ===============================================================================
# MUSDB18 STEM LOADING (for preprocessing)
# ===============================================================================
def load_musdb_stems(track_folder, sr=22050):
    """
    Loads the four MUSDB18 stems (vocals, drums, bass, other) from a track folder.
    Returns a dict: {'vocals': ..., 'drums': ..., 'bass': ..., 'other': ...} (all mono, resampled to sr)
    """
    import librosa
    from pathlib import Path
    stems = {}
    for stem in ['vocals', 'drums', 'bass', 'other']:
        # Try .wav first, fallback to .mp4
        wav_path = Path(track_folder) / f"{stem}.wav"
        mp4_path = Path(track_folder) / f"{stem}.mp4"
        if wav_path.exists():
            audio, file_sr = librosa.load(wav_path, sr=None, mono=True)
        elif mp4_path.exists():
            audio, file_sr = librosa.load(mp4_path, sr=None, mono=True)
        else:
            raise FileNotFoundError(f"Stem file not found for {stem} in {track_folder}")
        # Resample if needed
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        stems[stem] = audio
    return stems


# ===============================================================================
# MUSDB18 PREPROCESSING FUNCTIONS
# ===============================================================================

# Stage 1: mix = vocals+other, target = other (single stem output)
def process_stage1() -> None:
    print(f"\n{'='*70}\nSTAGE 1: vocals+other → other (single stem)\n{'='*70}")
    for split, folder in MUSDB_SPLITS.items():
        out_dir = DATA_DIR / 'stage1' / split
        mix_dir = out_dir / 'mixture'
        tgt_dir = out_dir / 'target'
        if mix_dir.exists() and tgt_dir.exists() and any(mix_dir.glob('*.npy')) and any(tgt_dir.glob('*.npy')):
            print(f"⏭️  {split}: already exists, skipping...")
            continue
        print(f"Processing {split} ({folder})...")
        mix_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)
        for track_folder in folder.iterdir():
            stems = load_musdb_stems(track_folder, sr=SAMPLE_RATE)  # expects dict: {'vocals', 'drums', 'bass', 'other'}
            vocals = stems['vocals']
            other = stems['other']
            # Mix = vocals + other
            mix = vocals + other
            # Target = other
            target = other
            # Chunking
            total_len: int = min(len(mix), len(target))
            step = int((CHUNK_DURATION - CHUNK_OVERLAP) * SAMPLE_RATE)
            chunk_len = int(CHUNK_DURATION * SAMPLE_RATE)
            for i, start in enumerate(range(0, total_len - chunk_len + 1, step)):
                mix_chunk = mix[start:start+chunk_len]
                tgt_chunk = target[start:start+chunk_len]
                np.save(mix_dir / f"{track_folder.name}_chunk{i}.npy", mix_chunk)
                np.save(tgt_dir / f"{track_folder.name}_chunk{i}.npy", tgt_chunk)
        print(f"✅ {split} complete: {len(list(mix_dir.glob('*.npy')))} chunks")

# Stage 2: mix = all 4 stems, target = full accompaniment (drums+bass+other)
def process_stage2() -> None:
    print(f"\n{'='*70}\nSTAGE 2: all 4 stems → accompaniment (3 stems)\n{'='*70}")
    for split, folder in MUSDB_SPLITS.items():
        out_dir = DATA_DIR / 'stage2' / split
        mix_dir = out_dir / 'mixture'
        tgt_dir = out_dir / 'target'
        if mix_dir.exists() and tgt_dir.exists() and any(mix_dir.glob('*.npy')) and any(tgt_dir.glob('*.npy')):
            print(f"⏭️  {split}: already exists, skipping...")
            continue
        print(f"Processing {split} ({folder})...")
        mix_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)
        for track_folder in folder.iterdir():
            stems = load_musdb_stems(track_folder, sr=SAMPLE_RATE)  # expects dict: {'vocals', 'drums', 'bass', 'other'}
            vocals = stems['vocals']
            drums = stems['drums']
            bass = stems['bass']
            other = stems['other']
            # Mix = vocals + drums + bass + other
            mix = vocals + drums + bass + other
            # Target = accompaniment (drums + bass + other)
            target = drums + bass + other
            # Chunking
            total_len: int = min(len(mix), len(target))
            step = int((CHUNK_DURATION - CHUNK_OVERLAP) * SAMPLE_RATE)
            chunk_len = int(CHUNK_DURATION * SAMPLE_RATE)
            for i, start in enumerate(range(0, total_len - chunk_len + 1, step)):
                mix_chunk = mix[start:start+chunk_len]
                tgt_chunk = target[start:start+chunk_len]
                np.save(mix_dir / f"{track_folder.name}_chunk{i}.npy", mix_chunk)
                np.save(tgt_dir / f"{track_folder.name}_chunk{i}.npy", tgt_chunk)
        print(f"✅ {split} complete: {len(list(mix_dir.glob('*.npy')))} chunks")


# Vocals Only: Clean vocal stems for Linearizer (voice conversion)
def process_vocals_for_linearizer() -> None:
    """
    Extracts clean vocal stems from MUSDB18 for Linearizer training.
    Organizes by singer (track name) for Phase 1 (identity) and Phase 2 (style transfer).
    
    Output structure:
        data/vocals/train/*.npy  (chunks named: SingerName_chunk0.npy, etc.)
        data/vocals/val/*.npy
        data/vocals/test/*.npy
    
    Each chunk is named with the singer/track name for easy pairing in Phase 2.
    """
    print(f"\n{'='*70}\nVOCALS PREPROCESSING: Clean vocal stems for Linearizer\n{'='*70}")
    
    for split, folder in MUSDB_SPLITS.items():
        out_dir = DATA_DIR / 'vocals' / split
        
        # Check if already processed
        if out_dir.exists() and any(out_dir.glob('*.npy')):
            print(f"⏭️  {split}: already exists, skipping...")
            continue
        
        print(f"Processing {split} ({folder})...")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for track_folder in folder.iterdir():
            # Load only vocals
            stems = load_musdb_stems(track_folder, sr=SAMPLE_RATE)
            vocals = stems['vocals']
            
            # Chunking
            step = int((CHUNK_DURATION - CHUNK_OVERLAP) * SAMPLE_RATE)
            chunk_len = int(CHUNK_DURATION * SAMPLE_RATE)
            
            for i, start in enumerate(range(0, len(vocals) - chunk_len + 1, step)):
                vocal_chunk = vocals[start:start+chunk_len]
                # Save with singer name for easy organization
                np.save(out_dir / f"{track_folder.name}_chunk{i}.npy", vocal_chunk)
        
        num_chunks = len(list(out_dir.glob('*.npy')))
        num_singers = len(set(f.stem.split('_chunk')[0] for f in out_dir.glob('*.npy')))
        print(f"✅ {split} complete: {num_chunks} chunks from {num_singers} singers")


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


# ===============================================================================
# MODEL A COMPARISON UTILITIES
# ===============================================================================

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
        
        min_len: int = min(len(mix_wav), len(tgt_wav))
        mix_wav = mix_wav[:min_len]
        tgt_wav = tgt_wav[:min_len]
        
        # Convert to spectrograms
        mix_mag, mix_phase = processor_lstm.to_spectrogram(torch.tensor(mix_wav))
        # mix_mag is already [1, freq, time] from to_spectrogram, just add batch dim
        mix_mag_in = mix_mag.unsqueeze(0).to(device)  # [1, 1, freq, time]
        
        # LSTM prediction
        with torch.no_grad():
            mask_lstm = model_lstm(mix_mag_in)
            # Apply mask in LINEAR domain
            est_linear_lstm = mask_lstm.squeeze(0) * torch.expm1(mix_mag.to(device))
            # Convert back to log domain for waveform reconstruction
            est_mag_lstm = torch.log1p(est_linear_lstm)
            est_wav_lstm = processor_lstm.to_waveform(est_mag_lstm.squeeze(0).cpu(), mix_phase.squeeze(0).cpu())
        
        # U-Net prediction
        with torch.no_grad():
            mask_unet = model_unet(mix_mag_in)
            # Apply mask in LINEAR domain
            est_linear_unet = mask_unet.squeeze(0) * torch.expm1(mix_mag.to(device))
            # Convert back to log domain for waveform reconstruction
            est_mag_unet = torch.log1p(est_linear_unet)
            est_wav_unet = processor_unet.to_waveform(est_mag_unet.squeeze(0).cpu(), mix_phase.squeeze(0).cpu())
        
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


# ==============================================================================
# Test Evaluation Functions
# ==============================================================================

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
    print(f"✅ Saved test results to: {ckpt_path.name}")


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

            # Check if data is already spectrograms (tuple) or waveforms (tensor)
            if isinstance(mix, tuple):
                # Already spectrograms - extract magnitude and move to device
                mix_mag = mix[0].to(device)
                tgt_mag = target[0].to(device)
                
                # Add channel dimension if needed (model expects 4D: batch, channel, freq, time)
                if mix_mag.dim() == 3:
                    mix_mag = mix_mag.unsqueeze(1)
                if tgt_mag.dim() == 3:
                    tgt_mag = tgt_mag.unsqueeze(1)
                
                mix_processed = mix_mag
                target_processed = tgt_mag
            else:
                # Waveforms - convert to spectrograms
                mix = mix.to(device)
                target = target.to(device)
                
                mix_spec = processor.to_spectrogram(mix)
                target_spec = processor.to_spectrogram(target)
                
                # Extract magnitude and add channel dimension
                mix_processed = mix_spec[0].unsqueeze(1) if mix_spec[0].dim() == 3 else mix_spec[0]
                target_processed = target_spec[0].unsqueeze(1) if target_spec[0].dim() == 3 else target_spec[0]

            # Forward pass (model outputs a mask)
            mask = model(mix_processed)
            
            # Ensure mask shape matches input
            if mask.shape != mix_processed.shape:
                mask = mask[:, :, :mix_processed.shape[2], :mix_processed.shape[3]]
            
            # Apply mask in LINEAR domain (same as training)
            est_linear = mask * torch.expm1(mix_processed)
            est_log = torch.log1p(est_linear)
            
            # Compute loss in LOG domain
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
            
            # Ensure shapes match
            if mask.shape != mag.shape:
                mask = mask[:mag.shape[0], :mag.shape[1]]
            
            est_mag = torch.log1p(mask * torch.expm1(mag))
            est_seg = processor.to_waveform(est_mag.cpu().numpy(), phase.cpu().numpy())
        
        valid_len = min(len(segment), len(audio) - pos)
        # Ensure est_seg matches the expected length
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
    # If first dimension is smaller, it's likely (freq, time) already - keep it
    # If first dimension is larger, it's likely (time, freq) - transpose it
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
    stride = chunk_samples // 2  # 50% overlap
    output = np.zeros_like(audio)
    weights = np.zeros_like(audio)
    
    # Step 1: Collect all chunks first
    chunks_data = []
    for pos in range(0, len(audio), stride):
        segment = audio[pos:pos + chunk_samples]
        if len(segment) < chunk_samples:
            segment = np.pad(segment, (0, chunk_samples - len(segment)))
        
        # Convert to spectrogram
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
    
    # Step 2: Process chunks in batches through forward pass
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, total_chunks, batch_size):
            if batch_start % (batch_size * 5) == 0:
                print(".", end="", flush=True)
            
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks_data[batch_start:batch_end]
            
            # Stack magnitudes into a batch tensor [batch_size, 1, freq, time]
            batch_mags = torch.stack([chunk['mag'].unsqueeze(0) for chunk in batch_chunks]).to(device)
            
            # Forward pass on entire batch (THIS IS THE KEY - parallel processing!)
            batch_masks = model(batch_mags)
            
            # Step 3: Reconstruct each chunk in the batch
            for i, chunk in enumerate(batch_chunks):
                mask = batch_masks[i].squeeze()
                mag = chunk['mag']
                phase = chunk['phase']
                pos = chunk['pos']
                valid_len = chunk['valid_len']
                
                # Ensure shapes match
                if mask.shape != mag.shape:
                    mask = mask[:mag.shape[0], :mag.shape[1]]
                
                # Apply mask in log domain
                est_mag = torch.log1p(mask * torch.expm1(mag))
                est_seg = processor.to_waveform(est_mag.cpu().numpy(), phase.cpu().numpy())
                
                # Ensure est_seg matches expected length
                if len(est_seg) > valid_len:
                    est_seg = est_seg[:valid_len]
                elif len(est_seg) < valid_len:
                    est_seg = np.pad(est_seg, (0, valid_len - len(est_seg)))
                
                # Apply window and accumulate
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
    print(f"\n📂 Loading: {file_path}")
    
    # Load audio file
    try:
        audio, file_sr = librosa.load(file_path, sr=None, mono=True)
        print(f"   Original SR: {file_sr} Hz, Duration: {len(audio)/file_sr:.2f}s")
        
        # Resample if needed
        if file_sr != sr:
            print(f"   Resampling to {sr} Hz...")
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        
        # Trim to specified duration
        if duration is not None:
            target_samples = int(duration * sr)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
                print(f"   Trimmed to {duration}s")
        
        print(f"✅ Loaded: {len(audio)/sr:.2f}s @ {sr} Hz")
        
    except Exception as e:
        print(f"❌ Error loading audio file: {e}")
        return
    
    # Run inference with both models
    model_lstm.eval()
    model_unet.eval()
    
    print("\n🚀 Running inference...")
    print("LSTM (sequential, batch_size=1):")
    est_lstm = sliding_window_inference(model_lstm, processor_lstm, audio, 
                                       chunk_len=8.0, sr=sr, device=device)
    print(f"U-Net (accelerated batched forward pass, batch_size={unet_batch_size}):")
    est_unet = unet_inference_accelerator(model_unet, processor_unet, audio,
                                         chunk_len=8.0, sr=sr, device=device, batch_size=unet_batch_size)
    
    # Visualization
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
    
    # Audio playback
    print("\n🔊 Audio Playback:\n")
    print("Original Audio:")
    display(Audio(audio, rate=sr))
    
    print("\nLSTM Separated Vocals:")
    display(Audio(est_lstm, rate=sr))
    
    print("\nU-Net Separated Vocals:")
    display(Audio(est_unet, rate=sr))
    
    print(f"\n{'='*70}")
    print("✅ INFERENCE COMPLETE")
    print(f"{'='*70}")