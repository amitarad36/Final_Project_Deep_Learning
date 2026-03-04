import torch
import os


class UniversalTrainer:
    def __init__(self, model, train_loader, val_loader, processor, optimizer, loss_fn, device='cpu', early_stopping=10,
                 input_type='spectrogram'):
        self.model = model.to(device)
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.early_stopping = early_stopping
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

        print(f"\n{'=' * 60}")
        print(f"Training: {num_epochs} epochs")
        print(f"{'=' * 60}\n")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch + 1, num_epochs)
            val_loss = self.validate()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{num_epochs} Complete → Train: {train_loss:.5f} | Val: {val_loss:.5f}")

            if log_file_path:
                try:
                    with open(log_file_path, 'a') as f:
                        f.write(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")
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
                            bf.write(
                                f"Best Epoch: {best_epoch}\nTrain Loss: {best_train_loss:.4f}\nVal Loss: {self.best_val_loss:.4f}\n")
                    except Exception as e:
                        print(f"[WARN] Could not write best_epoch.txt: {e}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
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

