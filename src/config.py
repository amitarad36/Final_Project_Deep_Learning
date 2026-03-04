GENERAL_CONFIG = {
    'sr': 22050,
    'n_fft': 2048,
    'seed': 42,
    'duration': 120,
    'hop_length': 4
}

SHARED_CONFIG = {
    'device': 'cuda',
    'chunk_duration': 8.0,
    'hop_length': 4.0,
    'chunk_overlap': 0.5,
    'skip_training': True,
    'lr':0.0001,
    'epochs':50,

    'early_stopping': 10
}

LSTM_HYPERPARAMETERS = {
    'channels': 32,
    'epochs': 5,
    'batch_size': 128,
    'learning_rate': 0.001,
    'skip_training': True,
    'freq_bins': 1025,
    'hidden_size': 512,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': True
}

UNET_HYPERPARAMETERS = {
    'channels': 64,
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'skip_training': True,
    'in_channels': 1,
    'out_channels':1,
    'base_filters':32,
    'num_layers':5,
    'batchnorm':True,
    'dropout': 0.1
}

UTEENTION_HYPERPARAMETERS = {
    'batch_size': 32,
    'base_filters': 32,
    'num_layers': 4,
    'num_heads': 4,
    'skip_training': True
}
"""
skip_voc_training = False

results_voc = utils.train_unetattention_vocals_only(
    data_dir=DATA_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    device=device,
    skip_training=skip_voc_training,
    batch_size=4,
    base_filters=32,
    num_layers=5,
    num_heads=4,
    learning_rate=1e-4,
    num_epochs=20
)

model_voc_attn = results_voc['model_voc_attn']
processor_voc = results_voc['processor_voc']
loss_fn_voc = results_voc['loss_fn_voc']
voc_history = results_voc['voc_history']
ckpt_voc_attn = results_voc['ckpt_voc_attn']




SR = 22050
DURATION = 120.0
CHUNK_LEN = 8.0
HOP_LENGTH = 4.0

result = utils.evaluate_vocals_extraction_unetattention(
    data_dir=DATA_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    device=device,
    model_voc_attn=model_voc_attn if 'model_voc_attn' in locals() else None,
    processor_voc=processor_voc if 'processor_voc' in locals() else None,
    #sr=SR,
    duration=DURATION,
    chunk_len=CHUNK_LEN,
    hop_length=HOP_LENGTH,
    base_filters=32,
    num_layers=5,
    num_heads=4,
    auto_select=False
)

extracted_vocals = result['extracted_vocals']
mix_wav = result['mix_wav']
vocal_wav_gt = result['vocal_wav_gt']
"""