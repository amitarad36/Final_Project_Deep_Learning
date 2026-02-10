# Final Project: Deep Learning on Computational Accelerators

## Music Source Separation with Deep Learning

This project implements deep learning architectures for music source separation using curriculum learning.

### Quick Start

**All you need: `local_main.ipynb`** - Single notebook for the entire project!

**Two modes:**
1. **Inference Mode** (`DOWNLOAD_DATA = False`): Use pre-trained checkpoints, no data download
2. **Training Mode** (`DOWNLOAD_DATA = True`): Download MUSDB18 (~4GB) and train from scratch

### Project Structure

```
Final_Project_Deep_Learning/
├── local_main.ipynb          # 👈 MAIN NOTEBOOK - Run this!
├── models/
│   ├── model_A.py            # Time-Frequency U-Net
│   ├── model_B1.py           # (Future)
│   ├── model_B2.py           # (Future)
│   └── utils.py              # All utilities, datasets, training
├── checkpoints/              # Pre-trained models
├── data/                     # Generated training data (if downloaded)
└── README.md
```

## Features

✅ **Realistic Mixture Weights** - Vocals: 35%, Drums: 30%, Bass: 20%, Other: 15%
✅ **ChunkedDataset** - 1-second segments with 30% overlap for consistent training
✅ **Curriculum Learning** - Stage 1 (Vocals+Other→Other), Stage 2 (Full Mix→Other)
✅ **Centralized Configuration** - All hyperparameters in `models/utils.py`
✅ **Optional Data Download** - Choose inference-only or full training mode

## Environment Setup

### Prerequisites
- Anaconda or Miniconda installed
- ~5GB disk space for environment and dependencies

### Installation Steps

1. **Create the conda environment from the yml file:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate final-dl-env
   ```

3. **Verify PyTorch installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   or use VS Code's notebook interface.

### GPU Support (Optional)

If you have an NVIDIA GPU and want CUDA support:

1. Edit `environment.yml` and remove the `- cpuonly` line
2. Add the appropriate CUDA toolkit version (e.g., `- cudatoolkit=11.8`)
3. Recreate the environment:
   ```bash
   conda env remove -n final-dl-env
   conda env create -f environment.yml
   ```

### Troubleshooting

**If PyTorch is not installed:**
```bash
conda activate final-dl-env
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**If kernel crashes in notebook:**
- Make sure you're using the `final-dl-env` kernel in Jupyter
- Restart the kernel and run cells in order
- Check that all dependencies are installed: `conda list`

## Project Structure

```
├── environment.yml          # Conda environment specification
├── models/
│   ├── model_a_unet_freq.py    # Model A implementation
│   └── model_b_demucs_time.py  # Model B implementation
├── notebooks/
│   ├── model_a_unet_freq.ipynb    # Model A training notebook
│   └── model_b_demucs_time.ipynb  # Model B training notebook
├── shared/
│   └── shared_utils.py      # Common utilities
├── checkpoints/             # Saved model checkpoints
├── outputs/                 # Training outputs and visualizations
└── data/                    # Dataset directory
```

## Usage

See the individual notebooks for detailed usage:
- [Model A Notebook](../notebooks/model_a_unet_freq.ipynb)
- [Model B Notebook](../notebooks/model_b_demucs_time.ipynb)
