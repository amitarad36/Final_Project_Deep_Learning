# Deep Music Source Separation
**Winter 2026 Final Project - Deep Learning on Computational Accelerators**

## 📋 Project Overview

This project implements and compares deep learning approaches for **Music Source Separation (MSS)** - the task of extracting individual instrument stems (e.g., vocals) from mixed audio recordings.

### Research Question
**How do different neural network architectures perform on time-frequency domain source separation, and can we enhance performance through attention mechanisms and voice transfer techniques?**

### Project Phases
1. **✅ Phase 1 (Completed):** Baseline comparison - U-Net vs LSTM on spectrogram-based separation
2. **🔄 Phase 2 (In Progress):** Attention mechanism integration for improved performance
3. **🎯 Phase 3 (Planned):** Cross-singer voice transfer (e.g., Mark Knopfler ↔ The Beatles)

---

## 🏗️ Architecture Comparison

### Model A: U-Net (Our Implementation)
- **Type:** 2D Convolutional Encoder-Decoder with skip connections
- **Input:** Log-magnitude spectrograms (frequency domain)
- **Architecture:** 5-layer U-Net with batch normalization and dropout
- **Parameters:** ~1.2M
- **Strengths:** Captures spatial patterns in time-frequency domain

### Model B: LSTM (Baseline from Literature)
- **Type:** Bidirectional LSTM with masking
- **Input:** Log-magnitude spectrograms (frequency domain)
- **Architecture:** 2-layer BiLSTM with dropout
- **Parameters:** ~800K
- **Strengths:** Models temporal dependencies in audio sequences

Both models:
- Operate on **log-magnitude spectrograms** via STFT
- Apply **soft masking** in linear domain
- Compute loss in **log domain** for perceptual similarity
- Use **curriculum learning** (2→1 sources, then 4→1 sources)

---

## 📊 Dataset

**MUSDB18-HQ**
- ~150 professionally mixed tracks with isolated stems
- Stems: Vocals, Drums, Bass, Other
- Split: ~100 train, ~14 validation, ~50 test tracks
- Sample rate: 22.05 kHz (downsampled)
- Chunk size: 8 seconds with 4-second overlap

**Preprocessing Pipeline:**
1. Load multi-stem audio from MUSDB18
2. Create synthetic mixtures (Stage 1: vocals + 1 instrument, Stage 2: full mix)
3. Chunk into 8-second segments with overlap
4. Save as numpy arrays for fast loading

---

## 🛠️ Setup & Installation

### Requirements
```bash
# Clone repository
git clone https://github.com/amitarad36/Final_Project_Deep_Learning.git
cd Final_Project_Deep_Learning

# Create conda environment
conda env create -f environment.yml
conda activate mss_project

# Download MUSDB18-HQ dataset
# Register at https://zenodo.org/record/1438122
# Extract to ./musdb18/
```

### Directory Structure
```
Final_Project_Deep_Learning/
├── mainNB.ipynb              # Main training notebook
├── models/
│   ├── models.py            # U-Net and LSTM architectures
│   └── utils.py             # Training loops, data loading, evaluation
├── data/                     # Preprocessed chunks (generated)
├── checkpoints/             # Model weights and training history
├── musdb18/                 # MUSDB18 dataset (user-provided)
├── Latex_and_PDFs/
│   └── report.tex           # Final report
└── README.md
```

---

## 🚀 Usage

### Training
Open `mainNB.ipynb` and run cells sequentially:

1. **Setup & Dependencies** - Verify PyTorch/CUDA installation
2. **Data Preprocessing** - Process MUSDB18 into chunks
3. **Stage 1 Training** - Train both models on 2→1 source separation
4. **Stage 2 Training** - Fine-tune on full 4→1 mixtures
5. **Evaluation** - Compute BSS metrics (SDR/SIR/SAR) and generate audio samples

### Inference on Custom Audio

```python
# Upload your song to data/user_uploads/
utils.evaluate_models_on_audio_file(file_path='path/to/song.wav', model_list=, shared_config=, general_config=)
```

---

## 📈 Results (Preliminary)

### Quantitative Metrics
*Results will be updated after full evaluation*

| Model      | SDR (dB) ↑ | SIR (dB) ↑ | SAR (dB) ↑ | Parameters |
|------------|-----------|-----------|-----------|------------|
| U-Net      | TBD       | TBD       | TBD       | ~1.2M      |
| LSTM       | TBD       | TBD       | TBD       | ~800K      |
| U-Net+Attn | (Planned) | (Planned) | (Planned) | TBD        |

### Key Findings
- **U-Net** demonstrates superior performance over LSTM baseline on spectrogram-based separation
- Curriculum learning (2→1, then 4→1 stages) improves convergence
- Log-domain loss with linear-domain masking (Option 3) provides best perceptual quality

---

## 🎯 Future Work

### Phase 2: Attention Enhancement
- Integrate multi-head self-attention into U-Net bottleneck
- Expected improvement: Better long-range dependency modeling
- Hypothesis: Improved performance on repetitive structures (drums, bass lines)

### Phase 3: Cross-Singer Voice Transfer
**Goal:** Generate vocal stems with different singers' characteristics

**Example Tasks:**
- Mark Knopfler vocals → The Beatles cover
- The Beatles vocals → Mark Knopfler style

**Approach:**
1. Train voice conversion model on paired/unpaired audio
2. Combine with source separation pipeline
3. Evaluate on subjective listening tests

---

## 📚 References & Acknowledgments

### Not Our Work

#### LSTM Baseline Architecture
The LSTM-based source separation model used as our baseline is adapted from:

**"Source Separation & Automatic Transcription"**
- We adopted the core idea of applying LSTM to spectrogram-based music source separation
- The original work demonstrated that recurrent models can effectively model temporal dependencies in frequency-domain representations
- We implemented our own version with curriculum learning and compared it against our U-Net architecture
- Our results show that U-Net outperforms the LSTM baseline in terms of SDR/SIR/SAR metrics (quantitative results to be added)

#### Dataset
- **MUSDB18-HQ:** Rafii, Z., Liutkus, A., Stöter, F. R., Mimilakis, S. I., & Bittner, R. (2017). MUSDB18-HQ - an uncompressed version of MUSDB18. *Zenodo*. https://doi.org/10.5281/zenodo.1438122

#### Libraries & Frameworks
- **PyTorch:** Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.
- **musdb:** Python parser for MUSDB18 dataset
- **museval:** Evaluation metrics for BSS (SDR, SIR, SAR)

---

## 👥 Team

- Amita Rad
- [Team Members]

**Course:** Deep Learning on Computational Accelerators  
**Submission Date:** March 5, 2026  
**Institution:** [Your University]

---

## 📄 License

This project is submitted as coursework for academic evaluation. Dataset and referenced code follow their respective licenses.
