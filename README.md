# Tactile User Authentication Experiments

This repository contains the code used for user authentication experiments on tactile force data collected during tele-operation tasks. The project evaluates both:

* direct time-series modelling, and
* image / frequency-based representations such as STFT, Mel spectrograms, CWT scalograms, and multi-resolution STFT.

The code is mainly written in Python using PyTorch, NumPy, pandas, SciPy, and matplotlib.

---

## Project Setup

It is recommended to run everything inside a Python virtual environment.

### 1. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install --upgrade pip
pip install numpy pandas matplotlib scipy torch torchvision torchaudio
```

If you are using an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build for your system.

---

## Expected Repository Structure

A typical structure is:

```text
tactile_model_training/
├── TAC/
│   ├── load_all.py
│   ├── train_user_transformer_features_paper.py
│   ├── bench_user_per_task_rawnn_stft1d.py
│   ├── bench_user_per_task_rawnn_mel1d.py
│   ├── bench_user_per_task_rawnn_cwt1d.py
│   ├── bench_user_per_task_rawnn_mrstft1d.py
│   └── ...
├── data/
│   ├── u1/
│   │   ├── a/
│   │   │   └── force.csv
│   │   ├── b/
│   │   │   └── force.csv
│   │   └── ...
│   ├── u2/
│   └── ...
├── runs/
└── README.md
```

The dataset is expected to contain user folders (`u1` to `u7`) and task folders (`a` to `g`) with a `force.csv` file in each task folder.

---

## Running Experiments

All commands below assume you are in the project root directory and the virtual environment is activated.

---

## Transformer time-series model

```powershell
python -m TAC.train_user_transformer_features_paper --seq_len 512 --per_user_train 100 --per_user_test 20 --slices_per_csv 120 --slice_frac 0.6 --epochs 100 --batch 16 --lr 1e-4 --d_model 256 --nhead 16 --num_layers 2 --dim_ff 256 --dropout 0.1
```

This runs the per-task Transformer model using the engineered two-stream temporal feature representation.

---

## STFT 1D CNN

```powershell
python -m TAC.bench_user_per_task_rawnn_stft1d --window_len 768 --stride 256 --use_ema --window_norm --stft_n_fft 128 --stft_hop 8 --stft_keep_bins 48 --cnn_base 192 --epochs 40 --batch_size 192 --out_csv runs/stft1d_fair_48.csv
```

---

## Mel Spectrogram 1D CNN

```powershell
python -m TAC.bench_user_per_task_rawnn_mel1d --window_len 768 --stride 256 --use_ema --window_norm --fs 250 --n_fft 128 --hop 8 --mel_bins 48 --mel_fmin 0 --mel_fmax 100 --cnn_base 192 --epochs 40 --batch_size 192 --out_csv runs/mel1d_fair_48.csv
```

---

## CWT 1D CNN

```powershell
python -m TAC.bench_user_per_task_rawnn_cwt1d --window_len 768 --stride 256 --use_ema --window_norm --cwt_scales 48 --cwt_smin 2 --cwt_smax 96 --cwt_w 6 --cnn_base 192 --epochs 40 --batch_size 192 --out_csv runs/cwt_fair_48.csv
```

---

## Multi-resolution STFT 1D CNN

```powershell
python -m TAC.bench_user_per_task_rawnn_mrstft1d --window_len 768 --stride 256 --use_ema --window_norm --stft1_n_fft 128 --stft1_hop 8 --stft1_keep_bins 24 --stft2_n_fft 256 --stft2_hop 16 --stft2_keep_bins 24 --cnn_base 192 --epochs 40 --batch_size 192 --out_csv runs/mrstft1d_fair_48.csv
```

---

## STFT 2D CNN

```powershell
python -m TAC.bench_user_per_task_rawnn_stft2d --window_len 768 --stride 256 --use_ema --window_norm --stft_n_fft 128 --stft_hop 8 --stft_keep_bins 48 --epochs 40 --batch_size 192 --cnn_base 32 --out_csv runs/stft2d_fair_48.csv
```

---

## Outputs

Most scripts will save:

* a CSV summary in the `runs/` folder
* confusion matrix images in `runs/confusion_matrices/`
* confusion matrix CSV files in `runs/confusion_matrices/`

Typical output files include:

```text
runs/stft1d_fair_48.csv
runs/mel1d_fair_48.csv
runs/cwt_fair_48.csv
runs/mrstft1d_fair_48.csv
runs/stft2d_fair_48.csv
runs/confusion_matrices/*.png
runs/confusion_matrices/*_counts.csv
runs/confusion_matrices/*_percent.csv
```

---

## GPU Notes

If CUDA is available, PyTorch should automatically use the GPU in most scripts.

You can check whether CUDA is available with:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
```

If CUDA is not available, the code will fall back to CPU, which may be much slower for larger experiments.

---

## Common Issues

### Virtual environment does not activate in PowerShell

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

### Missing Python packages

Install dependencies again:

```powershell
pip install numpy pandas matplotlib scipy torch torchvision torchaudio
```

### `ModuleNotFoundError: No module named TAC`

Make sure you are running the command from the project root, not from inside the `TAC/` folder.

Correct:

```powershell
python -m TAC.bench_user_per_task_rawnn_stft1d ...
```

from the repository root.

### GPU not being used

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

---

## Notes

The experiments in this repository were run under fixed window length and stride settings to ensure fair comparison across representations under hardware constraints. As a result, not every representation is individually tuned to its absolute best possible setting; instead, the focus is on controlled comparison across methods.

---