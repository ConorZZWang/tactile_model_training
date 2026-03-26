import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

import torch

from TAC.load_all import DATA_ROOT


# =========================================================
# Config
# =========================================================
WINDOW_LEN = 768
STRIDE = 256
USE_EMA = True
EMA_ALPHA = 0.001
WINDOW_NORM = True

# Pick same task for both users. Change if needed.
TASK_NAME = "a"        # e.g. a, b, c, d, e, f, g
USER_A = "u1"
USER_B = "u5"
WINDOW_INDEX = 0       # which extracted window to plot from each file

OUT_DIR = "runs/example_repr_plots"
os.makedirs(OUT_DIR, exist_ok=True)

FS = 250.0

# Fair settings matching your earlier runs
STFT_NFFT = 128
STFT_HOP = 8
STFT_KEEP_BINS = 48

MEL_NFFT = 128
MEL_HOP = 8
MEL_BINS = 48
MEL_FMIN = 0.0
MEL_FMAX = 100.0

CWT_SCALES = 48
CWT_SMIN = 2.0
CWT_SMAX = 96.0
CWT_W = 6.0

MRSTFT1 = (128, 8, 24)   # (n_fft, hop, keep_bins)
MRSTFT2 = (256, 16, 24)


# =========================================================
# Helpers
# =========================================================
def ema_1d(series: np.ndarray, alpha: float) -> np.ndarray:
    v = 0.0
    out = np.empty_like(series, dtype=np.float32)
    for i, s in enumerate(series.astype(np.float32, copy=False)):
        v = alpha * s + (1 - alpha) * (v if i > 0 else s)
        out[i] = v
    return out


def zwin_one(x: np.ndarray) -> np.ndarray:
    """
    x: (3, T)
    """
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mu) / sd


def load_force_csv(user_name: str, task_name: str) -> np.ndarray:
    """
    Returns raw force data as (3, T_full)
    Assumes structure DATA_ROOT / u1 / a / force.csv
    """
    csv_path = Path(DATA_ROOT) / user_name / task_name / "force.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find file: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("force_x", "force_y", "force_z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["force_x", "force_y", "force_z"])

    fx = df["force_x"].to_numpy(dtype=np.float32)
    fy = df["force_y"].to_numpy(dtype=np.float32)
    fz = df["force_z"].to_numpy(dtype=np.float32)

    return np.stack([fx, fy, fz], axis=0)


def preprocess_signal(x: np.ndarray) -> np.ndarray:
    """
    x: (3, T_full)
    """
    out = x.copy()
    if USE_EMA:
        for c in range(3):
            out[c] = ema_1d(out[c], EMA_ALPHA)
    return out


def extract_window(x: np.ndarray, window_len: int, stride: int, window_index: int) -> np.ndarray:
    """
    x: (3, T_full)
    returns: (3, window_len)
    """
    starts = list(range(0, x.shape[1] - window_len + 1, stride))
    if not starts:
        raise ValueError("Signal too short for requested window length.")
    if window_index >= len(starts):
        raise ValueError(f"window_index={window_index} out of range. Total windows={len(starts)}")

    s = starts[window_index]
    w = x[:, s:s + window_len].copy()

    if WINDOW_NORM:
        w = zwin_one(w)
    return w


# =========================================================
# STFT / Mel
# =========================================================
def stft_axis(x: np.ndarray, n_fft: int, hop: int, keep_bins: int) -> np.ndarray:
    """
    x: (T,)
    returns: (keep_bins, W) log-magnitude normalized per bin across time
    """
    xt = torch.tensor(x, dtype=torch.float32)
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        xt,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True
    )
    mag = torch.abs(spec)
    mag = torch.log(mag + 1e-6)

    kb = min(keep_bins, mag.shape[0])
    mag = mag[:kb, :]

    m = mag.mean(dim=1, keepdim=True)
    s = mag.std(dim=1, keepdim=True) + 1e-6
    mag = (mag - m) / s

    return mag.numpy()


def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(fs: float, n_fft: int, n_mels: int, fmin: float, fmax: float) -> torch.Tensor:
    freqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / fs)
    F = freqs.numel()

    mel_min = hz_to_mel(torch.tensor([fmin]))
    mel_max = hz_to_mel(torch.tensor([fmax]))
    mel_pts = torch.linspace(mel_min.item(), mel_max.item(), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    fb = torch.zeros((n_mels, F), dtype=torch.float32)
    for m in range(n_mels):
        f_left, f_center, f_right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        left_slope = (freqs - f_left) / (f_center - f_left + 1e-8)
        right_slope = (f_right - freqs) / (f_right - f_center + 1e-8)
        fb[m] = torch.clamp(torch.minimum(left_slope, right_slope), min=0.0)
    return fb


def mel_axis(x: np.ndarray, fs: float, n_fft: int, hop: int, mel_bins: int, fmin: float, fmax: float) -> np.ndarray:
    """
    x: (T,)
    returns: (mel_bins, W)
    """
    xt = torch.tensor(x, dtype=torch.float32)
    window = torch.hann_window(n_fft)

    spec = torch.stft(
        xt,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = (spec.real ** 2 + spec.imag ** 2)
    power = torch.log(power + 1e-6)

    fb = mel_filterbank(fs, n_fft, mel_bins, fmin, fmax)
    mel = fb @ power

    m = mel.mean(dim=1, keepdim=True)
    s = mel.std(dim=1, keepdim=True) + 1e-6
    mel = (mel - m) / s

    return mel.numpy()


# =========================================================
# CWT
# =========================================================
def morlet2(M: int, s: float, w: float = 6.0) -> np.ndarray:
    t = np.arange(M, dtype=np.float32) - (M - 1) / 2.0
    ts = t / float(s)
    A = np.pi ** (-0.25)
    wave = A * np.exp(1j * w * ts) * np.exp(-0.5 * ts * ts)
    wave = wave / np.sqrt(s)
    return wave.astype(np.complex64, copy=False)


def cwt_1d(x: np.ndarray, widths: np.ndarray, w: float = 6.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    out = np.empty((len(widths), T), dtype=np.complex64)

    for i, width in enumerate(widths):
        M = int(max(1, np.floor(10.0 * float(width))))
        if M % 2 == 0:
            M += 1
        if M > T:
            M = T if T % 2 == 1 else T - 1
            M = max(M, 1)

        wave = morlet2(M, s=float(width), w=w)
        conv = fftconvolve(x, wave[::-1].conj(), mode="same")
        out[i] = conv.astype(np.complex64, copy=False)

    return out


def cwt_axis(x: np.ndarray, scales: int, smin: float, smax: float, w: float) -> np.ndarray:
    widths = np.logspace(np.log10(smin), np.log10(smax), scales).astype(np.float32)
    W = cwt_1d(x, widths=widths, w=w)
    mag = np.abs(W).astype(np.float32, copy=False)
    mag = np.log(mag + 1e-6)

    m = mag.mean(axis=1, keepdims=True)
    s = mag.std(axis=1, keepdims=True) + 1e-6
    mag = (mag - m) / s
    return mag


# =========================================================
# MRSTFT
# =========================================================
def mrstft_axis(x: np.ndarray, stft1: Tuple[int, int, int], stft2: Tuple[int, int, int]) -> np.ndarray:
    a = stft_axis(x, n_fft=stft1[0], hop=stft1[1], keep_bins=stft1[2])
    b = stft_axis(x, n_fft=stft2[0], hop=stft2[1], keep_bins=stft2[2])

    W = min(a.shape[1], b.shape[1])
    a = a[:, :W]
    b = b[:, :W]

    # stack vertically so it appears as one figure
    return np.concatenate([a, b], axis=0)


# =========================================================
# Representation extraction
# =========================================================
def build_repr_maps(window_xyz: np.ndarray, axis: int = 2) -> dict:
    """
    window_xyz: (3, T)
    axis=2 means use force_z by default.
    Returns dict of 2D maps.
    """
    x = window_xyz[axis]

    stft_map = stft_axis(x, STFT_NFFT, STFT_HOP, STFT_KEEP_BINS)
    mel_map = mel_axis(x, FS, MEL_NFFT, MEL_HOP, MEL_BINS, MEL_FMIN, MEL_FMAX)
    cwt_map = cwt_axis(x, CWT_SCALES, CWT_SMIN, CWT_SMAX, CWT_W)
    mrstft_map = mrstft_axis(x, MRSTFT1, MRSTFT2)

    return {
        "STFT": stft_map,
        "Mel Spectrogram": mel_map,
        "CWT": cwt_map,
        "Multi-Resolution STFT": mrstft_map,
    }


# =========================================================
# Plotting
# =========================================================
def plot_comparison(maps_a: dict, maps_b: dict, user_a: str, user_b: str, task_name: str, axis_name: str):
    repr_names = ["STFT", "Mel Spectrogram", "CWT", "Multi-Resolution STFT"]

    fig, axes = plt.subplots(len(repr_names), 2, figsize=(12, 16))
    fig.suptitle(
        f"Time-Frequency Representation Comparison ({axis_name}, task={task_name}, window={WINDOW_INDEX})",
        fontsize=14
    )

    for r, name in enumerate(repr_names):
        m1 = maps_a[name]
        m2 = maps_b[name]

        ax1 = axes[r, 0]
        ax2 = axes[r, 1]

        im1 = ax1.imshow(m1, aspect="auto", origin="lower")
        ax1.set_title(f"{name} - {user_a}")
        ax1.set_xlabel("Time Frames")
        ax1.set_ylabel("Frequency / Scale Bins")

        im2 = ax2.imshow(m2, aspect="auto", origin="lower")
        ax2.set_title(f"{name} - {user_b}")
        ax2.set_xlabel("Time Frames")
        ax2.set_ylabel("Frequency / Scale Bins")

        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = os.path.join(OUT_DIR, f"repr_compare_{user_a}_vs_{user_b}_task_{task_name}_{axis_name}.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_individual_figures(maps: dict, user_name: str, task_name: str, axis_name: str):
    for name, arr in maps.items():
        plt.figure(figsize=(6, 4))
        plt.imshow(arr, aspect="auto", origin="lower")
        plt.title(f"{name} - {user_name} ({axis_name}, task={task_name}, window={WINDOW_INDEX})")
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency / Scale Bins")
        plt.colorbar()
        plt.tight_layout()

        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        out_path = os.path.join(OUT_DIR, f"{safe_name}_{user_name}_task_{task_name}_{axis_name}.png")
        plt.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    # load raw full signals
    xa_full = load_force_csv(USER_A, TASK_NAME)
    xb_full = load_force_csv(USER_B, TASK_NAME)

    # preprocess
    xa_full = preprocess_signal(xa_full)
    xb_full = preprocess_signal(xb_full)

    # extract same window index
    wa = extract_window(xa_full, WINDOW_LEN, STRIDE, WINDOW_INDEX)
    wb = extract_window(xb_full, WINDOW_LEN, STRIDE, WINDOW_INDEX)

    # choose one axis for clean paper figures
    # 0 = Fx, 1 = Fy, 2 = Fz
    axis_id = 2
    axis_name = ["Fx", "Fy", "Fz"][axis_id]

    maps_a = build_repr_maps(wa, axis=axis_id)
    maps_b = build_repr_maps(wb, axis=axis_id)

    plot_comparison(maps_a, maps_b, USER_A, USER_B, TASK_NAME, axis_name)

    # optional individual plots
    save_individual_figures(maps_a, USER_A, TASK_NAME, axis_name)
    save_individual_figures(maps_b, USER_B, TASK_NAME, axis_name)


if __name__ == "__main__":
    main()