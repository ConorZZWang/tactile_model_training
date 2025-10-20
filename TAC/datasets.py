# TAC/datasets.py
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from TAC.load_all import iter_force_files, DATA_ROOT, read_force_csv

# Map task folder name to an integer (a..g -> 0..6)
TASK_MAP = {chr(ord('a') + i): i for i in range(7)}  # {'a':0,...,'g':6}

def user_to_id(u: str) -> int:
    """'u1' -> 0, 'u2' -> 1, ..."""
    assert u.startswith("u")
    return int(u[1:]) - 1

def task_to_id(t: str) -> int:
    """'a'..'g' -> 0..6"""
    return TASK_MAP[t]

def ema_filter(x: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """Simple exponential moving average smoothing on [T, C]."""
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def compute_derivatives(x: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Enrich raw forces [Fx,Fy,Fz] with derivatives and magnitudes.
    Input x: [T,3] -> Output: [T,16]
    Channels: F(3), dF(3), d2F(3), d3F(3), |F|, |dF|, |d2F|, |d3F|
    """
    def diff(a):
        d = np.diff(a, axis=0, prepend=a[:1])
        return d * fs  # scale by sampling rate to get per-second units

    F   = x
    dF  = diff(F)
    d2F = diff(dF)
    d3F = diff(d2F)

    mag  = np.linalg.norm(F,   axis=1, keepdims=True)
    mag1 = np.linalg.norm(dF,  axis=1, keepdims=True)
    mag2 = np.linalg.norm(d2F, axis=1, keepdims=True)
    mag3 = np.linalg.norm(d3F, axis=1, keepdims=True)

    return np.concatenate([F, dF, d2F, d3F, mag, mag1, mag2, mag3], axis=1)

class ForceWindowDataset(Dataset):
    """
    Converts each force.csv stream into many fixed-length windows (slices of time).
    Each item returns:
      - x: [C, W] tensor (channels-first)
      - y_user: int (0..6)
      - y_task: int (0..6)
      - y_attack: float (0.0 normal, 1.0 synthetic attack)
    """
    def __init__(
        self,
        index: List[Tuple[str, str, Path]],  # (user_id, task_id, csv_path)
        window_len: int = 512,
        stride: int = 128,
        use_ema: bool = False,
        add_derivatives: bool = True,
        attack_gen: str = None,   # None | 'shuffle' | 'reverse' | 'drift'
        attack_ratio: float = 0.30,
        max_files: int = None,
    ):
        self.samples = []

        files = index if max_files is None else index[:max_files]
        rng = np.random.default_rng(2025)

        for (u, t, csv_path) in files:
            # Use your robust CSV reader (coerces types, drops NaNs, etc.)
            df = read_force_csv(csv_path)
            F = df[["force_x", "force_y", "force_z"]].to_numpy(dtype=np.float32)

            if len(F) < window_len:
                continue

            # Optional smoothing (EMA)
            if use_ema:
                F = ema_filter(F, alpha=0.001)

            # Optional channel expansion (derivatives & magnitudes)
            X = compute_derivatives(F) if add_derivatives else F  # [T, C]

            # Per-file z-score normalization (stabilizes across files)
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True) + 1e-6
            X = (X - mu) / sd

            # Slide a window over time to create many samples
            T = len(X)
            for start in range(0, T - window_len + 1, stride):
                end = start + window_len
                x_win = X[start:end]  # [W, C]

                y_user = user_to_id(u)
                y_task = task_to_id(t)

                # Normal window (attack=0)
                self.samples.append({
                    "x": x_win.astype(np.float32),
                    "y_user": y_user,
                    "y_task": y_task,
                    "y_attack": 0.0,
                })

                # Optionally create a synthetic "attack" version of the window
                if attack_gen is not None and rng.random() < attack_ratio:
                    xa = x_win.copy()
                    if attack_gen == "shuffle":
                        xa = xa[rng.permutation(xa.shape[0])]
                    elif attack_gen == "reverse":
                        xa = xa[::-1]
                    elif attack_gen == "drift":
                        drift = rng.normal(0, 0.1, size=xa.shape).astype(np.float32)
                        xa = xa + drift
                    self.samples.append({
                        "x": xa.astype(np.float32),
                        "y_user": y_user,
                        "y_task": y_task,
                        "y_attack": 1.0,
                    })

        if not self.samples:
            raise RuntimeError("No windows created — check data path, window_len, and stride.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        # Convert [W, C] -> [C, W] (PyTorch likes channels-first)
        x = torch.tensor(s["x"]).transpose(0, 1)
        y_user = torch.tensor(s["y_user"], dtype=torch.long)
        y_task = torch.tensor(s["y_task"], dtype=torch.long)
        y_attack = torch.tensor(s["y_attack"], dtype=torch.float32)
        return x, y_user, y_task, y_attack

def build_index():
    """
    Reuse your iterator to collect (user, task, csv_path) tuples,
    then split into train/val/test by files (prevents leakage).
    """
    idx = [(u, t, p) for (u, t, p) in iter_force_files(DATA_ROOT)]
    if not idx:
        raise RuntimeError("No files found via iter_force_files.")
    n = len(idx)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
