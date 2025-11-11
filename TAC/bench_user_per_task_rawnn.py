# TAC/bench_user_per_task_rawnn.py
# Deep nets (CNN / Transformer) for per-task USER authentication on RAW windows.
# - Windows are built directly from CSVs (force_x, force_y, force_z).
# - We split WITHIN-USER for each task (so each per-task model sees all users).
# - Options:
#     --window_norm  : per-window (channel-wise) z-normalization
#     --class_weight : inverse-frequency class weights for CE loss
#     --add_fft      : append simple log-magnitude FFT channels (low bins tiled to T)
# - Outputs a CSV summary with per-task and overall accuracy.

import os
import argparse
from collections import Counter
from typing import Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from TAC.load_all import iter_force_files, DATA_ROOT

torch.backends.cudnn.benchmark = True
# ---------------------------
# Helpers: per-window z-norm and FFT channel augmentation
# ---------------------------

def zwin(x: np.ndarray) -> np.ndarray:
    """
    Per-window z-normalization: (N, C, T) -> (N, C, T)
    Zero-mean, unit-std along time for each channel, each window.
    """
    mu = x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2, keepdims=True) + 1e-8
    return (x - mu) / sd


def add_fft_channels(x: np.ndarray, keep_bins: int = 64) -> np.ndarray:
    """
    Append simple log-magnitude FFT channels.

    x: (N, C, T) float32
    Returns: (N, C + C_fft, T) by repeating the first 'keep_bins' rFFT magnitudes to match T.
    """
    N, C, T = x.shape
    spec = np.fft.rfft(x, n=T, axis=2)  # (N, C, T//2+1)
    mag = np.log(np.abs(spec) + 1e-8).astype(np.float32)
    mag = mag[:, :, :keep_bins]  # (N, C, keep_bins)

    # Tile/repeat to time length T
    reps = int(np.ceil(T / keep_bins))
    mag_t = np.tile(mag, (1, 1, reps))[:, :, :T]  # (N, C, T)

    # Concatenate as extra channels
    x_aug = np.concatenate([x, mag_t], axis=1)  # (N, 2C, T)
    return x_aug


# ---------------------------
# Data pipeline (raw windows)
# ---------------------------

def windows_from_index(all_index,
                       window_len=512,
                       stride=512,
                       use_ema=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X, y_user, y_task) where:
      - X: (N, C, T), C=3 channels (force_x,y,z), T=window_len
      - y_user: (N,) mapped 0..U-1
      - y_task: (N,) mapped 0..6

    We read force.csv, apply an optional EMA, then cut into sliding windows.
    """
    import pandas as pd

    X_list, y_user_list, y_task_list = [], [], []
    user_map, task_map = {}, {}

    def ema(series, alpha=0.1):
        v = 0.0
        out = np.empty_like(series)
        for i, s in enumerate(series):
            v = alpha * s + (1 - alpha) * (v if i > 0 else s)
            out[i] = v
        return out

    for (user_id, task_id, csv_path) in all_index:
        if user_id not in user_map:
            user_map[user_id] = len(user_map)
        if task_id not in task_map:
            task_map[task_id] = len(task_map)
        u = user_map[user_id]
        t = task_map[task_id]

        df = pd.read_csv(csv_path)
        # Clean/coerce numerics
        for col in ("force_x", "force_y", "force_z"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["force_x", "force_y", "force_z"])
        fx = df["force_x"].values.astype(np.float32)
        fy = df["force_y"].values.astype(np.float32)
        fz = df["force_z"].values.astype(np.float32)

        if use_ema:
            fx = ema(fx); fy = ema(fy); fz = ema(fz)

        T = len(fx)
        if T < window_len:
            continue

        # slice windows
        for start in range(0, T - window_len + 1, stride):
            segx = fx[start:start + window_len]
            segy = fy[start:start + window_len]
            segz = fz[start:start + window_len]
            X_list.append(np.stack([segx, segy, segz], axis=0))  # (3, T)
            y_user_list.append(u)
            y_task_list.append(t)

    if not X_list:
        raise RuntimeError("No windows created. Check data path and window params.")

    X = np.stack(X_list, axis=0)  # (N, 3, T)
    y_user = np.array(y_user_list, dtype=np.int64)
    y_task = np.array(y_task_list, dtype=np.int64)

    print("Raw windows:", X.shape, "| users:", Counter(y_user), "| tasks:", Counter(y_task))
    return X, y_user, y_task


def split_per_task_within_user(N: int,
                               y_user: np.ndarray,
                               y_task: np.ndarray,
                               task_id: int,
                               seed=42,
                               ratios=(0.6, 0.2, 0.2)):
    """
    For a given task, split indices for each user into train/val/test by ratios.
    Ensures per-task models see data from all users in each split.
    """
    rng = np.random.default_rng(seed)
    idx_task = np.where(y_task == task_id)[0]
    users = np.unique(y_user[idx_task])

    tr_all, va_all, te_all = [], [], []
    for u in users:
        iu = idx_task[y_user[idx_task] == u]
        rng.shuffle(iu)
        n = len(iu)
        if n < 5:
            continue
        ntr = int(ratios[0] * n)
        nva = int(ratios[1] * n)
        tr, va, te = iu[:ntr], iu[ntr:ntr + nva], iu[ntr + nva:]
        if len(tr) and len(va) and len(te):
            tr_all.append(tr)
            va_all.append(va)
            te_all.append(te)

    if not tr_all:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    tr = np.concatenate(tr_all)
    va = np.concatenate(va_all)
    te = np.concatenate(te_all)
    return tr, va, te


# ---------------------------
# Models: CNN1D and TinyTransformer
# ---------------------------

class CNN1D(nn.Module):
    def __init__(self, in_channels=3, n_classes=7, base=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base, kernel_size=9, padding=4),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.Conv1d(base, base, kernel_size=9, padding=4),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # T/2

            nn.Conv1d(base, base * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base * 2), nn.ReLU(inplace=True),
            nn.Conv1d(base * 2, base * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base * 2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # T/4

            nn.Conv1d(base * 2, base * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # (B, C, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base * 4, base * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base * 2, n_classes)
        )

    def forward(self, x):  # x: (B, C, T)
        z = self.net(x)
        return self.head(z)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):  # (B, T, D)
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence longer than max_len")
        return x + self.pe[:, :x.size(1), :]


class TinyTransformer(nn.Module):
    def __init__(self, in_channels=3, n_classes=7, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)  # (B, d_model, T)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):  # x: (B, C, T)
        z = self.in_proj(x)            # (B, D, T)
        z = z.transpose(1, 2)          # (B, T, D)
        z = self.pos(z)                # (B, T, D)
        z = self.encoder(z)            # (B, T, D)
        z = z.mean(dim=1)              # (B, D) global average over time
        return self.head(z)


# ---------------------------
# Training helpers
# ---------------------------

def channel_standardize(train_x: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Per-channel standardization using train set stats.
      train_x: (Ntr, C, T)
      x:       (N,   C, T)  -> returns standardized x and stats dict
    """
    means = train_x.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    stds = train_x.std(axis=(0, 2), keepdims=True) + 1e-8
    x_std = (x - means) / stds
    stats = {"mean": means, "std": stds}
    return x_std, stats


def apply_stats(x: np.ndarray, stats: Dict) -> np.ndarray:
    return (x - stats["mean"]) / stats["std"]


def train_one_model(model_name: str, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray,
                    n_classes: int, seed=42, max_epochs=40, bs=256, lr=1e-3, wd=1e-4, patience=6,
                    cnn_base=128, class_weight=None):
    """
    Generic trainer for CNN/Transformer.
    Supports optional class-weighted CrossEntropy.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "cnn":
        model = CNN1D(in_channels=X.shape[1], n_classes=n_classes, base=cnn_base).to(device)  # <- use cnn_base
    elif model_name == "trans":
        model = TinyTransformer(in_channels=X.shape[1], n_classes=n_classes).to(device)
    else:
        raise ValueError("unknown model")

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # <- use class_weight (singular), not class_weights
    weight_t = None
    if class_weight is not None:
        weight_t = torch.tensor(class_weight, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight_t)

    ds_tr = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    ds_va = torch.utils.data.TensorDataset(torch.tensor(Xv, dtype=torch.float32), torch.tensor(yv, dtype=torch.long))
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=False)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False)

    best_va = -1.0
    best = None
    wait = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        with torch.no_grad():
            correct = 0
            n = 0
            for xb, yb in dl_va:
                xb = xb.to(device); yb = yb.to(device)
                pr = model(xb).argmax(1)
                correct += int((pr == yb).sum().item()); n += len(xb)
            va_acc = correct / max(1, n)

        if va_acc > best_va:
            best_va = va_acc
            best = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return model



def predict_model(model: nn.Module, X: np.ndarray, bs=256):
    device = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=bs, shuffle=False)
    out = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            logits = model(xb)
            out.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(out, axis=0)


def confusion_mat(y_true, y_pred, n_classes):
    mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[t, p] += 1
    return mat


# ---------------------------
# Main runner
# ---------------------------

def main():
    ap = argparse.ArgumentParser("Per-task USER authentication on RAW windows with deep nets (CNN/Transformer)")
    ap.add_argument("--models", nargs="+", default=["cnn", "trans"], help="cnn, trans")
    ap.add_argument("--window_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_csv", default="bench_user_per_task_rawnn.csv")

    # NEW switches
    ap.add_argument("--window_norm", action="store_true",
                    help="apply per-window z-normalization (zero-mean/unit-std per channel & window)")
    ap.add_argument("--class_weight", action="store_true",
                    help="use class-weighted cross-entropy from inverse-frequency")
    ap.add_argument("--cnn_base", type=int, default=128, help="CNN base channels")
    ap.add_argument("--add_fft", action="store_true",
                    help="append simple log-magnitude FFT channels per window")
    ap.add_argument("--fft_bins", type=int, default=64,
                    help="number of low-frequency FFT bins to keep when --add_fft is on")

    args = ap.parse_args()

    # Fewer OpenMP threads avoids CPU thrash
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Build raw windows
    all_index = tuple(iter_force_files(DATA_ROOT))
    X, y_user, y_task = windows_from_index(
        all_index,
        window_len=args.window_len,
        stride=args.stride,
        use_ema=args.use_ema
    )
    n_users = len(np.unique(y_user))
    tasks = sorted(np.unique(y_task).tolist())

    results = []
    for model_name in args.models:
        print(f"\n=== MODEL: {model_name} ===")
        per_task_acc = {}
        all_te_true = []
        all_te_pred = []

        for t in tasks:
            tr, va, te = split_per_task_within_user(len(X), y_user, y_task, task_id=t, seed=args.seed)
            if len(tr) == 0 or len(va) == 0 or len(te) == 0:
                print(f"[task {t}] not enough data")
                continue

            # Train/val/test cut for this task
            Xtr_raw = X[tr]; Xva_raw = X[va]; Xte_raw = X[te]
            ytr = y_user[tr]; yva = y_user[va]; yte = y_user[te]

            # Channel z-score using TRAIN stats
            # Xtr, stats = channel_standardize(Xtr_raw, Xtr_raw)
            # Xva = apply_stats(Xva_raw, stats)
            # Xte = apply_stats(Xte_raw, stats)

            # OPTIONAL: per-window z-norm
            if args.window_norm:
                Xtr = zwin(Xtr_raw)
                Xva = zwin(Xva_raw)
                Xte = zwin(Xte_raw)
            else:
                Xtr, Xva, Xte = Xtr_raw, Xva_raw, Xte_raw

            # OPTIONAL: FFT augmentation
            if args.add_fft:
                Xtr = add_fft_channels(Xtr, keep_bins=args.fft_bins)
                Xva = add_fft_channels(Xva, keep_bins=args.fft_bins)
                Xte = add_fft_channels(Xte, keep_bins=args.fft_bins)

            # OPTIONAL: class weights (inverse-frequency)
            class_weight = None
            if args.class_weight:
                n_users = len(np.unique(y_user))
                counts = np.bincount(ytr, minlength=n_users).astype(np.float32)
                counts[counts == 0] = 1.0
                inv = 1.0 / counts
                class_weight = inv * (len(counts) / inv.sum())  # normalize to ~1.0 mean weight

            # Train model
            model = train_one_model(
                model_name, Xtr, ytr, Xva, yva,
                n_classes=n_users, seed=args.seed,
                max_epochs=args.epochs, bs=args.batch_size,
                cnn_base=args.cnn_base,
                class_weight=class_weight,
            )

            # Evaluate on test
            yp = predict_model(model, Xte)
            acc = (yp == yte).mean()
            per_task_acc[t] = float(acc)
            all_te_true.append(yte)
            all_te_pred.append(yp)
            print(f"[task {t}] test_acc {acc:.3f}")

        if all_te_true:
            y_true = np.concatenate(all_te_true)
            y_pred = np.concatenate(all_te_pred)
            overall = (y_true == y_pred).mean()
        else:
            overall = float("nan")

        print(f"Overall TEST acc ({model_name}): {overall:.3f}")
        row = {"model": model_name, "overall_acc": overall}
        for t in tasks:
            row[f"task{t}_acc"] = per_task_acc.get(t, np.nan)
        results.append(row)

    df = pd.DataFrame(results)
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))
    df.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
