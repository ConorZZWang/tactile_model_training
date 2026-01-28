# TAC/bench_user_per_task_rawnn_stft1d.py
# Per-task USER authentication using a 1D-CNN on "image-like" STFT maps,
# channelized into (C, T) so we can reuse a 1D CNN pipeline.
#
# Representation:
#   For each window (3, T):
#     STFT per axis -> log-magnitude spectrogram (F, W)
#     Channelize: concat axes along "channel" => (C = 3*F, T = W)
#
# Example:
#   python -m TAC.bench_user_per_task_rawnn_stft1d \
#     --window_len 768 --stride 256 --use_ema --window_norm \
#     --cnn_base 192 --epochs 40 --batch_size 192 \
#     --stft_n_fft 128 --stft_hop 16 --stft_keep_bins 64

import os
import argparse
from collections import Counter
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from TAC.load_all import iter_force_files, DATA_ROOT

torch.backends.cudnn.benchmark = True


# ---------------------------
# Helpers
# ---------------------------

def zwin(x: np.ndarray) -> np.ndarray:
    """Per-window z-normalization: (N, C, T) -> (N, C, T)."""
    mu = x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2, keepdims=True) + 1e-8
    return (x - mu) / sd


def ema_1d(series: np.ndarray, alpha: float) -> np.ndarray:
    v = 0.0
    out = np.empty_like(series, dtype=np.float32)
    for i, s in enumerate(series.astype(np.float32, copy=False)):
        v = alpha * s + (1 - alpha) * (v if i > 0 else s)
        out[i] = v
    return out


def windows_from_index(
    all_index,
    window_len: int = 512,
    stride: int = 512,
    use_ema: bool = False,
    ema_alpha: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X, y_user, y_task):
      - X: (N, 3, T)
      - y_user: (N,)
      - y_task: (N,)
    """
    X_list, y_user_list, y_task_list = [], [], []
    user_map, task_map = {}, {}

    for (user_id, task_id, csv_path) in all_index:
        if user_id not in user_map:
            user_map[user_id] = len(user_map)
        if task_id not in task_map:
            task_map[task_id] = len(task_map)

        u = user_map[user_id]
        t = task_map[task_id]

        df = pd.read_csv(csv_path)
        for col in ("force_x", "force_y", "force_z"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["force_x", "force_y", "force_z"])

        fx = df["force_x"].values.astype(np.float32, copy=False)
        fy = df["force_y"].values.astype(np.float32, copy=False)
        fz = df["force_z"].values.astype(np.float32, copy=False)

        if use_ema:
            fx = ema_1d(fx, ema_alpha)
            fy = ema_1d(fy, ema_alpha)
            fz = ema_1d(fz, ema_alpha)

        T = len(fx)
        if T < window_len:
            continue

        for start in range(0, T - window_len + 1, stride):
            segx = fx[start:start + window_len]
            segy = fy[start:start + window_len]
            segz = fz[start:start + window_len]
            X_list.append(np.stack([segx, segy, segz], axis=0))
            y_user_list.append(u)
            y_task_list.append(t)

    if not X_list:
        raise RuntimeError("No windows created. Check data path and window params.")

    X = np.stack(X_list, axis=0)  # (N, 3, T)
    y_user = np.array(y_user_list, dtype=np.int64)
    y_task = np.array(y_task_list, dtype=np.int64)

    print("Raw windows:", X.shape, "| users:", Counter(y_user), "| tasks:", Counter(y_task))
    return X, y_user, y_task


def split_per_task_within_user(
    y_user: np.ndarray,
    y_task: np.ndarray,
    task_id: int,
    seed: int = 42,
    ratios=(0.6, 0.2, 0.2)
):
    """For a task, split indices for each user into train/val/test by ratios."""
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

    return np.concatenate(tr_all), np.concatenate(va_all), np.concatenate(te_all)


# ---------------------------
# STFT "image" -> 1D-CNN tensor (N, C, T)
# ---------------------------

@torch.no_grad()
def stft_channelize(
    X: np.ndarray,
    n_fft: int = 128,
    hop: int = 16,
    keep_bins: int = 64,
    log_eps: float = 1e-6
) -> np.ndarray:
    """
    X: (N, 3, T) float32
    Returns: (N, 3*keep_bins, W) float32 where W is #frames.
    """
    assert X.ndim == 3 and X.shape[1] == 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xt = torch.tensor(X, dtype=torch.float32, device=device)  # (N,3,T)
    N, C, T = xt.shape

    # Flatten (N*C, T) to compute stft in one call
    xc = xt.reshape(N * C, T)
    window = torch.hann_window(n_fft, device=device)

    # stft -> (N*C, F, W), complex
    spec = torch.stft(
        xc, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, center=True, return_complex=True
    )
    mag = torch.abs(spec)  # (N*C, F, W)

    # log magnitude
    mag = torch.log(mag + log_eps)
    kb = min(keep_bins, mag.shape[1])
    mag = mag[:, :kb, :]
    m = mag.mean(dim=2, keepdim=True)
    s = mag.std(dim=2, keepdim=True) + 1e-6
    mag = (mag - m) / s
    # keep low freq bins

    # reshape to (N, C*kb, W)
    mag = mag.reshape(N, C * kb, mag.shape[2])

    out = mag.detach().cpu().numpy().astype(np.float32, copy=False)
    return out


# ---------------------------
# Models
# ---------------------------

class CNN1D(nn.Module):
    """
    1D CNN that works for both:
      - raw: C=3
      - channelized maps: C can be large (e.g., 192)
    Uses a 1x1 stem to compress arbitrary input channels to `base`.
    """
    def __init__(self, in_channels: int, n_classes: int, base: int = 128, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, kernel_size=1),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv1d(base, base, kernel_size=9, padding=4),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.Conv1d(base, base, kernel_size=9, padding=4),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base, base * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base * 2), nn.ReLU(inplace=True),
            nn.Conv1d(base * 2, base * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base * 2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base * 2, base * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base * 4, base * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        z = self.net(x)
        return self.head(z)


# ---------------------------
# Training helpers
# ---------------------------

def train_one_model(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    n_classes: int,
    seed: int,
    max_epochs: int,
    bs: int,
    lr: float,
    wd: float,
    patience: int,
    cnn_base: int,
    class_weight: Optional[np.ndarray] = None
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN1D(in_channels=Xtr.shape[1], n_classes=n_classes, base=cnn_base).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    weight_t = None
    if class_weight is not None:
        weight_t = torch.tensor(class_weight, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight_t)

    ds_tr = torch.utils.data.TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    ds_va = torch.utils.data.TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.long))
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=False)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False)

    best_va = -1.0
    best_state = None
    wait = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            n = 0
            for xb, yb in dl_va:
                xb = xb.to(device); yb = yb.to(device)
                pr = model(xb).argmax(1)
                correct += int((pr == yb).sum().item())
                n += len(xb)
            va_acc = correct / max(1, n)

        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_model(model: nn.Module, X: np.ndarray, bs: int = 256) -> np.ndarray:
    device = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=bs, shuffle=False)
    out = []
    for xb in dl:
        xb = xb.to(device)
        logits = model(xb)
        out.append(logits.argmax(1).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser("Per-task USER authentication with 1D-CNN on STFT-channelized maps")
    ap.add_argument("--window_len", type=int, default=768)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--ema_alpha", type=float, default=0.001)
    ap.add_argument("--window_norm", action="store_true")
    ap.add_argument("--class_weight", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--cnn_base", type=int, default=192)

    ap.add_argument("--stft_n_fft", type=int, default=128)
    ap.add_argument("--stft_hop", type=int, default=16)
    ap.add_argument("--stft_keep_bins", type=int, default=64)

    ap.add_argument("--out_csv", default="bench_user_per_task_stft1d.csv")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Build raw windows
    all_index = tuple(iter_force_files(DATA_ROOT))
    X, y_user, y_task = windows_from_index(
        all_index,
        window_len=args.window_len,
        stride=args.stride,
        use_ema=args.use_ema,
        ema_alpha=args.ema_alpha
    )

    n_users = len(np.unique(y_user))
    tasks = sorted(np.unique(y_task).tolist())

    results = []
    print("\n=== MODEL: cnn(stft-channelized) ===")
    per_task_acc = {}
    all_te_true, all_te_pred = [], []

    for t in tasks:
        tr, va, te = split_per_task_within_user(y_user, y_task, task_id=t, seed=args.seed)
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            print(f"[task {t}] not enough data")
            continue

        Xtr_raw, Xva_raw, Xte_raw = X[tr], X[va], X[te]
        ytr, yva, yte = y_user[tr], y_user[va], y_user[te]

        if args.window_norm:
            Xtr_raw = zwin(Xtr_raw)
            Xva_raw = zwin(Xva_raw)
            Xte_raw = zwin(Xte_raw)

        # STFT channelize -> (N, C, T')
        Xtr = stft_channelize(Xtr_raw, n_fft=args.stft_n_fft, hop=args.stft_hop, keep_bins=args.stft_keep_bins)
        Xva = stft_channelize(Xva_raw, n_fft=args.stft_n_fft, hop=args.stft_hop, keep_bins=args.stft_keep_bins)
        Xte = stft_channelize(Xte_raw, n_fft=args.stft_n_fft, hop=args.stft_hop, keep_bins=args.stft_keep_bins)

        # Optional class weights
        class_weight = None
        if args.class_weight:
            counts = np.bincount(ytr, minlength=n_users).astype(np.float32)
            counts[counts == 0] = 1.0
            inv = 1.0 / counts
            class_weight = inv * (len(inv) / inv.sum())

        model = train_one_model(
            Xtr, ytr, Xva, yva,
            n_classes=n_users,
            seed=args.seed,
            max_epochs=args.epochs,
            bs=args.batch_size,
            lr=args.lr,
            wd=args.wd,
            patience=args.patience,
            cnn_base=args.cnn_base,
            class_weight=class_weight,
        )

        yp = predict_model(model, Xte, bs=args.batch_size)
        acc = float((yp == yte).mean())
        per_task_acc[t] = acc
        all_te_true.append(yte)
        all_te_pred.append(yp)
        print(f"[task {t}] test_acc {acc:.3f}")

    if all_te_true:
        y_true = np.concatenate(all_te_true)
        y_pred = np.concatenate(all_te_pred)
        overall = float((y_true == y_pred).mean())
    else:
        overall = float("nan")

    print(f"Overall TEST acc (stft1d): {overall:.3f}")

    row = {"model": "cnn_stft1d", "overall_acc": overall}
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