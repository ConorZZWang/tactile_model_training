# Per-task USER authentication using CWT (wavelet scalogram) channelized into (C, T)
#
# CWT per axis -> |W(s, t)|  (S, T)
# Channelize: concat axes along channel => (C = 3*S, T)
#
# Example:
#   python -m TAC.bench_user_per_task_rawnn_cwt1d `
#     --window_len 768 --stride 256 --use_ema --window_norm `
#     --cnn_base 192 --epochs 40 --batch_size 192 `
#     --fs 250 --cwt_w 6.0 --cwt_scales 64 --cwt_smin 1 --cwt_smax 128 `
#     --out_csv runs/cwt1d.csv

import os
import argparse
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.signal import fftconvolve  # robust across SciPy versions

from TAC.load_all import iter_force_files, DATA_ROOT

torch.backends.cudnn.benchmark = True


# -------------------------
# Helpers
# -------------------------
def zwin(x: np.ndarray) -> np.ndarray:
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


def windows_from_index(all_index, window_len=512, stride=512, use_ema=False, ema_alpha=0.001):
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

    X = np.stack(X_list, axis=0)
    y_user = np.array(y_user_list, dtype=np.int64)
    y_task = np.array(y_task_list, dtype=np.int64)

    print("Raw windows:", X.shape, "| users:", Counter(y_user), "| tasks:", Counter(y_task))
    return X, y_user, y_task


def split_per_task_within_user(y_user, y_task, task_id, seed=42, ratios=(0.6, 0.2, 0.2)):
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


# -------------------------
# CWT implementation (no scipy.signal.cwt dependency)
# -------------------------
def morlet2(M: int, s: float, w: float = 6.0) -> np.ndarray:
    """
    Morlet wavelet (similar to scipy.signal.morlet2).
    M: number of points
    s: scale
    w: omega0 (non-dimensional frequency)
    Returns complex wavelet of length M.
    """
    # center time vector at 0
    t = np.arange(M, dtype=np.float32) - (M - 1) / 2.0
    ts = t / float(s)
    # normalization roughly matching classic Morlet
    A = np.pi ** (-0.25)
    wave = A * np.exp(1j * w * ts) * np.exp(-0.5 * ts * ts)
    # scale normalization (keeps energy comparable across scales)
    wave = wave / np.sqrt(s)
    return wave.astype(np.complex64, copy=False)


def cwt_1d(x: np.ndarray, widths: np.ndarray, w: float = 6.0) -> np.ndarray:
    """
    Continuous Wavelet Transform using FFT convolution, compatible with old scipy.signal.cwt shape.
    x: (T,)
    widths: (S,) scales
    Returns: (S, T) complex array
    """
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    out = np.empty((len(widths), T), dtype=np.complex64)

    for i, width in enumerate(widths):
        # old scipy convention: wavelet length ~ 10*width (cap at signal length)
        M = int(max(1, np.floor(10.0 * float(width))))
        if M % 2 == 0:
            M += 1
        if M > T:
            M = T if T % 2 == 1 else T - 1  # keep odd, <= T
            M = max(M, 1)

        wave = morlet2(M, s=float(width), w=w)
        # convolution with time-reversed conjugate (correlation-like) is common for CWT
        # This matches the typical wavelet transform convention more closely.
        conv = fftconvolve(x, wave[::-1].conj(), mode="same")
        out[i] = conv.astype(np.complex64, copy=False)

    return out



def cwt_channelize(
    X: np.ndarray,
    fs: float = 250.0,
    scales: int = 64,
    smin: float = 1.0,
    smax: float = 128.0,
    w: float = 6.0,
    log_eps: float = 1e-6,
    per_scale_norm: bool = True,
    W_target: int = 96,          # <-- add this (or hardcode inside)
) -> np.ndarray:
    """
    X: (N, 3, T)
    Returns: (N, 3*scales, W_target) if W_target is used, else (N, 3*scales, T)
    """
    assert X.ndim == 3 and X.shape[1] == 3
    N, C, T = X.shape
    widths = np.logspace(np.log10(smin), np.log10(smax), scales).astype(np.float32)

    out_T = W_target if W_target is not None else T
    out = np.empty((N, 3 * scales, out_T), dtype=np.float32)

    for i in range(N):
        chans = []
        for c in range(3):
            x = X[i, c].astype(np.float32, copy=False)
            W = cwt_1d(x, widths=widths, w=w)  # (S,T) complex
            mag = np.abs(W).astype(np.float32, copy=False)
            mag = np.log(mag + log_eps)

            if per_scale_norm:
                m = mag.mean(axis=1, keepdims=True)
                s = mag.std(axis=1, keepdims=True) + 1e-6
                mag = (mag - m) / s

            # DOWNSAMPLING
            if W_target is not None and mag.shape[1] != W_target:
                t_old = np.linspace(0, 1, mag.shape[1], dtype=np.float32)
                t_new = np.linspace(0, 1, W_target, dtype=np.float32)
                mag = np.stack(
                    [np.interp(t_new, t_old, mag[s]).astype(np.float32)
                     for s in range(mag.shape[0])],
                    axis=0
                )

            chans.append(mag)  # (S, out_T)

        out[i] = np.concatenate(chans, axis=0)  # (3S, out_T)

    return out


# -------------------------
# Model
# -------------------------
class CNN1D(nn.Module):
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

    def forward(self, x):
        x = self.stem(x)
        z = self.net(x)
        return self.head(z)


def train_one_model(
    Xtr, ytr, Xva, yva,
    n_classes, seed, max_epochs, bs, lr, wd, patience, cnn_base,
    class_weight: Optional[np.ndarray] = None
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN1D(in_channels=Xtr.shape[1], n_classes=n_classes, base=cnn_base).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    w_t = None
    if class_weight is not None:
        w_t = torch.tensor(class_weight, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w_t)

    ds_tr = torch.utils.data.TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.long)
    )
    ds_va = torch.utils.data.TensorDataset(
        torch.tensor(Xva, dtype=torch.float32),
        torch.tensor(yva, dtype=torch.long)
    )
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=bs, shuffle=False)

    best_va = -1.0
    best_state = None
    wait = 0

    for _ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            n = 0
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
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
def predict_model(model: nn.Module, X: np.ndarray, bs: int):
    device = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=bs, shuffle=False)
    out = []
    for xb in dl:
        xb = xb.to(device)
        out.append(model(xb).argmax(1).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser("Per-task USER authentication with 1D-CNN on CWT scalograms")
    ap.add_argument("--window_len", type=int, default=768)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--ema_alpha", type=float, default=0.001)
    ap.add_argument("--window_norm", action="store_true")
    ap.add_argument("--class_weight", action="store_true")

    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--cwt_scales", type=int, default=64)
    ap.add_argument("--cwt_smin", type=float, default=1.0)
    ap.add_argument("--cwt_smax", type=float, default=128.0)
    ap.add_argument("--cwt_w", type=float, default=6.0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--cnn_base", type=int, default=192)

    ap.add_argument("--out_csv", default="runs/cwt1d.csv")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    all_index = tuple(iter_force_files(DATA_ROOT))
    X, y_user, y_task = windows_from_index(
        all_index,
        window_len=args.window_len,
        stride=args.stride,
        use_ema=args.use_ema,
        ema_alpha=args.ema_alpha,
    )

    n_users = len(np.unique(y_user))
    tasks = sorted(np.unique(y_task).tolist())

    print("\n=== MODEL: cnn(cwt-channelized) ===")
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

        Xtr_c = cwt_channelize(
            Xtr_raw, fs=args.fs, scales=args.cwt_scales, smin=args.cwt_smin, smax=args.cwt_smax, w=args.cwt_w
        )
        Xva_c = cwt_channelize(
            Xva_raw, fs=args.fs, scales=args.cwt_scales, smin=args.cwt_smin, smax=args.cwt_smax, w=args.cwt_w
        )
        Xte_c = cwt_channelize(
            Xte_raw, fs=args.fs, scales=args.cwt_scales, smin=args.cwt_smin, smax=args.cwt_smax, w=args.cwt_w
        )

        cw = None
        if args.class_weight:
            counts = np.bincount(ytr, minlength=n_users).astype(np.float32)
            counts[counts == 0] = 1.0
            inv = 1.0 / counts
            cw = inv * (len(inv) / inv.sum())

        model = train_one_model(
            Xtr_c, ytr, Xva_c, yva,
            n_classes=n_users,
            seed=args.seed,
            max_epochs=args.epochs,
            bs=args.batch_size,
            lr=args.lr,
            wd=args.wd,
            patience=args.patience,
            cnn_base=args.cnn_base,
            class_weight=cw,
        )

        yp = predict_model(model, Xte_c, bs=args.batch_size)
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

    print(f"Overall TEST acc (cwt1d): {overall:.3f}")

    row = {"model": "cnn_cwt1d", "overall_acc": overall}
    for t in tasks:
        row[f"task{t}_acc"] = per_task_acc.get(t, np.nan)

    df = pd.DataFrame([row])
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}")


if __name__ == "__main__":
    main()