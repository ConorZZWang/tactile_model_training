# Per-window raw force (1D) -> 1D CNN for per-task USER identification.

import os
import argparse
from collections import Counter
from typing import Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from TAC.load_all import iter_force_files, DATA_ROOT


# ----------------------------
# Helpers
# ----------------------------
def ema_1d(x: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """Exponential moving average (paper-like small alpha)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.empty_like(x)
    v = float(x[0])
    for i in range(len(x)):
        v = alpha * float(x[i]) + (1.0 - alpha) * v
        y[i] = v
    return y


def zwin_time(x: np.ndarray) -> np.ndarray:
    """Per-window, per-channel standardization along time: (C,T)->(C,T)."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mu) / sd


# ----------------------------
# Data: per-window 1D segments
# ----------------------------
def windows_from_csvs_1d(
    window_len: int,
    stride: int,
    use_ema: bool,
    window_norm: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (N, C, L) 1D force segments with labels y_user, y_task from all CSVs using sliding windows.
    C = 3 (force_x, force_y, force_z), L = window_len.
    """
    Xs, y_users, y_tasks = [], [], []
    user_map: Dict[int, int] = {}
    task_map: Dict[int, int] = {}

    for user_id, task_id, csv_path in iter_force_files(DATA_ROOT):
        if user_id not in user_map:
            user_map[user_id] = len(user_map)
        if task_id not in task_map:
            task_map[task_id] = len(task_map)
        u = user_map[user_id]
        t = task_map[task_id]

        df = pd.read_csv(csv_path)
        for c in ("force_x", "force_y", "force_z"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["force_x", "force_y", "force_z"])

        fx = df["force_x"].values.astype(np.float32)
        fy = df["force_y"].values.astype(np.float32)
        fz = df["force_z"].values.astype(np.float32)

        if use_ema:
            fx = ema_1d(fx); fy = ema_1d(fy); fz = ema_1d(fz)

        T = len(fx)
        if T < window_len:
            continue

        for s in range(0, T - window_len + 1, stride):
            ex = s + window_len
            segx = fx[s:ex]; segy = fy[s:ex]; segz = fz[s:ex]
            seg = np.stack([segx, segy, segz], axis=0)  # (3, L)

            if window_norm:
                seg = zwin_time(seg)

            Xs.append(seg)
            y_users.append(u)
            y_tasks.append(t)

    if not Xs:
        raise RuntimeError("No 1D windows created. Check paths or make window_len/stride smaller.")

    X = np.stack(Xs, axis=0).astype(np.float32)      # (N, 3, L)
    yu = np.array(y_users, dtype=np.int64)           # (N,)
    yt = np.array(y_tasks, dtype=np.int64)           # (N,)

    print(f"1D windows: {X.shape} | users: {Counter(yu)} | tasks: {Counter(yt)}")
    return X, yu, yt


def split_per_task_within_user(
    N: int,
    y_user: np.ndarray,
    y_task: np.ndarray,
    task_id: int,
    seed: int = 42,
    ratios=(0.6, 0.2, 0.2)
):
    """Within-user split per task—ensures train/val/test contain all users."""
    rng = np.random.default_rng(seed)
    idx_t = np.where(y_task == task_id)[0]
    users = np.unique(y_user[idx_t])

    tr_all, va_all, te_all = [], [], []
    for u in users:
        iu = idx_t[y_user[idx_t] == u]
        if len(iu) < 3:
            continue
        iu = rng.permutation(iu)
        n = len(iu)
        ntr = int(ratios[0] * n)
        nva = int(ratios[1] * n)
        tr, va, te = iu[:ntr], iu[ntr:ntr + nva], iu[ntr + nva:]
        if len(tr) and len(va) and len(te):
            tr_all.append(tr); va_all.append(va); te_all.append(te)

    if not tr_all:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    tr = np.concatenate(tr_all); va = np.concatenate(va_all); te = np.concatenate(te_all)
    return tr, va, te


# ----------------------------
# Model: 1D CNN
# ----------------------------
class Small1DCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=7):
        super().__init__()
        ch = 64
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # /2

            nn.Conv1d(ch, ch * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # /4

            nn.Conv1d(ch * 2, ch * 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),  # -> (B, ch*4, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),                  # (B, ch*4)
            nn.Linear(ch * 4, ch * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(ch * 4, n_classes),
        )

    def forward(self, x):
        z = self.features(x)
        return self.head(z)


# ----------------------------
# Train/Eval
# ----------------------------
def train_one(model, Xtr, ytr, Xva, yva, Xte, yte,
              epochs=40, bs=128, lr=5e-4, wd=1e-2, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    def mkdl(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

    dl_tr = mkdl(Xtr, ytr, True)
    dl_va = mkdl(Xva, yva, False)
    dl_te = mkdl(Xte, yte, False)

    best, best_va = None, -1.0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        # val
        model.eval()
        corr = n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                corr += int((pred == yb).sum()); n += len(yb)
        va_acc = corr / max(1, n)
        # print(f"[val] ep {ep+1} acc={va_acc:.3f}")  # optional

        if va_acc > best_va:
            best_va = va_acc
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best is not None:
        model.load_state_dict(best)

    # test
    model.eval()
    corr = n = 0
    with torch.no_grad():
        for xb, yb in dl_te:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            corr += int((pred == yb).sum()); n += len(yb)
    return corr / max(1, n)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Per-window 1D force CNN for per-task USER ID")
    # data/windowing
    ap.add_argument("--window_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--window_norm", action="store_true", help="per-window channel z-norm along time")

    # training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Build 1D segments
    X, yu, yt = windows_from_csvs_1d(
        window_len=args.window_len,
        stride=args.stride,
        use_ema=args.use_ema,
        window_norm=args.window_norm,
    )
    n_users = len(np.unique(yu))
    tasks = sorted(np.unique(yt).tolist())
    print(f"[INFO] users={n_users} tasks={tasks}")

    def make_model(in_ch):
        return Small1DCNN(in_ch=in_ch, n_classes=n_users)

    # Per-task within-user splits & training
    results: Dict[int, float] = {}
    for t in tasks:
        tr, va, te = split_per_task_within_user(len(X), yu, yt, task_id=t, seed=args.seed)
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            print(f"[task {t}] not enough data")
            results[t] = np.nan
            continue

        Xtr, ytr = X[tr], yu[tr]
        Xva, yva = X[va], yu[va]
        Xte, yte = X[te], yu[te]

        model = make_model(in_ch=X.shape[1])
        acc = train_one(model, Xtr, ytr, Xva, yva, Xte, yte,
                        epochs=args.epochs, bs=args.batch, lr=args.lr, wd=args.wd, seed=args.seed)
        results[t] = float(acc)
        print(f"[task {t}] TEST acc {acc:.3f}")

    # Summary
    vals = [v for v in results.values() if not np.isnan(v)]
    overall = float(np.mean(vals)) if vals else float("nan")
    print("\n=== SUMMARY (1D CNN) ===")
    for t in tasks:
        v = results.get(t, np.nan)
        print(f"task {t}: {v if isinstance(v, float) and not np.isnan(v) else np.nan:.3f}")
    print(f"OVERALL mean acc: {overall:.3f}")


if __name__ == "__main__":
    main()
