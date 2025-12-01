# TAC/train_user_transformer_features.py
# Paper-style USER identification with per-timestep features (Table I):
# ΔF(t) norm, velocity (Fx',Fy',Fz') + ||v||, acceleration (ax,ay,az) + ||a||,
# jerk (jx,jy,jz) + ||j||  -> total 13 channels per timestep.
# Per-task training; for each (user,task) sample up to N_train/N_test windows.

import os
import argparse
from collections import Counter
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math

from TAC.load_all import iter_force_files, DATA_ROOT

SAMPLE_RATE = 250.0  # Hz (paper)

# -----------------------
# Utilities
# -----------------------
def ema_1d(x: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """Exponential moving average (paper uses very small alpha)."""
    out = np.empty_like(x, dtype=np.float32)
    v = float(x[0])
    for i in range(len(x)):
        v = alpha * float(x[i]) + (1.0 - alpha) * v
        out[i] = v
    return out

def build_windows(seq_len: int, stride: int, arr: np.ndarray) -> np.ndarray:
    """Cut (T, D) into (N, D, L) windows with step=stride. No overlap if stride==seq_len."""
    T, D = arr.shape
    if T < seq_len:
        return np.empty((0, D, seq_len), dtype=np.float32)
    starts = range(0, T - seq_len + 1, stride)
    X = np.stack([arr[s:s + seq_len].T for s in starts], axis=0).astype(np.float32)  # (N, D, L)
    return X

def zwin_time(x: np.ndarray) -> np.ndarray:
    """Per-window, per-channel time z-norm: (N, C, L) -> (N, C, L)."""
    mu = x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2, keepdims=True) + 1e-8
    return (x - mu) / sd

def make_derivatives(F: np.ndarray, rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F: (T,3) force. Return v,a,j each (T,3) using first differences * rate, left-pad zeros.
    """
    def d1(x):
        dx = np.diff(x, axis=0, prepend=x[:1])
        return dx * rate
    v = d1(F)
    a = d1(v)
    j = d1(a)
    return v, a, j

def make_features(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray, use_ema: bool) -> np.ndarray:
    """
    Build per-timestep feature matrix (T,13) following Table I in the paper.
    Channels:
      [0] ΔF = ||F(t)-F(t-1)|| (first diff norm)
      [1:4] velocity (vx,vy,vz)
      [4]++ actually we’ll place velocity first then its norm, same for a, j.
    Final order (13):
      vx, vy, vz, ||v||,  ax, ay, az, ||a||,  jx, jy, jz, ||j||,  dF_norm
    """
    F = np.stack([Fx, Fy, Fz], axis=1).astype(np.float32)  # (T,3)

    if use_ema:
        F[:, 0] = ema_1d(F[:, 0])
        F[:, 1] = ema_1d(F[:, 1])
        F[:, 2] = ema_1d(F[:, 2])

    v, a, j = make_derivatives(F, SAMPLE_RATE)

    v_norm = np.linalg.norm(v, axis=1, keepdims=True)  # (T,1)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    j_norm = np.linalg.norm(j, axis=1, keepdims=True)

    dF = np.diff(F, axis=0, prepend=F[:1])
    dF_norm = np.linalg.norm(dF, axis=1, keepdims=True)  # (T,1)

    # Concatenate features: (T, 13)
    feat = np.concatenate([F, v, v_norm, a, a_norm, j, j_norm, dF_norm],axis=1).astype(np.float32)
    return feat

def load_feature_windows(seq_len: int, stride: int, use_ema: bool, window_norm: bool
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return X:(N,13,L), y_user:(N,), y_task:(N,)
    """
    Xs, y_users, y_tasks = [], [], []
    user_map: Dict[str, int] = {}
    task_map: Dict[str, int] = {}

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

        Fx = df["force_x"].values.astype(np.float32)
        Fy = df["force_y"].values.astype(np.float32)
        Fz = df["force_z"].values.astype(np.float32)

        feat = make_features(Fx, Fy, Fz, use_ema=use_ema)  # (T,13)

        X = build_windows(seq_len, stride, feat)  # (N,13,L)
        if X.shape[0] == 0:
            continue
        if window_norm:
            X = zwin_time(X)

        Xs.append(X)
        y_users.append(np.full(X.shape[0], u, dtype=np.int64))
        y_tasks.append(np.full(X.shape[0], t, dtype=np.int64))

    if not Xs:
        raise RuntimeError("No windows created. Try smaller --seq_len or stride.")

    X = np.concatenate(Xs, axis=0)
    yu = np.concatenate(y_users, axis=0)
    yt = np.concatenate(y_tasks, axis=0)

    print(f"Feature windows: {X.shape} | users: {Counter(yu)} | tasks: {Counter(yt)}")
    return X, yu, yt

def sample_per_user_task_indices(y_user: np.ndarray, y_task: np.ndarray,
                                 task_id: int, n_train: int, n_test: int, seed: int):
    """For task=t, for each user: shuffle that task's indices, take up to n_train/n_test."""
    rng = np.random.default_rng(seed)
    idx_t = np.where(y_task == task_id)[0]
    users = np.unique(y_user[idx_t])
    tr_idx, te_idx = [], []
    for u in users:
        iu = idx_t[y_user[idx_t] == u]
        if len(iu) == 0:
            continue
        iu = rng.permutation(iu)
        k = min(len(iu), n_train + n_test)
        if k == 0:
            continue
        cut = min(n_train, k)
        tr_u = iu[:cut]
        te_u = iu[cut:k]
        if len(tr_u) == 0 or len(te_u) == 0:
            if k >= 2:
                tr_u = iu[:k - 1]
                te_u = iu[k - 1:k]
            else:
                continue
        tr_idx.append(tr_u); te_idx.append(te_u)
    if not tr_idx or not te_idx:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(tr_idx), np.concatenate(te_idx)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape (1, max_len, d_model) so it broadcasts over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


# -----------------------
# Model (Transformer)
# -----------------------
class PaperTransformer(nn.Module):
    def __init__(self, in_channels=13, d_model=512, nhead=16,
                 num_layers=2, dim_ff=512, dropout=0.1, n_classes=7):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pre_ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):           # x: (B, C, L)
        z = self.in_proj(x).transpose(1, 2)  # (B, L, D)
        z = self.pos_enc(z)                  # add temporal info
        z = self.pre_ln(z)
        z = self.encoder(z)                  # (B, L, D)
        z = z.mean(dim=1)                    # (B, D)
        return self.head(z)


# -----------------------
# Train / eval one task
# -----------------------
def train_one_task(Xtr, ytr, Xva, yva, Xte, yte, n_users: int,
                   epochs=100, batch=16, lr=1e-4, weight_decay=1e-2, grad_clip=1.0,
                   d_model=512, nhead=16, num_layers=2, dim_ff=512, dropout=0.1, seed=42, device=None):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PaperTransformer(
        in_channels=Xtr.shape[1], d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_ff=dim_ff, dropout=dropout, n_classes=n_users
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    crit = nn.CrossEntropyLoss()

    def make_loader(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                            torch.tensor(y, dtype=torch.long))
        return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)

    # small val split from train if not provided
    if Xva is None or len(Xva) == 0:
        n = len(Xtr); nva = max(1, int(0.1 * n))
        perm = np.random.permutation(n)
        va = perm[:nva]; tr = perm[nva:]
        Xva, yva = Xtr[va], ytr[va]
        Xtr, ytr = Xtr[tr], ytr[tr]

    dl_tr = make_loader(Xtr, ytr, True)
    dl_va = make_loader(Xva, yva, False)

    best_state, best_va = None, -1.0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        # val
        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb).argmax(1)
                correct += int((pred == yb).sum().item())
                n += len(xb)
        va_acc = correct / max(1, n)
        sched.step()
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    dl_te = torch.utils.data.DataLoader(torch.tensor(Xte, dtype=torch.float32), batch_size=batch, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in dl_te:
            xb = xb.to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    ypred = np.concatenate(preds)
    test_acc = (ypred == yte).mean()
    return test_acc

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser("User identification (per-task) with per-timestep features (paper)")
    ap.add_argument("--use_ema", action="store_true", help="EMA on raw forces before derivatives")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=None, help="default: seq_len//2")
    ap.add_argument("--per_user_train", type=int, default=100)
    ap.add_argument("--per_user_test", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window_norm", action="store_true", help="per-window z-norm over time for features")

    # model size (paper defaults)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dim_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    stride = args.stride if args.stride is not None else max(1, args.seq_len // 2)

    # Build all feature windows once
    X, yu, yt = load_feature_windows(seq_len=args.seq_len, stride=stride,
                                     use_ema=args.use_ema, window_norm=args.window_norm)
    n_users = len(np.unique(yu))
    tasks = sorted(np.unique(yt).tolist())
    print(f"[INFO] seq_len={args.seq_len}, stride={stride}, users={n_users}, tasks={tasks}")

    results = {}
    for t in tasks:
        tr_idx, te_idx = sample_per_user_task_indices(
            y_user=yu, y_task=yt, task_id=t,
            n_train=args.per_user_train, n_test=args.per_user_test, seed=args.seed
        )
        print(f"[task {t}] train {len(tr_idx)}  test {len(te_idx)}")
        if len(tr_idx) == 0 or len(te_idx) == 0:
            print(f"[WARN] task {t} empty split; consider smaller --seq_len or stride or quotas.")
            results[t] = np.nan
            continue

        # make a small val from train (10%)
        ntr = len(tr_idx)
        nva = max(1, int(0.1 * ntr))
        perm = np.random.permutation(tr_idx)
        va_idx = perm[:nva]
        tr_idx2 = perm[nva:]

        Xtr, ytr = X[tr_idx2], yu[tr_idx2]
        Xva, yva = X[va_idx], yu[va_idx]
        Xte, yte = X[te_idx], yu[te_idx]

        acc = train_one_task(
            Xtr, ytr, Xva, yva, Xte, yte, n_users=n_users,
            epochs=args.epochs, batch=args.batch, lr=args.lr,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            dim_ff=args.dim_ff, dropout=args.dropout, seed=args.seed
        )
        results[t] = float(acc)
        print(f"[task {t}] TEST acc {acc:.3f}")

    # Summary
    vals = [v for v in results.values() if not np.isnan(v)]
    overall = float(np.mean(vals)) if vals else float("nan")
    print("\n=== SUMMARY (paper Transformer, per-timestep FEATURES, per-task USER ID) ===")
    for t in tasks:
        print(f"task {t}: {results.get(t, np.nan):.3f}")
    print(f"OVERALL mean acc: {overall:.3f}")

if __name__ == "__main__":
    main()
