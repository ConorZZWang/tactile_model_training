# TAC/train_user_transformer_paper.py
# "Paper-style" user identification with a small Transformer.
# - Per-task training: build a separate model per task (a..g).
# - For each (user, task), sample up to N_train train and N_test test sequences.
# - Windows built from force_x,y,z (optionally EMA) at --seq_len with overlap --stride.
# - Optimizer: Adam, lr=1e-4; CosineAnnealingLR; epochs default 100; batch 16.

import os
import argparse
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from TAC.load_all import iter_force_files, DATA_ROOT

# -----------------------
# Utilities
# -----------------------
def ema_1d(x: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """Exponential moving average like the paper (small alpha)."""
    out = np.empty_like(x)
    v = x[0]
    for i in range(len(x)):
        v = alpha * x[i] + (1 - alpha) * v
        out[i] = v
    return out

def build_windows(seq_len: int, stride: int, fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> np.ndarray:
    T = len(fx)
    if T < seq_len:
        return np.empty((0, 3, seq_len), dtype=np.float32)
    starts = range(0, T - seq_len + 1, stride)
    X = np.stack([np.stack([fx[s:s+seq_len], fy[s:s+seq_len], fz[s:s+seq_len]], axis=0)
                  for s in starts], axis=0).astype(np.float32)  # (N,3,L)
    return X

def zwin(x: np.ndarray) -> np.ndarray:
    """Per-window channel-wise z-norm (zero mean, unit std along time)."""
    mu = x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2, keepdims=True) + 1e-8
    return (x - mu) / sd

# -----------------------
# Model (paper-ish)
# -----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x):  # (B,L,D)
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence longer than max_len for positional encoding.")
        return x + self.pe[:, :x.size(1), :]

class PaperTransformer(nn.Module):
    def __init__(self, in_channels=3, d_model=512, nhead=16, num_layers=2, dim_ff=512, dropout=0.1, n_classes=7):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pre_ln = nn.LayerNorm(d_model)  # <<< add this
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):               # x: (B, C, T)
        z = self.in_proj(x).transpose(1, 2)  # (B, T, D)
        z = self.pre_ln(z)              # <<< add this
        z = self.encoder(z)             # (B, T, D)
        z = z.mean(dim=1)               # (B, D)
        return self.head(z)


# -----------------------
# Data assembly (per task)
# -----------------------
def load_all_windows(seq_len: int, stride: int, use_ema: bool, window_norm: bool
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Return X:(N,3,L), y_user:(N,), y_task:(N,), and a user_id map index->label
    """
    Xs, y_users, y_tasks = [], [], []
    user_map = {}
    task_map = {}
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

        X = build_windows(seq_len, stride, fx, fy, fz)  # (M,3,L)
        if X.shape[0] == 0:
            continue
        if window_norm:
            X = zwin(X)

        Xs.append(X)
        y_users.append(np.full(X.shape[0], u, dtype=np.int64))
        y_tasks.append(np.full(X.shape[0], t, dtype=np.int64))

    if not Xs:
        raise RuntimeError("No windows created. Try smaller --seq_len or smaller --stride / enable --use_ema.")

    X = np.concatenate(Xs, axis=0)
    yu = np.concatenate(y_users, axis=0)
    yt = np.concatenate(y_tasks, axis=0)
    inv_user_map = {v: k for k, v in user_map.items()}
    print(f"Windows: {X.shape} | users: {Counter(yu)} | tasks: {Counter(yt)}")
    return X, yu, yt, inv_user_map

def sample_per_user_task_indices(y_user: np.ndarray, y_task: np.ndarray,
                                 task_id: int, n_train: int, n_test: int, seed: int):
    """
    For a fixed task, for each user gather that task's indices, shuffle,
    and take up to n_train + n_test → split (train/test). Returns lists.
    """
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
        cut = min(n_train, k)  # allow fewer than quota if not enough
        tr_u = iu[:cut]
        te_u = iu[cut:k]
        if len(tr_u) == 0 or len(te_u) == 0:
            # try at least one test if possible
            if k >= 2:
                tr_u = iu[:k-1]
                te_u = iu[k-1:k]
            else:
                continue
        tr_idx.append(tr_u); te_idx.append(te_u)
    if not tr_idx or not te_idx:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(tr_idx), np.concatenate(te_idx)

# -----------------------
# Train / eval
# -----------------------
def train_one_task(Xtr, ytr, Xva, yva, Xte, yte, n_users: int,
                   epochs=100, batch=16, lr=1e-4, seed=42, device=None):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PaperTransformer(in_channels=Xtr.shape[1], n_classes=n_users).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    crit = nn.CrossEntropyLoss()

    def make_loader(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                            torch.tensor(y, dtype=torch.long))
        return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)

    # split a small val set from train if none provided
    if Xva is None or len(Xva) == 0:
        n = len(Xtr)
        nva = max(1, int(0.1 * n))
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    dl_te = torch.utils.data.DataLoader(torch.tensor(Xte, dtype=torch.float32),
                                        batch_size=batch, shuffle=False)
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
    ap = argparse.ArgumentParser("User identification with paper Transformer (per-task)")
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=None, help="window step; default seq_len//2")
    ap.add_argument("--per_user_train", type=int, default=100)
    ap.add_argument("--per_user_test", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window_norm", action="store_true",
                    help="per-window channel-wise z-norm before feeding model")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    stride = args.stride if args.stride is not None else max(1, args.seq_len // 2)

    # Build all windows once
    X, yu, yt, inv_user = load_all_windows(
        seq_len=args.seq_len, stride=stride, use_ema=args.use_ema, window_norm=args.window_norm
    )
    n_users = len(np.unique(yu))
    tasks = sorted(np.unique(yt).tolist())

    print(f"[INFO] seq_len={args.seq_len}, stride={stride}, users={n_users}, tasks={tasks}")

    # Per-task train/test sampling following paper-style quotas
    results = {}
    for t in tasks:
        tr_idx, te_idx = sample_per_user_task_indices(
            y_user=yu, y_task=yt, task_id=t,
            n_train=args.per_user_train, n_test=args.per_user_test, seed=args.seed
        )
        print(f"[task {t}] train {len(tr_idx)}  test {len(te_idx)}")
        if len(tr_idx) == 0 or len(te_idx) == 0:
            print(f"[WARN] task {t} has empty split. Consider smaller --seq_len or --stride, or lower per-user quotas.")
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

        acc = train_one_task(Xtr, ytr, Xva, yva, Xte, yte, n_users=n_users,
                             epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)
        results[t] = float(acc)
        print(f"[task {t}] TEST acc {acc:.3f}")

    # Summary
    if results:
        vals = [v for v in results.values() if not np.isnan(v)]
        overall = float(np.mean(vals)) if vals else float("nan")
    else:
        overall = float("nan")

    print("\n=== SUMMARY (paper Transformer, per-task USER ID) ===")
    for t in tasks:
        print(f"task {t}: {results.get(t, np.nan):.3f}")
    print(f"OVERALL mean acc: {overall:.3f}")

if __name__ == "__main__":
    main()
