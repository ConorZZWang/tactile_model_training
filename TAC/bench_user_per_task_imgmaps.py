# TAC/bench_user_per_task_imgmaps.py
# Per-window image encodings (RP/GAF/Force2D/STFT) -> CNN for per-task USER identification.

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

from PIL import Image
from scipy.signal import stft, get_window

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


def resize_hw(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize (C,H,W) with bilinear (PIL)."""
    if arr.shape[-2:] == (out_h, out_w):
        return arr
    C, H, W = arr.shape
    out = []
    for c in range(C):
        im = Image.fromarray(arr[c])
        im = im.resize((out_w, out_h), resample=Image.BILINEAR)
        out.append(np.array(im))
    return np.stack(out, axis=0)


def cmvn(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Cepstral mean/var norm style over spatial dims: (C,H,W)."""
    m = arr.mean(axis=(-2, -1), keepdims=True)
    s = arr.std(axis=(-2, -1), keepdims=True) + eps
    return (arr - m) / s


# ----------------------------
# Image builders
# ----------------------------
def stft_img_3axis(fx, fy, fz, fs, nperseg, noverlap) -> np.ndarray:
    def one(x):
        f, t, Z = stft(
            x, fs=fs, window=get_window("hann", nperseg),
            nperseg=nperseg, noverlap=noverlap, nfft=nperseg,
            padded=False, boundary=None
        )
        return np.log(np.abs(Z) + 1e-8).astype(np.float32)  # (F,Tt)
    Sx = one(fx); Sy = one(fy); Sz = one(fz)
    return np.stack([Sx, Sy, Sz], axis=0)  # (3,F,Tt)


def recur_plot(x: np.ndarray, sigma: float) -> np.ndarray:
    """Recurrence plot R[i,j] = exp(-|xi-xj|/sigma)."""
    x = (x - x.mean()) / (x.std() + 1e-8)
    d = np.abs(x[:, None] - x[None, :]).astype(np.float32)
    R = np.exp(-d / max(1e-8, sigma)).astype(np.float32)
    return R  # (T,T)


def make_rp_stack(fx, fy, fz, add_norm: bool, sigma: float) -> np.ndarray:
    C = [recur_plot(fx, sigma), recur_plot(fy, sigma), recur_plot(fz, sigma)]
    if add_norm:
        v = np.sqrt(fx**2 + fy**2 + fz**2)
        C.append(recur_plot(v, sigma))
    return np.stack(C, axis=0)  # (C, T, T)


def gaf(x: np.ndarray) -> np.ndarray:
    """Gramian Angular Field (summation)."""
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = 2 * (x - mn) / (mx - mn) - 1.0
    else:
        x = np.zeros_like(x)
    x = np.clip(x, -1.0, 1.0)
    phi = np.arccos(x)
    G = np.cos(phi[:, None] + phi[None, :]).astype(np.float32)  # (T,T)
    return G


def make_gaf_stack(fx, fy, fz, add_norm: bool) -> np.ndarray:
    C = [gaf(fx), gaf(fy), gaf(fz)]
    if add_norm:
        v = np.sqrt(fx**2 + fy**2 + fz**2)
        C.append(gaf(v))
    return np.stack(C, axis=0)  # (C, T, T)


def force_hist2d(x: np.ndarray, y: np.ndarray, bins: int, clip: float) -> np.ndarray:
    xs = np.clip((x - x.mean()) / (x.std() + 1e-8), -clip, clip)
    ys = np.clip((y - y.mean()) / (y.std() + 1e-8), -clip, clip)
    H, xe, ye = np.histogram2d(xs, ys, bins=bins, range=[[-clip, clip], [-clip, clip]])
    H = np.log1p(H).astype(np.float32)
    return H  # (bins,bins)


def make_force_maps(fx, fy, fz, bins: int, clip: float) -> np.ndarray:
    Hxy = force_hist2d(fx, fy, bins=bins, clip=clip)
    Hxz = force_hist2d(fx, fz, bins=bins, clip=clip)
    Hyz = force_hist2d(fy, fz, bins=bins, clip=clip)
    return np.stack([Hxy, Hxz, Hyz], axis=0)  # (3, bins, bins)


# ----------------------------
# Data: per-window image creation
# ----------------------------
def windows_from_csvs(
    window_len: int,
    stride: int,
    fs: float,
    use_ema: bool,
    window_norm: bool,
    image_mode: str,
    add_norm_chan: bool,
    nperseg: int,
    noverlap: int,
    rp_sigma: float,
    force_bins: int,
    force_clip: float,
    out_h: int,
    out_w: int,
    cmvn_flag: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (N, C, H, W) images with labels y_user, y_task from all CSVs using sliding windows.
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

            if image_mode == "stft":
                img = stft_img_3axis(seg[0], seg[1], seg[2], fs, nperseg, noverlap)  # (3,F,Tt)

            elif image_mode == "rp":
                img = make_rp_stack(seg[0], seg[1], seg[2], add_norm=add_norm_chan, sigma=rp_sigma)  # (C,L,L)

            elif image_mode == "gaf":
                img = make_gaf_stack(seg[0], seg[1], seg[2], add_norm=add_norm_chan)  # (C,L,L)

            elif image_mode == "force2d":
                img = make_force_maps(seg[0], seg[1], seg[2], bins=force_bins, clip=force_clip)  # (3,b,b)

            else:
                raise ValueError(f"Unknown image_mode: {image_mode}")

            # ensure float32 & resize to (out_h, out_w)
            if img.ndim != 3:
                raise RuntimeError(f"image builder returned shape {img.shape}, expected (C,H,W)")
            img = img.astype(np.float32)
            img = resize_hw(img, out_h, out_w)
            if cmvn_flag:
                img = cmvn(img)

            Xs.append(img)
            y_users.append(u)
            y_tasks.append(t)

    if not Xs:
        raise RuntimeError("No window-images created. Check paths or make window_len/stride smaller.")

    X = np.stack(Xs, axis=0).astype(np.float32)      # (N, C, H, W)
    yu = np.array(y_users, dtype=np.int64)           # (N,)
    yt = np.array(y_tasks, dtype=np.int64)           # (N,)

    print(f"Images: {X.shape} | users: {Counter(yu)} | tasks: {Counter(yt)}")
    return X, yu, yt


def split_per_task_within_user(N: int, y_user: np.ndarray, y_task: np.ndarray, task_id: int,
                               seed: int = 42, ratios=(0.6, 0.2, 0.2)):
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
# Models
# ----------------------------
class SmallImgCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=7):
        super().__init__()
        ch = 32
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(ch, ch * 2, 3, padding=1), nn.BatchNorm2d(ch * 2), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.BatchNorm2d(ch * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(ch * 2, ch * 4, 3, padding=1), nn.BatchNorm2d(ch * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 4, ch * 4), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(ch * 4, n_classes)
        )

    def forward(self, x):
        z = self.features(x)
        return self.head(z)


# ----------------------------
# Train/Eval
# ----------------------------
def train_one(model, Xtr, ytr, Xva, yva, Xte, yte, epochs=40, bs=128, lr=5e-4, wd=1e-2, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    def mkdl(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

    dl_tr = mkdl(Xtr, ytr, True)
    dl_va = mkdl(Xva, yva, False)
    dl_te = mkdl(Xte, yte, False)

    best, best_va = None, -1.0
    for _ in range(epochs):
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
    ap = argparse.ArgumentParser("Per-window image maps (RP/GAF/Force2D/STFT) for per-task USER ID with CNN")
    # data/windowing
    ap.add_argument("--window_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--window_norm", action="store_true", help="per-window channel z-norm along time")

    # image modes
    ap.add_argument("--image_mode", choices=["rp", "gaf", "force2d", "stft"], default="rp")
    ap.add_argument("--add_norm_chan", action="store_true", help="add ||F|| as extra channel (rp/gaf)")

    # STFT params
    ap.add_argument("--nperseg", type=int, default=64)
    ap.add_argument("--noverlap", type=int, default=48)

    # RP params
    ap.add_argument("--rp_sigma", type=float, default=0.1)

    # Force2D params
    ap.add_argument("--force_bins", type=int, default=64)
    ap.add_argument("--force_clip", type=float, default=3.0)

    # image output / normalization
    ap.add_argument("--img_h", type=int, default=128)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--cmvn", action="store_true")

    # training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)

    # backbone
    ap.add_argument("--backbone", choices=["small", "resnet18"], default="small")

    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Build images
    X, yu, yt = windows_from_csvs(
        window_len=args.window_len,
        stride=args.stride,
        fs=args.fs,
        use_ema=args.use_ema,
        window_norm=args.window_norm,
        image_mode=args.image_mode,
        add_norm_chan=args.add_norm_chan,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        rp_sigma=args.rp_sigma,
        force_bins=args.force_bins,
        force_clip=args.force_clip,
        out_h=args.img_h,
        out_w=args.img_w,
        cmvn_flag=args.cmvn,
    )
    n_users = len(np.unique(yu))
    tasks = sorted(np.unique(yt).tolist())
    print(f"[INFO] users={n_users} tasks={tasks}")

    # Model factory
    def make_model(in_ch):
        if args.backbone == "small":
            return SmallImgCNN(in_ch=in_ch, n_classes=n_users)
        else:
            import torchvision.models as models
            m = models.resnet18(pretrained=False)
            if in_ch != 3:
                # adapt first conv to in_ch
                w = m.conv1.weight
                m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    if in_ch < 3:
                        m.conv1.weight[:, :in_ch] = w[:, :in_ch]
                    else:
                        m.conv1.weight[:] = w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
            m.fc = nn.Linear(m.fc.in_features, n_users)
            return m

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
    print("\n=== SUMMARY (image maps) ===")
    for t in tasks:
        v = results.get(t, np.nan)
        print(f"task {t}: {v if isinstance(v, float) and not np.isnan(v) else np.nan:.3f}")
    print(f"OVERALL mean acc: {overall:.3f}")


if __name__ == "__main__":
    main()
