# Per-task USER authentication from spectrogram images (2D CNN / ResNet).
# - Reads force_x/y/z CSVs, optional EMA, windowing (window_len/stride)
# - Builds log-power spectrograms with scipy.signal.spectrogram
# - Options: 1-channel (merged) or 3-channel (Fx,Fy,Fz) spectrograms
# - Per-image CMVN (channel-wise cepstral mean/var norm) + optional ImageNet norm
# - Backbones: small (custom), resnet18, resnet34
# - Per-task within-user splits; reports per-task & overall accuracy; saves CSV

import os
import argparse
from collections import Counter
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from TAC.load_all import iter_force_files, DATA_ROOT

# ---- optional: use scipy for spectrogram ----
from scipy.signal import spectrogram

# ---------------------------
# Utils
# ---------------------------

def ema_1d(x: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    v = float(x[0])
    for i in range(len(x)):
        v = alpha * float(x[i]) + (1.0 - alpha) * v
        out[i] = v
    return out

def cut_windows(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray, win_len: int, stride: int):
    T = len(fx)
    if T < win_len:
        return []
    starts = range(0, T - win_len + 1, stride)
    return [(s, s+win_len) for s in starts]

def cmvn_per_image(x: np.ndarray) -> np.ndarray:
    """
    x: (C,H,W) float32
    Apply mean/var norm per-channel across (H,W).
    """
    C = x.shape[0]
    x2 = x.copy()
    for c in range(C):
        mu = x2[c].mean()
        sd = x2[c].std() + 1e-8
        x2[c] = (x2[c] - mu) / sd
    return x2

def to_log_power_spec(sig: np.ndarray, fs: float, nperseg: int, noverlap: int, out_hw: Tuple[int,int]) -> np.ndarray:
    """
    sig: (T,) -> (F,Tspec) log-power spectrogram resized to out_hw
    """
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False, scaling="spectrum", mode="psd")
    S = np.log1p(Sxx.astype(np.float32))  # log power
    # resize to (H,W) simple bilinear with torch (avoid skimage dep)
    H, W = out_hw
    ten = torch.tensor(S, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,F,T)
    S_res = torch.nn.functional.interpolate(ten, size=(H, W), mode="bilinear", align_corners=False)
    return S_res.squeeze(0).squeeze(0).cpu().numpy()  # (H,W)

def build_spec_image(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray,
                     fs: float, nperseg: int, noverlap: int,
                     out_hw: Tuple[int,int], channels: str = "3axis") -> np.ndarray:
    """
    Return image tensor (C,H,W):
      channels="3axis": stack Fx,Fy,Fz specs -> 3 channels
      channels="1merge": make one channel from sqrt(Fx^2+Fy^2+Fz^2)
    """
    if channels == "3axis":
        Sx = to_log_power_spec(fx, fs, nperseg, noverlap, out_hw)
        Sy = to_log_power_spec(fy, fs, nperseg, noverlap, out_hw)
        Sz = to_log_power_spec(fz, fs, nperseg, noverlap, out_hw)
        img = np.stack([Sx, Sy, Sz], axis=0)
    elif channels == "1merge":
        mag = np.sqrt(fx**2 + fy**2 + fz**2)
        Sm = to_log_power_spec(mag, fs, nperseg, noverlap, out_hw)
        img = Sm[None, ...]
    else:
        raise ValueError("channels must be '3axis' or '1merge'")
    return img.astype(np.float32)

def windows_to_images(all_index,
                      window_len: int,
                      stride: int,
                      use_ema: bool,
                      fs: float,
                      img_hw: Tuple[int,int],
                      nperseg: int,
                      noverlap: int,
                      channels: str,
                      cmvn: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build dataset of spectrogram images.
    Returns: X:(N,C,H,W), y_user:(N,), y_task:(N,)
    """
    Xs, yu, yt = [], [], []
    user_map, task_map = {}, {}

    for user_id, task_id, csv_path in iter_force_files(DATA_ROOT):
        if user_id not in user_map: user_map[user_id] = len(user_map)
        if task_id not in task_map: task_map[task_id] = len(task_map)
        u = user_map[user_id]; t = task_map[task_id]

        df = pd.read_csv(csv_path)
        for c in ("force_x","force_y","force_z"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["force_x","force_y","force_z"])

        fx = df["force_x"].values.astype(np.float32)
        fy = df["force_y"].values.astype(np.float32)
        fz = df["force_z"].values.astype(np.float32)
        if use_ema:
            fx = ema_1d(fx); fy = ema_1d(fy); fz = ema_1d(fz)

        for s, e in cut_windows(fx, fy, fz, window_len, stride):
            img = build_spec_image(fx[s:e], fy[s:e], fz[s:e], fs, nperseg, noverlap, img_hw, channels=channels)
            if cmvn:
                img = cmvn_per_image(img)
            Xs.append(img)
            yu.append(u)
            yt.append(t)

    if not Xs:
        raise RuntimeError("No windows → images created; try smaller window or stride.")
    X = np.stack(Xs, axis=0)
    yu = np.array(yu, dtype=np.int64)
    yt = np.array(yt, dtype=np.int64)
    print(f"Images: {X.shape} | users: {Counter(yu)} | tasks: {Counter(yt)}")
    return X, yu, yt

def split_per_task_within_user(N: int, y_user: np.ndarray, y_task: np.ndarray, task_id: int, seed=42, ratios=(0.6,0.2,0.2)):
    rng = np.random.default_rng(seed)
    idx_t = np.where(y_task == task_id)[0]
    users = np.unique(y_user[idx_t])
    tr_all, va_all, te_all = [], [], []
    for u in users:
        iu = idx_t[y_user[idx_t] == u]
        rng.shuffle(iu)
        n = len(iu)
        if n < 5:  # skip tiny bins
            continue
        ntr = int(ratios[0]*n); nva = int(ratios[1]*n)
        tr, va, te = iu[:ntr], iu[ntr:ntr+nva], iu[ntr+nva:]
        if len(tr) and len(va) and len(te):
            tr_all.append(tr); va_all.append(va); te_all.append(te)
    if not tr_all:
        return np.array([],dtype=int), np.array([],dtype=int), np.array([],dtype=int)
    return np.concatenate(tr_all), np.concatenate(va_all), np.concatenate(te_all)

# ---------------------------
# Models
# ---------------------------

class SmallImgCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, base: int = 32, drop: float = 0.2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, base, 5, padding=2), nn.BatchNorm2d(base), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(base, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(base*2, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(drop),
            nn.Linear(base*4, base*4), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(base*4, n_classes)
        )

    def forward(self, x):
        return self.head(self.feat(x))

def make_backbone(name: str, in_ch: int, n_classes: int):
    if name == "small":
        return SmallImgCNN(in_ch, n_classes)
    elif name in ("resnet18","resnet34"):
        import torchvision.models as tvm
        if name == "resnet18":
            net = tvm.resnet18(weights=None)
        else:
            net = tvm.resnet34(weights=None)
        # adapt first conv to in_ch
        if in_ch != 3:
            w = net.conv1.weight
            net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # heuristic: avg weights if 1ch
            with torch.no_grad():
                if in_ch == 1:
                    net.conv1.weight[:] = w.mean(dim=1, keepdim=True)
                elif in_ch == 2:
                    net.conv1.weight[:, :2] = w[:, :2]
                    net.conv1.weight[:, 2:] = 0
                else:
                    # in_ch==3 matches default; for >3 you could tile or reduce
                    pass
        # replace fc
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        return net
    else:
        raise ValueError("backbone must be small | resnet18 | resnet34")

# ---------------------------
# Training
# ---------------------------

def train_one(model: nn.Module, Xtr, ytr, Xva, yva, epochs=30, bs=64, lr=3e-4, wd=1e-2, patience=6, imagenet_norm=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def norm_batch(x):
        # if using torchvision resnet with no pretrained weights, ImageNet norm not required.
        # keep switch for experimentation.
        if imagenet_norm:
            mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
            if x.size(1) == 3:
                return (x-mean)/std
        return x

    ds_tr = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    ds_va = TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.long))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    best_state, best_va, wait = None, -1.0, 0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            xb = norm_batch(xb)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        # val
        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                xb = norm_batch(xb)
                pr = model(xb).argmax(1)
                correct += int((pr==yb).sum().item())
                n += len(xb)
        va = correct / max(1,n)
        if va > best_va:
            best_va, wait = va, 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # prediction helper
    @torch.no_grad()
    def predict(X):
        dl = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=bs, shuffle=False)
        out = []
        for xb in dl:
            xb = xb.to(device)
            xb = norm_batch(xb)
            out.append(model(xb).argmax(1).cpu().numpy())
        return np.concatenate(out, axis=0)

    return model, predict

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser("Per-task USER auth from spectrogram images")
    # data / imaging
    ap.add_argument("--window_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--fs", type=float, default=250.0, help="sampling rate")
    ap.add_argument("--img_h", type=int, default=160)
    ap.add_argument("--img_w", type=int, default=160)
    ap.add_argument("--nperseg", type=int, default=128)
    ap.add_argument("--noverlap", type=int, default=96)
    ap.add_argument("--channels", choices=["1merge","3axis"], default="3axis")
    ap.add_argument("--cmvn", action="store_true", help="per-image channel-wise mean/var norm")
    # model / train
    ap.add_argument("--backbone", choices=["small","resnet18","resnet34"], default="small")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--imagenet_norm", action="store_true", help="apply ImageNet mean/std (only sensible for 3ch)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="bench_user_per_task_imgcnn.csv")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")

    # 1) Build spectrogram images
    X, yu, yt = windows_to_images(
        all_index=tuple(iter_force_files(DATA_ROOT)),
        window_len=args.window_len, stride=args.stride, use_ema=args.use_ema,
        fs=args.fs, img_hw=(args.img_h, args.img_w),
        nperseg=args.nperseg, noverlap=args.noverlap,
        channels=args.channels, cmvn=args.cmvn
    )
    n_users = len(np.unique(yu))
    tasks = sorted(np.unique(yt).tolist())
    print(f"[INFO] users={n_users} tasks={tasks}")

    # 2) Per-task split & train
    results = []
    for t in tasks:
        tr, va, te = split_per_task_within_user(len(X), yu, yt, task_id=t, seed=args.seed)
        if len(tr)==0 or len(va)==0 or len(te)==0:
            print(f"[task {t}] not enough data")
            continue

        Xtr, Xva, Xte = X[tr], X[va], X[te]
        ytr, yva, yte = yu[tr], yu[va], yu[te]

        model = make_backbone(args.backbone, in_ch=X.shape[1], n_classes=n_users)
        model, pred_fn = train_one(model, Xtr, ytr, Xva, yva,
                                   epochs=args.epochs, bs=args.batch, lr=args.lr,
                                   wd=args.wd, patience=args.patience,
                                   imagenet_norm=args.imagenet_norm)
        yp = pred_fn(Xte)
        acc = (yp == yte).mean()
        print(f"[task {t}] TEST acc {acc:.3f}")
        results.append((t, acc))

    # 3) Summary
    if results:
        per = {t:a for t,a in results}
        overall = float(np.mean([a for _,a in results]))
    else:
        per, overall = {}, float("nan")

    print("\n=== SUMMARY (img spectrogram) ===")
    for t in tasks:
        print(f"task {t}: {per.get(t, np.nan):.3f}")
    print(f"OVERALL mean acc: {overall:.3f}")

    # save CSV
    df = pd.DataFrame([{"model": args.backbone, "overall_acc": overall, **{f"task{t}_acc": per.get(t, np.nan) for t in tasks}}])
    df.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}")

if __name__ == "__main__":
    main()
