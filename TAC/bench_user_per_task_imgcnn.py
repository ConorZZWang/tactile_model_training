# TAC/bench_user_per_task_imgcnn.py
import os, argparse
import numpy as np
from collections import Counter
from typing import Tuple
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import spectrogram
import torch.nn.functional as F

from TAC.load_all import iter_force_files, DATA_ROOT

def windows_from_index(all_index, window_len=512, stride=512, use_ema=False):
    def ema(x, a=0.1):
        y=np.empty_like(x); v=x[0]
        for i in range(len(x)):
            v = a*x[i] + (1-a)*v
            y[i]=v
        return y

    Xs, Yu, Yt = [], [], []
    user_map, task_map = {}, {}
    import pandas as pd
    for (user_id, task_id, csv_path) in all_index:
        u = user_map.setdefault(user_id, len(user_map))
        t = task_map.setdefault(task_id, len(task_map))
        df = pd.read_csv(csv_path)
        fx = pd.to_numeric(df["force_x"], errors="coerce").dropna().values.astype(np.float32)
        fy = pd.to_numeric(df["force_y"], errors="coerce").dropna().values.astype(np.float32)
        fz = pd.to_numeric(df["force_z"], errors="coerce").dropna().values.astype(np.float32)
        n = min(len(fx), len(fy), len(fz))
        fx,fy,fz = fx[:n],fy[:n],fz[:n]
        if use_ema:
            fx,fy,fz = ema(fx), ema(fy), ema(fz)
        if n < window_len: continue
        for s in range(0, n-window_len+1, stride):
            Xs.append(np.stack([fx[s:s+window_len], fy[s:s+window_len], fz[s:s+window_len]], 0))
            Yu.append(u); Yt.append(t)
    X=np.stack(Xs,0); Yu=np.array(Yu); Yt=np.array(Yt)
    print("Raw windows:", X.shape, "| users:", Counter(Yu), "| tasks:", Counter(Yt))
    return X, Yu, Yt

def to_spectrogram_rgb(x_3t, fs=250.0, nperseg=128, noverlap=96, out_hw=(128,128)):
    """
    x_3t: (3, T)
    Returns: (3, H, W) log-spectrogram stack
    """
    imgs=[]
    for c in range(3):
        f, t, Sxx = spectrogram(x_3t[c], fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum", mode="magnitude")
        S = np.log10(Sxx + 1e-10).astype(np.float32)  # (F, Tspec)
        # normalize per-channel image
        S = (S - S.mean()) / (S.std() + 1e-6)
        S = torch.from_numpy(S).unsqueeze(0).unsqueeze(0)  # (1,1,F,Tspec)
        S = F.interpolate(S, size=out_hw, mode="bilinear", align_corners=False)  # (1,1,H,W)
        imgs.append(S.squeeze(0))  # (1,H,W)
    img = torch.cat(imgs, dim=0)  # (3,H,W)
    return img

class SmallImgCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=7, base=32, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 if input 128x128

            nn.Conv2d(base, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(base*2, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(base*4, base*4), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(base*4, n_classes),
        )
    def forward(self, x):
        return self.head(self.net(x))

def split_per_task_within_user(N, y_user, y_task, task_id, seed=42, ratios=(0.6,0.2,0.2)):
    rng = np.random.default_rng(seed)
    idx = np.where(y_task==task_id)[0]
    users = np.unique(y_user[idx])
    tr, va, te = [], [], []
    for u in users:
        iu = idx[y_user[idx]==u]
        rng.shuffle(iu)
        n=len(iu); 
        if n<5: continue
        ntr=int(ratios[0]*n); nva=int(ratios[1]*n)
        tr.append(iu[:ntr]); va.append(iu[ntr:ntr+nva]); te.append(iu[ntr+nva:])
    if not tr: return np.array([],int), np.array([],int), np.array([],int)
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_h", type=int, default=128)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--nperseg", type=int, default=128)
    ap.add_argument("--noverlap", type=int, default=96)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    all_index = tuple(iter_force_files(DATA_ROOT))
    Xraw, y_user, y_task = windows_from_index(all_index, args.window_len, args.stride, args.use_ema)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_users = len(np.unique(y_user))
    tasks = sorted(np.unique(y_task).tolist())

    results = []
    for t in tasks:
        tr, va, te = split_per_task_within_user(len(Xraw), y_user, y_task, t, seed=args.seed)
        if len(tr)==0 or len(va)==0 or len(te)==0:
            print(f"[task {t}] not enough data"); continue

        # turn windows -> spectrogram images
        def make_imgs(idx):
            imgs = [to_spectrogram_rgb(Xraw[i], nperseg=args.nperseg, noverlap=args.noverlap,
                                       out_hw=(args.img_h,args.img_w)) for i in idx]
            Ximg = torch.stack(imgs, dim=0)         # (N,3,H,W)
            y = torch.from_numpy(y_user[idx]).long()
            return Ximg, y

        Xtr, ytr = make_imgs(tr)
        Xva, yva = make_imgs(va)
        Xte, yte = make_imgs(te)

        # standardize per-channel over TRAIN set (optional but helps)
        mean = Xtr.mean(dim=(0,2,3), keepdim=True)
        std  = Xtr.std(dim=(0,2,3), keepdim=True) + 1e-6
        Xtr = (Xtr - mean)/std; Xva = (Xva - mean)/std; Xte = (Xte - mean)/std

        dl_tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch, shuffle=True,
                           num_workers=4, pin_memory=True, persistent_workers=True)
        dl_va = DataLoader(TensorDataset(Xva, yva), batch_size=args.batch, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)

        model = SmallImgCNN(in_ch=3, n_classes=n_users, base=32).to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        crit = nn.CrossEntropyLoss()

        best, best_va = None, -1.0
        for ep in range(1, args.epochs+1):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            # val
            model.eval(); corr=n=0
            with torch.no_grad():
                for xb,yb in dl_va:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    pred = model(xb).argmax(1); corr += int((pred==yb).sum()); n += len(xb)
            va_acc = corr/max(1,n)
            if va_acc>best_va:
                best_va=va_acc; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if best: model.load_state_dict(best)

        # test
        dl_te = DataLoader(TensorDataset(Xte, yte), batch_size=args.batch, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)
        model.eval(); corr=n=0
        with torch.no_grad():
            for xb,yb in dl_te:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                pred = model(xb).argmax(1); corr += int((pred==yb).sum()); n+=len(xb)
        acc = corr/max(1,n)
        results.append(acc)
        print(f"[task {t}] test_acc {acc:.3f}")

    if results:
        print(f"Overall TEST acc (imgCNN): {float(np.mean(results)):.3f}")
    else:
        print("No results.")

if __name__ == "__main__":
    main()
