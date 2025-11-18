# Per-CSV spectrogram images -> CNN for per-task USER authentication.
import os, argparse, numpy as np, pandas as pd
from collections import Counter
from typing import Tuple
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import stft, get_window

from TAC.load_all import iter_force_files, DATA_ROOT

# ---------------- helpers ----------------
def ema_1d(x, alpha=0.001):
    y = np.empty_like(x, dtype=np.float32); v = float(x[0])
    for i in range(len(x)):
        v = alpha*float(x[i]) + (1.0-alpha)*v; y[i] = v
    return y

def cmvn(arr, eps=1e-8):
    m = arr.mean(axis=(-2,-1), keepdims=True)
    s = arr.std(axis=(-2,-1), keepdims=True) + eps
    return (arr - m) / s

def make_spec(x, fs, nperseg, noverlap, window="hann"):
    f, t, Z = stft(x, fs=fs, window=get_window(window, nperseg),
                   nperseg=nperseg, noverlap=noverlap, nfft=nperseg,
                   padded=False, boundary=None)
    S = np.log(np.abs(Z) + 1e-8).astype(np.float32)  # (F, Tt)
    return S

def resize_hw(a, out_h, out_w):
    if a.shape[-2:] == (out_h, out_w):
        return a
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("pip install pillow")
    C, H, W = a.shape
    out = []
    for c in range(C):
        im = Image.fromarray(a[c])
        im = im.resize((out_w, out_h), resample=Image.BILINEAR)
        out.append(np.array(im))
    return np.stack(out, axis=0)

# ---------------- model ----------------
class SmallImgCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=7):
        super().__init__()
        ch = 32
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ReLU(),
            nn.Conv2d(ch*2, ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ReLU(),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(ch*2, ch*4, 3, padding=1), nn.BatchNorm2d(ch*4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch*4, ch*4), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ch*4, n_classes),
        )
    def forward(self, x):
        return self.head(self.net(x))

# ---------------- data ----------------
def build_images_per_csv(fs=250.0, use_ema=False, channels="3axis",
                         nperseg=64, noverlap=48, img_h=80, img_w=40,
                         cmvn_flag=False, chunks=1, chunk_overlap=0.0):
    Xs, yu, yt = [], [], []
    user_map, task_map = {}, {}

    def stft_img_3axis(fx, fy, fz):
        Sx = make_spec(fx, fs, nperseg, noverlap)
        Sy = make_spec(fy, fs, nperseg, noverlap)
        Sz = make_spec(fz, fs, nperseg, noverlap)
        C = np.stack([Sx, Sy, Sz], axis=0)  # (3,F,Tt)
        return resize_hw(C, img_h, img_w)

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

        # --- chunking setup ---
        T = len(fx)
        min_len = max(nperseg * 2, 512)       # be sure each chunk has enough for STFT
        K = max(1, chunks)
        ov = float(np.clip(chunk_overlap, 0.0, 0.9))
        if K == 1:
            starts = [0]; chunk_len = T
        else:
            # choose chunk_len so that with given overlap we get ~K crops
            # step = chunk_len * (1 - ov)  and  1 + floor((T - chunk_len)/step) ≈ K
            # solve approx by trying a few chunk_len values
            best = None
            for frac in np.linspace(0.25, 0.75, 21):  # search chunk length ~ 25–75% of trial
                L = max(min_len, int(T * frac))
                step = max(1, int(L * (1 - ov)))
                n_crops = 1 + max(0, (T - L) // step)
                if best is None or abs(n_crops - K) < abs(best[0] - K):
                    best = (n_crops, L, step)
            _, chunk_len, step = best
            starts = list(range(0, max(1, T - chunk_len + 1), step))
            # trim if we overshot K a lot
            if len(starts) > K: starts = starts[:K]

        # --- make images per chunk ---
        made_any = False
        for s in starts:
            e = min(T, s + chunk_len)
            if e - s < min_len: continue
            if channels == "3axis":
                C = stft_img_3axis(fx[s:e], fy[s:e], fz[s:e])
            elif channels == "norm":
                v = np.sqrt(fx[s:e]**2 + fy[s:e]**2 + fz[s:e]**2)
                Sv = make_spec(v, fs, nperseg, noverlap)[None, ...]
                C = resize_hw(Sv, img_h, img_w)
            else:
                v = np.sqrt(fx[s:e]**2 + fy[s:e]**2 + fz[s:e]**2)
                S1 = make_spec(v, fs, 32, 24)
                S2 = make_spec(v, fs, 64, 48)
                S3 = make_spec(v, fs, 128, 96)
                C = resize_hw(np.stack([S1, S2, S3], axis=0), img_h, img_w)

            if cmvn_flag: C = cmvn(C)
            Xs.append(C); yu.append(u); yt.append(t); made_any = True

        if not made_any:
            # fallback: try full signal once if chunks failed min_len
            if T >= min_len:
                C = stft_img_3axis(fx, fy, fz) if channels=="3axis" else None
                if C is not None:
                    if cmvn_flag: C = cmvn(C)
                    Xs.append(C); yu.append(u); yt.append(t)

    if not Xs:
        raise RuntimeError("No images created. Use smaller nperseg/noverlap or fewer chunks/overlap.")

    X = np.stack(Xs, axis=0).astype(np.float32)
    yu = np.array(yu, dtype=np.int64)
    yt = np.array(yt, dtype=np.int64)

    print(f"Images: {X.shape} | users: Counter({Counter(yu)}) | tasks: Counter({Counter(yt)})")
    return X, yu, yt

def split_per_task_within_user(N, y_user, y_task, task_id, seed=42, ratios=(0.6,0.2,0.2)):
    rng = np.random.default_rng(seed)
    idx_t = np.where(y_task == task_id)[0]
    users = np.unique(y_user[idx_t])
    tr, va, te = [], [], []
    for u in users:
        iu = idx_t[y_user[idx_t] == u]
        iu = rng.permutation(iu)
        m = len(iu)
        if m < 2:             # cannot have both train & test
            continue
        ntr = max(1, int(ratios[0]*m))
        nva = max(0, int(ratios[1]*m))
        if ntr + nva >= m:    # leave at least 1 for test
            ntr = max(1, m-1); nva = 0
        tr.append(iu[:ntr]); va.append(iu[ntr:ntr+nva]); te.append(iu[ntr+nva:])
    if not tr:
        return np.array([], int), np.array([], int), np.array([], int)
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)

# ---------------- train ----------------
def train_one(model, Xtr, ytr, Xva, yva, Xte, yte, epochs=40, bs=128, lr=5e-4, wd=1e-2, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_users = int(ytr.max()) + 1
    model = model.to(dev)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    def mk(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

    dl_tr = mk(Xtr, ytr, True); dl_va = mk(Xva, yva, False); dl_te = mk(Xte, yte, False)

    best, best_va = None, -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb); loss.backward(); opt.step()
        # val
        model.eval(); corr = n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb).argmax(1)
                corr += int((pred==yb).sum()); n += len(yb)
        acc = corr / max(1,n)
        if acc > best_va:
            best_va = acc
            best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best is not None: model.load_state_dict(best)

    # test
    model.eval(); corr = n = 0
    with torch.no_grad():
        for xb, yb in dl_te:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb).argmax(1)
            corr += int((pred==yb).sum()); n += len(yb)
    return corr / max(1,n)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Per-CSV spectrogram CNN (per-task user ID)")
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--nperseg", type=int, default=64)
    ap.add_argument("--noverlap", type=int, default=48)
    ap.add_argument("--channels", choices=["3axis","norm","multires"], default="3axis")
    ap.add_argument("--cmvn", action="store_true")
    ap.add_argument("--img_h", type=int, default=80)
    ap.add_argument("--img_w", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", choices=["small","resnet18"], default="small")
    ap.add_argument("--chunks", type=int, default=1,
                help="number of time crops per CSV (>=1). If 1, behave like before.")
    ap.add_argument("--chunk_overlap", type=float, default=0.0,
                help="overlap fraction between crops in [0, 0.9]. e.g., 0.5 = 50% overlap")

    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")

    # Build one image per CSV
    X, yu, yt = build_images_per_csv(
        fs=args.fs, use_ema=args.use_ema, channels=args.channels,
        nperseg=args.nperseg, noverlap=args.noverlap,
        img_h=args.img_h, img_w=args.img_w, cmvn_flag=args.cmvn,
        chunks=args.chunks, chunk_overlap=args.chunk_overlap
    )

    # choose model factory
    def make_model(in_ch):
        if args.backbone == "small":
            return SmallImgCNN(in_ch=in_ch, n_classes=n_users)
        else:
            # torchvision resnet18
            import torchvision.models as models
            m = models.resnet18(pretrained=False)
            if in_ch != 3:  # adapt first conv
                w = m.conv1.weight
                m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
                if in_ch < 3:
                    m.conv1.weight.data[:, :in_ch] = w.data[:, :in_ch]
                else:
                    # repeat/avg to fill channels
                    m.conv1.weight.data = w.data.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
            m.fc = nn.Linear(m.fc.in_features, n_users)
            return m

    # Per-task within-user splits
    results = {}
    for t in tasks:
        tr, va, te = split_per_task_within_user(len(X), yu, yt, task_id=t, seed=args.seed)
        if len(tr)==0 or len(va)==0 or len(te)==0:
            print(f"[task {t}] not enough data"); results[t]=np.nan; continue
        Xtr, ytr = X[tr], yu[tr]
        Xva, yva = X[va], yu[va]
        Xte, yte = X[te], yu[te]
        model = make_model(in_ch=X.shape[1])
        acc = train_one(model, Xtr, ytr, Xva, yva, Xte, yte,
                        epochs=args.epochs, bs=args.batch, lr=args.lr, wd=args.wd, seed=args.seed)
        results[t] = float(acc)
        print(f"[task {t}] TEST acc {acc:.3f}")

    vals = [v for v in results.values() if not np.isnan(v)]
    overall = float(np.mean(vals)) if vals else float("nan")
    print("\n=== SUMMARY (per-CSV spectrogram) ===")
    for t in tasks: print(f"task {t}: {results.get(t, np.nan):.3f}")
    print(f"OVERALL mean acc: {overall:.3f}")

if __name__ == "__main__":
    main()
