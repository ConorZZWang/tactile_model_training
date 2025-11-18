import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import stft, get_window
from TAC.load_all import iter_force_files, DATA_ROOT

def ema_1d(x, alpha=0.001):
    y = np.empty_like(x, dtype=np.float32); v = float(x[0])
    for i in range(len(x)):
        v = alpha * float(x[i]) + (1.0 - alpha) * v; y[i] = v
    return y

def cmvn(arr, eps=1e-8):
    m = arr.mean(axis=(-2,-1), keepdims=True)
    s = arr.std(axis=(-2,-1), keepdims=True) + eps
    return (arr - m) / s

def force_to_frames(fx, fy, fz, L, stride):
    T = len(fx)
    if T < L: return []
    starts = range(0, T - L + 1, stride)
    return [(s, s+L) for s in starts]

def make_spec(x, fs, nperseg, noverlap, window="hann"):
    f, t, Z = stft(x, fs=fs, window=get_window(window, nperseg),
                   nperseg=nperseg, noverlap=noverlap, nfft=nperseg, padded=False, boundary=None)
    S = np.log(np.abs(Z) + 1e-8).astype(np.float32)  # (F, Tt)
    return S  # frequency x time

def stack_channels(specs, out_hw=None):
    # specs: list of (F,Tt) to stack as channels -> (C,H,W)
    H = specs[0].shape[0] if out_hw is None else out_hw[0]
    W = specs[0].shape[1] if out_hw is None else out_hw[1]
    out = []
    for s in specs:
        if out_hw is not None:
            s = np.array(Image.fromarray(s).resize((W, H), resample=Image.BILINEAR))
        out.append(s)
    x = np.stack(out, axis=0)  # (C,H,W)
    return x

def save_image(tensor_chw, path, vmin=None, vmax=None, cmap="magma", title=None):
    C,H,W = tensor_chw.shape
    fig, ax = plt.subplots(figsize=(W/50, H/50), dpi=100)
    if C == 1:
        ax.imshow(tensor_chw[0], origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        # visualise first three channels as RGB after min-max per-channel
        x = tensor_chw[:3].copy()
        for c in range(x.shape[0]):
            mn, mx = x[c].min(), x[c].max()
            if mx > mn: x[c] = (x[c]-mn)/(mx-mn)
        x = np.transpose(x, (1,2,0))  # HWC
        ax.imshow(x, origin="lower", aspect="auto")
    if title: ax.set_title(title, fontsize=8)
    ax.axis("off"); plt.tight_layout(pad=0); fig.savefig(path, bbox_inches="tight", pad_inches=0); plt.close(fig)

def main():
    ap = argparse.ArgumentParser("Preview spectrogram images")
    ap.add_argument("--window_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--nperseg", type=int, default=64)
    ap.add_argument("--noverlap", type=int, default=48)
    ap.add_argument("--channels", choices=["3axis","norm","multires"], default="3axis")
    ap.add_argument("--cmvn", action="store_true")
    ap.add_argument("--img_h", type=int, default=80)
    ap.add_argument("--img_w", type=int, default=40)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--limit", type=int, default=16, help="number of images to dump")
    ap.add_argument("--out_dir", default="preview_specs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    count = 0

    for user_id, task_id, csv_path in iter_force_files(DATA_ROOT):
        df = pd.read_csv(csv_path)
        for c in ("force_x","force_y","force_z"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["force_x","force_y","force_z"])
        fx = df["force_x"].values.astype(np.float32)
        fy = df["force_y"].values.astype(np.float32)
        fz = df["force_z"].values.astype(np.float32)
        if args.use_ema:
            fx = ema_1d(fx); fy = ema_1d(fy); fz = ema_1d(fz)

        for s,e in force_to_frames(fx, fy, fz, args.window_len, args.stride):
            segx, segy, segz = fx[s:e], fy[s:e], fz[s:e]

            if args.channels == "3axis":
                Sx = make_spec(segx, args.fs, args.nperseg, args.noverlap)
                Sy = make_spec(segy, args.fs, args.nperseg, args.noverlap)
                Sz = make_spec(segz, args.fs, args.nperseg, args.noverlap)
                C = np.stack([Sx, Sy, Sz], axis=0)  # (3,F,Tt)

            elif args.channels == "norm":
                v = np.sqrt(segx**2 + segy**2 + segz**2)
                Sv = make_spec(v, args.fs, args.nperseg, args.noverlap)
                C = Sv[None, ...]  # (1,F,Tt)

            else:  # multires on force-norm: three different (nperseg,noverlap)
                v = np.sqrt(segx**2 + segy**2 + segz**2)
                S1 = make_spec(v, args.fs, 32, 24)
                S2 = make_spec(v, args.fs, 64, 56)
                S3 = make_spec(v, args.fs, 128, 112)
                C = np.stack([S1, S2, S3], axis=0)

            # resize to (C,img_h,img_w)
            # use PIL only when size change is needed
            H,W = C.shape[-2], C.shape[-1]
            if (H, W) != (args.img_h, args.img_w):
                try:
                    from PIL import Image
                except ImportError:
                    raise SystemExit("pip install pillow to resize")
                C_resz = []
                for k in range(C.shape[0]):
                    im = Image.fromarray(C[k])
                    im = im.resize((args.img_w, args.img_h), resample=Image.BILINEAR)
                    C_resz.append(np.array(im))
                C = np.stack(C_resz, axis=0)

            if args.cmvn:
                C = cmvn(C)

            fn = f"user{user_id}_task{task_id}_start{s}.png"
            save_image(C, os.path.join(args.out_dir, fn),
                       title=f"user {user_id} task {task_id}")

            count += 1
            if count >= args.limit:
                print(f"[done] wrote {count} images to {args.out_dir}")
                return

    print(f"[done] wrote {count} images to {args.out_dir}")

if __name__ == "__main__":
    main()
