import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TAC.load_all import DATA_ROOT


def gaf(x: np.ndarray) -> np.ndarray:
    """Gramian Angular Field (summation), same as in bench_user_per_task_imgmaps.py."""
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = 2 * (x - mn) / (mx - mn) - 1.0
    else:
        x = np.zeros_like(x)
    x = np.clip(x, -1.0, 1.0)
    phi = np.arccos(x)
    G = np.cos(phi[:, None] + phi[None, :]).astype(np.float32)  # (T, T)
    return G


def main():
    ap = argparse.ArgumentParser("Plot a single GAF image from force.csv")
    ap.add_argument("--user", type=str, default="u1",
                    help="User folder name, e.g. u1, u2, ...")
    ap.add_argument("--task", type=str, default="a",
                    help="Task/action folder, e.g. a, b, c, ... g")
    ap.add_argument("--axis", type=str, default="norm",
                    choices=["x", "y", "z", "norm"],
                    help="Which signal to use for GAF: force_x, force_y, force_z, or ||F||")
    ap.add_argument("--window_len", type=int, default=512,
                    help="Window length in samples (0 to use full sequence)")
    ap.add_argument("--start", type=int, default=0,
                    help="Start index of the window inside the recording")
    args = ap.parse_args()

    # Build path: DATA_ROOT / user / task / force.csv
    csv_path = os.path.join(DATA_ROOT, args.user, args.task, "force.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")

    print(f"[INFO] Loading {csv_path}")
    df = pd.read_csv(csv_path)

    # Ensure numeric and drop NaNs
    for c in ("force_x", "force_y", "force_z"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["force_x", "force_y", "force_z"])

    fx = df["force_x"].values.astype(np.float32)
    fy = df["force_y"].values.astype(np.float32)
    fz = df["force_z"].values.astype(np.float32)

    if args.axis == "x":
        sig = fx
    elif args.axis == "y":
        sig = fy
    elif args.axis == "z":
        sig = fz
    else:  # "norm"
        sig = np.sqrt(fx**2 + fy**2 + fz**2)

    T = len(sig)
    print(f"[INFO] Signal length: {T} samples")

    # Choose segment: either full signal or a window
    if args.window_len > 0 and args.window_len < T:
        s = max(0, args.start)
        e = min(T, s + args.window_len)
        if e - s < args.window_len:
            print(f"[WARN] Requested window [{s}:{s + args.window_len}] "
                  f"truncated to [{s}:{e}] because recording is shorter.")
        seg = sig[s:e]
        print(f"[INFO] Using segment [{s}:{e}] of length {len(seg)}")
    else:
        seg = sig
        print(f"[INFO] Using full signal of length {len(seg)}")

    # Compute GAF
    G = gaf(seg)
    print(f"[INFO] GAF shape: {G.shape}")

    # Plot
    plt.figure(figsize=(5, 5))
    im = plt.imshow(G, origin="lower", aspect="equal")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"GAF | user={args.user}, task={args.task}, axis={args.axis}")
    plt.xlabel("time index")
    plt.ylabel("time index")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()