# TAC/bench_user_per_task_stft_xgb_feats.py
# Per-task USER authentication using STFT "graph features" + XGBoost.
#
# Idea:
#   Instead of feeding the full spectrogram map into a CNN, we extract a compact
#   feature vector from the STFT (bandpowers + spectral stats) and train XGBoost.
#
# Output:
#   CSV summary with per-task and overall accuracy (same format style as your other runners).
#
# Example:
#   python -m TAC.bench_user_per_task_stft_xgb_feats ^
#     --window_len 768 --stride 256 --use_ema --window_norm ^
#     --stft_n_fft 128 --stft_hop 8 --stft_keep_bins 64 ^
#     --out_csv runs/stft_feats_xgb.csv
#
# Requires:
#   pip install xgboost

import os
import argparse
from collections import Counter
from typing import Tuple, List

import numpy as np
import pandas as pd

import torch

from TAC.load_all import iter_force_files, DATA_ROOT

torch.backends.cudnn.benchmark = True

# Helpers 

def zwin(x: np.ndarray) -> np.ndarray:
    """Per-window z-normalization: (N, C, T) -> (N, C, T)."""
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


def windows_from_index(
    all_index,
    window_len: int = 512,
    stride: int = 512,
    use_ema: bool = False,
    ema_alpha: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X, y_user, y_task):
      - X: (N, 3, T)
      - y_user: (N,)
      - y_task: (N,)
    """
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

    X = np.stack(X_list, axis=0)  # (N, 3, T)
    y_user = np.array(y_user_list, dtype=np.int64)
    y_task = np.array(y_task_list, dtype=np.int64)

    print("Raw windows:", X.shape, "| users:", Counter(y_user), "| tasks:", Counter(y_task))
    return X, y_user, y_task


def split_per_task_within_user(
    y_user: np.ndarray,
    y_task: np.ndarray,
    task_id: int,
    seed: int = 42,
    ratios=(0.6, 0.2, 0.2)
):
    """For a task, split indices for each user into train/val/test by ratios."""
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


# ---------------------------
# STFT feature extraction
# ---------------------------
@torch.no_grad()
def stft_features(
    X: np.ndarray,
    fs: float = 250.0,
    n_fft: int = 128,
    hop: int = 8,
    keep_bins: int = 64,
    log_eps: float = 1e-6,
    bands_hz: List[Tuple[float, float]] = None,
    use_log_power: bool = True,
    add_band_dynamics: bool = True,   # NEW
) -> np.ndarray:
    """
    Compute STFT feature vectors per window.

    X: (N, 3, T)
    Returns: feats (N, D) float32

    Base features per axis:
      - bandpowers from mean spectrum S(f)=mean_t power(f,t): sum_f S(f) in each band
      - spectral centroid
      - spectral entropy
      - spectral rolloff (85%)
      - total frame energy mean/std (sum over f each frame)

    Optional (recommended) per axis band-dynamics:
      For each band, define band energy per frame:
        E_band(t) = sum_{f in band} power(f,t)
      Then add stats over time:
        mean, std, q25, q75, delta-mean, delta-std
      where delta is first difference over frames.

    D = 3 * (nbands + 5 + nbands*6) if add_band_dynamics else 3*(nbands+5)
    """
    assert X.ndim == 3 and X.shape[1] == 3
    if bands_hz is None:
        bands_hz = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 80.0)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.tensor(X, dtype=torch.float32, device=device)  # (N,3,T)
    N, C, T = xt.shape

    xc = xt.reshape(N * C, T)
    window = torch.hann_window(n_fft, device=device)

    spec = torch.stft(
        xc,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True
    )  # (N*C, F, W)

    # Power spectrogram (non-negative)
    power = (spec.real ** 2 + spec.imag ** 2)  # (N*C, F, W)

    # keep low freq bins
    kb = min(keep_bins, power.shape[1])
    power = power[:, :kb, :]  # (N*C, kb, W)

    # Optionally use log-power for stability
    if use_log_power:
        power = torch.log(power + log_eps)

    # Frequency axis (Hz) for kept bins
    freqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / fs).to(device)[:kb]  # (kb,)

    # Mean spectrum over time frames: S(f) = mean_t power(f,t)
    S = power.mean(dim=2)  # (N*C, kb)

    # Total power per frame (energy dynamics)
    frame_energy = power.sum(dim=1)              # (N*C, W)
    frame_mean = frame_energy.mean(dim=1)        # (N*C,)
    frame_std = frame_energy.std(dim=1)          # (N*C,)

    # For centroid/entropy/rolloff we want positive weights
    Wpos = torch.exp(S) if use_log_power else S.clamp_min(0.0)

    denom = Wpos.sum(dim=1) + 1e-8
    centroid = (Wpos * freqs[None, :]).sum(dim=1) / denom

    p = Wpos / denom[:, None]
    entropy = -(p * (p + 1e-12).log()).sum(dim=1)

    cumsum = torch.cumsum(Wpos, dim=1)
    thr = 0.85 * denom
    roll_idx = torch.searchsorted(cumsum, thr[:, None]).squeeze(1).clamp(0, kb - 1)
    rolloff = freqs[roll_idx]

    # ---- Bandpowers from mean spectrum S(f) ----
    band_feats = []
    band_masks = []  # reuse masks for dynamics
    for (lo, hi) in bands_hz:
        mask = (freqs >= lo) & (freqs < hi)
        band_masks.append(mask)
        if mask.any():
            bp = S[:, mask].sum(dim=1)  # (N*C,)
        else:
            bp = torch.zeros((N * C,), device=device)
        band_feats.append(bp)

    feat_list = band_feats + [centroid, entropy, rolloff, frame_mean, frame_std]

    # ---- NEW: Band dynamics over time frames ----
    if add_band_dynamics:
        W = power.shape[2]
        for mask in band_masks:
            if mask.any():
                e = power[:, mask, :].sum(dim=1)         # (N*C, W)
            else:
                e = torch.zeros((N * C, W), device=device)

            e_mean = e.mean(dim=1)
            e_std  = e.std(dim=1)

            # quantiles over frames (robust to spikes)
            e_q25  = e.quantile(0.25, dim=1)
            e_q75  = e.quantile(0.75, dim=1)

            # delta over frames
            de = e[:, 1:] - e[:, :-1]                    # (N*C, W-1)
            de = torch.cat([de[:, :1], de], dim=1)       # pad -> (N*C, W)
            de_mean = de.mean(dim=1)
            de_std  = de.std(dim=1)

            feat_list.extend([e_mean, e_std, e_q25, e_q75, de_mean, de_std])

    feats_nc = torch.stack(feat_list, dim=1)             # (N*C, D_axis)
    feats = feats_nc.reshape(N, C * feats_nc.shape[1])   # (N, 3*D_axis)
    return feats.detach().cpu().numpy().astype(np.float32, copy=False)


# ---------------------------
# XGBoost trainer
# ---------------------------

def train_xgb(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    n_classes: int,
    seed: int,
    n_estimators: int = 2000,
    lr: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample: float = 0.8,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    early_stopping_rounds: int = 50,
):
    import xgboost as xgb

    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)

    params = {
        "objective": "multi:softprob",
        "num_class": int(n_classes),
        "eta": float(lr),
        "max_depth": int(max_depth),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample),
        "lambda": float(reg_lambda),
        "min_child_weight": float(min_child_weight),
        "gamma": float(gamma),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": int(seed),
    }

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=int(n_estimators),
        evals=[(dva, "val")],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )
    return booster


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser("Per-task USER authentication using STFT features + XGBoost")

    ap.add_argument("--window_len", type=int, default=768)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--ema_alpha", type=float, default=0.001)
    ap.add_argument("--window_norm", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--band_dynamics", action="store_true", help="add per-band time-dynamics stats (recommended).")
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--stft_n_fft", type=int, default=128)
    ap.add_argument("--stft_hop", type=int, default=8)
    ap.add_argument("--stft_keep_bins", type=int, default=64)
    ap.add_argument("--use_log_power", action="store_true",
                    help="use log-power spectrogram for features (recommended).")

    # XGBoost knobs
    ap.add_argument("--xgb_estimators", type=int, default=2000)
    ap.add_argument("--xgb_lr", type=float, default=0.05)
    ap.add_argument("--xgb_depth", type=int, default=6)
    ap.add_argument("--xgb_subsample", type=float, default=0.8)
    ap.add_argument("--xgb_colsample", type=float, default=0.8)
    ap.add_argument("--xgb_lambda", type=float, default=1.0)
    ap.add_argument("--xgb_min_child", type=float, default=1.0)
    ap.add_argument("--xgb_gamma", type=float, default=0.0)
    ap.add_argument("--early_stop", type=int, default=50)

    ap.add_argument("--out_csv", default="bench_user_per_task_stft_feats_xgb.csv")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    all_index = tuple(iter_force_files(DATA_ROOT))
    X, y_user, y_task = windows_from_index(
        all_index,
        window_len=args.window_len,
        stride=args.stride,
        use_ema=args.use_ema,
        ema_alpha=args.ema_alpha
    )

    n_users = len(np.unique(y_user))
    tasks = sorted(np.unique(y_task).tolist())

    results = []
    per_task_acc = {}
    all_te_true, all_te_pred = [], []

    print("\n=== MODEL: xgb(stft-features) ===")

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

        # Featurize STFT "graph" into tabular vectors
        Xtr_f = stft_features(
            Xtr_raw,
            fs=args.fs,
            n_fft=args.stft_n_fft,
            hop=args.stft_hop,
            keep_bins=args.stft_keep_bins,
            use_log_power=args.use_log_power,
            add_band_dynamics=args.band_dynamics,
        )

        Xva_f = stft_features(
            Xva_raw,
            fs=args.fs,
            n_fft=args.stft_n_fft,
            hop=args.stft_hop,
            keep_bins=args.stft_keep_bins,
            use_log_power=args.use_log_power,
            add_band_dynamics=args.band_dynamics,
        )

        Xte_f = stft_features(
            Xte_raw,
            fs=args.fs,
            n_fft=args.stft_n_fft,
            hop=args.stft_hop,
            keep_bins=args.stft_keep_bins,
            use_log_power=args.use_log_power,
            add_band_dynamics=args.band_dynamics,
        )


        model = train_xgb(
            Xtr_f, ytr, Xva_f, yva,
            n_classes=n_users,
            seed=args.seed,
            n_estimators=args.xgb_estimators,
            lr=args.xgb_lr,
            max_depth=args.xgb_depth,
            subsample=args.xgb_subsample,
            colsample=args.xgb_colsample,
            reg_lambda=args.xgb_lambda,
            min_child_weight=args.xgb_min_child,
            gamma=args.xgb_gamma,
            early_stopping_rounds=args.early_stop,
        )

        import xgboost as xgb
        dte = xgb.DMatrix(Xte_f)
        probs = model.predict(dte)          # (N, n_classes)
        yp = probs.argmax(axis=1)
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

    print(f"Overall TEST acc (xgb_stft_feats): {overall:.3f}")

    row = {"model": "xgb_stft_feats", "overall_acc": overall}
    for t in tasks:
        row[f"task{t}_acc"] = per_task_acc.get(t, np.nan)
    results.append(row)

    df = pd.DataFrame(results)
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
