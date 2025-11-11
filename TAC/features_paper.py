from pathlib import Path
import numpy as np
import pandas as pd

SAMPLE_RATE = 250.0  # Hz

def ema(x, alpha=0.001):
    y = np.empty_like(x, dtype=np.float32)
    acc = x[0]
    a = float(alpha)
    one_ma = 1.0 - a
    for i, xi in enumerate(x):
        acc = a * xi + one_ma * acc
        y[i] = acc
    return y

def build_paper_features(df: pd.DataFrame, use_ema: bool) -> np.ndarray:
    """Return array of shape (T, D) with paper features."""
    fx = df["force_x"].astype(np.float32).to_numpy()
    fy = df["force_y"].astype(np.float32).to_numpy()
    fz = df["force_z"].astype(np.float32).to_numpy()

    if use_ema:
        fx, fy, fz = ema(fx), ema(fy), ema(fz)

    F = np.stack([fx, fy, fz], axis=1)  # (T,3)

    # finite differences
    dF  = np.diff(F, axis=0)                               # (T-1,3)
    v   = dF * SAMPLE_RATE                                 # velocity
    a   = np.diff(v, axis=0) * SAMPLE_RATE                 # acceleration (T-2,3)
    j   = np.diff(a, axis=0) * SAMPLE_RATE                 # jerk (T-3,3)

    # align all to the same length (T-3)
    F3  = F[3: ]                     # (T-3,3) raw forces
    dFn = np.linalg.norm(dF[2: ], axis=1, keepdims=True)   # (T-3,1)
    vn  = np.linalg.norm(v [2: ], axis=1, keepdims=True)   # (T-3,1)
    an  = np.linalg.norm(a [1: ], axis=1, keepdims=True)   # (T-3,1)
    jn  = np.linalg.norm(j      , axis=1, keepdims=True)   # (T-3,1)

    feat = np.concatenate([F3, dFn, v[2:], vn, a[1:], an, j, jn], axis=1).astype(np.float32)
    # columns = [Fx,Fy,Fz, |ΔF|, Vx,Vy,Vz, |V|, Ax,Ay,Az, |A|, Jx,Jy,Jz, |J|] => 3+1+3+1+3+1+3+1 = 16 dims
    return feat  # (T-3, 16)

def load_force_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("force_x","force_y","force_z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["force_x","force_y","force_z"]).reset_index(drop=True)
    return df
