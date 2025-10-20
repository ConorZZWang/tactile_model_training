# TAC/features.py
import numpy as np

def _percentiles(x, ps=(10,25,50,75,90)):
    return np.percentile(x, ps)

def _iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

def _zcr(x):
    # zero-crossing rate of detrended signal
    x = x - x.mean()
    return ((x[:-1] * x[1:]) < 0).mean() if len(x) > 1 else 0.0

def _lin_trend(x):
    # slope from simple linear fit
    n = len(x)
    if n < 2: return 0.0
    t = np.arange(n, dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return float(np.dot(t, x) / (n - 1))

def _autocorr(x, lag):
    if len(x) <= lag: return 0.0
    x = x - x.mean()
    denom = (x * x).sum() + 1e-8
    return float(np.dot(x[:-lag], x[lag:]) / denom)

def _rms(x):
    return float(np.sqrt((x * x).mean()))

def _energy(x):
    return float((x * x).sum())

def _fft_features(x, fs=250.0):
    """
    Compute magnitude spectrum features: centroid, spread, band powers.
    """
    n = len(x)
    if n < 4:
        return dict(spec_centroid=0.0, spec_spread=0.0, **{f"bp_{i}":0.0 for i in range(5)})
    xz = x - x.mean()
    # real FFT
    X = np.fft.rfft(xz, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mag = np.abs(X)
    mag_sum = mag.sum() + 1e-8
    centroid = float((freqs * mag).sum() / mag_sum)
    spread = float(np.sqrt(((freqs - centroid)**2 * mag).sum() / mag_sum))

    # simple bands (Hz): [0-2), [2-5), [5-10), [10-20), [20-50)
    bands = [(0,2),(2,5),(5,10),(10,20),(20,50)]
    bp = {}
    for i,(f0,f1) in enumerate(bands):
        m = (freqs >= f0) & (freqs < f1)
        bp[f"bp_{i}"] = float(mag[m].sum() / mag_sum)
    return dict(spec_centroid=centroid, spec_spread=spread, **bp)

def _derivatives(F, fs=250.0):
    # F: [W,3]
    def diff(a):
        d = np.diff(a, axis=0, prepend=a[:1])
        return d * fs
    d1 = diff(F)
    d2 = diff(d1)
    d3 = diff(d2)
    mag1 = np.linalg.norm(d1, axis=1)
    mag2 = np.linalg.norm(d2, axis=1)
    mag3 = np.linalg.norm(d3, axis=1)
    return d1, d2, d3, mag1, mag2, mag3

def window_to_features(x_win, fs=250.0):
    """
    x_win: [W,3] array with columns [Fx, Fy, Fz] (raw window).
    Returns: 1D numpy array of engineered features (float32).
    """
    W = x_win.shape[0]
    Fx, Fy, Fz = x_win[:,0], x_win[:,1], x_win[:,2]
    mag = np.linalg.norm(x_win, axis=1)  # resultant |F|

    # Base signals to featurize
    series = {
        "Fx": Fx, "Fy": Fy, "Fz": Fz, "Fmag": mag,
    }

    # Derivative magnitudes (summary only — not full waveform)
    d1, d2, d3, mag1, mag2, mag3 = _derivatives(x_win, fs)
    series_short = {
        "Vmag": mag1, "Amag": mag2, "Jmag": mag3,
    }

    feats = []
    names = []

    # Per-signal time features
    for name, s in {**series, **series_short}.items():
        s = s.astype(np.float32)
        mu = float(s.mean()); sd = float(s.std() + 1e-8)
        p10, p25, p50, p75, p90 = _percentiles(s)
        iqr = _iqr(s)
        feats.extend([mu, sd, float(s.min()), float(s.max()), p10, p25, p50, p75, p90, iqr, _rms(s), _zcr(s), _lin_trend(s)])
        names.extend([f"{name}_mean", f"{name}_std", f"{name}_min", f"{name}_max",
                      f"{name}_p10", f"{name}_p25", f"{name}_p50", f"{name}_p75", f"{name}_p90",
                      f"{name}_iqr", f"{name}_rms", f"{name}_zcr", f"{name}_trend"])
        # autocorr lags
        for lag in (1,2,4):
            feats.append(_autocorr(s, lag))
            names.append(f"{name}_ac{lag}")
        # energy
        feats.append(_energy(s))
        names.append(f"{name}_energy")
        # spectrum (only for main series, not for V/A/J)
        if name in ("Fx","Fy","Fz","Fmag"):
            sp = _fft_features(s, fs=fs)
            feats.extend([sp["spec_centroid"], sp["spec_spread"],
                          sp["bp_0"], sp["bp_1"], sp["bp_2"], sp["bp_3"], sp["bp_4"]])
            names.extend([f"{name}_spec_centroid", f"{name}_spec_spread",
                          f"{name}_bp0", f"{name}_bp1", f"{name}_bp2", f"{name}_bp3", f"{name}_bp4"])

    # Cross-axis correlations
    for (a, A), (b, B) in [(("Fx",Fx),("Fy",Fy)), (("Fy",Fy),("Fz",Fz)), (("Fz",Fz),("Fx",Fx))]:
        A = A - A.mean(); B = B - B.mean()
        denom = (np.sqrt((A*A).sum()) * np.sqrt((B*B).sum()) + 1e-8)
        corr = float(np.dot(A,B) / denom)
        feats.append(corr); names.append(f"corr_{a}_{b}")

    return np.asarray(feats, dtype=np.float32), names
