import os, argparse, numpy as np, pandas as pd
from collections import Counter
from typing import Tuple, Dict
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

from TAC.load_all import iter_force_files, DATA_ROOT

SAMPLE_RATE = 250.0

# ---------------- utils ----------------
def set_seed(seed:int):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ema_1d(x: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    y = np.empty_like(x, dtype=np.float32); v = float(x[0])
    for i in range(len(x)): v = alpha*float(x[i]) + (1-alpha)*v; y[i] = v
    return y

def derivs(F: np.ndarray, rate: float):
    def d1(a): return np.diff(a, axis=0, prepend=a[:1]) * rate
    v = d1(F); a = d1(v); j = d1(a); return v,a,j

def confusion_matrix_np(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def row_normalise_percent(cm):
    cm = cm.astype(np.float32)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return 100.0 * cm / row_sums

def prediction_distribution_percent(y_pred, n_classes):
    """
    For one task: percentage distribution of predicted users.
    Returns shape (n_classes,), summing to 100.
    """
    counts = np.bincount(y_pred, minlength=n_classes).astype(np.float32)
    total = counts.sum()
    if total == 0:
        return np.zeros(n_classes, dtype=np.float32)
    return 100.0 * counts / total

def print_confusion_matrix_percent(cm_pct, title="Confusion Matrix (%)", labels=None):
    print(f"\n{title}")
    n = cm_pct.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]

    header = "true\\pred".ljust(10) + " " + " ".join([f"{lab:>8}" for lab in labels])
    print(header)
    for i in range(n):
        row_str = " ".join([f"{cm_pct[i, j]:8.1f}" for j in range(n)])
        print(f"{labels[i]:>10} {row_str}")

def print_task_summary_matrix(task_user_pct, task_labels=None, user_labels=None):
    """
    task_user_pct: shape (num_tasks, n_users)
    """
    print("\n=== TASK -> PREDICTED USER SUMMARY MATRIX (%) ===")
    if task_labels is None:
        task_labels = [f"task{i}" for i in range(task_user_pct.shape[0])]
    if user_labels is None:
        user_labels = [f"u{i+1}" for i in range(task_user_pct.shape[1])]

    header = "task\\user".ljust(10) + " " + " ".join([f"{u:>8}" for u in user_labels])
    print(header)
    for i, row in enumerate(task_user_pct):
        row_str = " ".join([f"{v:8.1f}" for v in row])
        print(f"{task_labels[i]:>10} {row_str}")

def save_confusion_matrix_plot(cm_pct, task_id, labels=None, save_dir="runs/confusion_matrices"):
    os.makedirs(save_dir, exist_ok=True)

    n = cm_pct.shape[0]
    if labels is None:
        labels = [f"u{i+1}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_title(f"Task {task_id} User Confusion Matrix (%)")
    ax.set_xlabel("Predicted User")
    ax.set_ylabel("True User")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}",
                    ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Percentage")
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"task_{task_id}_user_confusion.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

def save_task_summary_plot(task_user_pct, task_labels=None, user_labels=None,
                           save_path="runs/task_user_summary_matrix.png"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    n_tasks, n_users = task_user_pct.shape
    if task_labels is None:
        task_labels = [f"task{i}" for i in range(n_tasks)]
    if user_labels is None:
        user_labels = [f"u{i+1}" for i in range(n_users)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(task_user_pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_title("Task-to-Predicted-User Summary Matrix (%)")
    ax.set_xlabel("Predicted User")
    ax.set_ylabel("Task")

    ax.set_xticks(np.arange(n_users))
    ax.set_yticks(np.arange(n_tasks))
    ax.set_xticklabels(user_labels)
    ax.set_yticklabels(task_labels)

    for i in range(n_tasks):
        for j in range(n_users):
            ax.text(j, i, f"{task_user_pct[i, j]:.1f}",
                    ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Percentage")
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")

def features_13(Fx, Fy, Fz):
    F = np.stack([Fx, Fy, Fz], axis=1).astype(np.float32)  # (T,3)

    # finite diffs * sample rate
    def d1(a):  # (T,3) -> (T,3)
        return (np.diff(a, axis=0, prepend=a[:1])) * SAMPLE_RATE

    v = d1(F)         # velocity (T,3)
    a = d1(v)         # acceleration (T,3)
    j = d1(a)         # jerk (T,3)

    # IMPORTANT: use axis=1, not ord=1
    v_n  = np.linalg.norm(v, axis=1, keepdims=True)   # (T,1)
    a_n  = np.linalg.norm(a, axis=1, keepdims=True)   # (T,1)
    j_n  = np.linalg.norm(j, axis=1, keepdims=True)   # (T,1)

    dF   = np.diff(F, axis=0, prepend=F[:1])          # (T,3)
    dF_n = np.linalg.norm(dF, axis=1, keepdims=True)  # (T,1)

    # concatenate to (T,13): vx,vy,vz,|v|, ax,ay,az,|a|, jx,jy,jz,|j|, |ΔF|
    return np.concatenate([v, v_n, a, a_n, j, j_n, dF_n], axis=1).astype(np.float32)

def resample_to_len(arr: np.ndarray, L_out: int) -> np.ndarray:
    T,D = arr.shape
    if T == L_out: return arr
    x_in  = np.linspace(0.0,1.0,T, dtype=np.float32)
    x_out = np.linspace(0.0,1.0,L_out, dtype=np.float32)
    return np.stack([np.interp(x_out, x_in, arr[:,d]) for d in range(D)],1).astype(np.float32)

def zwin_time(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True); sd = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mu) / sd

# ------------- data: many crops per CSV -------------
def load_cropped_sequences(seq_len:int, two_stream:bool, window_norm:bool,
                           slices_per_csv:int, slice_frac:float, seed:int):
    """
    Build MANY samples per CSV by cropping the original time series.
    For each CSV:
      - choose crop length = max(512, int(slice_frac*T))
      - place 'slices_per_csv' starts evenly (with overlap allowed)
      - for each crop: build RAW 13 (+ EMA 13 if two_stream), resample to seq_len, z-norm (optional)
    Returns X:(N,C,L), yu:(N,), yt:(N,)
    """
    Xs, y_users, y_tasks = [], [], []
    user_map: Dict[int,int] = {}; task_map: Dict[int,int] = {}
    rng = np.random.default_rng(seed)

    for user_id, task_id, csv_path in iter_force_files(DATA_ROOT):
        if user_id not in user_map: user_map[user_id] = len(user_map)
        if task_id not in task_map: task_map[task_id] = len(task_map)
        u = user_map[user_id]; t = task_map[task_id]

        df = pd.read_csv(csv_path)
        for c in ("force_x","force_y","force_z"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["force_x","force_y","force_z"])
        Fx = df["force_x"].values.astype(np.float32)
        Fy = df["force_y"].values.astype(np.float32)
        Fz = df["force_z"].values.astype(np.float32)
        T = len(Fx)
        if T < 16: continue

        # crop plan
        Lcrop = max(512, int(max(0.1, min(1.0, slice_frac)) * T))
        if Lcrop > T: Lcrop = T
        if slices_per_csv <= 1:
            starts = [0]
        else:
            if T == Lcrop:
                starts = [0]*slices_per_csv   # identical crops if sequence is tiny
            else:
                # evenly spaced starts to get exactly slices_per_csv crops
                max_start = T - Lcrop
                starts = np.linspace(0, max_start, num=slices_per_csv).astype(int).tolist()

        # precompute EMA forces if needed
        if two_stream:
            FxE, FyE, FzE = ema_1d(Fx), ema_1d(Fy), ema_1d(Fz)

        for s in starts:
            e = min(T, s + Lcrop)
            Fx_c, Fy_c, Fz_c = Fx[s:e], Fy[s:e], Fz[s:e]

            # stream A
            feat_raw = features_13(Fx_c, Fy_c, Fz_c)
            # stream B (EMA)
            if two_stream:
                feat_ema = features_13(FxE[s:e], FyE[s:e], FzE[s:e])
                A = resample_to_len(feat_raw, seq_len)
                B = resample_to_len(feat_ema, seq_len)
                feat = np.concatenate([A,B], axis=1)  # (L,26)
            else:
                feat = resample_to_len(feat_raw, seq_len)  # (L,13)

            sample = feat.T  # (C,L)
            if window_norm: sample = zwin_time(sample)
            Xs.append(sample); y_users.append(u); y_tasks.append(t)

    if not Xs:
        raise RuntimeError("No sequences created. Try smaller --slice_frac or --slices_per_csv.")

    X  = np.stack(Xs,0).astype(np.float32)
    yu = np.array(y_users, dtype=np.int64)
    yt = np.array(y_tasks, dtype=np.int64)

    print(f"Sequences: {X.shape} | users: {Counter(yu)} | tasks: {Counter(yt)}")
    return X, yu, yt

def sample_per_user_task_indices(y_user, y_task, task_id, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    idx_t = np.where(y_task == task_id)[0]
    users = np.unique(y_user[idx_t])
    tr_idx, te_idx = [], []
    for u in users:
        iu = idx_t[y_user[idx_t] == u]
        if len(iu) == 0: continue
        iu = rng.permutation(iu)
        k  = min(len(iu), n_train + n_test)
        if k == 0: continue
        cut = min(n_train, k)
        tr_u, te_u = iu[:cut], iu[cut:k]
        if len(tr_u)==0 or len(te_u)==0:
            if k >= 2: tr_u, te_u = iu[:k-1], iu[k-1:k]
            else: continue
        tr_idx.append(tr_u); te_idx.append(te_u)
    if not tr_idx or not te_idx: return np.array([],int), np.array([],int)
    return np.concatenate(tr_idx), np.concatenate(te_idx)

# ------------- model -------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):  # (B,L,D)
        return x + self.pe[:, :x.size(1), :]

class PaperTransformer(nn.Module):
    def __init__(self, in_channels=26, d_model=256, nhead=16, num_layers=2, dim_ff=256,
                 dropout=0.1, n_classes=7):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, d_model, 1)
        self.pre_mlp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(d_model, d_model)
        )
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(d_model, n_classes)
        )
    def forward(self, x):  # (B,C,L)
        z = self.in_proj(x).transpose(1,2)  # (B,L,D)
        z = self.pre_mlp(z); z = self.pos(z); z = self.encoder(z)
        z = z.mean(1)
        return self.head(z)

# ------------- train/eval -------------
def make_loader(X,y,batch,shuffle):
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                        torch.tensor(y, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)

def train_one_task(Xtr,ytr,Xva,yva,Xte,yte,n_users,epochs,batch,lr,grad_clip,
                   d_model,nhead,num_layers,dim_ff,dropout,seed,device=None):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperTransformer(Xtr.shape[1], d_model, nhead, num_layers, dim_ff, dropout, n_users).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,epochs))
    crit = nn.CrossEntropyLoss()

    if Xva is None or len(Xva)==0:
        n = len(Xtr)
        nva = max(1, int(0.1*n))
        perm = np.random.permutation(n)
        va, tr = perm[:nva], perm[nva:]
        Xva, yva = Xtr[va], ytr[va]
        Xtr, ytr = Xtr[tr], ytr[tr]

    dl_tr = make_loader(Xtr, ytr, batch, True)
    dl_va = make_loader(Xva, yva, batch, False)

    best, best_va = None, -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        model.eval()
        corr = 0
        n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                corr += int((pred == yb).sum())
                n += len(yb)

        va_acc = corr / max(1, n)
        sched.step()

        if va_acc > best_va:
            best_va = va_acc
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best is not None:
        model.load_state_dict(best)

    dl_te = make_loader(Xte, yte, batch, False)
    model.eval()
    corr = 0
    n = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in dl_te:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            corr += int((pred == yb).sum())
            n += len(yb)
            all_true.append(yb.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    acc = corr / max(1, n)

    return acc, y_true, y_pred

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser("Paper-style user ID with per-CSV crops (two-stream features)")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--per_user_train", type=int, default=100)
    ap.add_argument("--per_user_test", type=int, default=20)
    ap.add_argument("--slices_per_csv", type=int, default=120,
                    help="how many crops to extract per CSV (>= per_user_train+per_user_test)")
    ap.add_argument("--slice_frac", type=float, default=0.6,
                    help="crop length as a fraction of the CSV length (0.1..1.0)")
    ap.add_argument("--window_norm", action="store_true",
                    help="per-sequence z-norm over time (optional)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dim_ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")

    X, yu, yt = load_cropped_sequences(
        seq_len=args.seq_len, two_stream=True, window_norm=args.window_norm,
        slices_per_csv=max(args.slices_per_csv, args.per_user_train + args.per_user_test),
        slice_frac=args.slice_frac, seed=args.seed
    )

    n_users = len(np.unique(yu))
    tasks = sorted(np.unique(yt).tolist())
    print(f"[INFO] seq_len={args.seq_len}, users={n_users}, tasks={tasks}")

    user_labels = [f"u{i+1}" for i in range(n_users)]
    task_labels = [f"task{t}" for t in tasks]

    results = {}
    task_user_summary_rows = []

    for t in tasks:
        tr_idx, te_idx = sample_per_user_task_indices(
            y_user=yu, y_task=yt, task_id=t,
            n_train=args.per_user_train, n_test=args.per_user_test, seed=args.seed
        )
        print(f"[task {t}] train {len(tr_idx)}  test {len(te_idx)}")

        if len(tr_idx) == 0 or len(te_idx) == 0:
            print(f"[WARN] task {t} empty split.")
            results[t] = np.nan
            task_user_summary_rows.append(np.full(n_users, np.nan, dtype=np.float32))
            continue

        ntr = len(tr_idx)
        nva = max(1, int(0.1 * ntr))
        perm = np.random.permutation(tr_idx)
        va_idx = perm[:nva]
        tr_idx2 = perm[nva:]

        Xtr, ytr = X[tr_idx2], yu[tr_idx2]
        Xva, yva = X[va_idx],  yu[va_idx]
        Xte, yte = X[te_idx],  yu[te_idx]

        acc, y_true, y_pred = train_one_task(
            Xtr, ytr, Xva, yva, Xte, yte, n_users,
            epochs=args.epochs, batch=args.batch, lr=args.lr, grad_clip=args.grad_clip,
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            dim_ff=args.dim_ff, dropout=args.dropout, seed=args.seed
        )

        results[t] = float(acc)
        print(f"[task {t}] TEST acc {acc:.3f}")

        # Proper user confusion matrix for this task
        cm = confusion_matrix_np(y_true, y_pred, n_users)
        cm_pct = row_normalise_percent(cm)
        print_confusion_matrix_percent(cm_pct, title=f"[task {t}] User Confusion Matrix (%)", labels=user_labels)
        save_confusion_matrix_plot(cm_pct, task_id=t, labels=user_labels)

        # Your requested 1x7 row: distribution of predicted users for this task
        pred_dist = prediction_distribution_percent(y_pred, n_users)
        task_user_summary_rows.append(pred_dist)

        print(f"\n[task {t}] Predicted-user distribution (%)")
        row_txt = " ".join([f"{lab}:{pred_dist[i]:.1f}" for i, lab in enumerate(user_labels)])
        print(row_txt)

    vals = [v for v in results.values() if not np.isnan(v)]
    overall = float(np.mean(vals)) if vals else float("nan")

    print("\n=== SUMMARY (paper Transformer, per-CSV CROPS, two-stream FEATURES) ===")
    for t in tasks:
        print(f"task {t}: {results.get(t, np.nan):.3f}")
    print(f"OVERALL mean acc: {overall:.3f}")

    # Final 7x7 task-by-user summary matrix
    task_user_summary = np.stack(task_user_summary_rows, axis=0)
    print_task_summary_matrix(task_user_summary, task_labels=task_labels, user_labels=user_labels)
    save_task_summary_plot(task_user_summary, task_labels=task_labels, user_labels=user_labels,
                           save_path="runs/task_user_summary_matrix.png")

if __name__ == "__main__":
    main()
