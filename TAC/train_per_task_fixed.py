import os
import argparse
import numpy as np
from collections import defaultdict, Counter

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Reuse your feature builder but call it ONCE for all files
from TAC.train_tabular import make_feature_table
from TAC.load_all import iter_force_files, DATA_ROOT

def pick_model(name: str, n_estimators: int, max_depth: int | None):
    name = name.lower()
    if name in ("et", "extratrees", "extra_trees"):
        return ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            n_jobs=-1, random_state=42, class_weight="balanced"
        )
    elif name in ("rf", "randomforest", "random_forest"):
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            n_jobs=-1, random_state=42, class_weight="balanced_subsample"
        )
    else:
        raise ValueError("model must be 'et' or 'rf'")

def split_within_user(idx, users, seed=42, ratios=(0.6, 0.2, 0.2)):
    """
    Split indices so that *each user* contributes to train/val/test.
    idx: np.array of row indices into the task subset
    users: np.array of user labels aligned with idx
    returns three np.array index masks (relative to idx) for train/val/test
    """
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for u in np.unique(users):
        u_mask = (users == u)
        u_idx = np.where(u_mask)[0]
        rng.shuffle(u_idx)
        n = len(u_idx)
        n_tr = int(ratios[0] * n)
        n_va = int(ratios[1] * n)
        # ensure at least 1 goes to each split if possible
        n_tr = max(1, n_tr) if n >= 3 else max(1, n-1)
        n_va = max(1, n_va) if n >= 3 else 0
        n_te = max(1, n - n_tr - n_va)
        # adjust if sums exceed
        while n_tr + n_va + n_te > n:
            if n_te > 1: n_te -= 1
            elif n_va > 1: n_va -= 1
            else: n_tr -= 1
        tr_idx = u_idx[:n_tr]
        va_idx = u_idx[n_tr:n_tr+n_va]
        te_idx = u_idx[n_tr+n_va:n_tr+n_va+n_te]
        tr.extend(tr_idx); va.extend(va_idx); te.extend(te_idx)
    return np.array(tr), np.array(va), np.array(te)

def main():
    parser = argparse.ArgumentParser(description="Per-task user ID with within-user splits (features).")
    parser.add_argument("--model", default="et", choices=["et","rf"])
    parser.add_argument("--window_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--attack_ratio", type=float, default=0.10)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")

    # 1) Build features ONCE for "all files". We do that by passing a list of all (user,task,csv) triplets
    #    to the existing dataset path through its index_split form. Easiest way: collect all file paths first.
    all_index = []
    for user_id, task_id, csv_path in iter_force_files(DATA_ROOT):
        # index item is (user_id, task_id, csv_path) — exactly what your ForceWindowDataset expects
        all_index.append((user_id, task_id, csv_path))
    all_index = tuple(all_index)

    X, y_user, y_attack, y_task, _ = make_feature_table(
        all_index,
        window_len=args.window_len, stride=args.stride,
        use_ema=args.use_ema,
        add_attacks=True, attack_gen="shuffle", attack_ratio=args.attack_ratio,
        n_jobs=-1
    )
    print("All features:", X.shape)
    print("User dist:", Counter(y_user))
    print("Task dist:", Counter(y_task))

    # 2) For each task, select subset and do within-user split
    per_task_val_acc = {}
    per_task_test_acc = {}
    preds_test = np.full_like(y_user, fill_value=-1)  # store stitched TEST preds in global space (we'll fill only task slices)

    for t in range(7):
        task_mask = (y_task == t)
        if not task_mask.any():
            print(f"[task {t}] no samples, skipping.")
            continue

        X_t = X[task_mask]
        y_u_t = y_user[task_mask]

        # split within each user
        idx_all = np.arange(len(X_t))
        tr_rel, va_rel, te_rel = split_within_user(idx_all, y_u_t, seed=args.seed)

        Xtr, ytr = X_t[tr_rel], y_u_t[tr_rel]
        Xva, yva = X_t[va_rel], y_u_t[va_rel]
        Xte, yte = X_t[te_rel], y_u_t[te_rel]

        # train model for this task
        clf = pick_model(args.model, args.n_estimators, args.max_depth)
        clf.fit(Xtr, ytr)

        # per-task metrics
        if len(Xva) > 0:
            yp_va = clf.predict(Xva)
            per_task_val_acc[t] = accuracy_score(yva, yp_va)
        if len(Xte) > 0:
            yp_te = clf.predict(Xte)
            per_task_test_acc[t] = accuracy_score(yte, yp_te)

        # stitch test preds back into the global array (only for this task's test rows)
        global_task_indices = np.where(task_mask)[0]
        global_te_idx = global_task_indices[te_rel]
        preds_test[global_te_idx] = yp_te if len(Xte) > 0 else preds_test[global_te_idx]

        print(f"[task {t}] train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}  val_acc {per_task_val_acc.get(t, np.nan):.3f}  test_acc {per_task_test_acc.get(t, np.nan):.3f}")

    # 3) Overall user accuracy on TEST (only over rows we filled)
    filled = (preds_test != -1)
    overall_test_acc = accuracy_score(y_user[filled], preds_test[filled])
    cm_test = confusion_matrix(y_user[filled], preds_test[filled])

    print("\n=== USER ID (per-task, within-user splits) ===")
    print("Per-task VAL acc:", {k: round(v,3) for k,v in per_task_val_acc.items()})
    print("Per-task TEST acc:", {k: round(v,3) for k,v in per_task_test_acc.items()})
    print("Overall TEST acc :", f"{overall_test_acc:.3f}")
    print("TEST confusion matrix:\n", cm_test)

    # 4) Optional: Global attack detector (just to report)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    # crude split (80/20) just to compute a robust AUC; you already have ~0.99 earlier
    n = len(X)
    ntr = int(0.8*n)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    tr, te = perm[:ntr], perm[ntr:]
    atk = ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced")
    atk.fit(Xs[tr], y_attack[tr])
    ps = atk.predict_proba(Xs[te])[:,1]
    auc = roc_auc_score(y_attack[te], ps)
    print(f"[ATTACK] AUC (80/20 split): {auc:.3f}")

if __name__ == "__main__":
    main()
