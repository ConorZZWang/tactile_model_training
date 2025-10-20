# TAC/train_tabular.py
import os
import hashlib
import numpy as np
from collections import Counter
from pathlib import Path
from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from TAC.load_all import iter_force_files, DATA_ROOT
from TAC.datasets import ForceWindowDataset
from TAC.features import window_to_features

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def build_index_stratified(seed: int = 42):
    import random
    by_user = {}
    for (u, t, p) in iter_force_files(DATA_ROOT):
        by_user.setdefault(u, []).append((u, t, p))
    train, val, test = [], [], []
    rng = random.Random(seed)
    for u, files in by_user.items():
        rng.shuffle(files)
        n = len(files)
        if n == 1:
            train.extend(files); print(f"[warn] Only 1 file for {u}; train only."); continue
        if n == 2:
            train.append(files[0]); test.append(files[1]); continue
        n_tr = max(1, int(0.70*n)); n_va = max(1, int(0.15*n)); n_te = max(1, n - n_tr - n_va)
        while n_tr + n_va + n_te > n: n_tr -= 1
        s_tr=0; s_va=s_tr+n_tr; s_te=s_va+n_va
        train.extend(files[s_tr:s_va]); val.extend(files[s_va:s_te]); test.extend(files[s_te:])
    if not train or not val or not test:
        raise RuntimeError("Stratified split failed.")
    return train, val, test

def _featurize_window(x_cw):
    # x_cw: [C,W] -> [W,C]
    return window_to_features(x_cw.T)[0]  # only feats, not names

def _cache_key(index_split, window_len, stride, use_ema, add_attacks, attack_gen, attack_ratio):
    h = hashlib.md5()
    parts = [str(window_len), str(stride), str(use_ema), str(add_attacks), str(attack_gen), str(attack_ratio)]
    for (u, t, p) in index_split:
        parts.append(f"{u}/{t}/{p}")
    h.update("|".join(parts).encode())
    return h.hexdigest()

def make_feature_table(index_split, window_len=256, stride=512, use_ema=False,
                       add_attacks=True, attack_gen="shuffle", attack_ratio=0.10,
                       max_files=None, n_jobs=-1):
    """
    Build features for a split, with caching + parallel featurization.
    RETURNS: X, y_user, y_attack, y_task, names
    Cache format: X, y_user, y_attack, y_task
    """
    key = _cache_key(index_split, window_len, stride, use_ema, add_attacks, attack_gen, attack_ratio)
    cache_path = CACHE_DIR / f"feat_{key}.npz"

    # ---- try to load cache (supports old cache that might lack y_task) ----
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        have_task = ("y_task" in data.files)
        if have_task:
            X = data["X"]; y_user = data["y_user"]; y_attack = data["y_attack"]; y_task = data["y_task"]
            print(f"[cache] loaded {cache_path.name}: X={X.shape}, y_task present")
            return X, y_user, y_attack, y_task, None
        else:
            print(f"[cache] {cache_path.name} missing y_task → rebuilding this split...")

    # ---- build dataset and collect windows ----
    ds = ForceWindowDataset(
        index_split,
        window_len=window_len, stride=stride,
        use_ema=use_ema, add_derivatives=False,  # we engineer features ourselves
        attack_gen=(attack_gen if add_attacks else None),
        attack_ratio=attack_ratio,
        max_files=max_files
    )

    # Collect tensors first (faster to parallelize)
    X_list = []
    y_user_list, y_attack_list, y_task_list = [], [], []
    for i in range(len(ds)):
        x, yu, yt, ya = ds[i]   # x: [C,W], yu: user, yt: task, ya: attack
        X_list.append(x.numpy())
        y_user_list.append(int(yu.item()))
        y_attack_list.append(int(ya.item()))
        y_task_list.append(int(yt.item()))

    # ---- parallel featurization ----
    feats = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_featurize_window)(xcw) for xcw in X_list
    )
    X = np.stack(feats).astype(np.float32)
    y_user  = np.asarray(y_user_list,  dtype=np.int64)
    y_attack= np.asarray(y_attack_list, dtype=np.int64)
    y_task  = np.asarray(y_task_list,  dtype=np.int64)

    np.savez_compressed(cache_path, X=X, y_user=y_user, y_attack=y_attack, y_task=y_task)
    print(f"[cache] saved {cache_path.name}: X={X.shape}, y_task saved")
    return X, y_user, y_attack, y_task, None

def main():
    # Speed preset — tweak here for faster runs
    WINDOW_LEN = 256
    STRIDE     = 512
    ATTACK_RATIO = 0.10
    N_JOBS = -1  # use all cores

    train_idx, val_idx, test_idx = build_index_stratified(seed=42)

    Xtr, ytr_u, ytr_a, _ = make_feature_table(train_idx, window_len=WINDOW_LEN, stride=STRIDE,
                                              add_attacks=True, attack_gen="shuffle", attack_ratio=ATTACK_RATIO,
                                              n_jobs=N_JOBS)
    Xva, yva_u, yva_a, _ = make_feature_table(val_idx,   window_len=WINDOW_LEN, stride=STRIDE,
                                              add_attacks=True, attack_gen="shuffle", attack_ratio=ATTACK_RATIO,
                                              n_jobs=N_JOBS)
    Xte, yte_u, yte_a, _ = make_feature_table(test_idx,  window_len=WINDOW_LEN, stride=STRIDE,
                                              add_attacks=True, attack_gen="shuffle", attack_ratio=ATTACK_RATIO,
                                              n_jobs=N_JOBS)

    print("Feature dim:", Xtr.shape[1])
    print("[train] user dist:", Counter(ytr_u))
    print("[val]   user dist:", Counter(yva_u))
    print("[test]  user dist:", Counter(yte_u))

    clf_user = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced_subsample"
    )
    clf_attack = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    clf_user.fit(Xtr, ytr_u)
    clf_attack.fit(Xtr, ytr_a)

    for split_name, X, y in [("val", Xva, yva_u), ("test", Xte, yte_u)]:
        yp = clf_user.predict(X)
        acc = accuracy_score(y, yp)
        cm = confusion_matrix(y, yp)
        print(f"[{split_name}] USER acc: {acc:.3f}")
        print(f"[{split_name}] USER confusion:\n{cm}")

    for split_name, X, y in [("val", Xva, yva_a), ("test", Xte, yte_a)]:
        ps = clf_attack.predict_proba(X)[:,1]
        auc = roc_auc_score(y, ps)
        print(f"[{split_name}] ATTACK AUC: {auc:.3f}")

if __name__ == "__main__":
    # Optional: limit BLAS threads for fairness/speed predictability
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
