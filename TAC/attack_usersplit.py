import os
import argparse
import numpy as np
from collections import Counter, defaultdict

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix
)

from TAC.train_tabular import make_feature_table
from TAC.load_all import iter_force_files, DATA_ROOT


def pick_model(name: str, seed: int, n_estimators: int = 400, max_depth=None):
    name = name.lower()
    if name in ("et", "extratrees", "extra_trees"):
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced"
        )
    if name in ("rf", "randomforest", "random_forest"):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample"
        )
    if name in ("lr", "logreg", "logistic"):
        # LR needs scaling; wrap in pipeline
        return make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                max_iter=2000,
                n_jobs=-1,
                random_state=seed,
                class_weight="balanced"
            )
        )
    raise ValueError("model must be one of: et, rf, lr")


def tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # find last index where fpr <= target; if none, take smallest
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0, thr[-1]
    i = idx[-1]
    return float(tpr[i]), float(thr[i])


def main():
    parser = argparse.ArgumentParser(
        "Attack detection where u1..u5 are genuine and u6..u7 are attackers"
    )
    parser.add_argument("--window_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="et", choices=["et","rf","lr"])
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=None)
    # split ratios on the whole pool (we’ll shuffle globally)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # 1) Build features for ALL files (no synthetic attacks here)
    all_index = tuple(iter_force_files(DATA_ROOT))
    X, y_user, _, y_task, _ = make_feature_table(
        all_index,
        window_len=args.window_len,
        stride=args.stride,
        use_ema=args.use_ema,
        add_attacks=False,   # <<< important
        n_jobs=-1
    )
    print("X:", X.shape, "| user dist:", Counter(y_user), "| task dist:", Counter(y_task))

    # NOTE: load_all encodes users as 0..6 matching sorted(u1..u7)
    # We’ll assume: 0→u1, 1→u2, ..., 6→u7
    genuine_users = {0,1,2,5,6}   # u1..u5
    attacker_users = {3,4}        # u6..u7

    y_attack = np.array([1 if u in attacker_users else 0 for u in y_user], dtype=int)
    print("attack dist (0=genuine,1=attacker):", Counter(y_attack))

    # 2) Global shuffle split (keeps it simple & fast)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n = len(idx)
    ntr = int(args.train_ratio * n)
    nva = int(args.val_ratio * n)
    tr, va, te = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]

    # 3) Train binary classifier
    clf = pick_model(args.model, args.seed, args.n_estimators, args.max_depth)
    clf.fit(X[tr], y_attack[tr])

    # 4) Scores
    def score_split(name, ids, choose_thresh=False, ref_threshold=None):
        # predict_proba for ET/RF/LR; pipeline .predict_proba works too
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X[ids])[:,1]
        else:
            # fallback - decision_function if available
            if hasattr(clf, "decision_function"):
                z = clf.decision_function(X[ids])
                # map to 0..1 via min-max; not strictly calibrated but OK for ROC
                zmin, zmax = z.min(), z.max()
                p = (z - zmin) / max(1e-9, (zmax - zmin))
            else:
                # last resort: raw predictions (bad for AUC)
                p = clf.predict(X[ids]).astype(float)

        y = y_attack[ids]
        auc = roc_auc_score(y, p)
        ap  = average_precision_score(y, p)

        # thresholds
        tpr1, thr1 = tpr_at_fpr(y, p, target_fpr=0.01)
        tpr01, thr01 = tpr_at_fpr(y, p, target_fpr=0.001)

        # if we need a single operating threshold from VAL, pass that in ref_threshold
        if ref_threshold is not None:
            thr = ref_threshold
        elif choose_thresh:
            thr = thr1   # choose VAL threshold at 1% FPR (common choice)
        else:
            thr = 0.5

        yhat = (p >= thr).astype(int)
        cm = confusion_matrix(y, yhat)

        print(f"[{name}] AUC {auc:.3f} | PR-AUC {ap:.3f} | TPR@1%FPR {tpr1:.3f} | TPR@0.1%FPR {tpr01:.3f}")
        print(f"[{name}] threshold used: {thr:.6f}")
        print(f"[{name}] Confusion matrix @thr:\n{cm}")
        return auc, ap, thr, p, y

    print("\n--- VALIDATION ---")
    auc_va, ap_va, thr_va, p_va, y_va = score_split("VAL", va, choose_thresh=True)

    print("\n--- TEST (using VAL threshold @ ~1% FPR) ---")
    auc_te, ap_te, _, p_te, y_te = score_split("TEST", te, ref_threshold=thr_va)

    # 5) Per-task AUCs (on TEST) for insight
    print("\nPer-task AUCs on TEST:")
    per_task = defaultdict(list)
    for i in te:
        per_task[y_task[i]].append(i)
    for t, ids in per_task.items():
        ids = np.array(ids, dtype=int)
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X[ids])[:,1]
        else:
            if hasattr(clf, "decision_function"):
                z = clf.decision_function(X[ids])
                zmin, zmax = z.min(), z.max()
                p = (z - zmin) / max(1e-9, (zmax - zmin))
            else:
                p = clf.predict(X[ids]).astype(float)
        y = y_attack[ids]
        auc_t = roc_auc_score(y, p)
        print(f"  task {t}: AUC {auc_t:.3f}")


if __name__ == "__main__":
    main()
