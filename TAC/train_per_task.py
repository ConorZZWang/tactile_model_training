import os
import argparse
import numpy as np
from collections import Counter, defaultdict

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Reuse your split + cached-features builder (with y_task!)
from TAC.train_tabular import build_index_stratified, make_feature_table

def pick_user_model(name: str, n_estimators: int, max_depth: int | None):
    name = name.lower()
    if name in ("rf", "randomforest", "random_forest"):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
        )
    elif name in ("et", "extratrees", "extra_trees"):
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Use 'rf' or 'et'.")

def main():
    parser = argparse.ArgumentParser(description="Per-task user classifier + global attack detector (features).")
    parser.add_argument("--model", default="et", choices=["rf", "et"], help="User classifier type")
    parser.add_argument("--window_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--use_ema", action="store_true", help="EMA smoothing before features")
    parser.add_argument("--attack_ratio", type=float, default=0.10, help="synthetic attack ratio")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # 1) Consistent split
    train_idx, val_idx, test_idx = build_index_stratified(seed=42)

    # 2) Build/load cached features (now includes y_task)
    kw = dict(window_len=args.window_len, stride=args.stride,
              use_ema=args.use_ema, add_attacks=True,
              attack_gen="shuffle", attack_ratio=args.attack_ratio, n_jobs=-1)

    Xtr, ytr_u, ytr_a, ytr_t, _ = make_feature_table(train_idx, **kw)
    Xva, yva_u, yva_a, yva_t, _ = make_feature_table(val_idx,   **kw)
    Xte, yte_u, yte_a, yte_t, _ = make_feature_table(test_idx,  **kw)

    print(f"Feature dim: {Xtr.shape[1]}")
    print("[train] user dist:", Counter(ytr_u))
    print("[val]   user dist:", Counter(yva_u))
    print("[test]  user dist:", Counter(yte_u))
    print("[train] task dist:", Counter(ytr_t))
    print("[val]   task dist:", Counter(yva_t))
    print("[test]  task dist:", Counter(yte_t))

    # 3) Train ONE small user classifier per task (a..g => 0..6)
    user_models = {}
    per_task_val_acc = {}
    per_task_test_acc = {}
    Model = lambda: pick_user_model(args.model, args.n_estimators, args.max_depth)

    for task_id in range(7):
        tr_mask = (ytr_t == task_id)
        va_mask = (yva_t == task_id)
        te_mask = (yte_t == task_id)

        if not tr_mask.any():
            print(f"[warn] no train samples for task {task_id} — skipping.")
            continue

        clf = Model()
        clf.fit(Xtr[tr_mask], ytr_u[tr_mask])
        user_models[task_id] = clf

        if va_mask.any():
            yp_va = clf.predict(Xva[va_mask])
            per_task_val_acc[task_id] = accuracy_score(yva_u[va_mask], yp_va)
        if te_mask.any():
            yp_te = clf.predict(Xte[te_mask])
            per_task_test_acc[task_id] = accuracy_score(yte_u[te_mask], yp_te)

    # Stitch overall predictions (choose model by the known task)
    yp_va_all = np.empty_like(yva_u)
    yp_te_all = np.empty_like(yte_u)
    for t, clf in user_models.items():
        vm = (yva_t == t); tm = (yte_t == t)
        if vm.any(): yp_va_all[vm] = clf.predict(Xva[vm])
        if tm.any(): yp_te_all[tm] = clf.predict(Xte[tm])

    overall_val_acc = accuracy_score(yva_u, yp_va_all)
    overall_test_acc = accuracy_score(yte_u, yp_te_all)
    cm_test = confusion_matrix(yte_u, yp_te_all)

    print("\n=== USER ID (per-task models) ===")
    print("Per-task VAL acc:", {k: round(v, 3) for k, v in per_task_val_acc.items()})
    print("Overall VAL acc :", f"{overall_val_acc:.3f}")
    print("Per-task TEST acc:", {k: round(v, 3) for k, v in per_task_test_acc.items()})
    print("Overall TEST acc :", f"{overall_test_acc:.3f}")
    print("TEST confusion matrix:\n", cm_test)

    # 4) Global attack detector (simple, strong): scale + logistic regression
    #    (You could also make it per-task; global works well in practice.)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    atk = LogisticRegression(max_iter=1000, class_weight="balanced")
    atk.fit(Xtr_s, ytr_a)

    for name, X, y in [("VAL", Xva_s, yva_a), ("TEST", Xte_s, yte_a)]:
        ps = atk.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, ps)
        print(f"[{name}] ATTACK AUC: {auc:.3f}")

if __name__ == "__main__":
    main()
