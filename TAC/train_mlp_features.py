# TAC/train_mlp_features.py
import os
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# we reuse the split + feature builder from your tabular pipeline (and its cache)
from TAC.train_tabular import build_index_stratified, make_feature_table

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_dim: int, n_users: int = 7, hidden=(256,128), drop=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head_user = nn.Linear(d, n_users)
        self.head_attack = nn.Linear(d, 1)

    def forward(self, x):  # x: [B, D]
        h = self.backbone(x)
        return {
            "logits_user": self.head_user(h),             # [B, n_users]
            "logits_attack": self.head_attack(h).squeeze(-1),  # [B]
        }

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_u_true = []; y_u_pred = []
    y_a_true = []; y_a_score = []
    for xb, y_u, y_a in loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        y_u_true.append(y_u.numpy())
        y_u_pred.append(out["logits_user"].argmax(1).cpu().numpy())
        y_a_true.append(y_a.numpy())
        y_a_score.append(torch.sigmoid(out["logits_attack"]).cpu().numpy())

    y_u_true = np.concatenate(y_u_true); y_u_pred = np.concatenate(y_u_pred)
    y_a_true = np.concatenate(y_a_true); y_a_score = np.concatenate(y_a_score)
    acc = accuracy_score(y_u_true, y_u_pred)
    try:
        auc = roc_auc_score(y_a_true, y_a_score)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_u_true, y_u_pred)
    return acc, auc, cm

def main():
    # hyperparams (fast defaults)
    BATCH = 512
    LR = 1e-3
    EPOCHS = 25
    HIDDEN = (256, 128)
    DROP = 0.2

    # same split as tabular; cached features will be reused
    train_idx, val_idx, test_idx = build_index_stratified(seed=42)

    # IMPORTANT: to test the "task conditioning" trick, set add_task_oh=True
    add_task_oh = True

    def add_task_onehot(X, y_task, n_tasks=7):
        oh = np.zeros((y_task.shape[0], n_tasks), dtype=np.float32)
        oh[np.arange(y_task.shape[0]), y_task] = 1.0
        return np.concatenate([X, oh], axis=1)

    # Build features (loads from cache if exists)
    Xtr, ytr_u, ytr_a, ytr_t, _ = make_feature_table(train_idx, window_len=256, stride=512,
                                                     add_attacks=True, attack_gen="shuffle", attack_ratio=0.10, n_jobs=-1)
    Xva, yva_u, yva_a, yva_t, _ = make_feature_table(val_idx,   window_len=256, stride=512,
                                                     add_attacks=True, attack_gen="shuffle", attack_ratio=0.10, n_jobs=-1)
    Xte, yte_u, yte_a, yte_t, _ = make_feature_table(test_idx,  window_len=256, stride=512,
                                                     add_attacks=True, attack_gen="shuffle", attack_ratio=0.10, n_jobs=-1)

    if add_task_oh:
        Xtr = add_task_onehot(Xtr, ytr_t)
        Xva = add_task_onehot(Xva, yva_t)
        Xte = add_task_onehot(Xte, yte_t)

    print("Feature dim:", Xtr.shape[1])
    print("[train] user dist:", Counter(ytr_u))
    print("[val]   user dist:", Counter(yva_u))
    print("[test]  user dist:", Counter(yte_u))

    # Torch datasets
    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr_u), torch.from_numpy(ytr_a))
    va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva_u), torch.from_numpy(yva_a))
    te_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte_u), torch.from_numpy(yte_a))

    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

    # Model
    model = MLP(in_dim=Xtr.shape[1], n_users=7, hidden=HIDDEN, drop=DROP).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_state = None
    for epoch in range(1, EPOCHS+1):
        model.train()
        losses=[]
        for xb, y_u, y_a in tr_dl:
            xb = xb.to(DEVICE); y_u = y_u.to(DEVICE); y_a = y_a.to(DEVICE, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            l_u = ce(out["logits_user"], y_u)
            l_a = bce(out["logits_attack"], y_a)
            loss = l_u + 0.5*l_a
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va_acc, va_auc, _ = evaluate(model, va_dl)
        print(f"epoch {epoch:02d}  train_loss {np.mean(losses):.4f}  val_user_acc {va_acc:.3f}  val_attack_auc {va_auc:.3f}")

        # early stopping on val accuracy
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    te_acc, te_auc, cm = evaluate(model, te_dl)
    print(f"[TEST] user_acc {te_acc:.3f}  attack_auc {te_auc:.3f}")
    print("User confusion matrix:\n", cm)

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    main()
