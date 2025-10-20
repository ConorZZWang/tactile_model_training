# TAC/train_cnn.py
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from TAC.datasets import build_index, ForceWindowDataset

# ----- Model (tiny, fast CNN) -----
class CNN1D(nn.Module):
    def __init__(self, in_ch: int, n_users: int = 7, n_tasks: int = 7):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> [B, 128, 1]
        )
        self.head_user   = nn.Linear(128, n_users)
        self.head_task   = nn.Linear(128, n_tasks)
        self.head_attack = nn.Linear(128, 1)

    def forward(self, x):  # x: [B, C, W]
        h = self.fe(x).squeeze(-1)  # [B, 128]
        return {
            "logits_user":   self.head_user(h),         # [B, n_users]
            "logits_task":   self.head_task(h),         # [B, n_tasks]
            "logits_attack": self.head_attack(h).squeeze(-1),  # [B]
        }

# ----- Device & mixed precision handling -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (DEVICE.type == "cuda")

def make_loaders(use_ema=False, add_deriv=True, attack_gen="shuffle"):
    """
    Build train/val/test datasets and dataloaders.
    NOTE: we now also generate synthetic attacks in val/test so AUC is defined.
    """
    train_idx, val_idx, test_idx = build_index()

    ds_tr = ForceWindowDataset(train_idx, window_len=512, stride=256,
                               use_ema=use_ema, add_derivatives=add_deriv,
                               attack_gen=attack_gen, attack_ratio=0.30)
    ds_va = ForceWindowDataset(val_idx, window_len=512, stride=256,
                               use_ema=use_ema, add_derivatives=add_deriv,
                               attack_gen=attack_gen, attack_ratio=0.30)
    ds_te = ForceWindowDataset(test_idx, window_len=512, stride=256,
                               use_ema=use_ema, add_derivatives=add_deriv,
                               attack_gen=attack_gen, attack_ratio=0.30)

    dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True,  num_workers=0, pin_memory=USE_CUDA)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False, num_workers=0, pin_memory=USE_CUDA)
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False, num_workers=0, pin_memory=USE_CUDA)

    in_ch = ds_tr[0][0].shape[0]  # number of input channels: 16 (with derivatives) or 3 (raw only)
    return (dl_tr, dl_va, dl_te, in_ch, ds_tr, ds_va, ds_te)

def split_stats(ds, name):
    """Quick check: how many windows per user in a split."""
    ys = [int(ds[i][1]) for i in range(len(ds))]  # user labels
    counts = np.bincount(ys, minlength=7)
    print(f"[{name}] windows per user: {counts.tolist()}")

@torch.no_grad()
def evaluate(model, loader):
    """Compute loss, user accuracy, attack AUC, and confusion matrix."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    tot = 0
    loss_u = loss_t = loss_a = 0.0
    y_true_u, y_pred_u = [], []
    y_true_a, y_score_a = [], []

    for x, yu, yt, ya in loader:
        x, yu, yt, ya = x.to(DEVICE), yu.to(DEVICE), yt.to(DEVICE), ya.to(DEVICE)
        out = model(x)

        loss_u += ce(out["logits_user"], yu).item()
        loss_t += ce(out["logits_task"], yt).item()
        loss_a += bce(out["logits_attack"], ya).item()
        tot += len(x)

        y_true_u.append(yu.cpu().numpy())
        y_pred_u.append(out["logits_user"].argmax(1).cpu().numpy())
        y_true_a.append(ya.cpu().numpy())
        y_score_a.append(torch.sigmoid(out["logits_attack"]).cpu().numpy())

    y_true_u = np.concatenate(y_true_u)
    y_pred_u = np.concatenate(y_pred_u)
    y_true_a = np.concatenate(y_true_a)
    y_score_a = np.concatenate(y_score_a)

    acc_u = accuracy_score(y_true_u, y_pred_u)
    try:
        auc_a = roc_auc_score(y_true_a, y_score_a)
    except Exception:
        auc_a = float("nan")  # happens if y_true has only one class (shouldn't now)

    loss = (loss_u + 0.25 * loss_t + 0.5 * loss_a) / tot
    cm = confusion_matrix(y_true_u, y_pred_u)
    return loss, acc_u, auc_a, cm

def main():
    # --- Build data ---
    dl_tr, dl_va, dl_te, in_ch, ds_tr, ds_va, ds_te = make_loaders(
        use_ema=False, add_deriv=True, attack_gen="shuffle"
    )
    split_stats(ds_tr, "train")
    split_stats(ds_va, "val")
    split_stats(ds_te, "test")

    # --- Build model/opt ---
    model = CNN1D(in_ch, n_users=7, n_tasks=7).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Mixed precision scaler (only on CUDA)
    scaler = torch.amp.GradScaler("cuda") if USE_CUDA else None
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    patience = 7
    wait = 0

    # --- Train loop ---
    for epoch in range(1, 41):
        model.train()
        batch_losses = []

        for x, yu, yt, ya in dl_tr:
            x, yu, yt, ya = x.to(DEVICE), yu.to(DEVICE), yt.to(DEVICE), ya.to(DEVICE)
            opt.zero_grad(set_to_none=True)

            if USE_CUDA:
                with torch.amp.autocast("cuda"):
                    out = model(x)
                    l_u = ce(out["logits_user"], yu)
                    l_t = ce(out["logits_task"], yt)
                    l_a = bce(out["logits_attack"], ya)
                    loss = l_u + 0.25 * l_t + 0.5 * l_a
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(x)
                l_u = ce(out["logits_user"], yu)
                l_t = ce(out["logits_task"], yt)
                l_a = bce(out["logits_attack"], ya)
                loss = l_u + 0.25 * l_t + 0.5 * l_a
                loss.backward()
                opt.step()

            batch_losses.append(loss.item())

        val_loss, val_acc, val_auc, _ = evaluate(model, dl_va)
        print(f"epoch {epoch:02d}  train_loss {np.mean(batch_losses):.4f}  "
              f"val_loss {val_loss:.4f}  val_user_acc {val_acc:.3f}  val_attack_auc {val_auc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # --- Load best and test ---
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    test_loss, test_acc, test_auc, cm = evaluate(model, dl_te)
    print(f"[TEST] loss {test_loss:.4f}  user_acc {test_acc:.3f}  attack_auc {test_auc:.3f}")
    print("User confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
