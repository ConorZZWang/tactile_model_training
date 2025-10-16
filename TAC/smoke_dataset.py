# TAC/smoke_dataset.py
from torch.utils.data import DataLoader
from TAC.datasets import build_index, ForceWindowDataset

if __name__ == "__main__":
    train_idx, val_idx, test_idx = build_index()
    print("files ->", len(train_idx), len(val_idx), len(test_idx))

    ds = ForceWindowDataset(
        train_idx,
        window_len=512,
        stride=256,
        use_ema=False,
        add_derivatives=True,
        attack_gen="shuffle",   # try None first if you want
        attack_ratio=0.3,
        max_files=3             # keep small for first test
    )
    print("windows:", len(ds))

    x, yu, yt, ya = ds[0]
    print("one sample shapes:", x.shape, "user:", yu.item(), "task:", yt.item(), "attack:", ya.item())

    dl = DataLoader(ds, batch_size=16, shuffle=True)
    xb, yub, ytb, yab = next(iter(dl))
    print("batch shapes:", xb.shape, yub.shape, ytb.shape, yab.shape)
