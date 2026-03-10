import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TAC.load_all import iter_force_files, DATA_ROOT


def main():
    ap = argparse.ArgumentParser("Plot one raw force signal example")
    ap.add_argument("--user", type=int, default=None, help="User id to plot, e.g. 1 for u1")
    ap.add_argument("--task", type=str, default=None, help="Task letter/name to plot, e.g. a")
    ap.add_argument("--max_samples", type=int, default=3000, help="Max number of samples to plot")
    ap.add_argument("--out", type=str, default="figures/transformer/raw_force_example.png")
    args = ap.parse_args()

    chosen = None

    # Try to find a matching file if user/task provided
    for user_id, task_id, csv_path in iter_force_files(DATA_ROOT):
        match_user = args.user is None or user_id == args.user
        match_task = args.task is None or str(task_id) == str(args.task)
        if match_user and match_task:
            chosen = (user_id, task_id, csv_path)
            break

    # Fallback: just use the first file
    if chosen is None:
        all_files = list(iter_force_files(DATA_ROOT))
        if not all_files:
            raise RuntimeError("No CSV files found.")
        chosen = all_files[0]

    user_id, task_id, csv_path = chosen
    print(f"Using file: {csv_path}")

    df = pd.read_csv(csv_path)
    for c in ("force_x", "force_y", "force_z"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["force_x", "force_y", "force_z"]).reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("Selected CSV has no valid force samples.")

    n = min(args.max_samples, len(df))
    x = range(n)

    fx = df["force_x"].values[:n]
    fy = df["force_y"].values[:n]
    fz = df["force_z"].values[:n]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(11, 4.8))
    plt.plot(x, fx, label=r"$F_x$", linewidth=1.0)
    plt.plot(x, fy, label=r"$F_y$", linewidth=1.0)
    plt.plot(x, fz, label=r"$F_z$", linewidth=1.0)

    plt.xlabel("Sample Index")
    plt.ylabel("Force")
    plt.title(f"Example Raw Force Signal (user={user_id}, task={task_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()