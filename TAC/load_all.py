from pathlib import Path
from typing import Iterator, Tuple
import pandas as pd

# --- 1) Locate the data directory robustly ---
# If this file lives at .../ML/TAC/load_all.py, then data is .../ML/TAC/data
HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"              # -> TAC/data

# --- 2) Required columns we expect inside each force.csv ---
REQUIRED_COLS = {"time", "force_x", "force_y", "force_z", "key_state"}  # we won't fail if some are missing, but we check

def iter_force_files(root: Path) -> Iterator[Tuple[str, str, Path]]:
    """
    Yield (user_id, task_id, csv_path) for every TAC/data/u*/[a-g]/force.csv

    root/u1/a/force.csv  -> ("u1", "a", Path(...))
    """
    for user_dir in sorted(root.glob("u*")):
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name  # e.g., "u3"
        for task_dir in sorted(user_dir.glob("[a-g]")):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name  # e.g., "c"
            csv_path = task_dir / "force.csv"
            if csv_path.exists():
                yield user_id, task_id, csv_path
            else:
                print(f"[warn] missing file: {csv_path}")

def read_force_csv(csv_path: Path) -> pd.DataFrame:
    """
    Robust CSV reader:
      - read with default dtypes
      - coerce numeric cols
      - make key_state a clean 0/1 int (if present)
    """
    df = pd.read_csv(
        csv_path,
        na_values=["", "NA", "NaN", "None"],  # treat blanks as NaN
        low_memory=False,
    )

    # Coerce numeric columns if present
    for col in ("time", "force_x", "force_y", "force_z", "key_state"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean key_state: convert floats/NaNs to 0/1 ints
    if "key_state" in df.columns:
        # fill NaN with 0, threshold 0.5 -> 1 else 0, store as small int
        ks = df["key_state"].fillna(0)
        df["key_state"] = (ks >= 0.5).astype("int8")

    # Optional: drop rows where any force value is NaN
    force_cols = [c for c in ("force_x", "force_y", "force_z") if c in df.columns]
    if force_cols:
        df = df.dropna(subset=force_cols)

    return df

def load_all() -> pd.DataFrame:
    """
    Walk the directory tree, load every CSV, attach labels:
      - user_id: u1..u7
      - task_id: a..g
    Return one concatenated DataFrame.
    """
    rows = []
    n_files = 0

    for user_id, task_id, csv_path in iter_force_files(DATA_ROOT):
        n_files += 1
        df = read_force_csv(csv_path)

        # --- 3) light schema check (inform, don't crash) ---
        missing = REQUIRED_COLS.difference(df.columns)
        if missing:
            print(f"[warn] {csv_path.name}: missing columns {sorted(missing)}; present: {list(df.columns)}")

        # --- 4) attach labels (these are your targets/meta) ---
        df["user_id"] = user_id
        df["task_id"] = task_id

        rows.append(df)

    if not rows:
        raise RuntimeError(f"No CSVs found under {DATA_ROOT}/u*/[a-g]/force.csv")

    # --- 5) concatenate into a single table ---
    df_all = pd.concat(rows, ignore_index=True)

    # --- 6) optional basic cleaning/sorting ---
    # Ensure a consistent column order (helpful for later)
    # Keep whatever columns exist, but order known ones first
    known_order = ["user_id", "task_id", "time", "force_x", "force_y", "force_z", "key_state"]
    ordered_cols = [c for c in known_order if c in df_all.columns] + [c for c in df_all.columns if c not in known_order]
    df_all = df_all[ordered_cols]

    # sort for reproducibility (not required)
    df_all = df_all.sort_values(["user_id", "task_id", "time"], ignore_index=True)

    print(f"[info] Loaded {n_files} files → {len(df_all):,} rows")
    return df_all

# --- 7) Run this file directly to see a quick summary ---
if __name__ == "__main__":
    print("Script folder :", HERE)
    print("Data root     :", DATA_ROOT, "exists?", DATA_ROOT.exists())

    if DATA_ROOT.exists():
        # Show top-level entries under data (expect u1..u7)
        print("Children of data/:", [p.name for p in DATA_ROOT.iterdir()])

    try:
        df = load_all()
        print("Columns      :", df.columns.tolist())
        print("Users        :", df['user_id'].nunique(), sorted(df['user_id'].unique()))
        print("Tasks        :", df['task_id'].nunique(), sorted(df['task_id'].unique()))
        # basic force magnitude sanity check
        if all(col in df.columns for col in ("force_x","force_y","force_z")):
            mag = (df["force_x"]**2 + df["force_y"]**2 + df["force_z"]**2) ** 0.5
            print("Force |F| mean/std:", float(mag.mean()), float(mag.std()))
        print(df.head())
    except Exception as e:
        print("ERROR:", e)
