import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLS = {"time", "force_x", "force_y", "force_z", "key_state"}


def load_force_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the force.csv file and ensure required columns exist.
    Let pandas auto-detect the delimiter (commas, tabs, spaces, etc.).
    """
    print(f"Loading CSV: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Auto-detect delimiter
    df = pd.read_csv(csv_path, sep=None, engine="python")
    print("Detected columns:", list(df.columns))

    # Strip possible leading/trailing spaces from column names
    df.columns = [c.strip() for c in df.columns]
    print("Stripped columns:", list(df.columns))

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def plot_force_space(csv_path: Path, include_all_states: bool = False) -> None:
    """
    Plot force_x vs force_y on a 2D plane.

    If include_all_states is False, only samples with key_state == 1 (pen down)
    are plotted.
    """
    df = load_force_csv(csv_path)

    if include_all_states:
        draw_df = df
        title_state = "all samples"
    else:
        draw_df = df[df["key_state"] == 1]
        title_state = "pen down (key_state = 1)"

    print(f"Number of samples plotted: {len(draw_df)}")

    plt.figure(figsize=(6, 6))
    plt.scatter(draw_df["force_x"], draw_df["force_y"], s=4)

    plt.xlabel("force_x")
    plt.ylabel("force_y")
    plt.title(f"Force-space pattern ({title_state})\n{csv_path.name}")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 2D force-space (force_x vs force_y) from "
            "TAC/data/<user>/<action>/force.csv"
        )
    )
    parser.add_argument(
        "user",
        help="User ID, e.g. u1, u2, u3 ..."
    )
    parser.add_argument(
        "action",
        help="Action folder, e.g. a, b, c ..."
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Include all samples, not just key_state == 1",
    )

    args = parser.parse_args()

    # Directory containing this script (TAC/)
    this_dir = Path(__file__).resolve().parent

    # data/ is inside TAC/
    base_dir = this_dir / "data"

    csv_path = base_dir / args.user / args.action / "force.csv"

    print(f"Resolved CSV path: {csv_path}")
    print(f"Exists? {csv_path.exists()}")

    plot_force_space(csv_path, include_all_states=args.all_states)


if __name__ == "__main__":
    main()
