from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE/ "data"  #TAC/data

def load_all():
    rows = []
    for user_dir in sorted(DATA_ROOT.glob("u")):
        user = user_dir.name  #eg., 'u3'
        for task_dir in sorted(user_dir.glob("[a-g]")):
            task = task_dir.name
            csv_path = task_dir / "force.csv"
            if not csv_path.exists():
                print("Missing:", csv_path); continue
            df = pd.read_csv(csv_path)
            df["user_id"] = user
            df["task_id"] = task
            rows.append(df)
        if not rows:
            raise RuntimeError("No CSVs found")
        df_all = pd.concat(rows, ignore_index = True)
        return df_all
    
if __name__ == "_main_":
    df = load_all()
    print(df.head())
    print("cols:",df.columns.tolist())
    print("records:", len(df), "users:", df['user_id'].nunique(),"tasks:", df['task_id'].nunique())