from pathlib import Path

HERE = Path(__file__).resolve().parent        # -> .../ML/TAC
root = HERE / "data"                          # -> .../ML/TAC/data

print("Script folder:", HERE)
print("Data root:", root, "exists?", root.exists())

# List users
users = sorted([p.name for p in root.glob("u*") if p.is_dir()])
print("Users:", users)

for u in users:
    user_dir = root / u
    tasks = sorted([p.name for p in user_dir.glob("[a-g]") if p.is_dir()])
    have_csv = all((user_dir / t / "force.csv").exists() for t in tasks)
    print(u, "tasks:", tasks, "have_csv:", have_csv)
