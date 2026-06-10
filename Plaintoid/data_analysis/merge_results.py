"""
Merge per-run result CSVs into one file per experiment group.
Deduplicates by keeping the latest run per (exp_name, seed).

Output files:
  ../Cora/results_ExpA.csv
  ../Cora/results_ExpB.csv
  ../Cora/results_ExpBudget.csv
"""
import glob
import pandas as pd

RESULTS_DIR = "/fs/gpfs41/lv11/fileset01/pool/pool-shao/Hierarchial_RW/Plaintoid/Cora"

files = glob.glob(f"{RESULTS_DIR}/result_*.csv")
print(f"Found {len(files)} CSV files")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Deduplicate: keep last run per (exp_name, seed)
df = df.sort_values("run_id").groupby(["exp_name", "seed"], as_index=False).last()
df = df.sort_values(["exp_name", "seed"]).reset_index(drop=True)

groups = {
    "ExpA":      df[df["exp_name"].str.startswith("ExpA")],
    "ExpB":      df[df["exp_name"].str.startswith("ExpB") & ~df["exp_name"].str.startswith("ExpBudget")],
    "ExpBudget": df[df["exp_name"].str.startswith("ExpBudget")],
}

for name, group in groups.items():
    if group.empty:
        continue
    out = f"{RESULTS_DIR}/results_{name}.csv"
    group.to_csv(out, index=False)
    print(f"\n[{name}] {len(group)} rows → {out}")
    metrics = [c for c in ["best_val_MRR", "best_test_MRR", "best_test_Hits@10", "best_test_Hits@50"] if c in group.columns]
    print(group.groupby("exp_name")[metrics].agg(["mean", "std"]).round(4).to_string())
