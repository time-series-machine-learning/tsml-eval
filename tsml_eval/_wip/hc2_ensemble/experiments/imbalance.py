"""Class distribution of each UCR dataset, taken from the results files."""
import os, glob
import numpy as np
import pandas as pd

R = "D:/Results/UCR/Hybrid/HC2/Predictions"
rows = []
for d in sorted(os.path.basename(p) for p in glob.glob(R + "/*") if os.path.isdir(p)):
    f = f"{R}/{d}/testResample0.csv"
    if not os.path.exists(f):
        continue
    with open(f) as fh:
        lines = fh.readlines()
    y = np.array([int(ln.split(",", 1)[0]) for ln in lines[3:] if ln.strip()])
    _, counts = np.unique(y, return_counts=True)
    rows.append({
        "dataset": d,
        "n_test": len(y),
        "n_classes": len(counts),
        "min_class": int(counts.min()),
        "max_class": int(counts.max()),
        "imbalance_ratio": float(counts.max() / counts.min()),
        # 1.0 = perfectly balanced, lower = more skewed
        "evenness": float(len(counts) * counts.min() / counts.sum()),
    })
df = pd.DataFrame(rows)
here = os.path.dirname(os.path.abspath(__file__))
df.to_csv(os.path.join(here, "imbalance.csv"), index=False)
print(df.sort_values("imbalance_ratio", ascending=False).head(20).to_string(index=False))
print("\nbalanced (ratio==1):", (df.imbalance_ratio == 1).sum(), "of", len(df))
