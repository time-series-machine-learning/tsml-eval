"""Collect accuracy and balanced accuracy for HC2, MrHydra and the HC2 components."""
import os, sys, glob
import numpy as np
import pandas as pd

R = "D:/Results/UCR"
CLFS = {
    "HC2": R + "/Hybrid/HC2",
    "MrHydra": R + "/ConvolutionBased/MrHydra",
    "STC": R + "/ShapeletBased/STC",
    "DrCIF": R + "/IntervalBased/DrCIF",
    "Arsenal": R + "/ConvolutionBased/Arsenal",
    "TDE": R + "/DictionaryBased/TDE",
}
NR = 30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hc2_vs_mrhydra.csv")


def read_preds(path):
    """Return (true, pred) label indices from a tsml results file."""
    with open(path) as f:
        lines = f.readlines()
    rows = [ln.split(",", 2) for ln in lines[3:] if ln.strip()]
    y = np.fromiter((int(r[0]) for r in rows), dtype=int, count=len(rows))
    p = np.fromiter((int(r[1]) for r in rows), dtype=int, count=len(rows))
    return y, p


def stats(y, p):
    acc = float((y == p).mean())
    classes = np.unique(y)
    recalls = [float((p[y == c] == c).mean()) for c in classes]
    return acc, float(np.mean(recalls))


datasets = sorted(
    {os.path.basename(d) for d in glob.glob(CLFS["HC2"] + "/Predictions/*")
     if os.path.isdir(d)}
)

rows = []
for i, d in enumerate(datasets):
    for r in range(NR):
        rec = {"dataset": d, "resample": r}
        ok = True
        for name, path in CLFS.items():
            f = f"{path}/Predictions/{d}/testResample{r}.csv"
            if not os.path.exists(f):
                ok = False
                break
            a, ba = stats(*read_preds(f))
            rec[name + "_acc"] = a
            rec[name + "_ba"] = ba
        if ok:
            rows.append(rec)
    print(f"{i+1}/{len(datasets)} {d}", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("written", OUT)
