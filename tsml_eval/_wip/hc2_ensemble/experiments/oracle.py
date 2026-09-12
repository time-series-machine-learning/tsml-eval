"""Oracle-weighted HC2 variants, built from stored component files.

Asks two things:
  a) how much does HC2 improve if CAWPE weights use test accuracy (an oracle)?
  b) does the imbalance bias in balanced accuracy reduce or disappear?

Also evaluates externally debiasing TDE's train estimate by a fixed offset, which is a
legitimate (non-oracle) compensation if the offset is fitted on other datasets.
"""
import os, glob, sys
import numpy as np
import pandas as pd

R = "D:/Results/UCR"
COMPS = {
    "STC": R + "/ShapeletBased/STC",
    "DrCIF": R + "/IntervalBased/DrCIF",
    "Arsenal": R + "/ConvolutionBased/Arsenal",
    "TDE": R + "/DictionaryBased/TDE",
}
NAMES = list(COMPS)
TDE_I = NAMES.index("TDE")
NR = 1
DEBIAS = [0.02, 0.04, 0.06, 0.08, 0.10]
HERE = os.path.dirname(os.path.abspath(__file__))


def read(path):
    """Return (y, probabilities) from a tsml results file."""
    with open(path) as f:
        head = [next(f) for _ in range(3)]
    n_classes = int(head[2].split(",")[5])
    arr = pd.read_csv(
        path, skiprows=3, header=None, usecols=list(range(3 + n_classes))
    ).to_numpy()
    return arr[:, 0].astype(int), arr[:, 3:3 + n_classes].astype(float)


def acc(y, p):
    return float((y == p).mean())


def ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def vote(probas, weights, seed):
    d = np.zeros(probas[0].shape)
    for pr, w in zip(probas, weights):
        d = d + pr * w
    d = d / d.sum(axis=1, keepdims=True)
    rng = np.random.RandomState(seed)
    return np.array([int(rng.choice(np.flatnonzero(r == r.max()))) for r in d])


datasets = sorted(
    os.path.basename(d) for d in glob.glob(R + "/Hybrid/HC2/Predictions/*")
    if os.path.isdir(d)
)
if len(sys.argv) > 1:
    datasets = sys.argv[1:]

rows = []
for i, ds in enumerate(datasets):
    for r in range(NR):
        files = {n: (f"{p}/Predictions/{ds}/trainResample{r}.csv",
                     f"{p}/Predictions/{ds}/testResample{r}.csv")
                 for n, p in COMPS.items()}
        if not all(os.path.exists(f) for pair in files.values() for f in pair):
            continue
        tr = {n: read(files[n][0]) for n in NAMES}
        te = {n: read(files[n][1]) for n in NAMES}
        y = te[NAMES[0]][0]
        probas = [te[n][1] for n in NAMES]

        rec = {"dataset": ds, "resample": r, "n_train": len(tr[NAMES[0]][0])}
        tr_acc, tr_ba, te_acc, te_ba = [], [], [], []
        for n in NAMES:
            ytr, ptr = tr[n]
            ptr = np.argmax(ptr, axis=1)
            yte, pte = te[n]
            pte = np.argmax(pte, axis=1)
            tr_acc.append(acc(ytr, ptr)); tr_ba.append(ba(ytr, ptr))
            te_acc.append(acc(yte, pte)); te_ba.append(ba(yte, pte))
            rec[f"{n}_train_acc"] = tr_acc[-1]
            rec[f"{n}_train_ba"] = tr_ba[-1]
            rec[f"{n}_test_acc"] = te_acc[-1]
            rec[f"{n}_test_ba"] = te_ba[-1]
        tr_acc = np.array(tr_acc); tr_ba = np.array(tr_ba)
        te_acc = np.array(te_acc); te_ba = np.array(te_ba)

        variants = {
            "HC2": (probas, tr_acc ** 4),
            "HC2_equal": (probas, np.ones(4)),
            "HC2_trainba": (probas, tr_ba ** 4),
            "HC2_oracle_acc": (probas, te_acc ** 4),
            "HC2_oracle_ba": (probas, te_ba ** 4),
        }
        # Oracle applied to TDE only, to see how much of the oracle gain is TDE.
        w = tr_acc.copy(); w[TDE_I] = te_acc[TDE_I]
        variants["HC2_oracle_tde"] = (probas, w ** 4)
        # Externally debias TDE's train estimate by a fixed offset.
        for d in DEBIAS:
            w = tr_acc.copy()
            w[TDE_I] = max(w[TDE_I] - d, 0.01)
            variants[f"HC2_tde-{d:.2f}"] = (probas, w ** 4)
        # Leave-one-component-out.
        for j, n in enumerate(NAMES):
            keep = [k for k in range(4) if k != j]
            variants[f"HC2_no{n}"] = ([probas[k] for k in keep], (tr_acc ** 4)[keep])

        for v, (pr, wt) in variants.items():
            p = vote(pr, wt, r)
            rec[v + "_acc"] = acc(y, p)
            rec[v + "_ba"] = ba(y, p)
        rows.append(rec)
    print(f"{i+1}/{len(datasets)} {ds}", flush=True)

out = os.path.join(HERE, "oracle.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("written", out)
