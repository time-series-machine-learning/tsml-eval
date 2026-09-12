"""Measured signal characteristics of each UCR train set.

Aim: predict which representation suits a dataset, since representation fit is what
drives the MrHydra vs HC2 gap.
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from aeon.datasets import load_from_ts_file

DATA="D:/Data/All"
HERE=os.path.dirname(os.path.abspath(__file__))

def feats(X, y):
    X = np.asarray(X)[:, 0, :] if np.asarray(X).ndim == 3 else np.asarray(X)
    n, m = X.shape
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    # smoothness: lag-1 autocorrelation of the normalised series
    ac1 = np.mean([np.corrcoef(z[:-1], z[1:])[0, 1] for z in Z if np.std(z) > 0])
    # spectral: normalised centroid and high-frequency energy share
    P = np.abs(np.fft.rfft(Z, axis=1)) ** 2
    P = P[:, 1:]
    f = np.arange(1, P.shape[1] + 1) / P.shape[1]
    tot = P.sum(1) + 1e-12
    centroid = float(np.mean((P * f).sum(1) / tot))
    hf = float(np.mean(P[:, P.shape[1] // 4:].sum(1) / tot))
    # first-difference energy: how much of the signal is step-to-step change
    d1 = float(np.mean(np.abs(np.diff(Z, axis=1)).mean(1)))
    # class separability of the raw level and of the shape
    cls = np.unique(y)
    mu = np.stack([Z[y == c].mean(0) for c in cls])
    between = float(np.mean([np.linalg.norm(mu[i] - mu[j])
                             for i in range(len(cls)) for j in range(i + 1, len(cls))])) \
        if len(cls) > 1 else 0.0
    within = float(np.mean([np.linalg.norm(Z[y == c] - Z[y == c].mean(0), axis=1).mean()
                            for c in cls]))
    # how much discriminative signal survives shuffling time (order sensitivity proxy)
    rng = np.random.RandomState(0)
    Zs = Z[:, rng.permutation(m)]
    mus = np.stack([Zs[y == c].mean(0) for c in cls])
    between_s = float(np.mean([np.linalg.norm(mus[i] - mus[j])
                               for i in range(len(cls)) for j in range(i + 1, len(cls))])) \
        if len(cls) > 1 else 0.0
    return {"length": m, "n_train": n, "ac1": ac1, "spec_centroid": centroid,
            "hf_energy": hf, "diff_energy": d1, "between": between, "within": within,
            "sep_ratio": between / (within + 1e-12), "between_shuffled": between_s,
            "order_gain": between - between_s}

rows=[]
# the 112 UCR problems we have results for
names=sorted(os.path.basename(p) for p in
             glob.glob("D:/Results/UCR/Hybrid/HC2/Predictions/*") if os.path.isdir(p))
for i,ds in enumerate(names):
    f=f"{DATA}/{ds}/{ds}_TRAIN.ts"
    try:
        X,y=load_from_ts_file(f)
        r=feats(X,y); r["dataset"]=ds; rows.append(r)
    except Exception as e:
        print("skip",ds,str(e)[:60],flush=True); continue
    print(f"{i+1}/{len(names)} {ds}",flush=True)
pd.DataFrame(rows).set_index("dataset").to_csv(os.path.join(HERE,"signal.csv"))
print("written")
