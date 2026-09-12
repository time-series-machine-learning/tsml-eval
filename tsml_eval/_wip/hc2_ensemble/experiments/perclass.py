"""Per-class CAWPE weighting for HC2, built from stored component files.

Scalar CAWPE gives component j one weight. Here weight becomes W[j, c], a weight for
component j on class c. Three schemes, which behave differently:

  recall   W[j,c] = train recall of j on class c, ** alpha.
           Rewards a component for finding class c. But minority classes have low
           recall for every component, so this shrinks total mass on minority classes.
  precision W[j,c] = train precision of j on class c, ** alpha.
           Rewards a component for being right when it says class c. Does not
           systematically shrink minority mass.
  prior    scalar CAWPE, then divide by the train class prior (balanced prior
           correction), which is the textbook fix for a skew-biased posterior.
"""
import os, glob, sys
import numpy as np, pandas as pd

R = "D:/Results/UCR"
COMPS = {"STC": R + "/ShapeletBased/STC", "DrCIF": R + "/IntervalBased/DrCIF",
         "Arsenal": R + "/ConvolutionBased/Arsenal", "TDE": R + "/DictionaryBased/TDE"}
N = list(COMPS)
HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-3


def read(path):
    with open(path) as f:
        head = [next(f) for _ in range(3)]
    nc = int(head[2].split(",")[5])
    a = pd.read_csv(path, skiprows=3, header=None,
                    usecols=list(range(3 + nc))).to_numpy()
    return a[:, 0].astype(int), a[:, 3:3 + nc].astype(float)


def acc(y, p):
    return float((y == p).mean())


def ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def per_class(y, p, n_classes):
    """Per-class recall and precision, floored away from zero."""
    rec = np.full(n_classes, EPS)
    prec = np.full(n_classes, EPS)
    for c in range(n_classes):
        if (y == c).any():
            rec[c] = max((p[y == c] == c).mean(), EPS)
        if (p == c).any():
            prec[c] = max((y[p == c] == c).mean(), EPS)
    return rec, prec


def vote(P, W, seed, prior=None):
    """P is (n_comp, n_cases, n_classes); W is (n_comp,) or (n_comp, n_classes)."""
    W = np.asarray(W)
    if W.ndim == 1:
        W = W[:, None] * np.ones(P.shape[2])
    d = np.einsum("jnc,jc->nc", P, W)
    if prior is not None:
        d = d / prior
    s = d.sum(1, keepdims=True)
    s[s == 0] = 1.0
    d = d / s
    rng = np.random.RandomState(seed)
    return np.array([int(rng.choice(np.flatnonzero(r == r.max()))) for r in d])


datasets = sorted(os.path.basename(x)
                  for x in glob.glob(R + "/Hybrid/HC2/Predictions/*") if os.path.isdir(x))
if len(sys.argv) > 1:
    datasets = sys.argv[1:]

rows = []
for i, ds in enumerate(datasets):
    r = 0
    try:
        tr = {n: read(f"{COMPS[n]}/Predictions/{ds}/trainResample{r}.csv") for n in N}
        te = {n: read(f"{COMPS[n]}/Predictions/{ds}/testResample{r}.csv") for n in N}
    except Exception as e:
        print("skip", ds, e, flush=True)
        continue
    y = te[N[0]][0]
    P = np.stack([te[n][1] for n in N])
    nc = P.shape[2]

    w_acc, w_ba, REC, PREC, OREC = [], [], [], [], []
    for n in N:
        ytr, ptr = tr[n]
        ptr = np.argmax(ptr, 1)
        w_acc.append(acc(ytr, ptr))
        w_ba.append(ba(ytr, ptr))
        rec, prec = per_class(ytr, ptr, nc)
        REC.append(rec)
        PREC.append(prec)
        yte, pte = te[n]
        OREC.append(per_class(yte, np.argmax(pte, 1), nc)[0])   # oracle per-class
    w_acc = np.array(w_acc); w_ba = np.array(w_ba)
    REC = np.array(REC); PREC = np.array(PREC); OREC = np.array(OREC)

    _, cnt = np.unique(tr[N[0]][0], return_counts=True)
    prior = cnt / cnt.sum()

    V = {"HC2": (w_acc ** 4, None), "HC2_trainba": (w_ba ** 4, None),
         "HC2_prior": (w_acc ** 4, prior),
         "HC2_trainba_prior": (w_ba ** 4, prior)}
    for a in (1, 2, 4):
        V[f"HC2_recall{a}"] = (REC ** a, None)
        V[f"HC2_prec{a}"] = (PREC ** a, None)
    V["HC2_recall4_prior"] = (REC ** 4, prior)
    V["HC2_prec4_prior"] = (PREC ** 4, prior)
    V["HC2_oracle_recall4"] = (OREC ** 4, None)

    rec = {"dataset": ds, "n_classes": nc}
    for v, (W, pr) in V.items():
        p = vote(P, W, r, prior=pr)
        rec[v + "_acc"] = acc(y, p)
        rec[v + "_ba"] = ba(y, p)
    rows.append(rec)
    print(f"{i+1}/{len(datasets)} {ds}", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(HERE, "perclass.csv"), index=False)
print("written")
