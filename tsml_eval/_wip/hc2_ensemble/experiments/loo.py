"""Leave-one-component-out HC2, built from files, to attribute balanced accuracy loss.

All variants are built from the same stored component files, so they are internally
consistent even though the stored HC2 itself is not reproducible.
"""
import os, sys
import numpy as np
import pandas as pd

from tsml_eval.estimators.classification.hybrid.hivecote_from_results import (
    _load_split, component_prediction_paths,
)
from tsml_eval.estimators.classification.hybrid import FromFileHIVECOTE

R = "D:/Results/UCR"
COMPS = {
    "STC": R + "/ShapeletBased/STC",
    "DrCIF": R + "/IntervalBased/DrCIF",
    "Arsenal": R + "/ConvolutionBased/Arsenal",
    "TDE": R + "/DictionaryBased/TDE",
}
NAMES = list(COMPS)


def balanced_accuracy(y, p):
    classes = np.unique(y)
    return float(np.mean([(p[y == c] == c).mean() for c in classes]))


def accuracy(y, p):
    return float((y == p).mean())


def vote(probas, weights, seed):
    d = np.zeros(probas[0].shape)
    for p, w in zip(probas, weights):
        d = d + p * w
    d = d / d.sum(axis=1, keepdims=True)
    rng = np.random.RandomState(seed)
    return np.array([int(rng.choice(np.flatnonzero(r == r.max()))) for r in d]), d


def analyse(dataset, resamples, subsets=None):
    rows = []
    for r in resamples:
        paths = component_prediction_paths([COMPS[n] for n in NAMES], dataset)
        try:
            tr = _load_split(paths, r, "TRAIN")
            te = _load_split(paths, r, "TEST")
        except Exception as e:
            print(f"skip {dataset} r{r}: {e}")
            continue
        y = np.asarray(te[0].class_labels).astype(int)
        w = np.array([c.accuracy ** 4 for c in tr])
        probas = [c.probabilities for c in te]

        for i, n in enumerate(NAMES):
            p = np.argmax(probas[i], axis=1)
            rows.append({"dataset": dataset, "resample": r, "variant": n,
                         "acc": accuracy(y, p), "ba": balanced_accuracy(y, p),
                         "weight": w[i]})

        p, _ = vote(probas, w, r)
        rows.append({"dataset": dataset, "resample": r, "variant": "HC2",
                     "acc": accuracy(y, p), "ba": balanced_accuracy(y, p),
                     "weight": np.nan})

        for i, n in enumerate(NAMES):
            keep = [j for j in range(len(NAMES)) if j != i]
            p, _ = vote([probas[j] for j in keep], w[keep], r)
            rows.append({"dataset": dataset, "resample": r, "variant": f"HC2-no{n}",
                         "acc": accuracy(y, p), "ba": balanced_accuracy(y, p),
                         "weight": np.nan})

        # Equal weights, to separate the weighting from the components.
        p, _ = vote(probas, np.ones(len(NAMES)), r)
        rows.append({"dataset": dataset, "resample": r, "variant": "HC2-equalw",
                     "acc": accuracy(y, p), "ba": balanced_accuracy(y, p),
                     "weight": np.nan})
    return rows


if __name__ == "__main__":
    datasets = sys.argv[1:]
    resamples = range(30)
    out = []
    for d in datasets:
        out += analyse(d, resamples)
        print("done", d, flush=True)
    df = pd.DataFrame(out)
    here = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(here, "loo.csv"), index=False)
    piv = df.groupby(["dataset", "variant"])[["acc", "ba"]].mean()
    print(piv.to_string())
