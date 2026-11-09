"""Global xgboost forecasters for M4: one model for all series, or one per frequency.

python m4_global_xgboost.py <data_dir> <out_dir> <val|test> <config>[,<config>...]
    [--n-jobs N] [--trees 3000] [--lr 0.05] [--depth 8] [--seed 0] [--tag SUFFIX]
    [--cache] [--limit K]

Writes <out_dir>/<mode>_<config><tag>.npz with the forecasts of every series (preds,
in M4 order yearly..hourly), fr, names, fits, targets and the early-stopping iteration
of each model; write_global_results.py turns test forecasts into tsml result files.
--seed changes the model's random state (row/column subsampling and the early-stopping
series split) but not the training rows, so seeds can be averaged as a bagged model.
--cache keeps the training rows in <out_dir>/cache (several GB) for later runs.

val : each series' fit part is its training series minus the last H points, and the
      targets are those H points (the validation split used for all other members).
test: fit on the full training series, forecast the M4 test horizon.
Training rows come only from the fit parts.

Preprocessing per series (ES-RNN / N-BEATS style, M4 conventions):
  * classical multiplicative deseasonalisation when the M4 90% seasonality test
    passes (as Naive2: m = 1, 4, 12, 1, 1, 24), then log;
  * for a forecast origin t, the input is the last L deseasonalised log values
    minus the value at t (left-padded with NaN for short histories, which xgboost
    treats as missing), and the target for step h is z[t-1+h] - z[t-1];
  * extra inputs: step h, horizon H, the forecasters' seasonal period, log history
    length, and optionally the 17 statistics of ens/features.py computed on the
    history up to the origin only.
Forecast = exp(z[n-1] + prediction) x the seasonal index continued periodically.

Training rows: the last K origins of every series, every step with an observed
target; per frequency at most ROW_CAP rows are kept at random. The all-frequency
and per-frequency models are trained on exactly the same rows. 10% of series are
held out for early stopping.
"""

import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m4_series_features import feats as series_stats  # noqa: E402

FR = ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly"]
M4M = {"yearly": 1, "quarterly": 4, "monthly": 12, "weekly": 1, "daily": 1, "hourly": 24}
FCM = {"yearly": 1, "quarterly": 4, "monthly": 12, "weekly": 52, "daily": 7, "hourly": 24}
L = {"yearly": 20, "quarterly": 24, "monthly": 36, "weekly": 52, "daily": 28, "hourly": 120}
LMAX = max(L.values())
K = 30
ROW_CAP = 4_000_000
N_STATS = 17
TREES, LR, DEPTH, SEED, TAG = 3000, 0.05, 8, 0, ""  # set from the command line
D = OUT = None
LIMIT = 0

CONFIGS = {
    # name: (scope, objective, stats)
    "all_l2": ("all", "reg:squarederror", False),
    "freq_l2": ("freq", "reg:squarederror", False),
    "all_l1": ("all", "reg:absoluteerror", False),
    "freq_l1": ("freq", "reg:absoluteerror", False),
    "all_l2_stats": ("all", "reg:squarederror", True),
    "freq_l2_stats": ("freq", "reg:squarederror", True),
    "all_l1_stats": ("all", "reg:absoluteerror", True),
    "freq_l1_stats": ("freq", "reg:absoluteerror", True),
}


def seasonal(y, m, n_future):
    """Seasonal indices for y and the next n_future points (all 1 if not seasonal)."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf

    ones = np.ones(len(y) + n_future)
    if m <= 1 or len(y) < 3 * m:
        return ones
    r = acf(y, nlags=m, fft=False)
    if not abs(r[m]) > 1.645 * np.sqrt((1 + 2 * np.sum(r[1:m] ** 2)) / len(y)):
        return ones
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = np.asarray(seasonal_decompose(y, model="multiplicative", period=m).seasonal)
    fut = np.array([s[len(y) - m + (i % m)] for i in range(n_future)])
    return np.concatenate([s, fut])


def prep(args):
    fr, y, H = args
    s = seasonal(y, M4M[fr], H)
    z = np.log(np.maximum(y / s[: len(y)], 1e-8))
    return z, s


def window(z, t, l):
    x = np.full(LMAX, np.nan)
    w = z[max(0, t - l): t]
    x[LMAX - len(w):] = w - z[t - 1]
    return x


def load(mode):
    from aeon.datasets import load_from_tsf_file

    S = []
    for fr in FR:
        ds = f"m4_{fr}_dataset"
        df, md = load_from_tsf_file(f"{D}/{ds}/{ds}.tsf")
        H = int(md["forecast_horizon"])
        lim = LIMIT or None
        for name, v in list(zip(df.series_name, df.series_value))[:lim]:
            v = np.asarray(v, float)
            train = v[:-H]
            fit, target = (train[:-H], train[-H:]) if mode == "val" else (train, v[-H:])
            S.append(dict(fr=fr, name=name, fit=fit, target=target, H=H))
    return S


def stats_job(a):
    y, m = a
    return series_stats((y, m))


def build_rows(S, Z, rng, with_stats, nj):
    """Training rows (X, y, freq index, series index) from the fit parts."""
    pairs = {fr: [] for fr in FR}
    for i, s in enumerate(S):
        n, H = len(s["fit"]), s["H"]
        for t in range(n - 1, max(2, n - K - 1), -1):  # last K origins, t >= 3
            for h in range(1, min(H, n - t) + 1):
                pairs[s["fr"]].append((i, t, h))
    rows = []
    for fr in FR:
        p = np.array(pairs[fr])
        if len(p) > ROW_CAP:
            p = p[rng.choice(len(p), ROW_CAP, replace=False)]
        rows.append(p)
        print(f"  {fr}: {len(pairs[fr])} candidate rows, {len(p)} kept", flush=True)
    rows = np.concatenate(rows)
    rows = rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]
    X = np.empty((len(rows), LMAX + 4 + (N_STATS if with_stats else 0)), np.float32)
    y = np.empty(len(rows), np.float32)
    if with_stats:
        uo = np.unique(rows[:, :2], axis=0)
        with Pool(nj) as pool:
            st = pool.map(stats_job, [(S[i]["fit"][:t], M4M[S[i]["fr"]]) for i, t in uo], chunksize=2000)
        stat_of = {(int(i), int(t)): v for (i, t), v in zip(uo, st)}
    for r, (i, t, h) in enumerate(rows):
        s, z = S[i], Z[i][0]
        X[r, :LMAX] = window(z, t, L[s["fr"]])
        X[r, LMAX:LMAX + 4] = (h, s["H"], FCM[s["fr"]], np.log(t))
        if with_stats:
            X[r, LMAX + 4:] = stat_of[(int(i), int(t))]
        y[r] = z[t - 1 + h] - z[t - 1]
    fidx = np.array([FR.index(S[i]["fr"]) for i in rows[:, 0]])
    return np.nan_to_num(X, nan=np.nan, posinf=0, neginf=0), y, fidx, rows[:, 0]


def forecast_rows(S, Z, idx, with_stats, nj):
    blocks, owner = [], []
    if with_stats:
        with Pool(nj) as pool:
            st = pool.map(stats_job, [(S[i]["fit"], M4M[S[i]["fr"]]) for i in idx], chunksize=2000)
    for k, i in enumerate(idx):
        s, z = S[i], Z[i][0]
        n, H = len(s["fit"]), s["H"]
        x = window(z, n, L[s["fr"]])
        B = np.empty((H, LMAX + 4 + (N_STATS if with_stats else 0)), np.float32)
        B[:, :LMAX] = x
        B[:, LMAX] = np.arange(1, H + 1)
        B[:, LMAX + 1:LMAX + 4] = (H, FCM[s["fr"]], np.log(n))
        if with_stats:
            B[:, LMAX + 4:] = st[k]
        blocks.append(B)
        owner.append(np.full(H, i))
    return np.concatenate(blocks), np.concatenate(owner)


def fit_model(X, y, groups, objective, nj, seed=0):
    from xgboost import XGBRegressor

    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    val_g = set(rng.choice(ug, max(1, len(ug) // 10), replace=False).tolist())
    va = np.array([g in val_g for g in groups])
    m = XGBRegressor(
        tree_method="hist", max_depth=DEPTH, learning_rate=LR, n_estimators=TREES,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20, max_bin=256,
        objective=objective, early_stopping_rounds=50, n_jobs=nj, random_state=seed,
    )
    m.fit(X[~va], y[~va], eval_set=[(X[va], y[va])], verbose=False)
    return m


def main():
    import argparse

    global TREES, LR, DEPTH, SEED, TAG, D, OUT, LIMIT
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("mode", choices=["val", "test"])
    ap.add_argument("configs")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--trees", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first K series per frequency (testing)")
    a = ap.parse_args()
    D, OUT, mode, names, nj = a.data_dir, a.out_dir, a.mode, a.configs.split(","), a.n_jobs
    TREES, LR, DEPTH, SEED, TAG, LIMIT = a.trees, a.lr, a.depth, a.seed, a.tag, a.limit
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    S = load(mode)
    with Pool(nj) as pool:
        Z = pool.map(prep, [(s["fr"], s["fit"], s["H"]) for s in S], chunksize=500)
    print(f"{len(S)} series loaded and deseasonalised ({time.time() - t0:.0f}s)", flush=True)
    cache = {}
    for name in names:
        scope, objective, with_stats = CONFIGS[name]
        if with_stats not in cache:
            cf = f"{OUT}/cache/rows_{mode}_{int(with_stats)}{f'_limit{LIMIT}' if LIMIT else ''}.npz"
            if not a.cache:
                cache[with_stats] = build_rows(S, Z, np.random.default_rng(0), with_stats, nj)
            elif os.path.exists(cf):
                c = np.load(cf)
                cache[with_stats] = (c["X"], c["y"], c["fidx"], c["sidx"])
            else:
                rng = np.random.default_rng(0)  # rows are fixed; SEED varies the model only
                cache[with_stats] = build_rows(S, Z, rng, with_stats, nj)
                os.makedirs(f"{OUT}/cache", exist_ok=True)
                np.savez(cf, X=cache[with_stats][0], y=cache[with_stats][1],
                         fidx=cache[with_stats][2], sidx=cache[with_stats][3])
            print(f"rows built: {cache[with_stats][0].shape} ({time.time() - t0:.0f}s)", flush=True)
        X, y, fidx, sidx = cache[with_stats]
        all_idx = np.arange(len(S))
        Xf, owner = forecast_rows(S, Z, all_idx, with_stats, nj)
        ofr = np.array([FR.index(S[i]["fr"]) for i in owner])
        pred = np.empty(len(Xf))
        info = []
        groups = [np.arange(6)] if scope == "all" else [[j] for j in range(6)]
        for g in groups:
            sel, fsel = np.isin(fidx, g), np.isin(ofr, g)
            m = fit_model(X[sel], y[sel], sidx[sel], objective, nj, seed=SEED)
            pred[fsel] = m.predict(Xf[fsel])
            info.append(f"{','.join(FR[j][0] for j in g)}:{m.best_iteration}")
        # forecast rows are grouped by series in order, so split by horizon lengths
        Hs = np.array([S[i]["H"] for i in all_idx])
        splits = np.split(pred, np.cumsum(Hs)[:-1])
        F = []
        for i, p in zip(all_idx, splits):
            s, (z, sea) = S[i], Z[i]
            n = len(s["fit"])
            F.append(np.exp(z[n - 1] + p) * sea[n:n + s["H"]])
        np.savez(
            f"{OUT}/{mode}_{name}{TAG}.npz",
            preds=np.array(F, dtype=object), fr=np.array([s["fr"] for s in S]),
            names=np.array([s["name"] for s in S]), fits=np.array([s["fit"] for s in S], dtype=object),
            targets=np.array([s["target"] for s in S], dtype=object), best_iter=np.array(info),
        )
        print(f"{name}{TAG}: done, best iterations {info} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
