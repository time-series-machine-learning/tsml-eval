"""Per-series statistics of M4 series, used as inputs by m4_global_xgboost.py and for
data-driven ensemble selection.

None of the features is the frequency, the seasonal period or the horizon themselves.
The seasonal period m is used only where a statistic needs it (seasonal strength and
seasonal autocorrelation, which are 0 when m = 1), as the forecasters themselves use it.
"""
import sys
from multiprocessing import Pool

import numpy as np

FR = {"yearly": 1, "quarterly": 4, "monthly": 12, "weekly": 1, "daily": 1, "hourly": 24}
NAMES = [
    "log_length", "cv", "log_growth", "log_growth_last", "trend_r2", "volatility",
    "vol_ratio", "acf1", "acf1_diff", "acf_m_diff", "seasonal_strength",
    "spectral_entropy", "level_shift", "last_vs_max", "skew_diff", "crossing_rate",
    "nonlin_curvature",
]


def acf(x, lag):
    x = x - x.mean()
    d = (x * x).sum()
    return 0.0 if lag >= len(x) or d == 0 else float((x[lag:] * x[:-lag]).sum() / d)


def feats(args):
    y, m = args
    y = np.asarray(y, float)
    n = len(y)
    ly = np.log(np.maximum(y, 1e-8))
    dl = np.diff(ly)
    t = np.arange(n)
    # linear trend on log scale
    A = np.vstack([t, np.ones(n)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    sst = ((ly - ly.mean()) ** 2).sum()
    r2 = 1 - (res[0] / sst) if len(res) and sst > 0 else 0.0
    # quadratic curvature relative to the linear trend
    q = np.polyfit(t / max(n - 1, 1), ly, 2)[0] if n >= 4 else 0.0
    k = max(2, n // 5)
    vol = dl.std() if n > 2 else 0.0
    vol_last = dl[-k:].std() if n > 3 else vol
    # classical seasonal strength on detrended log series (moving average trend)
    ss, acfm = 0.0, 0.0
    if m > 1 and n >= 2 * m + 1:
        tr = np.convolve(ly, np.ones(m) / m, mode="same")
        de = (ly - tr)[m // 2: n - m // 2]
        idx = np.arange(len(de)) % m
        seas = np.array([de[idx == j].mean() for j in range(m)])[idx]
        rem = de - seas
        ss = max(0.0, 1 - rem.var() / max((de).var(), 1e-12))
        acfm = acf(dl, m)
    # spectral entropy of the differenced log series
    se = 1.0
    if n > 8:
        p = np.abs(np.fft.rfft(dl - dl.mean())) ** 2
        p = p[1:]
        if p.sum() > 0:
            p = p / p.sum()
            se = float(-(p * np.log(p + 1e-300)).sum() / np.log(len(p)))
    sd = y.std()
    dd = np.diff(y)
    skew = float(((dd - dd.mean()) ** 3).mean() / (dd.std() ** 3)) if n > 3 and dd.std() > 0 else 0.0
    return [
        np.log(n),
        sd / max(abs(y.mean()), 1e-8),
        float(np.median(dl)) if n > 1 else 0.0,
        float(dl[-k:].mean()) if n > 2 else 0.0,
        r2,
        vol,
        vol_last / vol if vol > 0 else 1.0,
        acf(ly, 1),
        acf(dl, 1),
        acfm,
        ss,
        se,
        (y[-k:].mean() - y.mean()) / sd if sd > 0 else 0.0,
        y[-1] / y.max(),
        skew,
        float(np.mean(np.diff(np.sign(ly - ly.mean())) != 0)),
        q,
    ]


if __name__ == "__main__":
    # python m4_series_features.py <data_dir> <out.npz>: statistics of every M4 training series
    from aeon.datasets import load_from_tsf_file

    D, out = sys.argv[1], sys.argv[2]
    X, FREQ, KEYS = [], [], []
    for fr, m in FR.items():
        ds = f"m4_{fr}_dataset"
        df, md = load_from_tsf_file(f"{D}/{ds}/{ds}.tsf")
        H = int(md["forecast_horizon"])
        with Pool(4) as p:
            X += p.map(feats, [(np.asarray(v, float)[:-H], m) for v in df.series_value], chunksize=500)
        FREQ += [fr] * len(df)
        KEYS += [f"{ds}_{s}" for s in df.series_name]
        print(fr, len(df), flush=True)
    X = np.nan_to_num(np.array(X, float))
    np.savez(out, X=X, names=np.array(NAMES), freq=np.array(FREQ), keys=np.array(KEYS))
    print(X.shape)
