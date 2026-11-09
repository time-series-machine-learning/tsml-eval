"""Write global-model M4 test forecasts (npz from m4_global_xgboost.py / m4_nbeats.py) as
tsml forecasting results.

python write_global_results.py <results_dir> <name> <member> [<member> ...]
    [--method mean|median] [--label TEXT] [--kind "global xgboost"]

Each <member> is one npz file, or several comma-separated npz files whose forecasts are
averaged first (seed bagging). With more than one member, the members are combined per
step with --method (default median). Examples:

    # one model
    write_global_results.py results global-xgb-freq-l1-stats out/test_freq_l1_stats.npz
    # a seed-bagged model
    write_global_results.py results global-xgb-freq-l1-stats-bag \\
        out/test_freq_l1_stats.npz,out/test_freq_l1_stats_s1.npz,out/test_freq_l1_stats_s2.npz
    # median of seed-bagged configs
    write_global_results.py results global-xgb-ens-bag a.npz,a_s1.npz b.npz,b_s1.npz

Results go to <results_dir>/<name>/Predictions/m4_<freq>_dataset_<series>/testResample0.csv
with the M4 test values as targets; negative forecasts are clipped to 0. Existing files
are skipped. Fit and predict times are not recorded per series (-1).
"""

import argparse
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from tsml_eval.utils.results_writing import write_forecasting_results


def load_member(spec):
    zs = [np.load(f, allow_pickle=True) for f in spec.split(",")]
    for z in zs[1:]:
        assert (z["names"] == zs[0]["names"]).all() and (z["fr"] == zs[0]["fr"]).all()
    preds = [np.mean([np.asarray(z["preds"][i], float) for z in zs], 0) for i in range(len(zs[0]["fr"]))]
    return preds, zs[0]


def main(results_dir, name, members, method="median", label=None, kind="global xgboost"):
    loaded = [load_member(m) for m in members]
    ref = loaded[0][1]
    for _, z in loaded[1:]:
        assert (z["names"] == ref["names"]).all() and (z["fr"] == ref["fr"]).all()
    comb = np.median if method == "median" else np.mean
    short = [",".join(os.path.basename(f) for f in m.split(",")) for m in members]
    label = label or (short[0] if len(short) == 1 else f"{method} of {' + '.join(short)}")
    comment = (f"{kind[0].upper() + kind[1:]} forecaster ({label}) trained across M4 series, "
               f"written {datetime.now():%m/%d/%Y, %H:%M:%S}")
    for i, (fr, s, y) in enumerate(zip(ref["fr"], ref["names"], ref["targets"])):
        ds = f"m4_{fr}_dataset_{s}"
        if os.path.exists(f"{results_dir}/{name}/Predictions/{ds}/testResample0.csv"):
            continue
        p = comb([P[i] for P, _ in loaded], 0) if len(loaded) > 1 else loaded[0][0][i]
        p = np.maximum(np.asarray(p, float), 0)
        y = np.asarray(y, float)
        write_forecasting_results(
            p, y, name, ds, results_dir, full_path=False,
            first_line_forecaster_name=f"{name} ({kind})", split="TEST",
            random_seed=0, time_unit="MILLISECONDS", first_line_comment=comment,
            parameter_info=str({"config": label}),
            mape=mean_absolute_percentage_error(y, p), fit_time=-1, predict_time=-1,
            benchmark_time=-1, memory_usage=-1,
        )
    print(name, len(ref["fr"]), "series", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("name")
    ap.add_argument("members", nargs="+")
    ap.add_argument("--method", default="median", choices=["mean", "median"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--kind", default="global xgboost")
    a = ap.parse_args()
    main(a.results_dir, a.name, a.members, a.method, a.label, a.kind)
