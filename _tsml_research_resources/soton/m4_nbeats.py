"""N-BEATS-style (generic) global forecasters for M4, one model per frequency and member.

python m4_nbeats.py <data_dir> <out_dir> <val|test> [--gpus 0,1,2,3] [--seeds 0,1,2,3,4]
    [--iters N] [--freqs yearly,...] [--ensemble]

Members per frequency: lookback L = k*H for k in LOOKBACKS, loss in (smape, mase), seeds.
Each member is a generic N-BEATS network (BLOCKS residual blocks of 4 x WIDTH FC layers,
linear backcast/forecast heads). Training follows the N-BEATS M4 setup: sample a series
uniformly, then a forecast origin among the last LH*H points of its fit part, input = the
L points before the origin (zero-padded on the left for short histories), target = the
next H points (masked where they run past the fit part). Each window is divided by the
mean absolute value of its observed input points, so the network sees scale-free inputs.

val : fit part = training series minus the last H points (targets = those H points);
test: fit part = full training series, forecasting the M4 test horizon.
Writes <out_dir>/members/<mode>_<freq>_L<k>_<loss>_s<seed>.npz (preds for that
frequency); finished members are skipped, so runs can be resumed or extended with
more seeds. Training iterations per frequency default to BASE_ITERS (yearly is cut
to 1500 because longer training overfits on the validation split).
--ensemble (after training) writes <out_dir>/<mode>_nbeats.npz: the per-step median of
all members found, for all 100k series in M4 order, in the layout of
m4_global_xgboost.py (preds, fr, names, fits, targets) for write_global_results.py.
BLOCKS and WIDTH can be overridden from the environment.
"""
import argparse
import os
import sys
import time

import numpy as np

D = OUT = None  # set from the command line
FR = ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly"]
M4M = {"yearly": 1, "quarterly": 4, "monthly": 12, "weekly": 1, "daily": 1, "hourly": 24}
LH = {"yearly": 1.5, "quarterly": 1.5, "monthly": 1.5, "weekly": 10, "daily": 10, "hourly": 10}
LOOKBACKS = [2, 3, 5]
LOSSES = ["smape", "mase"]
BLOCKS, WIDTH, BATCH = int(os.environ.get("BLOCKS", 10)), int(os.environ.get("WIDTH", 512)), 1024


BASE_ITERS = {"yearly": 1500, "quarterly": 15000, "monthly": 15000, "weekly": 5000, "daily": 8000, "hourly": 5000}


def load(fr, mode):
    from aeon.datasets import load_from_tsf_file

    ds = f"m4_{fr}_dataset"
    df, md = load_from_tsf_file(f"{D}/{ds}/{ds}.tsf")
    H = int(md["forecast_horizon"])
    fits, targets = [], []
    for v in df.series_value:
        v = np.asarray(v, float)
        tr = v[:-H]
        f, t = (tr[:-H], tr[-H:]) if mode == "val" else (tr, v[-H:])
        fits.append(f)
        targets.append(t)
    return list(df.series_name), fits, targets, H


def mase_scale(y, m):
    m = m if len(y) > m else 1
    s = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.nan
    return s if np.isfinite(s) and s > 0 else max(np.mean(np.abs(y)), 1e-8)


class NBeats:
    def __init__(self, L, H, torch):
        nn = torch.nn

        class Block(nn.Module):
            def __init__(s):
                super().__init__()
                layers, d = [], L
                for _ in range(4):
                    layers += [nn.Linear(d, WIDTH), nn.ReLU()]
                    d = WIDTH
                s.fc = nn.Sequential(*layers)
                s.back = nn.Linear(WIDTH, L)
                s.fore = nn.Linear(WIDTH, H)

            def forward(s, x):
                h = s.fc(x)
                return s.back(h), s.fore(h)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.blocks = nn.ModuleList([Block() for _ in range(BLOCKS)])

            def forward(s, x, mask):
                r, f = x, 0
                for b in s.blocks:
                    bc, fc = b(r)
                    r = (r - bc) * mask
                    f = f + fc
                return f

        self.net = Net()


def run_member(fr, k, loss, seed, mode, iters, device, data):
    import torch

    names, fits, targets, H = data
    L = k * H
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(fits)
    # padded series array for fast window gathering: P[i, PAD + j] = fits[i][j]
    lens = np.array([len(f) for f in fits])
    PAD = L
    P = np.zeros((n, PAD + lens.max() + H), np.float32)
    O = np.zeros_like(P)  # observed mask
    for i, f in enumerate(fits):
        P[i, PAD:PAD + len(f)] = f
        O[i, PAD:PAD + len(f)] = 1
    scale = np.array([mase_scale(f, M4M[fr]) for f in fits], np.float32)
    Pt, Ot = torch.tensor(P, device=device), torch.tensor(O, device=device)
    sc = torch.tensor(scale, device=device)
    lens_t = torch.tensor(lens, device=device)
    ar_in = torch.arange(-L, 0, device=device)
    ar_out = torch.arange(0, H, device=device)

    model = NBeats(L, H, torch).net.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.6), int(iters * 0.85)], 0.3)
    hist = int(np.ceil(LH[fr] * H))

    def windows(idx, origin):
        pos_in = PAD + origin[:, None] + ar_in[None]
        pos_out = PAD + origin[:, None] + ar_out[None]
        x = Pt[idx[:, None], pos_in]
        mx = Ot[idx[:, None], pos_in]
        norm = (x.abs() * mx).sum(1) / mx.sum(1).clamp(min=1)
        norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        return x / norm[:, None], mx, norm, pos_out

    for it in range(iters):
        idx = torch.tensor(rng.integers(0, n, BATCH), device=device)
        ln = lens_t[idx]
        lo = torch.clamp(ln - hist, min=1)
        u = torch.tensor(rng.random(BATCH), device=device, dtype=torch.float32)
        origin = (lo + (u * (ln - lo).float()).long()).clamp(max=ln - 1)  # 1..len-1
        x, mx, norm, pos_out = windows(idx, origin)
        y = Pt[idx[:, None], pos_out] / norm[:, None]
        my = Ot[idx[:, None], pos_out]
        f = model(x, mx)
        if loss == "smape":
            e = 2 * (f - y).abs() / (f.abs() + y.abs()).clamp(min=1e-8)
        else:
            e = (f - y).abs() / (sc[idx] / norm)[:, None]
        l = (e * my).sum() / my.sum()
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, n, 8192):
            idx = torch.arange(s, min(n, s + 8192), device=device)
            x, mx, norm, _ = windows(idx, lens_t[idx])
            out.append((model(x, mx) * norm[:, None]).cpu().numpy())
    return np.concatenate(out).astype(float), float(l)


def ensemble(mode):
    """Median of all members per frequency, saved in the m4_global_xgboost.py layout."""
    import glob

    preds, fr_, names, fits, targets = [], [], [], [], []
    for fr in FR:
        files = sorted(glob.glob(f"{OUT}/members/{mode}_{fr}_L*.npz"))
        nm, ft, tg, _ = load(fr, mode)
        if not files:
            raise FileNotFoundError(f"no {mode} members for {fr}")
        P = np.stack([np.load(f)["preds"] for f in files])
        assert all((np.load(f)["names"] == np.array(nm)).all() for f in files)
        med = np.maximum(np.median(P, 0), 0)
        preds += list(med)
        fr_ += [fr] * len(nm)
        names += nm
        fits += ft
        targets += tg
        print(f"{fr}: median of {len(files)} members", flush=True)
    obj = lambda x: np.array(x + [None], dtype=object)[:-1]  # noqa: E731 ragged arrays
    np.savez(f"{OUT}/{mode}_nbeats.npz", preds=obj(preds), fr=np.array(fr_), names=np.array(names),
             fits=obj(fits), targets=obj(targets))


def worker(gpu, jobs, mode, iters, d, out):
    import torch

    global D, OUT
    D, OUT = d, out

    torch.set_num_threads(4)
    device = f"cuda:{gpu}"
    cache = {}
    for fr, k, loss, seed in jobs:
        f = f"{OUT}/members/{mode}_{fr}_L{k}_{loss}_s{seed}.npz"
        if os.path.exists(f):
            continue
        if fr not in cache:
            cache = {fr: load(fr, mode)}
        t0 = time.time()
        preds, last = run_member(fr, k, loss, seed, mode, iters[fr], device, cache[fr])
        np.savez(f, preds=preds, names=np.array(cache[fr][0]))
        print(f"gpu{gpu} {mode} {fr} L{k} {loss} s{seed}: loss {last:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("mode", choices=["val", "test"])
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--freqs", default=",".join(FR))
    ap.add_argument("--ensemble", action="store_true")
    a = ap.parse_args()
    D, OUT = a.data_dir, a.out_dir
    os.makedirs(f"{OUT}/members", exist_ok=True)
    iters = {fr: a.iters or BASE_ITERS[fr] for fr in FR}
    jobs = [(fr, k, loss, int(s)) for fr in a.freqs.split(",") for s in a.seeds.split(",")
            for k in LOOKBACKS for loss in LOSSES]
    gpus = [int(g) for g in a.gpus.split(",")]
    # contiguous chunks per GPU so each worker loads few frequencies
    chunks = np.array_split(np.arange(len(jobs)), len(gpus))
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=(g, [jobs[i] for i in c], a.mode, iters, D, OUT)) for g, c in zip(gpus, chunks)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    if a.ensemble:
        ensemble(a.mode)
