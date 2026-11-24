"""Build time-shift and phase-surrogate variants of event-locked Cognitive EEG problems.

Cognitive case study (24 Nov 2026): do classifiers rely on where in the epoch a feature
occurs (phase dependence), and is the class information in the time-locked waveform or in
band power? Interval methods (DrCIF, QUANT) summarise fixed positions, so they should be
phase dependent; STC (best match anywhere), Arsenal and MRHydra (convolution with
pooling) and TS-LR (covariance) largely not.

From each trial of length T (original train/test split, same labels and order):
- Crop50: the middle T/2 (offset T/4), the same for every trial. The reference for the
  shifted variants, and with the main results it shows what cropping alone costs.
- Shift50S: length T/2 at offset T/4 + u, u uniform on [-T/20, T/20] per trial (small
  jitter, +-5% of the epoch, e.g. +-50 ms for a 1 s epoch).
- Shift50L: length T/2 at an offset uniform on [0, T/2] per trial (+-25% of the epoch).
- Surr: phase-randomised surrogate of the whole trial: its Fourier phases are replaced by
  random ones, the same random phases for every channel, so each channel's power spectrum
  and the cross-channel covariance are kept and the waveform is destroyed.
- SurrIndep: as Surr with independent random phases per channel, which also destroys the
  cross-channel phase relations (so covariance methods should suffer).
Offsets and seeds are written to cases_<split>.tsv. Shifts never pad and never cross the
epoch boundary. Train and test cases are both transformed.

Usage: python build_cognitive_shifts.py SRC_DIR NAME OUT_ROOT [--seed 0] [--digits 6]
"""

import argparse
import csv
import json
import time
import zlib
from pathlib import Path

import numpy as np

VARIANTS = ("Crop50", "Shift50S", "Shift50L", "Surr", "SurrIndep")


def read_ts(path):
    with open(path) as f:
        header = []
        for line in f:
            header.append(line)
            if line.lower().startswith("@data"):
                break
        yield header, None
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(":")
                yield np.array([np.array(p.split(","), dtype=float) for p in parts[:-1]]), parts[-1]


def surrogate(x, rng, shared):
    """Phase-randomised surrogate of x (channels x time) with the same amplitude spectrum."""
    n = x.shape[1]
    f = np.fft.rfft(x, axis=1)
    k = f.shape[1]
    phases = rng.uniform(0, 2 * np.pi, size=(1 if shared else x.shape[0], k))
    phases[:, 0] = 0.0  # keep the mean
    if n % 2 == 0:
        phases[:, -1] = 0.0  # the Nyquist bin must stay real
    return np.fft.irfft(f * np.exp(1j * phases), n=n, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("name")
    p.add_argument("out_root", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--digits", type=int, default=6)
    a = p.parse_args()
    t0 = time.time()
    spec = f"{{:.{a.digits}g}}".format
    summary = {"source": a.name, "seed": a.seed, "digits": a.digits}
    for split in ("train", "test"):
        gen = read_ts(a.src / f"{a.name}_{split.upper()}.ts")
        header, _ = next(gen)
        T = int(next(l for l in header if l.lower().startswith("@serieslength")).split()[1])
        half, quarter, small = T // 2, T // 4, T // 20
        rng = np.random.default_rng([a.seed, zlib.crc32(a.name.encode()), split == "test"])
        outs, cases = {}, {}
        for v in VARIANTS:
            name = f"{a.name}-{v}"
            d = a.out_root / name
            d.mkdir(parents=True, exist_ok=True)
            outs[v] = open(d / f"{name}_{split.upper()}.ts.part", "w")
            cases[v] = []
            length = half if v.startswith(("Crop", "Shift")) else T
            for line in header:
                low = line.lower()
                if low.startswith("@problemname"):
                    line = f"@problemName {name}\n"
                elif low.startswith("@serieslength"):
                    line = f"@seriesLength {length}\n"
                elif line.startswith("#"):
                    line = line.rstrip("\n") + f" Cognitive shift/surrogate variant {v}.\n"
                outs[v].write(line)
        n = 0
        for x, label in gen:
            offsets = {"Crop50": quarter,
                       "Shift50S": quarter + int(rng.integers(-small, small + 1)),
                       "Shift50L": int(rng.integers(0, T - half + 1))}
            for v, o in offsets.items():
                seg = x[:, o:o + half]
                outs[v].write(":".join(",".join(map(spec, ch.tolist())) for ch in seg) + f":{label}\n")
                cases[v].append({"case": n, "label": label, "offset": o, "length": half})
            for v, shared in (("Surr", True), ("SurrIndep", False)):
                s = surrogate(x, rng, shared)
                outs[v].write(":".join(",".join(map(spec, ch.tolist())) for ch in s) + f":{label}\n")
                cases[v].append({"case": n, "label": label, "offset": 0, "length": T})
            n += 1
        for v in VARIANTS:
            outs[v].close()
            name = f"{a.name}-{v}"
            d = a.out_root / name
            (d / f"{name}_{split.upper()}.ts.part").rename(d / f"{name}_{split.upper()}.ts")
            with open(d / f"cases_{split}.tsv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(cases[v][0]), delimiter="\t")
                w.writeheader()
                w.writerows(cases[v])
            ids = a.src / f"{a.name}_id_{split.upper()}.txt"
            if ids.exists():
                (d / f"{name}_id_{split.upper()}.txt").write_text(ids.read_text())
        summary[split] = {"cases": n, "series_length": T, "crop_length": half}
        print(f"{a.name} {split}: {n} cases, T={T}, crop {half}; {time.time() - t0:.0f} s", flush=True)
    for v in VARIANTS:
        (a.out_root / f"{a.name}-{v}" / "build_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
