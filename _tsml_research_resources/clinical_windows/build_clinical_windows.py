"""Build longer-window, 256 Hz variants of segmented Clinical EEG problems.

Clinical window-length case study (24 Nov 2026). Seventeen Clinical problems were made by
cutting continuous recordings into back-to-back 4 s windows. Each window's recording and
start sample are in cases_<split>.tsv, row i of which is line i of the .ts file (checked
here with each row's signal_sha256). This script joins the windows of each contiguous run
(same recording and label, each window starting where the previous one ended), resamples
the run to 256 Hz with a polyphase anti-aliasing filter (scipy resample_poly), and cuts it
into non-overlapping windows of each requested length, dropping the remainder. Runs shorter
than a length give no windows of that length. With --full, each run is also kept whole,
cut to the shortest run in the problem, so there is one series per run (per subject when a
subject has one run).

The train/test split is the original one. For each length L the output is a new problem
<Name>-256Hz-W<L>s (and <Name>-256Hz-Full), with _TRAIN.ts, _TEST.ts, _id_TRAIN.txt and
_id_TEST.txt (participant per case, in file order) and cases_<split>.tsv (participant,
recording, run start and the window's offset in the run, in seconds).

Usage: python build_clinical_windows.py SRC_DIR NAME OUT_ROOT [--lengths 4 10 30 120]
                                        [--full] [--sfreq 256] [--digits 4]
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def read_ts(path):
    """Yield (header lines, None) once, then (array channels x time, label) per case."""
    with open(path) as f:
        header = []
        for line in f:
            header.append(line)
            if line.lower().startswith("@data"):
                break
        yield header, None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            x = np.array([np.array(p.split(","), dtype=float) for p in parts[:-1]])
            yield x, parts[-1]


def native_rate(row):
    n = int(row["end_sample"]) - int(row["start_sample"])
    seconds = float(row["end_time_seconds"]) - float(row["start_time_seconds"])
    return round(n / seconds)


def fmt(run, digits):
    """Each channel of a run as a list of strings, formatted once and reused."""
    spec = f"{{:.{digits}g}}".format
    return [list(map(spec, ch.tolist())) for ch in run]


class Writer:
    def __init__(self, out_root, name, tag, split, header, n_channels, length, sfreq):
        self.name = f"{name}-{int(sfreq)}Hz-{tag}"
        self.dir = Path(out_root) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.ts = open(self.dir / f"{self.name}_{split.upper()}.ts.part", "w")
        self.ids = []
        self.cases = []
        self.labels = set()
        self.length = length
        for line in header:
            low = line.lower()
            if low.startswith("@problemname"):
                line = f"@problemName {self.name}\n"
            elif low.startswith("@serieslength"):
                line = f"@seriesLength {length}\n"
            elif low.startswith("@dimensions"):
                line = f"@dimensions {n_channels}\n"
            elif line.startswith("#") and not self.cases:
                line = line.rstrip("\n") + (
                    f" Window-length variant: contiguous 4 s windows joined, resampled "
                    f"to {int(sfreq)} Hz, cut into {tag} windows.\n"
                )
            self.ts.write(line)

    def add(self, strs, a, b, label, participant, recording, run_start_s, offset_s):
        self.ts.write(":".join(",".join(ch[a:b]) for ch in strs) + f":{label}\n")
        self.ids.append(participant)
        self.labels.add(label)
        self.cases.append(
            dict(participant_id=participant, recording_id=recording,
                 run_start_seconds=run_start_s, offset_seconds=offset_s, label=label)
        )

    def close(self):
        self.ts.close()
        part = self.dir / f"{self.name}_{self.split.upper()}.ts.part"
        part.rename(self.dir / f"{self.name}_{self.split.upper()}.ts")
        (self.dir / f"{self.name}_id_{self.split.upper()}.txt").write_text(
            "\n".join(self.ids) + "\n"
        )
        with open(self.dir / f"cases_{self.split}.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.cases[0]), delimiter="\t")
            w.writeheader()
            w.writerows(self.cases)
        return len(self.ids)


def runs(src, name, split):
    """Yield (header, run array at native rate, rows of the run)."""
    rows = list(csv.DictReader(open(src / f"cases_{split}.tsv"), delimiter="\t"))
    reader = read_ts(src / f"{name}_{split.upper()}.ts")
    header, _ = next(reader)
    yield header, None, None
    cur, cur_rows = [], []
    for i, (x, label) in enumerate(reader):
        row = rows[i]
        h = hashlib.sha256(np.ascontiguousarray(x, dtype="<f8").tobytes()).hexdigest()
        if h != row["signal_sha256"]:
            sys.exit(f"{name} {split} line {i}: hash differs from cases row {i}")
        if label != str(row["label"]):
            sys.exit(f"{name} {split} line {i}: label {label} vs cases {row['label']}")
        if cur_rows:
            last = cur_rows[-1]
            joined = (row["recording_id"] == last["recording_id"]
                      and row["label"] == last["label"]
                      and row["participant_id"] == last["participant_id"]
                      and int(row["start_sample"]) == int(last["end_sample"]))
            if not joined:
                yield header, np.concatenate(cur, axis=1), cur_rows
                cur, cur_rows = [], []
        cur.append(x)
        cur_rows.append(row)
    if len(rows) != i + 1:
        sys.exit(f"{name} {split}: {i + 1} cases in .ts but {len(rows)} rows in cases")
    if cur:
        yield header, np.concatenate(cur, axis=1), cur_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("name")
    p.add_argument("out_root", type=Path)
    p.add_argument("--lengths", type=int, nargs="+", default=[4, 10, 30, 120])
    p.add_argument("--full", action="store_true")
    p.add_argument("--sfreq", type=int, default=256)
    p.add_argument("--digits", type=int, default=4)
    a = p.parse_args()
    t0 = time.time()
    rows = list(csv.DictReader(open(a.src / "cases_train.tsv"), delimiter="\t"))
    native = native_rate(rows[0])
    ratio = Fraction(a.sfreq, native)
    print(f"{a.name}: native {native} Hz -> {a.sfreq} Hz (up {ratio.numerator}, "
          f"down {ratio.denominator}), lengths {a.lengths} s, full {a.full}", flush=True)

    # shortest run over both splits, for --full (window counts from the case tables)
    full_len = None
    if a.full:
        shortest = None
        for split in ("train", "test"):
            rs = list(csv.DictReader(open(a.src / f"cases_{split}.tsv"), delimiter="\t"))
            n = 0
            for i, r in enumerate(rs):
                prev = rs[i - 1] if i else None
                if prev and (r["recording_id"] == prev["recording_id"]
                             and r["label"] == prev["label"]
                             and int(r["start_sample"]) == int(prev["end_sample"])):
                    n += int(r["end_sample"]) - int(r["start_sample"])
                else:
                    if n:
                        shortest = n if shortest is None else min(shortest, n)
                    n = int(r["end_sample"]) - int(r["start_sample"])
            shortest = n if shortest is None else min(shortest, n)
        full_len = int(shortest * ratio)
        print(f"  full: shortest run {shortest / native:.1f} s -> {full_len} points", flush=True)

    summary = {"source": a.name, "native_sfreq": native, "sfreq": a.sfreq,
               "lengths_s": a.lengths, "full_points": full_len, "digits": a.digits}
    for split in ("train", "test"):
        gen = runs(a.src, a.name, split)
        header, _, _ = next(gen)
        writers, n_runs, short = None, 0, 0
        for _, run, run_rows in gen:
            n_runs += 1
            x = run if ratio == 1 else resample_poly(run, ratio.numerator, ratio.denominator, axis=1)
            if writers is None:
                specs = [(f"W{L}s", L * a.sfreq) for L in a.lengths]
                if full_len:
                    specs.append(("Full", full_len))
                writers = [(Writer(a.out_root, a.name, tag, split, header, x.shape[0], n,
                                   a.sfreq), n) for tag, n in specs]
            strs = fmt(x, a.digits)
            first = run_rows[0]
            start_s = float(first["start_time_seconds"])
            label = str(first["label"])
            for w, n in writers:
                k = x.shape[1] // n
                if k == 0:
                    short += 1
                for j in range(k):
                    w.add(strs, j * n, (j + 1) * n, label, first["participant_id"],
                          first["recording_id"], start_s, j * n / a.sfreq)
        counts = {w.name: w.close() for w, _ in writers}
        summary[split] = {"runs": n_runs, "cases": counts}
        print(f"  {split}: {n_runs} runs; cases {counts}; {time.time() - t0:.0f} s", flush=True)
    for w, _ in writers:
        (w.dir / "build_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
