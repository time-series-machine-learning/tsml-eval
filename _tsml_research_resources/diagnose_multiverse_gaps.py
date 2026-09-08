"""Quantify what is missing from a results tree, and why.

Two questions come up whenever a long run stops short. Which experiments are
missing, and is rerunning them going to help? This answers both: it counts the
gap against a dataset list at a given number of resamples, and then reads the job
output log of every missing experiment to classify what happened to it.

The distinction that matters is between work that failed and work that never
started. An out-of-memory kill escalates a memory tier on the next attempt and
usually succeeds; a Python traceback reproduces on the same seed and needs a fix;
an experiment with no log at all was simply never reached before its round ended,
and will run on the next invocation with no intervention.

A worked example. PULSAR on the multiverse core was missing 164 of 1980
experiments, spread over 31 problems with none untouched, which looked like a
systematic per-problem failure. The logs showed 118 out-of-memory kills at the
4 and 8 GiB opening tiers, 35 experiments never started, and 10 node-level
resource exhaustions reported as thread and mmap failures rather than as memory.
Nothing was wrong with the estimator; the run had simply stopped after one round,
so nothing escalated.

Usage::

    python diagnose_multiverse_gaps.py --results-root ~/Results/Multiverse/IntervalBased
    python diagnose_multiverse_gaps.py --results-root ... --estimator PULSAR --why
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

DEFAULT_LIST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset_lists",
    "MultivariateClassification65-MultiverseMini.txt",
)

#: Job logs are named ``output-<dataset>-<resample>-<timestamp>-<pid>-<label>.txt``.
#: Dataset names contain both hyphens and digits, so the resample number can only
#: be identified by anchoring the 14-digit timestamp that follows it; a looser
#: pattern lets the dataset group swallow the resample and silently reports every
#: experiment as having no log.
LOG_NAME = re.compile(r"output-(.+)-(\d+)-(\d{14})-")

OOM = re.compile(
    r"out[ -]?of[ -]?memory|OUT_OF_MEMORY|oom[_-]kill|Killed process|MemoryError"
    r"|Cannot allocate memory|std::bad_alloc|Unable to allocate"
    r"|failed to map segment|can't start new thread",
    re.I,
)
ERROR = re.compile(
    r"Traceback \(most recent call last\)|Segmentation fault|slurmstepd: error:"
    r"|CANCELLED|DUE TO TIME LIMIT",
    re.I,
)


def read_list(path):
    """Dataset names, one per line; blank lines and # comments are skipped."""
    with open(path, encoding="utf-8") as fh:
        names = (line.strip() for line in fh)
        return sorted(n for n in names if n and not n.startswith("#"))


def completed(results_root, estimator, datasets, resamples):
    """The (dataset, resample) pairs with a non-empty test result file."""
    predictions = os.path.join(results_root, estimator, "Predictions")
    found = set()
    if not os.path.isdir(predictions):
        return found
    wanted = set(datasets)
    for dataset in os.listdir(predictions):
        if dataset not in wanted:
            continue
        folder = os.path.join(predictions, dataset)
        for name in os.listdir(folder):
            m = re.fullmatch(r"testResample(\d+)\.csv", name)
            if not m:
                continue
            index = int(m.group(1))
            if index < resamples and os.path.getsize(os.path.join(folder, name)) > 0:
                found.add((dataset, index))
    return found


def gap_report(results_root, estimators, datasets, resamples):
    target = len(datasets) * resamples
    print("%d problems x %d resamples = %d per estimator\n"
          % (len(datasets), resamples, target))
    print("%-20s %12s %9s %10s %9s" % (
        "ESTIMATOR", "COMPLETE", "MISSING", "UNTOUCHED", "PARTIAL"))
    print("-" * 64)
    detail = {}
    for estimator in estimators:
        have = completed(results_root, estimator, datasets, resamples)
        untouched, partial = [], []
        for dataset in datasets:
            n = sum(1 for r in range(resamples) if (dataset, r) in have)
            if n == 0:
                untouched.append(dataset)
            elif n < resamples:
                partial.append((dataset, n))
        detail[estimator] = (have, untouched, partial)
        print("%-20s %5d/%-6d %9d %10d %9d" % (
            estimator, len(have), target, target - len(have),
            len(untouched), len(partial)))
    return detail


def why_report(results_root, estimator, datasets, resamples, show=3):
    """Classify every missing experiment from its job output log."""
    have = completed(results_root, estimator, datasets, resamples)
    missing = [(d, r) for d in datasets for r in range(resamples)
               if (d, r) not in have]
    print("\n%s: %d missing experiments" % (estimator, len(missing)))
    if not missing:
        return

    out_dir = os.path.join(results_root, "output", estimator)
    logs = defaultdict(list)
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            m = LOG_NAME.match(name)
            if m:
                logs[(m.group(1), int(m.group(2)))].append(
                    os.path.join(out_dir, name))
    else:
        print("  no output directory at %s" % out_dir)

    verdicts = Counter()
    examples = defaultdict(list)
    tiers = Counter()
    for key in missing:
        paths = logs.get(key, [])
        if not paths:
            verdicts["never started"] += 1
            continue
        path = max(paths, key=os.path.getmtime)
        tier = re.search(r"-(mem\d+)-", os.path.basename(path))
        if tier:
            tiers[tier.group(1)] += 1
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if not text.strip():
            verdict = "started, log empty"
        elif OOM.search(text):
            verdict = "out of memory"
        elif ERROR.search(text):
            verdict = "error or cancelled"
        else:
            verdict = "ran, no result, no error"
        verdicts[verdict] += 1
        examples[verdict].append((key, path))

    for verdict, n in verdicts.most_common():
        print("  %-26s %4d" % (verdict, n))
    if tiers:
        print("  memory tiers attempted: %s"
              % ", ".join("%s x%d" % kv for kv in sorted(tiers.items())))

    print("\n  rerunning will help for: never started, out of memory")
    print("  rerunning will not help for: error or cancelled, unless the cause "
          "was the allocation rather than the code")

    for verdict in ("error or cancelled", "ran, no result, no error"):
        for key, path in examples.get(verdict, [])[:show]:
            print("\n  %s, %s resample %d" % (verdict, key[0], key[1]))
            with open(path, encoding="utf-8", errors="replace") as fh:
                lines = [l for l in fh.read().split("\n") if l.strip()]
            for line in lines[-8:]:
                print("      " + line[:140])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True,
                        help="directory holding <estimator>/Predictions")
    parser.add_argument("--dataset-list", default=DEFAULT_LIST)
    parser.add_argument("--resamples", type=int, default=30)
    parser.add_argument("--estimator", action="append", default=None,
                        help="restrict to these; default is every directory")
    parser.add_argument("--why", action="store_true",
                        help="also classify the missing experiments from their logs")
    args = parser.parse_args(argv)

    datasets = read_list(args.dataset_list)
    estimators = args.estimator or sorted(
        d for d in os.listdir(args.results_root)
        if os.path.isdir(os.path.join(args.results_root, d, "Predictions"))
    )

    detail = gap_report(args.results_root, estimators, datasets, args.resamples)

    for estimator in estimators:
        have, untouched, partial = detail[estimator]
        if untouched:
            print("\n%s never attempted (%d problems): %s"
                  % (estimator, len(untouched), " ".join(untouched)))
        if partial:
            print("\n%s part done (%d problems): %s"
                  % (estimator, len(partial),
                     " ".join("%s %d/%d" % (d, n, args.resamples)
                              for d, n in partial)))

    if args.why:
        for estimator in estimators:
            why_report(args.results_root, estimator, datasets, args.resamples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
