"""Compare one estimator with every other one, pairwise, from results files.

For each other estimator, the datasets both have test files for are compared on
test accuracy and balanced accuracy, averaged over the resamples both have. Reports
wins, draws and losses for the reference estimator, the mean difference, and a
Wilcoxon signed-rank p value. Reads the layout pulled by pull_results.sh:

    <results>/<Category>/<Estimator>/Predictions/<Dataset>/testResample<i>.csv

    python head_to_head.py Hybrid/HC2FromFile
    python head_to_head.py Hybrid/HC2FromFile --results C:/Results/EEG --csv h2h.csv
    python head_to_head.py Hybrid/HC2FromFile --only CTNet-NB TSLR-NB         --datasets ../dataset_lists/eeg_motor_imagery.txt

Folders whose name starts with _ (e.g. _superseded) are skipped.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score

_TEST = re.compile(r"^testResample(\d+)\.csv$")


def scores(file):
    """Accuracy and balanced accuracy from the label and prediction columns."""
    rows = [line.split(",", 2) for line in file.read_text().splitlines()[3:]]
    y = np.array([int(r[0]) for r in rows if len(r) > 1])
    p = np.array([int(r[1]) for r in rows if len(r) > 1])
    if len(y) == 0:
        return None
    return float(np.mean(y == p)), float(balanced_accuracy_score(y, p))


def load(estimator_dir):
    """{dataset: {resample: (accuracy, balanced accuracy)}} for one estimator."""
    out = {}
    pred = estimator_dir / "Predictions"
    for d in sorted(pred.iterdir()) if pred.is_dir() else []:
        for f in d.glob("testResample*.csv"):
            m = _TEST.match(f.name)
            if m and f.stat().st_size > 0:
                found = scores(f)
                if found is not None:  # skip a file with no case rows
                    out.setdefault(d.name, {})[int(m.group(1))] = found
    return out


def compare(ref, other):
    """Mean score per shared dataset, over the resamples both have."""
    a, b = [], []
    for dataset in sorted(set(ref) & set(other)):
        shared = sorted(set(ref[dataset]) & set(other[dataset]))
        if shared:
            a.append(np.mean([ref[dataset][i] for i in shared], axis=0))
            b.append(np.mean([other[dataset][i] for i in shared], axis=0))
    return np.array(a), np.array(b)


def main(argv=None):
    """Print the pairwise table for the reference estimator."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference", help="<Category>/<Estimator> under --results")
    parser.add_argument("--results", default=str(Path.home() / "Results" / "EEG"))
    parser.add_argument("--csv", help="also write the table to this file")
    parser.add_argument(
        "--min-datasets", type=int, default=5, help="skip pairs with fewer shared"
    )
    parser.add_argument(
        "--datasets", help="file of dataset names, one per line: compare only these"
    )
    parser.add_argument(
        "--only", nargs="+", help="estimator names (no category) to compare with"
    )
    args = parser.parse_args(argv)

    results = Path(args.results).expanduser()
    keep = None
    if args.datasets:
        keep = set(Path(args.datasets).expanduser().read_text().split())

    def load_kept(estimator_dir):
        found = load(estimator_dir)
        return found if keep is None else {d: r for d, r in found.items() if d in keep}

    ref = load_kept(results / args.reference)
    if not ref:
        sys.exit(f"No test files for {args.reference} in {results}")
    print(f"{args.reference}: {len(ref)} datasets\n")

    rows = []
    for est in sorted(results.glob("*/*/Predictions")):
        name = f"{est.parent.parent.name}/{est.parent.name}"
        if name == args.reference or any(p.startswith("_") for p in name.split("/")):
            continue
        if args.only and est.parent.name not in args.only:
            continue
        a, b = compare(ref, load_kept(est.parent))
        if len(a) < args.min_datasets:
            continue
        row = [name, len(a)]
        for k in range(2):  # accuracy, balanced accuracy
            diff = a[:, k] - b[:, k]
            w, d, lo = (diff > 1e-9).sum(), (abs(diff) <= 1e-9).sum(), (diff < -1e-9).sum()
            p = wilcoxon(diff).pvalue if np.any(abs(diff) > 1e-9) else 1.0
            row += [f"{w}/{d}/{lo}", diff.mean(), p]
        rows.append(row)

    header = ["vs", "n", "acc W/D/L", "acc diff", "acc p", "bal W/D/L", "bal diff",
              "bal p"]
    print(f"{header[0]:<36}{header[1]:>4}  {header[2]:>10}{header[3]:>9}{header[4]:>8}"
          f"  {header[5]:>10}{header[6]:>9}{header[7]:>8}")
    for r in sorted(rows, key=lambda r: r[3]):
        print(f"{r[0]:<36}{r[1]:>4}  {r[2]:>10}{r[3]:>+9.4f}{r[4]:>8.3f}"
              f"  {r[5]:>10}{r[6]:>+9.4f}{r[7]:>8.3f}")
    print("\nW/D/L and diff are for the reference; p is Wilcoxon signed-rank.")
    if args.csv:
        with open(args.csv, "w") as f:
            f.write(",".join(header) + "\n")
            f.writelines(",".join(str(x) for x in r) + "\n" for r in rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
