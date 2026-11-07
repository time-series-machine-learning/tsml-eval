"""Move EEGBench-layout results into the aeon category layout used by eeg-aeon.

EEGBench (eegbench-cpu) writes <src>/<Name>/Predictions/<Dataset>/. The EEG
experiments write <dst>/<Category>/<Name>/Predictions/<Dataset>/, grouped by aeon
module. This moves each dataset folder across, keeping estimator names, so that
eeg-aeon and eeg-hc2-components see the work as done.

    python move_eegbench_results.py                 # dry run: show what would move
    python move_eegbench_results.py --apply         # move

Defaults: --src ~/Results/EEGBench, --dst ~/Results/EEG. Run it only when no
eegbench-cpu job is running (stop the experiment first).

A dataset folder already present at the destination is left alone; if its files
match the source the source copy is removed, otherwise both are reported. R-MDM and
R-KNN duplicate DistanceBased/RiemannianMDM and RiemannianKNN, and CSP-SVM is the
superseded CSP+SVM, so those go to <dst>/_superseded/ rather than being deleted,
as does <dst>/ConvolutionBased/MultiRocketHydra (MRHydra under its old name).
"""

import argparse
import filecmp
import shutil
from pathlib import Path

CATEGORY = {
    "HC2": "Hybrid",
    "MRHydra": "ConvolutionBased",
    "ROCKET": "ConvolutionBased",
    "MiniROCKET": "ConvolutionBased",
    "Arsenal": "ConvolutionBased",
    "DrCIF": "IntervalBased",
    "QUANT": "IntervalBased",
    "STC": "ShapeletBased",
    "TDE": "DictionaryBased",
    "FreshPRINCE": "FeatureBased",
    "RotationForest": "Tabular",
    "RandomForest": "Tabular",
    "SVM": "Tabular",
    "LogisticRegression": "Tabular",
    "R-MDM": "_superseded",
    "R-KNN": "_superseded",
    "CSP-SVM": "_superseded",
}

# Folders already under the destination that duplicate a moved estimator under an
# old name: MultiRocketHydra is MRHydra, run by an earlier eeg-cpu.
SUPERSEDED_IN_DST = ["ConvolutionBased/MultiRocketHydra"]


def _predictions(path):
    """Return a results file without its header and timings.

    Keeps the parameter line, the accuracy and the predictions.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        return lines
    return [lines[1], lines[2].split(",")[0]] + lines[3:]


def same_files(a, b):
    """Whether two dataset folders hold the same results.

    Results files count as the same if their parameters, accuracy and predictions
    match; the generation time and timings in the header may differ.
    """
    names = sorted(p.name for p in a.iterdir() if p.is_file())
    if names != sorted(p.name for p in b.iterdir() if p.is_file()):
        return False
    for name in names:
        if name.endswith(".csv"):
            if _predictions(a / name) != _predictions(b / name):
                return False
        elif not filecmp.cmp(a / name, b / name, shallow=False):
            return False
    return True


def main():
    """Move or report every dataset folder under the source."""
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=home / "Results" / "EEGBench")
    parser.add_argument("--dst", type=Path, default=home / "Results" / "EEG")
    parser.add_argument("--apply", action="store_true", help="move (default: dry run)")
    args = parser.parse_args()

    moved = duplicate = conflict = 0
    for estimator_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        name = estimator_dir.name
        if name not in CATEGORY:
            print(f"skip {name}: no category (add it to CATEGORY)")
            continue
        predictions = estimator_dir / "Predictions"
        if not predictions.is_dir():
            continue
        target_root = args.dst / CATEGORY[name] / name / "Predictions"
        for dataset_dir in sorted(p for p in predictions.iterdir() if p.is_dir()):
            target = target_root / dataset_dir.name
            label = f"{name}/{dataset_dir.name}"
            if target.exists():
                if same_files(dataset_dir, target):
                    duplicate += 1
                    print(f"same     {label} (already at {target}); remove source")
                    if args.apply:
                        shutil.rmtree(dataset_dir)
                else:
                    conflict += 1
                    print(f"CONFLICT {label}: {target} differs; both kept")
                continue
            moved += 1
            print(f"move     {label} -> {target}")
            if args.apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dataset_dir), str(target))

    for old in SUPERSEDED_IN_DST:
        source = args.dst / old
        if source.is_dir():
            target = args.dst / "_superseded" / source.name
            print(f"retire   {old} -> {target}")
            if args.apply and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))

    verb = "Moved" if args.apply else "Would move"
    print(
        f"\n{verb} {moved} folder(s); {duplicate} already present and identical; "
        f"{conflict} conflict(s)."
    )
    if not args.apply:
        print("Dry run: nothing changed. Add --apply to move.")


if __name__ == "__main__":
    main()
