"""Build HC2 results from its components' results files, without the data.

HC2's CAWPE weights come from each component's train estimate, so with train and
test files for Arsenal, DrCIF, STC and TDE the ensemble can be formed from the files
alone: no data is loaded and nothing is refit. Results are written in the usual
layout, by default to

    <results>/Hybrid/HC2FromFile/Predictions/<Dataset>/{train,test}Resample<i>.csv

A dataset is built for every resample that all four components have both files for;
the rest are listed as incomplete, so run this again as the components finish.
Datasets already built are skipped unless --overwrite is given. Run it on a login
node from this directory, in the tsml-eval env:

    python build_hc2_from_files.py               # build what is complete
    python build_hc2_from_files.py --dry-run     # only report what would be built
"""

import argparse
import re
import sys
from pathlib import Path

from tsml_eval.estimators.classification.hybrid.hivecote_from_results import (
    build_hivecote_from_results,
)

# the layout of iridis6_eeg_hc2_components.toml
COMPONENTS = [
    "ConvolutionBased/Arsenal",
    "IntervalBased/DrCIF",
    "ShapeletBased/STC-aeon",
    "DictionaryBased/TDE",
]
_RESAMPLE = re.compile(r"^(?:train|test)Resample(\d+)\.csv$")


def resamples(directory, split):
    """Resample ids with a results file for the split in a Predictions directory."""
    if not directory.is_dir():
        return set()
    found = set()
    for f in directory.glob(f"{split}Resample*.csv"):
        m = _RESAMPLE.match(f.name)
        if m and f.stat().st_size > 0:
            found.add(int(m.group(1)))
    return found


def main(argv=None):
    """Build every HC2 the component files allow and report the rest."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results",
        default=str(Path.home() / "Results" / "EEG"),
        help="results directory holding the components (default ~/Results/EEG)",
    )
    parser.add_argument("--name", default="HC2FromFile", help="classifier name")
    parser.add_argument(
        "--category", default="Hybrid", help="directory under --results to write to"
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=COMPONENTS,
        help="component directories relative to --results",
    )
    parser.add_argument(
        "--datasets", help="file of dataset names, one per line (default: all found)"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    results = Path(args.results).expanduser()
    components = [results / c for c in args.components]
    output = results / args.category
    for c in components:
        if not (c / "Predictions").is_dir():
            sys.exit(f"No Predictions directory in {c}")

    if args.datasets:
        lines = Path(args.datasets).expanduser().read_text().split()
        datasets = [d for d in lines if d and not d.startswith("#")]
    else:
        datasets = sorted(
            {d.name for c in components for d in (c / "Predictions").iterdir()}
        )

    built, existing, incomplete, failed = [], [], [], []
    for dataset in datasets:
        have = {}
        for c in components:
            p = c / "Predictions" / dataset
            have[c.name] = resamples(p, "train") & resamples(p, "test")
        ready = set.intersection(*have.values())
        if not ready:
            missing = [n for n, r in have.items() if not r]
            incomplete.append(
                f"{dataset}: no train+test files for {', '.join(missing)}"
            )
            continue
        for i in sorted(ready):
            out = output / args.name / "Predictions" / dataset / f"testResample{i}.csv"
            if out.exists() and not args.overwrite:
                existing.append(f"{dataset} {i}")
                continue
            if args.dry_run:
                built.append(f"{dataset} {i}")
                continue
            try:
                build_hivecote_from_results(
                    [str(c) for c in components],
                    dataset,
                    i,
                    str(output),
                    classifier_name=args.name,
                    write_train_file=True,
                )
                built.append(f"{dataset} {i}")
                print(f"built {dataset} resample {i}", flush=True)
            except Exception as err:  # noqa: BLE001, report and carry on
                failed.append(f"{dataset} {i}: {type(err).__name__}: {err}")

    verb = "would build" if args.dry_run else "built"
    print(
        f"\n{len(built)} {verb}, {len(existing)} already built, "
        f"{len(incomplete)} datasets incomplete, {len(failed)} failed"
    )
    print(f"output {output / args.name / 'Predictions'}")
    for line in incomplete:
        print(f"  incomplete  {line}")
    for line in failed:
        print(f"  FAILED      {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
