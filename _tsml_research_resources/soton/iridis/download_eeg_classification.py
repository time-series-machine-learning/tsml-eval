"""Download aeon-neuro EEG classification problems for tsml-eval experiments.

Run this on an Iridis node with internet access. The output dataset list can be passed
to the Iridis classification submission scripts.
"""

import argparse
from pathlib import Path


def main():
    """Download published problems and write a list of usable datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-list", type=Path, required=True)
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", help="Download selected datasets first"
    )
    args = parser.parse_args()

    from aeon_neuro.datasets.classification_datasets import dataset_map, dataset_names

    names = args.only if args.only else dataset_names
    unknown = sorted(set(names) - set(dataset_names))
    if unknown:
        parser.error(f"Unknown aeon-neuro datasets: {', '.join(unknown)}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_list.parent.mkdir(parents=True, exist_ok=True)
    ready = []
    unavailable = []
    failed = []

    for name in names:
        problem_dir = args.data_dir / name
        train = problem_dir / f"{name}_TRAIN.ts"
        test = problem_dir / f"{name}_TEST.ts"
        if train.is_file() and test.is_file():
            print(f"READY {name} (already downloaded)", flush=True)
            ready.append(name)
            continue

        if dataset_map.get(name) == 49:
            print(f"UNAVAILABLE {name} (no download record yet)", flush=True)
            unavailable.append(name)
            continue

        try:
            # The loader downloads both splits; loading only TRAIN limits memory use.
            from aeon_neuro.datasets import load_eeg_classification

            load_eeg_classification(name, split="train", extract_path=str(args.data_dir))
            if not train.is_file() or not test.is_file():
                raise FileNotFoundError(f"Missing TRAIN or TEST file in {problem_dir}")
        except Exception as exc:
            print(f"FAILED {name}: {exc}", flush=True)
            failed.append(name)
            continue

        print(f"READY {name}", flush=True)
        ready.append(name)

    args.dataset_list.write_text("".join(f"{name}\n" for name in ready))
    print(
        f"Ready: {len(ready)}; unavailable: {len(unavailable)}; failed: "
        f"{len(failed)}. Dataset list: {args.dataset_list}",
        flush=True,
    )
    if unavailable:
        print(f"Unavailable: {', '.join(unavailable)}", flush=True)
    if failed:
        print(f"Failed: {', '.join(failed)}", flush=True)
    if unavailable or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
