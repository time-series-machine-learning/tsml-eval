"""Reorder a dataset list by real on-disk size, largest first.

Usage:
    python reorder_by_size_desc.py <dataset_file> <data_dir> [--pin-first] [-o OUTPUT]

Measures the total size of every file under ``<data_dir>/<dataset>`` for each
dataset named in ``dataset_file`` (one per line; blank lines and ``#`` comments
are preserved verbatim at the top of the output and otherwise ignored), then
rewrites the list largest-first. This is the same measurement
``multiverse_controller.py``'s ``small_datasets_first`` option uses
(``_dataset_size_bytes``), just sorted in the opposite direction and run
standalone rather than as part of a Slurm controller.

With --pin-first, the first dataset line in the input is kept at the very top
of the output regardless of its measured size (for a small sanity-check
dataset that should run before the large ones, even though it would otherwise
sort to the bottom).

Datasets whose directory cannot be found under data_dir are reported on
stderr and appended at the end of the output, unsized, rather than silently
dropped -- missing data is worth noticing, not hiding.
"""

import argparse
import sys
from pathlib import Path


def dataset_size_bytes(data_dir, dataset):
    dataset_dir = data_dir / dataset
    if not dataset_dir.is_dir():
        return None
    try:
        return sum(
            path.stat().st_size for path in dataset_dir.rglob("*") if path.is_file()
        )
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_file", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--pin-first",
        action="store_true",
        help="Keep the first dataset line at the top regardless of its size.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write here instead of overwriting dataset_file.",
    )
    args = parser.parse_args()

    header_lines = []
    datasets = []
    with args.dataset_file.open() as f:
        for line in f:
            stripped = line.rstrip("\n")
            content = stripped.strip()
            if not content or content.startswith("#"):
                if not datasets:
                    header_lines.append(stripped)
                continue
            datasets.append(content)

    if not datasets:
        print("ERROR: no datasets found in the input file.", file=sys.stderr)
        raise SystemExit(1)

    pinned = datasets[0] if args.pin_first else None
    remaining = datasets[1:] if args.pin_first else list(datasets)

    sizes = {}
    missing = []
    for dataset in remaining:
        size = dataset_size_bytes(args.data_dir, dataset)
        if size is None:
            missing.append(dataset)
        else:
            sizes[dataset] = size

    ordered = sorted(sizes, key=lambda d: sizes[d], reverse=True)

    if missing:
        print(
            f"WARNING: {len(missing)} dataset(s) not found under {args.data_dir}, "
            "appended unsized at the end:",
            file=sys.stderr,
        )
        for dataset in missing:
            print(f"  {dataset}", file=sys.stderr)

    output_path = args.output or args.dataset_file
    with output_path.open("w") as f:
        for line in header_lines:
            f.write(line + "\n")
        if header_lines:
            f.write("\n")
        if pinned is not None:
            f.write(pinned + "\n")
        for dataset in ordered:
            size_mb = sizes[dataset] / 1e6
            print(f"{size_mb:10.2f} MB  {dataset}")
            f.write(dataset + "\n")
        for dataset in missing:
            f.write(dataset + "\n")

    print(f"\nWrote {len(datasets)} datasets to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
