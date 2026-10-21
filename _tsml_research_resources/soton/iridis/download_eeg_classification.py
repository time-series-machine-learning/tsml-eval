"""Download aeon-neuro EEG classification problems for tsml-eval experiments.

Run this on an Iridis node with internet access. The output dataset list can be passed
to the Iridis classification submission scripts.
"""

import argparse
import shutil
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen


def download_id_files(name, record_id, problem_dir):
    """Fetch subject/case ID sidecars when they are present in the Zenodo record."""
    for split in ("TRAIN", "TEST"):
        filename = f"{name}_id_{split}.txt"
        target = problem_dir / filename
        if target.is_file():
            continue

        url = (
            f"https://zenodo.org/records/{record_id}/files/"
            f"{quote(filename)}?download=1"
        )
        partial = problem_dir / f"{filename}.part"
        try:
            with urlopen(url, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            partial.replace(target)
        except HTTPError as exc:
            if exc.code != 404:
                raise
        finally:
            partial.unlink(missing_ok=True)

    return all(
        (problem_dir / f"{name}_id_{split}.txt").is_file()
        for split in ("TRAIN", "TEST")
    )


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
    id_complete = []
    id_failed = []

    for name in names:
        problem_dir = args.data_dir / name
        train = problem_dir / f"{name}_TRAIN.ts"
        test = problem_dir / f"{name}_TEST.ts"
        record_id = dataset_map.get(name)
        if not (train.is_file() and test.is_file()):
            if record_id in (None, 49):
                print(f"UNAVAILABLE {name} (no download record yet)", flush=True)
                unavailable.append(name)
                continue

            try:
                # The loader downloads both splits; load only TRAIN to limit memory.
                from aeon_neuro.datasets import load_eeg_classification

                load_eeg_classification(
                    name, split="train", extract_path=str(args.data_dir)
                )
                if not train.is_file() or not test.is_file():
                    raise FileNotFoundError(
                        f"Missing TRAIN or TEST file in {problem_dir}"
                    )
            except Exception as exc:
                print(f"FAILED {name}: {exc}", flush=True)
                failed.append(name)
                continue

        if record_id not in (None, 49):
            try:
                if download_id_files(name, record_id, problem_dir):
                    id_complete.append(name)
            except Exception as exc:
                print(f"ID DOWNLOAD FAILED {name}: {exc}", flush=True)
                id_failed.append(name)

        print(f"READY {name}", flush=True)
        ready.append(name)

    args.dataset_list.write_text("".join(f"{name}\n" for name in ready))
    print(
        f"Ready: {len(ready)}; unavailable: {len(unavailable)}; failed: "
        f"{len(failed)}; paired ID files: {len(id_complete)}. "
        f"Dataset list: {args.dataset_list}",
        flush=True,
    )
    if unavailable:
        print(f"Unavailable: {', '.join(unavailable)}", flush=True)
    if failed:
        print(f"Failed: {', '.join(failed)}", flush=True)
    if id_failed:
        print(f"ID download failed: {', '.join(id_failed)}", flush=True)
    if unavailable or failed or id_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
