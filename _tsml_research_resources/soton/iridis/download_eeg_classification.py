"""Download aeon-neuro EEG classification problems for tsml-eval experiments.

Run this on an Iridis node with internet access. The output dataset list can be passed
to the Iridis classification submission scripts.
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    """Download published problems and write a list of usable datasets."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
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

    for name in names:
        problem_dir = args.data_dir / name
        train = problem_dir / f"{name}_TRAIN.ts"
        test = problem_dir / f"{name}_TEST.ts"
        record_id = dataset_map.get(name)
        if not (train.is_file() and test.is_file()):
            if record_id in (None, 49):
                logger.info("UNAVAILABLE %s (no download record yet)", name)
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
                logger.info("FAILED %s: %s", name, exc)
                failed.append(name)
                continue

        if all(
            (problem_dir / f"{name}_id_{split}.txt").is_file()
            for split in ("TRAIN", "TEST")
        ):
            id_complete.append(name)

        logger.info("READY %s", name)
        ready.append(name)

    args.dataset_list.write_text("".join(f"{name}\n" for name in ready))
    logger.info(
        "Ready: %s; unavailable: %s; failed: %s; paired ID files: %s. "
        "Dataset list: %s",
        len(ready),
        len(unavailable),
        len(failed),
        len(id_complete),
        args.dataset_list,
    )
    if unavailable:
        logger.info("Unavailable: %s", ", ".join(unavailable))
    if failed:
        logger.info("Failed: %s", ", ".join(failed))
    if unavailable or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
