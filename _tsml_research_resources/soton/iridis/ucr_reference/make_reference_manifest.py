"""Build a reference manifest by inventorying a local results tree.

The controller runs only what a reference tree does not already hold. That tree
is normally the D drive, which is not visible from Iridis, so the inventory is
taken here and checked in as a manifest: a list of the
``classifier|dataset|resample`` keys that are already complete, and the
complement that still has to run. It records metadata only, never predictions.

The manifest was previously produced by hand, which is why the README could
only say "regenerate the manifest" without naming a tool. This is that tool, and
it is deliberately generic: it reads the same configuration the controller does,
so the universe it enumerates is the same one the controller validates against.

A result counts as complete when its prediction file exists and is nonempty,
matching the controller's own rule. Classifiers with ``train`` set need both the
test and train file for a resample to count.

Usage:

    python make_reference_manifest.py --config multiverse_reference.json \\
        --source D:/Results/Multiverse --out multiverse_reference_manifest.json

    python make_reference_manifest.py --config ucr_reference.json \\
        --source D:/Results/UCR --out ucr_reference_manifest.json

Add ``--check`` to compare against an existing manifest and report what moved,
rather than writing. That is the safe thing to run after a sync from Iridis,
before deciding whether a rerun is needed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCHEMA = 1
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def read_json(path: Path):
    """Return parsed JSON from a path."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def datasets(config, config_path: Path) -> list[str]:
    """Read the configured dataset list, preserving its order.

    The configured path is the one the cluster will use, so it is resolved
    against this checkout when that absolute path does not exist locally.
    """
    raw = config["dataset_list"].format(
        username=config.get("username", ""), repo_dir=config.get("repo_dir", "")
    )
    path = Path(raw)
    if not path.is_file():
        marker = "_tsml_research_resources"
        if marker in raw:
            root = config_path.resolve().parents[4]
            path = root / raw[raw.index(marker):]
    if not path.is_file():
        raise FileNotFoundError(f"Dataset list not found: {raw}")
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def resamples_present(directory: Path, split: str) -> set[int]:
    """Return the resample ids with a nonempty prediction file for one split."""
    found = set()
    if not directory.is_dir():
        return found
    for entry in directory.iterdir():
        match = re.fullmatch(rf"{split}Resample(\d+)\.csv", entry.name)
        if match and entry.is_file() and entry.stat().st_size > 0:
            found.add(int(match.group(1)))
    return found


def inventory(config, names: list[str], source: Path):
    """Return the complete keys and a per-classifier count summary."""
    complete: set[str] = set()
    counts = []
    wanted = range(config["resamples"])
    for row in config["classifiers"]:
        directory = source / row["category"] / row["name"] / "Predictions"
        tests = trains = 0
        full_datasets = 0
        for dataset in names:
            here = directory / dataset
            test = resamples_present(here, "test")
            train = resamples_present(here, "train") if row["train"] else set()
            tests += len(test)
            trains += len(train)
            done = (test & train) if row["train"] else test
            done = {i for i in done if i in wanted}
            if len(done) == len(wanted):
                full_datasets += 1
            complete.update(f"{row['name']}|{dataset}|{i}" for i in done)
        counts.append(
            {
                "name": row["name"],
                "category": row["category"],
                "device": row["device"],
                "test": tests,
                "train": trains,
                "complete": sum(
                    1 for k in complete if k.startswith(f"{row['name']}|")
                ),
                "datasets": full_datasets,
            }
        )
    return complete, counts


def build(config, config_path: Path, source: Path):
    """Return a manifest dictionary for this configuration and results tree."""
    names = datasets(config, config_path)
    for value in names + [r["name"] for r in config["classifiers"]]:
        if not SAFE.fullmatch(value):
            raise ValueError(f"Unsafe name in the universe: {value!r}")

    expected = {
        f"{row['name']}|{dataset}|{i}"
        for row in config["classifiers"]
        for dataset in names
        for i in range(config["resamples"])
    }
    complete, counts = inventory(config, names, source)
    stray = complete - expected
    if stray:
        raise ValueError(
            f"{len(stray)} complete keys are outside the configured universe, "
            f"for example {sorted(stray)[:3]}"
        )
    return {
        "schema": SCHEMA,
        "source": str(source).replace("\\", "/"),
        "dataset_list": Path(config["dataset_list"]).name,
        "resamples": config["resamples"],
        "reference_complete": sorted(complete),
        "target_tasks": sorted(expected - complete),
        "total_tasks": len(expected),
        "classifier_counts": counts,
        "required_splits": sorted(
            {"test/train" if r["train"] else "test" for r in config["classifiers"]}
        ),
    }


def report(manifest, previous=None) -> None:
    """Print the summary, and what changed if there is a manifest to compare."""
    total = manifest["total_tasks"]
    done = len(manifest["reference_complete"])
    print(f"source     {manifest['source']}")
    print(f"datasets   {manifest['dataset_list']}")
    print(f"resamples  {manifest['resamples']}")
    print(f"complete   {done} of {total} ({100 * done / total:.1f}%)")
    print(f"missing    {len(manifest['target_tasks'])}\n")
    width = max(len(row["name"]) for row in manifest["classifier_counts"])
    per = total // len(manifest["classifier_counts"])
    for row in sorted(manifest["classifier_counts"], key=lambda r: -r["complete"]):
        print(
            f"  {row['name']:{width}s} {row['device']:3s} "
            f"{row['complete']:5d}/{per} complete"
        )
    if previous is None:
        return
    gained = set(manifest["reference_complete"]) - set(previous["reference_complete"])
    lost = set(previous["reference_complete"]) - set(manifest["reference_complete"])
    print(f"\nsince the checked-in manifest: +{len(gained)} complete, -{len(lost)}")
    for key in sorted(gained)[:20]:
        print(f"  + {key}")
    if len(gained) > 20:
        print(f"  ... and {len(gained) - 20} more")
    for key in sorted(lost)[:20]:
        print(f"  - {key}")


def main(argv=None) -> int:
    """Build or check a reference manifest."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "ucr_reference.json")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Results tree to inventory, for example D:/Results/Multiverse",
    )
    parser.add_argument("--out", type=Path, help="Manifest to write; defaults to the "
                        "reference_manifest named in the configuration")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report against the existing manifest without writing",
    )
    args = parser.parse_args(argv)

    config = read_json(args.config)
    if not args.source.is_dir():
        raise SystemExit(f"Results tree not found: {args.source}")

    out = args.out or (here / Path(config["reference_manifest"]).name)
    manifest = build(config, args.config, args.source)
    previous = read_json(out) if out.is_file() else None
    report(manifest, previous if args.check else None)

    if args.check:
        print("\n--check: nothing written")
        return 0
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
