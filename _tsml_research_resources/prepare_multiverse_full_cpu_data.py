"""Prepare locally cached clean Multiverse data for the Hali CPU pass.

The Multiverse experiment entry point deliberately requests equal-length and
no-missing-value data. This script downloads every file attached to a missing
Zenodo record on the login node, verifies the exact clean variant named by the
archive list, and writes base problem names for the Slurm controller.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from aeon.datasets._data_loaders import _download_all_zenodo_files
from aeon.datasets.tsc_datasets import tsc_zenodo

_CLEAN_SUFFIXES = ("_eq_nmv", "_eq", "_nmv")


def _base_name(clean_name: str) -> str:
    for suffix in _CLEAN_SUFFIXES:
        if clean_name.endswith(suffix):
            return clean_name[: -len(suffix)]
    return clean_name


def _read_names(path: Path) -> tuple[str, ...]:
    names = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Dataset list is empty or contains duplicates: {path}")
    return names


def _required_files(data_dir: Path, clean_name: str) -> tuple[Path, ...]:
    base = _base_name(clean_name)
    directory = data_dir / base
    names = {base, clean_name}
    return tuple(
        directory / f"{name}_{split}.ts"
        for name in names
        for split in ("TRAIN", "TEST")
    )


def _is_ready(data_dir: Path, clean_name: str) -> bool:
    try:
        return all(
            path.is_file() and path.stat().st_size > 0
            for path in _required_files(data_dir, clean_name)
        )
    except OSError:
        return False


def _download_record(data_dir: Path, clean_name: str) -> None:
    base = _base_name(clean_name)
    if base not in tsc_zenodo:
        raise ValueError(f"{base} has no classification Zenodo record in aeon")

    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{base}-", dir=data_dir) as temporary:
        temporary_path = Path(temporary)
        downloaded = _download_all_zenodo_files(
            tsc_zenodo[base], os.fspath(temporary_path)
        )
        expected_names = {path.name for path in _required_files(data_dir, clean_name)}
        missing = expected_names - downloaded
        if missing:
            raise ValueError(
                f"Zenodo record for {base} lacks required files: {sorted(missing)}"
            )

        target = data_dir / base
        target.mkdir(parents=True, exist_ok=True)
        for source in temporary_path.iterdir():
            if source.is_file():
                os.replace(source, target / source.name)

    if not _is_ready(data_dir, clean_name):
        raise ValueError(f"Downloaded files for {base} did not pass verification")


def _write_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{value}\n" for value in values), encoding="utf-8", newline="\n"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--available", type=Path, required=True)
    parser.add_argument("--unavailable", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    available: list[str] = []
    unavailable: list[str] = []
    excluded: list[str] = []
    clean_names = _read_names(args.source)

    for position, clean_name in enumerate(clean_names, start=1):
        base = _base_name(clean_name)
        if (
            base == "LenDB"
            or base.startswith("DREAM")
            or base.startswith("S2Agri-")
        ):
            excluded.append(base)
            print(f"[{position}/{len(clean_names)}] excluded: {base}", flush=True)
            continue

        if not _is_ready(args.data_dir, clean_name) and not args.no_download:
            print(f"[{position}/{len(clean_names)}] downloading: {base}", flush=True)
            try:
                _download_record(args.data_dir, clean_name)
            except Exception as error:
                print(f"  FAILED: {type(error).__name__}: {error}", flush=True)

        if _is_ready(args.data_dir, clean_name):
            available.append(base)
            print(f"[{position}/{len(clean_names)}] ready: {base}", flush=True)
        else:
            unavailable.append(base)
            print(f"[{position}/{len(clean_names)}] unavailable: {base}", flush=True)

    _write_list(args.available, available)
    _write_list(args.unavailable, unavailable)
    _write_list(args.excluded, excluded)

    print(f"Archive datasets:   {len(clean_names)}")
    print(f"Ready for CPU pass: {len(available)}")
    print(f"Unavailable:        {len(unavailable)}")
    print(f"Explicitly excluded:{len(excluded):>3}")
    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())
