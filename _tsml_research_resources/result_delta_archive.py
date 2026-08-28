#!/usr/bin/env python3
"""Create a ZIP containing prediction results absent from a local snapshot.

The cluster cannot inspect a workstation drive directly, so the workflow has
two stages:

1. Create a small JSON manifest from the local result tree.
2. Copy that manifest to the cluster and archive only missing/changed results.

Only standard tsml-eval prediction files are considered:

    .../Predictions/<dataset>/{test,train}Resample<N>.csv

Output logs, controller state, existing archives and other CSV files are
deliberately ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


_FORMAT = "tsml-eval-result-manifest-v1"
_RESULT_NAME = re.compile(r"^(?:test|train)Resample\d+\.csv$")


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalise_relative_path(path):
    """Return a platform-independent archive/manifest path."""
    return path.as_posix()


def _is_prediction_result(relative_path):
    parts = relative_path.parts
    return "Predictions" in parts and bool(_RESULT_NAME.fullmatch(relative_path.name))


def _prediction_files(root):
    """Yield ``(relative path, absolute path)`` pairs in stable order."""
    discovered = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        # These directories can be extremely large and never contain result CSVs.
        directory_names[:] = [
            name
            for name in directory_names
            if name not in {"output", ".controller", "batch-submissions"}
        ]
        directory_path = Path(directory)
        for file_name in file_names:
            if not _RESULT_NAME.fullmatch(file_name):
                continue
            absolute_path = directory_path / file_name
            relative_path = absolute_path.relative_to(root)
            if _is_prediction_result(relative_path):
                discovered.append((relative_path, absolute_path))

    print(f"Discovered {len(discovered):,} prediction result files.", flush=True)
    yield from sorted(discovered, key=lambda item: item[0].as_posix())


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_write(path, value):
    serialised = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(serialised)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def create_manifest(
    results_root, manifest_path, include_hashes=False, paths_only=False
):
    if include_hashes and paths_only:
        raise ValueError("--hash and --paths-only cannot be used together.")

    results_root = results_root.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    entries = []
    for index, (relative_path, absolute_path) in enumerate(
        _prediction_files(results_root), start=1
    ):
        entry = {"path": _normalise_relative_path(relative_path)}
        if not paths_only:
            entry["size"] = absolute_path.stat().st_size
        if include_hashes:
            entry["sha256"] = _sha256(absolute_path)
        entries.append(entry)
        if index % 5000 == 0:
            print(f"Indexed {index:,} local result files...", flush=True)

    manifest = {
        "format": _FORMAT,
        "created_utc": _utc_now(),
        "results_root": str(results_root),
        "sizes": not paths_only,
        "hashes": "sha256" if include_hashes else None,
        "entries": entries,
    }
    print("Writing manifest...", flush=True)
    _atomic_json_write(manifest_path.resolve(), manifest)
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Prediction result files: {len(entries):,}")
    print(f"File sizes: {'no' if paths_only else 'yes'}")
    print(f"Content hashes: {'yes' if include_hashes else 'no'}")


def _load_manifest(path):
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("format") != _FORMAT:
        raise ValueError(
            f"Unsupported manifest format in {path}: {manifest.get('format')!r}"
        )

    entries = {}
    for entry in manifest.get("entries", []):
        relative_path = entry.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Invalid result path in manifest: {entry!r}")
        if relative_path in entries:
            raise ValueError(f"Duplicate result path in manifest: {relative_path}")
        entries[relative_path] = entry
    return manifest, entries


def _difference_reason(absolute_path, relative_name, local_entries, compare_mode):
    local_entry = local_entries.get(relative_name)
    if local_entry is None:
        return "new"
    if compare_mode == "path":
        return None

    if absolute_path.stat().st_size != local_entry.get("size"):
        return "changed-size"
    if compare_mode == "size":
        return None

    local_hash = local_entry.get("sha256")
    if not local_hash:
        raise ValueError(
            "Hash comparison requested, but the local manifest has no hashes. "
            "Recreate it with --hash."
        )
    if _sha256(absolute_path) != local_hash:
        return "changed-content"
    return None


def create_delta_archive(
    results_root,
    manifest_path,
    output_path,
    compare_mode="size",
    overwrite=False,
):
    results_root = results_root.resolve()
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Local manifest does not exist: {manifest_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    local_manifest, local_entries = _load_manifest(manifest_path)
    if compare_mode != "path" and local_manifest.get("sizes", True) is not True:
        raise ValueError(
            "Size or hash comparison requested, but the manifest was created with "
            "--paths-only. Use --compare path or recreate the manifest without "
            "--paths-only."
        )
    if compare_mode == "hash" and local_manifest.get("hashes") != "sha256":
        raise ValueError(
            "Hash comparison requested, but the manifest was not created with --hash."
        )

    selected = []
    scanned = 0
    for relative_path, absolute_path in _prediction_files(results_root):
        scanned += 1
        relative_name = _normalise_relative_path(relative_path)
        reason = _difference_reason(
            absolute_path, relative_name, local_entries, compare_mode
        )
        if reason is not None:
            selected.append((relative_name, absolute_path, reason))
        if scanned % 5000 == 0:
            print(
                f"Scanned {scanned:,}; selected {len(selected):,}...", flush=True
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    metadata = {
        "format": "tsml-eval-result-delta-v1",
        "created_utc": _utc_now(),
        "cluster_results_root": str(results_root),
        "local_manifest_created_utc": local_manifest.get("created_utc"),
        "local_manifest_results_root": local_manifest.get("results_root"),
        "compare_mode": compare_mode,
        "cluster_files_scanned": scanned,
        "files_selected": len(selected),
        "entries": [
            {
                "path": relative_name,
                "size": absolute_path.stat().st_size,
                "reason": reason,
            }
            for relative_name, absolute_path, reason in selected
        ],
    }

    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            archive.writestr(
                "_result_delta_manifest.json",
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            )
            for index, (relative_name, absolute_path, _) in enumerate(
                selected, start=1
            ):
                archive.write(absolute_path, arcname=relative_name)
                if index % 1000 == 0:
                    print(f"Archived {index:,}/{len(selected):,}...", flush=True)
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise

    reason_counts = {}
    for _, _, reason in selected:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print(f"Archive: {output_path}")
    print(f"Cluster prediction files scanned: {scanned:,}")
    print(f"Files archived: {len(selected):,}")
    for reason in sorted(reason_counts):
        print(f"  {reason}: {reason_counts[reason]:,}")
    print(f"Archive size: {output_path.stat().st_size / (1024 * 1024):,.1f} MiB")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest", help="Inventory prediction results already held locally."
    )
    manifest_parser.add_argument("results_root", type=Path)
    manifest_parser.add_argument("manifest", type=Path)
    manifest_parser.add_argument(
        "--hash",
        action="store_true",
        help="Store SHA-256 hashes for exact, but slower, comparison.",
    )
    manifest_parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Store paths only for the fastest missing-file comparison.",
    )

    archive_parser = subparsers.add_parser(
        "archive", help="Archive cluster results absent or changed locally."
    )
    archive_parser.add_argument("results_root", type=Path)
    archive_parser.add_argument("manifest", type=Path)
    archive_parser.add_argument("output", type=Path)
    archive_parser.add_argument(
        "--compare",
        choices=("path", "size", "hash"),
        default="size",
        help="Comparison strength (default: size).",
    )
    archive_parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output ZIP."
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "manifest":
            create_manifest(
                args.results_root,
                args.manifest,
                include_hashes=args.hash,
                paths_only=args.paths_only,
            )
        else:
            create_delta_archive(
                args.results_root,
                args.manifest,
                args.output,
                args.compare,
                args.overwrite,
            )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
