"""Cancel 30-resample CPU work and submit paper resample-0 OOM tasks at 128 GB."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _tsml_research_resources import multiverse_controller as controller

_MEMORY_MB = 128000
_PRESERVED_MEMORY_RANGES = (
    (31000, 33000),
    (63000, 66000),
    (127000, 130000),
)
_PAPER_CLASSIFIERS = (
    "CIF-500",
    "DrCIF-500",
    "QUANT",
    "TDE",
    "RDST",
    "STC",
    "Arsenal",
    "MRHydra",
    "ROCKET",
    "1NN-DTW",
    "Catch22",
    "FreshPRINCE",
    "HC2",
    "RIST",
    "H-InceptionTime",
    "LITETime-MV",
    "Dummy",
)
_OOM_TASKS = (
    ("FeatureBased", "FreshPRINCE", "AustraliaRainfall_disc"),
    ("FeatureBased", "FreshPRINCE", "DuckDuckGeese"),
    ("FeatureBased", "FreshPRINCE", "FaceDetection"),
    ("FeatureBased", "FreshPRINCE", "FordChallenge"),
    ("FeatureBased", "FreshPRINCE", "PEMS-SF"),
    ("FeatureBased", "FreshPRINCE", "Skoda"),
    ("FeatureBased", "FreshPRINCE", "STEW"),
    ("Hybrid", "HC2", "AustraliaRainfall_disc"),
    ("Hybrid", "HC2", "BIDMC32HR_disc"),
    ("Hybrid", "HC2", "BIDMC32SpO2_disc"),
    ("Hybrid", "HC2", "CrowdSourced"),
    ("Hybrid", "HC2", "FordChallenge"),
    ("Hybrid", "HC2", "STEW"),
    ("Hybrid", "HC2", "Tiselac"),
    ("Hybrid", "HC2", "USCActivity"),
    ("ConvolutionBased", "MRHydra", "BIDMC32HR_disc"),
    ("ConvolutionBased", "MRHydra", "BIDMC32SpO2_disc"),
    ("ConvolutionBased", "MRHydra", "USCActivity"),
    ("ShapeletBased", "STC", "BIDMC32HR_disc"),
    ("ShapeletBased", "STC", "BIDMC32SpO2_disc"),
    ("ShapeletBased", "STC", "Tiselac"),
    ("ShapeletBased", "STC", "USCActivity"),
    ("ConvolutionBased", "ROCKET", "Tiselac"),
    ("DictionaryBased", "TDE", "AustraliaRainfall_disc"),
    ("DictionaryBased", "TDE", "CrowdSourced"),
    ("DictionaryBased", "TDE", "STEW"),
    ("DictionaryBased", "TDE", "Tiselac"),
    ("DictionaryBased", "TDE", "USCActivity"),
    ("DistanceBased", "1NN-DTW", "BIDMC32HR_disc"),
    ("DistanceBased", "1NN-DTW", "BIDMC32SpO2_disc"),
)


def _queue_rows(config):
    """Return expanded active compute-queue array tasks."""
    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--array",
            f"--user={config.username}",
            f"--partition={config.partition}",
            "--states=RUNNING,PENDING",
            "--format=%i|%200j|%K|%T|%m",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.rsplit("|", maxsplit=4)
        if len(fields) != 5:
            continue
        job_id, name, array_index, state, memory = (
            field.strip() for field in fields
        )
        try:
            array_index = int(array_index)
        except ValueError:
            continue
        rows.append(
            {
                "job_id": job_id,
                "name": name,
                "array_index": array_index,
                "state": state,
                "memory_mb": controller._parse_memory_mb(memory),
            }
        )
    return rows


def _is_paper_job(name):
    """Return whether a Slurm name belongs to a configured paper classifier."""
    return any(name.startswith(f"{classifier}_") for classifier in _PAPER_CLASSIFIERS)


def _preserve_active_resample0(row):
    """Keep running resample-0 OOM jobs already using 32, 64, or 128 GB."""
    memory_mb = row["memory_mb"]
    return row["state"] == "RUNNING" and memory_mb is not None and any(
        lower <= memory_mb <= upper
        for lower, upper in _PRESERVED_MEMORY_RANGES
    )


def _cancel_jobs(job_ids, dry_run):
    """Cancel expanded Slurm task IDs in bounded command-line chunks."""
    if not job_ids:
        print("No 30-resample tasks or replaceable OOM tasks needed cancellation.")
        return
    print(f"Cancelling {len(job_ids)} active Slurm array tasks.")
    if dry_run:
        return
    for start in range(0, len(job_ids), 500):
        subprocess.run(["scancel", *job_ids[start : start + 500]], check=True)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name(
            "multiverse_paper_30resamples_cpu.toml"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    """Perform the guarded cancellation and one-off submission."""
    args = _parse_args()
    config = controller._load_config(args.config)
    branch, commit = controller._git_revision(config.repo_dir)
    if branch != "ajb/hc2":
        raise RuntimeError(f"Expected ajb/hc2, found {branch}")

    oom_names = {
        controller._job_name(classifier, dataset)
        for _, classifier, dataset in _OOM_TASKS
    }
    initial_rows = _queue_rows(config)
    cancel_ids = []
    preserved = []
    for row in initial_rows:
        if not _is_paper_job(row["name"]):
            continue
        if row["array_index"] > 1:
            cancel_ids.append(row["job_id"])
        elif row["name"] in oom_names:
            if _preserve_active_resample0(row):
                preserved.append(row)
            else:
                cancel_ids.append(row["job_id"])

    for row in preserved:
        print(
            "Preserving active resample 0: "
            f"{row['name']} at {row['memory_mb']} MB"
        )
    _cancel_jobs(cancel_ids, args.dry_run)
    if cancel_ids and not args.dry_run:
        time.sleep(5)

    active = {}
    for row in _queue_rows(config):
        if row["array_index"] == 1:
            active.setdefault(row["name"], []).append(row)

    submitted = 0
    complete = 0
    skipped_active = 0
    for category, classifier, dataset in _OOM_TASKS:
        task = controller.Task(category, classifier, dataset, 0)
        if controller._is_complete(config, task):
            print(f"Complete; skipping: {classifier}/{dataset}/resample0")
            complete += 1
            continue
        task_name = controller._job_name(classifier, dataset)
        if task_name in active:
            descriptions = ", ".join(
                f"{row['state']} at {row['memory_mb'] or 'unknown'} MB"
                for row in active[task_name]
            )
            print(
                f"Active; skipping: {classifier}/{dataset}/resample0 "
                f"({descriptions})"
            )
            skipped_active += 1
            continue
        script = controller._batch_script(
            config,
            task,
            [1],
            commit,
            _MEMORY_MB,
            prepare_directories=not args.dry_run,
        )
        job_id = controller._submit_array(config, script, args.dry_run)
        print(
            f"Submitted {job_id}: {classifier}/{dataset}/resample0 "
            f"at {_MEMORY_MB} MB"
        )
        submitted += 1

    print()
    print(f"Completed results skipped: {complete}")
    print(f"Active resample-0 jobs preserved/skipped: {skipped_active}")
    print(f"New 128 GB jobs submitted: {submitted}")
    print("Historical non-default-resample OOM tasks were intentionally ignored.")


if __name__ == "__main__":
    main()
