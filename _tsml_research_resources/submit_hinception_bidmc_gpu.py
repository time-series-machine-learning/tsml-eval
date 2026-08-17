"""One-off submission of the missing H-InceptionTime BIDMC32 tasks on Hali GPU.

BIDMC32HR_disc and BIDMC32SpO2_disc are not in the current Hali GPU queue for
H-InceptionTime despite neither having a result yet. Rather than wait on the next
completion-pass cycle, submit both directly at the config's usual memory tier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _tsml_research_resources import multiverse_controller as controller

_CATEGORY = "DeepLearning"
_CLASSIFIER = "H-InceptionTime"
_DATASETS = ("BIDMC32HR_disc", "BIDMC32SpO2_disc")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name(
            "multiverse_core_resample0_hinception_gpu.toml"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    """Submit each task if it is not already complete or active."""
    args = _parse_args()
    config = controller._load_config(args.config)
    branch, commit = controller._git_revision(config.repo_dir)
    if config.expected_branch and branch != config.expected_branch:
        raise RuntimeError(f"Expected {config.expected_branch}, found {branch}")

    snapshot = controller._query_slurm(config)
    memory_mb = config.memory_mb_levels[0]

    for dataset in _DATASETS:
        task = controller.Task(_CATEGORY, _CLASSIFIER, dataset, 0)
        if controller._is_complete(config, task):
            print(f"Complete; skipping: {_CLASSIFIER}/{dataset}/resample0")
            continue
        if task.job_key in snapshot.states:
            print(
                f"Active; skipping: {_CLASSIFIER}/{dataset}/resample0 "
                f"({snapshot.states[task.job_key]})"
            )
            continue
        script = controller._batch_script(
            config,
            task,
            [1],
            commit,
            memory_mb,
            prepare_directories=not args.dry_run,
        )
        job_id = controller._submit_array(config, script, args.dry_run)
        print(f"Submitted {job_id}: {_CLASSIFIER}/{dataset}/resample0 at {memory_mb} MB")


if __name__ == "__main__":
    main()
