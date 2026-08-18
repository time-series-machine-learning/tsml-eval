"""One-off submission of H-InceptionTime/BIDMC32SpO2_disc on IridisX.

BIDMC32HR_disc is already running on Hali. BIDMC32SpO2_disc has been stuck PENDING
there on QOSMaxGRESPerUser, so this submits it on IridisX's spare a100 node instead,
against multiverse_core_resample0_hinception_gpu_iridisx.toml's config.

Unlike the full completion-pass config, this only ever submits the one task: IridisX
has its own separate results tree, and running the full iridisx.toml as-is would
resubmit every one of the 66 core datasets if H-InceptionTime has never run there
before.
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
_DATASET = "BIDMC32SpO2_disc"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name(
            "multiverse_core_resample0_hinception_gpu_iridisx.toml"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    """Submit the single task if it is not already complete or active."""
    args = _parse_args()
    config = controller._load_config(args.config)
    branch, commit = controller._git_revision(config.repo_dir)
    if config.expected_branch and branch != config.expected_branch:
        raise RuntimeError(f"Expected {config.expected_branch}, found {branch}")

    task = controller.Task(_CATEGORY, _CLASSIFIER, _DATASET, 0)
    if controller._is_complete(config, task):
        print(f"Complete; skipping: {_CLASSIFIER}/{_DATASET}/resample0")
        return

    snapshot = controller._query_slurm(config)
    if task.job_key in snapshot.states:
        print(
            f"Active; skipping: {_CLASSIFIER}/{_DATASET}/resample0 "
            f"({snapshot.states[task.job_key]})"
        )
        return

    memory_mb = config.memory_mb_levels[0]
    script = controller._batch_script(
        config,
        task,
        [1],
        commit,
        memory_mb,
        prepare_directories=not args.dry_run,
    )
    job_id = controller._submit_array(config, script, args.dry_run)
    print(f"Submitted {job_id}: {_CLASSIFIER}/{_DATASET}/resample0 at {memory_mb} MB")


if __name__ == "__main__":
    main()
