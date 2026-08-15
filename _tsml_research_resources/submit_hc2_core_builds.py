"""Submit ready Core HC2-from-file builds without duplicating active HC2 jobs."""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import subprocess
from pathlib import Path

_COMPONENTS = (
    ("ShapeletBased", "STC"),
    ("IntervalBased", "DrCIF-500"),
    ("ConvolutionBased", "Arsenal"),
    ("DictionaryBased", "TDE"),
)


def _read_datasets(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8-sig") as file:
        datasets = tuple(
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        )
    if len(datasets) != len(set(datasets)):
        raise ValueError(f"Duplicate dataset names in {path}")
    return datasets


def _active_job_names(username: str) -> set[str]:
    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--array",
            f"--user={username}",
            "--partition=compute",
            "--states=RUNNING,PENDING",
            "--format=%200j",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _components_ready(results_root: Path, dataset: str, resample: int) -> bool:
    return all(
        _nonempty(
            results_root
            / category
            / classifier
            / "Predictions"
            / dataset
            / f"{split}Resample{resample}.csv"
        )
        for category, classifier in _COMPONENTS
        for split in ("train", "test")
    )


def _batch_script(
    repo_dir: Path,
    results_root: Path,
    output_dir: Path,
    dataset: str,
    resample: int,
    commit: str,
) -> str:
    q = shlex.quote
    job_name = f"HC2Build_{dataset}"
    builder = repo_dir / "_tsml_research_resources" / "build_hc2_from_components.py"
    return f"""#!/bin/bash
#SBATCH --account=cmp
#SBATCH --partition=compute
#SBATCH --qos=uea-core-default
#SBATCH --time=7-00:00:00
#SBATCH --job-name={job_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M
#SBATCH --output={output_dir}/%j.out
#SBATCH --error={output_dir}/%j.err

set -eo pipefail
source /etc/profile
module purge
module load python/anaconda/2024.10/3.12.7
source /gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh
conda activate tsml-eval

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

cd {q(str(repo_dir))}
actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != {q(commit)} ]]; then
    echo "ERROR: repository changed after this job was submitted."
    echo "Expected commit: {commit}"
    echo "Current commit:  $actual_commit"
    exit 1
fi

python -u {q(str(builder))} \\
    {q(str(results_root))} \\
    {q(dataset)} \\
    {resample}
"""


def main() -> None:
    username = os.environ.get("USER") or getpass.getuser()
    repo_default = Path(__file__).resolve().parents[1]
    home = Path("/gpfs/home") / username
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-list",
        type=Path,
        default=home / "DataSetLists" / "MultiverseCoreMissingHC2.txt",
    )
    parser.add_argument("--repo-dir", type=Path, default=repo_default)
    parser.add_argument(
        "--results-root", type=Path, default=home / "Results" / "Multiverse"
    )
    parser.add_argument("--resample", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    branch = subprocess.run(
        ["git", "-C", str(args.repo_dir), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != "ajb/hc2":
        raise RuntimeError(f"CPU HC2 builds require ajb/hc2; found {branch}")
    commit = subprocess.run(
        ["git", "-C", str(args.repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    datasets = _read_datasets(args.dataset_list)
    active = _active_job_names(username)
    submitted = 0
    waiting = 0
    skipped = 0
    for dataset in datasets:
        final_result = (
            args.results_root
            / "Hybrid"
            / "HC2"
            / "Predictions"
            / dataset
            / f"testResample{args.resample}.csv"
        )
        build_job = f"HC2Build_{dataset}"
        direct_job = f"HC2_{dataset}"
        if _nonempty(final_result) or build_job in active or direct_job in active:
            skipped += 1
            continue
        if not _components_ready(args.results_root, dataset, args.resample):
            waiting += 1
            continue

        output_dir = args.results_root / "Hybrid" / "output" / "HC2Build" / dataset
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        script = _batch_script(
            args.repo_dir,
            args.results_root,
            output_dir,
            dataset,
            args.resample,
            commit,
        )
        if args.dry_run:
            print(f"DRY RUN: {build_job}")
        else:
            result = subprocess.run(
                ["sbatch", "--parsable"],
                input=script,
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"{result.stdout.strip()}: {build_job}")
            active.add(build_job)
        submitted += 1

    print(
        f"HC2 builds: submitted={submitted}, waiting_for_components={waiting}, "
        f"complete_or_active={skipped}"
    )


if __name__ == "__main__":
    main()
