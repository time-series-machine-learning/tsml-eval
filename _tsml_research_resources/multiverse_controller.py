"""Reconcile, submit, monitor, and report Multiverse Slurm experiments.

The controller performs one reconciliation cycle and exits. Run it periodically with
``run_multiverse_controller.sh`` so a failed cycle is restarted without leaving a
long-lived Python process. It supports either ordered categories or a breadth-first
pass across every category.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_RESULT_PATTERN = re.compile(r"testResample(\d+)\.csv$")


@dataclass(frozen=True)
class Category:
    """One ordered classifier category and its results-directory name."""

    name: str
    classifiers: tuple[str, ...]


@dataclass(frozen=True)
class ControllerConfig:
    """Validated controller configuration."""

    username: str
    email: str
    repo_dir: Path
    data_dir: Path
    dataset_file: Path
    results_root: Path
    state_dir: Path
    resamples: int
    max_attempts: int
    all_categories_first_pass: bool
    excluded_datasets: tuple[str, ...]
    validate_results: bool
    account: str
    partition: str
    qos: str
    max_active_tasks: int
    memory_mb_levels: tuple[int, ...]
    time_limit: str
    module: str
    conda_sh: Path
    env_name: str
    numba_cache_dir: Path
    categories: tuple[Category, ...]


@dataclass(frozen=True)
class Task:
    """A classifier, dataset, and zero-based resample to run."""

    category: str
    classifier: str
    dataset: str
    resample: int

    @property
    def state_key(self) -> str:
        """Return a stable JSON key for submission-attempt accounting."""
        return "|".join(
            (self.category, self.classifier, self.dataset, str(self.resample))
        )

    @property
    def job_key(self) -> tuple[str, int]:
        """Return the Slurm job name and one-based array index."""
        return _job_name(self.classifier, self.dataset), self.resample + 1


@dataclass(frozen=True)
class SlurmSnapshot:
    """Expanded running and pending Slurm tasks."""

    states: dict[tuple[str, int], str]
    total_user_tasks: int
    error: str | None = None
    memory_mb: dict[tuple[str, int], int] = field(default_factory=dict)


def _format_value(value, username):
    """Expand username placeholders and environment variables."""
    return os.path.expandvars(str(value).format(username=username))


def _load_config(config_file):
    """Load and validate TOML controller configuration."""
    with config_file.open("rb") as file:
        raw = tomllib.load(file)

    controller = raw.get("controller", {})
    slurm = raw.get("slurm", {})
    environment = raw.get("environment", {})
    category_rows = raw.get("categories", [])
    username = str(controller.get("username") or getpass.getuser())

    def path_from(section, key, default=None):
        value = section.get(key, default)
        if value is None:
            raise ValueError(f"Missing required path setting: {key}")
        return Path(_format_value(value, username)).expanduser()

    categories = tuple(
        Category(
            str(row["name"]),
            tuple(str(classifier) for classifier in row["classifiers"]),
        )
        for row in category_rows
    )
    results_root = path_from(controller, "results_root")
    config = ControllerConfig(
        username=username,
        email=str(controller.get("email", "")),
        repo_dir=path_from(controller, "repo_dir"),
        data_dir=path_from(controller, "data_dir"),
        dataset_file=path_from(controller, "dataset_file"),
        results_root=results_root,
        state_dir=path_from(controller, "state_dir", results_root / ".controller"),
        resamples=int(controller.get("resamples", 30)),
        max_attempts=int(controller.get("max_attempts", 2)),
        all_categories_first_pass=bool(
            controller.get("all_categories_first_pass", False)
        ),
        excluded_datasets=tuple(
            str(dataset) for dataset in controller.get("excluded_datasets", ())
        ),
        validate_results=bool(controller.get("validate_results", False)),
        account=str(slurm.get("account", "cmp")),
        partition=str(slurm.get("partition", "compute")),
        qos=str(slurm.get("qos", "uea-core-default")),
        max_active_tasks=int(slurm.get("max_active_tasks", 200)),
        memory_mb_levels=tuple(
            int(value)
            for value in slurm.get("memory_mb_levels", (16000, 32000, 64000, 128000))
        ),
        time_limit=str(slurm.get("time_limit", "7-00:00:00")),
        module=str(environment.get("module", "python/anaconda/2024.10/3.12.7")),
        conda_sh=path_from(
            environment,
            "conda_sh",
            "/gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh",
        ),
        env_name=str(environment.get("env_name", "tsml-eval")),
        numba_cache_dir=path_from(
            environment,
            "numba_cache_dir",
            "/gpfs/home/{username}/Code/.cache/numba/tsml-eval",
        ),
        categories=categories,
    )
    _validate_config(config)
    return config


def _validate_config(config):
    """Reject unsafe or ambiguous controller configuration."""
    if config.resamples < 1:
        raise ValueError("resamples must be at least 1")
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if len(config.excluded_datasets) != len(set(config.excluded_datasets)):
        raise ValueError("excluded_datasets must be unique")
    if config.max_active_tasks < 1:
        raise ValueError("max_active_tasks must be at least 1")
    if (
        not config.memory_mb_levels
        or any(value < 1 for value in config.memory_mb_levels)
        or tuple(sorted(set(config.memory_mb_levels))) != config.memory_mb_levels
    ):
        raise ValueError("memory_mb_levels must be unique positive increasing values")
    if not config.categories:
        raise ValueError("At least one [[categories]] entry is required")

    category_names = [category.name for category in config.categories]
    if len(category_names) != len(set(category_names)):
        raise ValueError("Category names must be unique")
    classifiers = [
        classifier
        for category in config.categories
        for classifier in category.classifiers
    ]
    if len(classifiers) != len(set(classifiers)):
        raise ValueError("A classifier must occur in only one category")
    if any(not category.classifiers for category in config.categories):
        raise ValueError("Every category must contain at least one classifier")
    for value in category_names + classifiers + list(config.excluded_datasets):
        if not value or "\n" in value or "\r" in value or "|" in value:
            raise ValueError(f"Unsafe category or classifier name: {value!r}")


def _read_datasets(dataset_file):
    """Read a unique, nonblank dataset list."""
    with dataset_file.open(encoding="utf-8-sig") as file:
        datasets = tuple(
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not datasets:
        raise ValueError(f"No datasets found in {dataset_file}")
    if len(datasets) != len(set(datasets)):
        raise ValueError(f"Duplicate dataset names found in {dataset_file}")
    unsafe_names = (
        "\n" in dataset or "\r" in dataset or "|" in dataset for dataset in datasets
    )
    if any(unsafe_names):
        raise ValueError("Dataset names must not contain newlines or '|'")
    return datasets


def _included_datasets(config, datasets):
    """Remove explicitly deferred datasets while preserving list order."""
    excluded = set(config.excluded_datasets)
    included = tuple(dataset for dataset in datasets if dataset not in excluded)
    if not included:
        raise ValueError("Every dataset was excluded")
    return included


def _job_name(classifier, dataset):
    """Return the job name shared with the existing submission architecture."""
    return f"{classifier}_{dataset}"


def _result_file(config, task):
    """Return the expected test-result path for a task."""
    return (
        config.results_root
        / task.category
        / task.classifier
        / "Predictions"
        / task.dataset
        / f"testResample{task.resample}.csv"
    )


def _is_complete(config, task):
    """Check whether a result exists and, optionally, fully validate it."""
    result_file = _result_file(config, task)
    try:
        if not result_file.is_file() or result_file.stat().st_size == 0:
            return False
    except OSError:
        return False
    if not config.validate_results:
        return True
    try:
        from tsml_eval.utils.results_validation import validate_results_file

        return validate_results_file(result_file)
    except (IndexError, OSError, ValueError):
        return False


def _iter_tasks(category, datasets, resamples):
    """Yield expected tasks in stable classifier/dataset/resample order."""
    for classifier in category.classifiers:
        for dataset in datasets:
            for resample in range(resamples):
                yield Task(category.name, classifier, dataset, resample)


def _query_slurm(config):
    """Return expanded running/pending array tasks for the user's partition."""
    if shutil.which("squeue") is None:
        return SlurmSnapshot({}, 0, "squeue was not found")
    command = [
        "squeue",
        "--noheader",
        "--array",
        f"--user={config.username}",
        f"--partition={config.partition}",
        "--states=RUNNING,PENDING",
        "--format=%200j|%a|%T|%m",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        return SlurmSnapshot({}, 0, f"squeue failed: {error}")

    states = {}
    memory_mb = {}
    total = 0
    for line in result.stdout.splitlines():
        total += 1
        fields = line.rsplit("|", maxsplit=3)
        if len(fields) != 4:
            continue
        name, array_index, state, memory = (field.strip() for field in fields)
        try:
            index = int(array_index)
        except ValueError:
            continue
        states[(name, index)] = state
        parsed_memory = _parse_memory_mb(memory)
        if parsed_memory is not None:
            memory_mb[(name, index)] = parsed_memory
    return SlurmSnapshot(states, total, memory_mb=memory_mb)


def _parse_memory_mb(value):
    """Convert a Slurm memory value such as 16000M or 64G to MB."""
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)(?:[cn])?", value)
    if match is None:
        return None
    number = float(match.group(1))
    factor = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}
    return int(number * factor[match.group(2)])


def _load_state(state_file):
    """Load persistent attempt counts, recovering safely from no prior state."""
    if not state_file.is_file():
        return {
            "version": 1,
            "attempts": {},
            "memory_levels": {},
            "last_submitted_memory": {},
            "failures": {},
        }
    with state_file.open(encoding="utf-8") as file:
        state = json.load(file)
    if state.get("version") != 1 or not isinstance(state.get("attempts"), dict):
        raise ValueError(f"Unsupported controller state in {state_file}")
    state.setdefault("memory_levels", {})
    state.setdefault("last_submitted_memory", {})
    state.setdefault("failures", {})
    return state


def _save_state(state_file, state):
    """Atomically save controller state."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(state, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(state_file)


def _git_revision(repo_dir):
    """Return the exact repository branch and commit used for new jobs."""
    commit = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(repo_dir), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return branch or "DETACHED", commit


def _batch_script(
    config,
    task,
    array_indices,
    commit,
    memory_mb,
    prepare_directories=True,
):
    """Create a CPU-only Slurm array script for one classifier/dataset pair."""
    category_results = config.results_root / task.category
    output_dir = category_results / "output" / task.classifier / task.dataset
    if prepare_directories:
        output_dir.mkdir(parents=True, exist_ok=True)
        config.numba_cache_dir.mkdir(parents=True, exist_ok=True)
    array_spec = ",".join(str(index) for index in array_indices)
    q = shlex.quote
    return f"""#!/bin/bash
#SBATCH --account={config.account}
#SBATCH --partition={config.partition}
#SBATCH --qos={config.qos}
#SBATCH --time={config.time_limit}
#SBATCH --job-name={_job_name(task.classifier, task.dataset)}
#SBATCH --array={array_spec}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory_mb}M
#SBATCH --output={output_dir}/%A-%a.out
#SBATCH --error={output_dir}/%A-%a.err

set -eo pipefail
source /etc/profile
module purge
module load {q(config.module)}
source {q(str(config.conda_sh))}
conda activate {q(config.env_name)}

export NUMBA_CACHE_DIR={q(str(config.numba_cache_dir))}
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPI_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export PYTHONUNBUFFERED=1

cd {q(str(config.repo_dir))}
actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != {q(commit)} ]]; then
    echo "ERROR: repository changed after this job was submitted."
    echo "Expected commit: {commit}"
    echo "Current commit:  $actual_commit"
    exit 1
fi

echo "Host:              $(hostname)"
echo "Slurm job:         ${{SLURM_JOB_ID}}"
echo "Slurm array task:  ${{SLURM_ARRAY_TASK_ID}}"
echo "Classifier:        {task.classifier}"
echo "Dataset:           {task.dataset}"
echo "Resample ID:       $((SLURM_ARRAY_TASK_ID - 1))"
echo "Requested memory:  ${{SLURM_MEM_PER_NODE:-unknown}} MB"
echo "CPU-only:          true"
echo "tsml-eval commit:  $actual_commit"

python -u -m tsml_eval.experiments.classification_experiments \\
    {q(str(config.data_dir))} \\
    {q(str(category_results))} \\
    {q(task.classifier)} \\
    {q(task.dataset)} \\
    $((SLURM_ARRAY_TASK_ID - 1))
"""


def _submit_array(config, script, dry_run):
    """Submit a generated script and return its Slurm job ID."""
    if dry_run:
        return "DRY-RUN"
    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().split(";", maxsplit=1)[0]


def _find_current_category(config, datasets, snapshot, state):
    """Return the first category with active or retryable missing tasks."""
    for category in config.categories:
        missing = [
            task
            for task in _iter_tasks(category, datasets, config.resamples)
            if not _is_complete(config, task)
        ]
        for task in missing:
            if task.job_key in snapshot.states:
                _record_active_submission(config, state, task, snapshot)
            else:
                _refresh_failure_record(config, state, task)
        actionable = any(
            task.job_key in snapshot.states or _task_retryable(config, state, task)
            for task in missing
        )
        if actionable:
            return category, missing
    return None, []


def _find_work_scope(config, datasets, snapshot, state):
    """Return the configured ordered-category or all-category work scope."""
    if not config.all_categories_first_pass:
        return _find_current_category(config, datasets, snapshot, state)

    missing = []
    actionable = False
    for category in config.categories:
        category_missing = [
            task
            for task in _iter_tasks(category, datasets, config.resamples)
            if not _is_complete(config, task)
        ]
        missing.extend(category_missing)
        for task in category_missing:
            if task.job_key in snapshot.states:
                _record_active_submission(config, state, task, snapshot)
                actionable = True
            else:
                _refresh_failure_record(config, state, task)
                actionable = actionable or _task_retryable(config, state, task)
    if actionable:
        return Category("AllCategoriesFirstPass", ()), missing
    return None, []


def _round_robin_categories(config, tasks):
    """Interleave eligible tasks across categories in configured order."""
    buckets = {
        category.name: [task for task in tasks if task.category == category.name]
        for category in config.categories
    }
    ordered = []
    offset = 0
    while any(offset < len(bucket) for bucket in buckets.values()):
        for category in config.categories:
            bucket = buckets[category.name]
            if offset < len(bucket):
                ordered.append(bucket[offset])
        offset += 1
    return ordered


def _count_complete(config, category, datasets):
    """Count existing result files without reading their full CSV contents."""
    if config.validate_results:
        return sum(
            _is_complete(config, task)
            for task in _iter_tasks(category, datasets, config.resamples)
        )
    complete = 0
    for classifier in category.classifiers:
        for dataset in datasets:
            result_dir = (
                config.results_root
                / category.name
                / classifier
                / "Predictions"
                / dataset
            )
            try:
                files = result_dir.glob("testResample*.csv")
                indices = {
                    int(match.group(1))
                    for result_file in files
                    if (match := _RESULT_PATTERN.match(result_file.name))
                    and 0 <= int(match.group(1)) < config.resamples
                    and result_file.stat().st_size > 0
                }
            except OSError:
                indices = set()
            complete += len(indices)
    return complete


def _category_rows(config, datasets, snapshot, state_data):
    """Build concise progress rows for every configured category."""
    rows = []
    tasks_per_classifier = len(datasets) * config.resamples
    for category in config.categories:
        complete = _count_complete(config, category, datasets)
        running = 0
        pending = 0
        oom = 0
        timeouts = 0
        failed = 0
        for task in _iter_tasks(category, datasets, config.resamples):
            job_state = snapshot.states.get(task.job_key)
            if job_state == "RUNNING":
                running += 1
            elif job_state == "PENDING":
                pending += 1
            events = state_data["failures"].get(task.state_key, {}).get("events", [])
            if any(event.get("reason") == "OOM" for event in events):
                oom += 1
            if (
                job_state is None
                and task.state_key in state_data["attempts"]
                and not _is_complete(config, task)
            ):
                terminal_reason = _task_terminal_reason(config, state_data, task)
                if terminal_reason == "Time limit":
                    timeouts += 1
                elif terminal_reason not in {None, "OOM"}:
                    failed += 1
        total = len(category.classifiers) * tasks_per_classifier
        rows.append(
            (category.name, complete, running, pending, oom, timeouts, failed, total)
        )
    return rows


def _exhausted_tasks(config, snapshot, state, datasets=None):
    """Return all inactive, still-missing tasks at the submission-attempt limit."""
    exhausted = []
    allowed_datasets = set(datasets) if datasets is not None else None
    allowed_pairs = {
        (category.name, classifier)
        for category in config.categories
        for classifier in category.classifiers
    }
    for key in state["attempts"]:
        fields = key.split("|")
        if len(fields) != 4:
            continue
        category, classifier, dataset, resample = fields
        try:
            task = Task(category, classifier, dataset, int(resample))
        except ValueError:
            continue
        if (
            (category, classifier) in allowed_pairs
            and (allowed_datasets is None or dataset in allowed_datasets)
            and task.job_key not in snapshot.states
            and not _is_complete(config, task)
            and _task_terminal_reason(config, state, task) is not None
        ):
            exhausted.append(task)
    return exhausted


def _latest_failure_details(config, task):
    """Diagnose the latest output pair and return its reason and signature."""
    output_dir = (
        config.results_root / task.category / "output" / task.classifier / task.dataset
    )
    array_index = task.resample + 1
    try:
        logs = list(output_dir.glob(f"*-{array_index}.out")) + list(
            output_dir.glob(f"*-{array_index}.err")
        )
    except OSError:
        return "No readable logs", None
    if not logs:
        return "No logs", None

    attempts = {}
    for log in logs:
        attempts.setdefault(log.stem, []).append(log)
    try:
        latest = max(
            attempts.values(),
            key=lambda files: max(path.stat().st_mtime_ns for path in files),
        )
        modified = max(path.stat().st_mtime_ns for path in latest)
    except OSError:
        return "No readable logs", None

    text = ""
    for log in latest:
        try:
            text += log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    lower = text.lower()
    signature = f"{latest[0].stem}:{modified}"
    if re.search(r"oom[_ -]?kill|out[ -]?of[ -]?memory", lower):
        return "OOM", signature
    if "due to time limit" in lower or "time limit exceeded" in lower:
        return "Time limit", signature
    if "traceback (most recent call last)" in lower:
        return "Python exception", signature
    if "cancelled" in lower:
        return "Cancelled", signature
    if re.search(r"\bkilled\b", lower):
        return "Killed", signature
    if re.search(r"(^|\n).*error:", lower):
        return "Slurm/runtime error", signature
    return "No terminal status", signature


def _latest_failure_reason(config, task):
    """Return only the latest diagnosed terminal reason."""
    return _latest_failure_details(config, task)[0]


def _task_memory(config, state, task):
    """Return the currently selected memory tier for a task."""
    level = int(state["memory_levels"].get(task.state_key, 0))
    level = min(max(level, 0), len(config.memory_mb_levels) - 1)
    return config.memory_mb_levels[level]


def _record_active_submission(config, state, task, snapshot):
    """Capture attempts and requested memory for jobs from any queue feeder."""
    state["attempts"].setdefault(task.state_key, 1)
    memory_mb = snapshot.memory_mb.get(task.job_key)
    if memory_mb is None:
        return
    state["last_submitted_memory"][task.state_key] = memory_mb
    eligible_levels = [
        index
        for index, configured in enumerate(config.memory_mb_levels)
        if configured <= memory_mb
    ]
    if eligible_levels:
        state["memory_levels"][task.state_key] = eligible_levels[-1]


def _record_all_active_submissions(config, datasets, snapshot, state):
    """Capture active job attempts and memory across every configured category."""
    for category in config.categories:
        for task in _iter_tasks(category, datasets, config.resamples):
            if task.job_key in snapshot.states:
                _record_active_submission(config, state, task, snapshot)


def _refresh_failure_record(config, state, task):
    """Record a newly observed failure and escalate confirmed OOM memory."""
    if (
        config.all_categories_first_pass
        and int(state["attempts"].get(task.state_key, 0)) == 0
    ):
        # A clean first-pass state deliberately ignores logs left by older runs.
        return
    reason, signature = _latest_failure_details(config, task)
    if signature is None:
        return
    record = state["failures"].setdefault(task.state_key, {"events": []})
    if record.get("last_signature") == signature:
        return

    attempts = state["attempts"]
    if attempts.get(task.state_key, 0) == 0:
        # Account for jobs submitted by an older queue feeder.
        attempts[task.state_key] = 1
    failed_memory = int(
        state["last_submitted_memory"].get(
            task.state_key, _task_memory(config, state, task)
        )
    )
    event = {
        "reason": reason,
        "memory_mb": failed_memory,
        "signature": signature,
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    record.setdefault("events", []).append(event)
    record.update(
        {
            "last_reason": reason,
            "last_signature": signature,
            "failed_memory_mb": failed_memory,
        }
    )
    if reason == "OOM":
        current_level = int(state["memory_levels"].get(task.state_key, 0))
        if current_level < len(config.memory_mb_levels) - 1:
            state["memory_levels"][task.state_key] = current_level + 1
            event["next_memory_mb"] = config.memory_mb_levels[current_level + 1]
        else:
            event["next_memory_mb"] = None


def _task_retryable(config, state, task):
    """Return whether an inactive missing task has another permitted attempt."""
    attempts = int(state["attempts"].get(task.state_key, 0))
    if attempts == 0:
        return True
    record = state["failures"].get(task.state_key, {})
    reason = record.get("last_reason")
    if reason == "Time limit":
        return False
    if reason == "OOM":
        failed_memory = int(record.get("failed_memory_mb", 0))
        selected_memory = _task_memory(config, state, task)
        submitted_memory = int(state["last_submitted_memory"].get(task.state_key, 0))
        return (
            failed_memory < config.memory_mb_levels[-1]
            and selected_memory > failed_memory
            and submitted_memory <= failed_memory
        )
    return attempts < config.max_attempts


def _task_terminal_reason(config, state, task):
    """Return a terminal outcome or None while a retry remains possible."""
    if _task_retryable(config, state, task):
        return None
    record = state["failures"].get(task.state_key, {})
    reason = record.get("last_reason")
    if reason == "Time limit":
        return "Time limit"
    if reason == "OOM":
        failed_memory = int(record.get("failed_memory_mb", 0))
        if failed_memory >= config.memory_mb_levels[-1]:
            return "OOM"
        return "Failed after OOM retry"
    return reason or "No terminal status"


def _compose_report(
    config,
    branch,
    commit,
    category,
    missing,
    active_missing,
    retryable,
    current_exhausted,
    all_exhausted,
    snapshot,
    rows,
    submissions,
    submission_errors,
    state,
):
    """Compose the stdout, saved, and emailed progress report."""
    lines = [
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Repository: {config.repo_dir}",
        f"Revision: {branch} {commit}",
        f"Results root: {config.results_root}",
        f"Datasets: {config.dataset_file}",
        f"Resamples: {config.resamples}",
        "Scheduling mode: "
        + (
            "breadth-first across all categories"
            if config.all_categories_first_pass
            else "ordered categories"
        ),
        "Memory tiers (MB): "
        + ", ".join(str(memory) for memory in config.memory_mb_levels),
        "Excluded datasets: " + (", ".join(config.excluded_datasets) or "none"),
        "CPU-only jobs: yes",
        "",
    ]
    if snapshot.error:
        lines.append(f"SLURM ERROR: {snapshot.error}")
    lines.extend(
        [
            f"User running/pending tasks on {config.partition}: "
            f"{snapshot.total_user_tasks}/{config.max_active_tasks}",
            "Current scheduling scope: "
            + (
                category.name
                if category
                else ("ALL SETTLED WITH FAILURES" if all_exhausted else "ALL COMPLETE")
            ),
        ]
    )
    if category:
        lines.extend(
            [
                f"In-scope missing results: {len(missing)}",
                f"Already running/pending: {len(active_missing)}",
                f"Eligible for submission: {len(retryable)}",
                f"Terminal outcomes in scope: {len(current_exhausted)}",
            ]
        )
    lines.extend(
        [
            f"Newly submitted tasks: {sum(item[3] for item in submissions)}",
            "",
            f"{'Category':<22} {'Complete':>10} {'Running':>9} "
            f"{'Pending':>9} {'OOM':>6} {'Timeout':>8} {'Failed':>8} "
            f"{'Total':>10} {'Progress':>9}",
            "-" * 101,
        ]
    )
    for name, complete, running, pending, oom, timeouts, failed, total in rows:
        progress = 100 * complete / total if total else 100.0
        lines.append(
            f"{name:<22} {complete:>10} {running:>9} "
            f"{pending:>9} {oom:>6} {timeouts:>8} {failed:>8} "
            f"{total:>10} {progress:>8.1f}%"
        )
    if submissions:
        lines.append("\nSubmitted arrays:")
        for classifier, dataset, job_id, count, memory_mb in submissions:
            lines.append(
                f"  {job_id}: {classifier}/{dataset} "
                f"({count} tasks at {memory_mb} MB)"
            )
    if submission_errors:
        lines.append("\nSubmission errors:")
        lines.extend(f"  {error}" for error in submission_errors)
    if all_exhausted:
        lines.append("\nTerminal outcomes (first 50):")
        for task in all_exhausted[:50]:
            reason = _task_terminal_reason(config, state, task)
            lines.append(
                f"  {task.classifier}/{task.dataset}/resample{task.resample}: "
                f"{reason}"
            )
        if len(all_exhausted) > 50:
            lines.append(f"  ...and {len(all_exhausted) - 50} more")

    failure_events = []
    for key, record in state["failures"].items():
        fields = key.split("|")
        if len(fields) != 4:
            continue
        for event in record.get("events", []):
            failure_events.append((fields, event))
    failure_events.sort(key=lambda item: item[1].get("recorded_at", ""))
    if failure_events:
        counts = {}
        for _, event in failure_events:
            reason = event.get("reason", "Unknown")
            counts[reason] = counts.get(reason, 0) + 1
        lines.append("\nRecorded failure events:")
        lines.append(
            "  " + ", ".join(f"{reason}: {count}" for reason, count in counts.items())
        )
        for fields, event in failure_events[-100:]:
            category_name, classifier, dataset, resample = fields
            detail = ""
            if event.get("reason") == "OOM":
                next_memory = event.get("next_memory_mb")
                if next_memory is not None:
                    detail = f"; next tier {next_memory} MB"
                else:
                    detail = "; maximum memory tier exhausted"
            lines.append(
                f"  {category_name}/{classifier}/{dataset}/resample{resample}: "
                f"{event.get('reason', 'Unknown')} at "
                f"{event.get('memory_mb', 'unknown')} MB{detail}"
            )
        if len(failure_events) > 100:
            lines.append(f"  ...showing latest 100 of {len(failure_events)} events")
    return "\n".join(lines) + "\n"


def _send_email(address, subject, report):
    """Send a report with mail/mailx/sendmail, returning a status message."""
    if not address:
        return "Email disabled: no address configured"
    errors = []
    for executable in ("mail", "mailx"):
        if shutil.which(executable):
            try:
                subprocess.run(
                    [executable, "-s", subject, address],
                    input=report,
                    check=True,
                    text=True,
                    capture_output=True,
                )
                return f"Email sent to {address} with {executable}"
            except (OSError, subprocess.CalledProcessError) as error:
                errors.append(f"{executable}: {error}")
    if shutil.which("sendmail"):
        message = f"To: {address}\nSubject: {subject}\n\n{report}"
        try:
            subprocess.run(
                ["sendmail", "-t"],
                input=message,
                check=True,
                text=True,
                capture_output=True,
            )
            return f"Email sent to {address} with sendmail"
        except (OSError, subprocess.CalledProcessError) as error:
            errors.append(f"sendmail: {error}")
    if errors:
        return "Email failed: " + "; ".join(errors)
    return "Email not sent: mail, mailx, and sendmail were not found"


def _save_report(state_dir, report):
    """Save the latest report and append it to a controller history log."""
    state_dir.mkdir(parents=True, exist_ok=True)
    latest = state_dir / "latest_report.txt"
    latest.write_text(report, encoding="utf-8", newline="\n")
    with (state_dir / "report_history.txt").open(
        "a", encoding="utf-8", newline="\n"
    ) as file:
        file.write(report)
        file.write("\n")


def _acquire_lock(lock_file):
    """Acquire a nonblocking Linux advisory lock for this reconciliation cycle."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    file = lock_file.open("a+", encoding="utf-8")
    try:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:  # pragma: no cover - Windows development only
        return file
    except BlockingIOError:
        file.close()
        raise RuntimeError("Another Multiverse controller cycle is already running")
    return file


def run_cycle(config, dry_run=False, report_only=False, no_email=False):
    """Run one restart-safe monitor and queue-refill cycle."""
    lock = None
    if not dry_run:
        config.state_dir.mkdir(parents=True, exist_ok=True)
        lock = _acquire_lock(config.state_dir / "controller.lock")
    try:
        if not config.repo_dir.is_dir():
            raise FileNotFoundError(f"Repository not found: {config.repo_dir}")
        if not config.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {config.data_dir}")
        datasets = _included_datasets(config, _read_datasets(config.dataset_file))
        state_file = config.state_dir / "state.json"
        state = _load_state(state_file)
        attempts = state["attempts"]
        branch, commit = _git_revision(config.repo_dir)
        snapshot = _query_slurm(config)
        _record_all_active_submissions(config, datasets, snapshot, state)
        category, missing = _find_work_scope(config, datasets, snapshot, state)

        active_missing = [task for task in missing if task.job_key in snapshot.states]
        inactive_missing = [
            task for task in missing if task.job_key not in snapshot.states
        ]
        retryable = [
            task for task in inactive_missing if _task_retryable(config, state, task)
        ]
        if config.all_categories_first_pass:
            retryable = _round_robin_categories(config, retryable)
        current_exhausted = [
            task
            for task in inactive_missing
            if _task_terminal_reason(config, state, task) is not None
        ]

        submissions = []
        submitted_tasks = []
        submission_errors = []
        capacity = max(0, config.max_active_tasks - snapshot.total_user_tasks)
        if category and capacity and not report_only and snapshot.error is None:
            selected = retryable[:capacity]
            grouped = OrderedDict()
            for task in selected:
                memory_mb = _task_memory(config, state, task)
                grouped.setdefault(
                    (task.classifier, task.dataset, memory_mb), []
                ).append(task)
            for (classifier, dataset, memory_mb), tasks in grouped.items():
                indices = [task.resample + 1 for task in tasks]
                script = _batch_script(
                    config,
                    tasks[0],
                    indices,
                    commit,
                    memory_mb,
                    prepare_directories=not dry_run,
                )
                try:
                    job_id = _submit_array(config, script, dry_run)
                except (OSError, subprocess.CalledProcessError) as error:
                    submission_errors.append(f"{classifier}/{dataset}: {error}")
                    continue
                submissions.append((classifier, dataset, job_id, len(tasks), memory_mb))
                if not dry_run:
                    submitted_tasks.extend(tasks)
                    for task in tasks:
                        attempts[task.state_key] = attempts.get(task.state_key, 0) + 1
                        state["last_submitted_memory"][task.state_key] = memory_mb
                    _save_state(state_file, state)

        report_states = dict(snapshot.states)
        report_states.update({task.job_key: "PENDING" for task in submitted_tasks})
        report_snapshot = SlurmSnapshot(
            report_states,
            snapshot.total_user_tasks + len(submitted_tasks),
            snapshot.error,
        )
        all_exhausted = _exhausted_tasks(config, report_snapshot, state, datasets)
        rows = _category_rows(config, datasets, report_snapshot, state)
        report = _compose_report(
            config,
            branch,
            commit,
            category,
            missing,
            active_missing,
            retryable,
            current_exhausted,
            all_exhausted,
            report_snapshot,
            rows,
            submissions,
            submission_errors,
            state,
        )
        print(report, end="")  # noqa: T201
        if not dry_run:
            _save_state(state_file, state)
            _save_report(config.state_dir, report)
            if not no_email:
                status = (
                    "settled-with-failures"
                    if category is None and all_exhausted
                    else ("complete" if category is None else category.name)
                )
                subject = f"Multiverse progress: {status}"
                print(_send_email(config.email, subject, report))  # noqa: T201
        return 0 if snapshot.error is None and not submission_errors else 1
    finally:
        if lock is not None:
            lock.close()


def _parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("multiverse_controller.toml"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without changing state or emailing.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Report and email without submitting jobs.",
    )
    parser.add_argument(
        "--no-email", action="store_true", help="Do not send this cycle's email."
    )
    return parser.parse_args(args)


def main(args=None):
    """Run one controller cycle from command-line arguments."""
    parsed = _parse_args(args)
    try:
        config = _load_config(parsed.config)
        return run_cycle(
            config,
            dry_run=parsed.dry_run,
            report_only=parsed.report_only,
            no_email=parsed.no_email,
        )
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)  # noqa: T201
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
