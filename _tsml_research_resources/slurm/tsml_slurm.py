#!/usr/bin/env python3
"""Run tsml-eval experiments on Slurm clusters (Hali, Iridis 6, Iridis X).

One controller for every cluster, task and device. An experiment is described by a
TOML file which names a cluster profile (``clusters/<cluster>.toml``) and one of its
resource sections. The controller compares the results on disk with the results the
experiment asks for, submits only the missing work, tracks the jobs it submitted by
Slurm job ID, diagnoses the ones that end without a result, raises the memory of
tasks killed for running out of memory, and stops once everything is settled.

Commands (see README.md in this directory for the user guide)::

    python tsml_slurm.py check  EXPERIMENT.toml   # validate, count, show a job script
    python tsml_slurm.py submit EXPERIMENT.toml   # one reconcile-and-submit cycle
    python tsml_slurm.py start  EXPERIMENT.toml   # submit, then keep refilling
    python tsml_slurm.py status EXPERIMENT.toml   # progress report
    python tsml_slurm.py stop   EXPERIMENT.toml   # stop refilling (--cancel kills jobs)
    python tsml_slurm.py retry  EXPERIMENT.toml   # make failed tasks eligible again

Only the Python standard library is used, so any Python 3.11+ can run it. Use the
interpreter of the experiment's environment if ``check`` should also construct the
estimators.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
CLUSTER_DIR = SCRIPT_PATH.parent / "clusters"
STATE_VERSION = 1

# Entry module, threaded entry module, factory module and factory function per task.
TASKS = {
    "classification": (
        "classification_experiments",
        "threaded_classification_experiments",
        "_get_classifier",
        "get_classifier_by_name",
    ),
    "regression": (
        "regression_experiments",
        "threaded_regression_experiments",
        "_get_regressor",
        "get_regressor_by_name",
    ),
    "clustering": (
        "clustering_experiments",
        "threaded_clustering_experiments",
        "_get_clusterer",
        "get_clusterer_by_name",
    ),
    "forecasting": (
        "forecasting_experiments",
        "threaded_forecasting_experiments",
        "_get_forecaster",
        "get_forecaster_by_name",
    ),
}

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]*$")
_SAFE_PATH = re.compile(r"^[^\s'\"`$\\;|&<>]+$")

# Failure reasons. See _diagnose and _apply_outcome for how each is treated.
OOM = "OOM"
TIMEOUT = "Time limit"
GPU_OOM = "GPU out of memory"
ERROR = "Python exception"
CANCELLED = "Cancelled"
NO_RESULT = "Finished without a result file"
INFRA = "Node failure or preemption"
NOT_STARTED = "Never started"
KILLED = "Killed"
UNKNOWN = "Unknown failure"


class ConfigError(ValueError):
    """An experiment or cluster profile is invalid."""


# ======================================================================================
# Configuration
# ======================================================================================


@dataclass(frozen=True)
class Resources:
    """Where and how jobs run. Comes from a cluster profile, optionally overridden."""

    partition: str
    time_limit: str
    memory_gb: tuple[int, ...]
    mode: str = "array"  # array: one Slurm array task per experiment; farm: packed
    account: str = ""
    qos: str = ""
    cpus_per_task: int = 1
    gres: str = ""
    max_active: int = 100  # this experiment's tasks running or pending at once
    max_user_tasks: int = 0  # all of the user's tasks in the partition, 0 = no cap
    # farm mode only
    max_jobs: int = 4
    node_cpus: int = 192
    node_memory_gb: int = 630
    waves: int = 4
    # environment
    module: str = ""
    conda_sh: str = ""  # empty: derived from the conda on PATH after module load
    env_name: str = ""
    container: str = ""  # apptainer image or sandbox; replaces conda when set
    gpu_check: str = "none"  # none, tensorflow or torch
    extra_sbatch: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()  # extra shell lines run before the experiment

    @property
    def gpu(self) -> bool:
        """Whether these resources request a GPU."""
        return bool(self.gres)


@dataclass(frozen=True)
class Supervisor:
    """How the refill loop runs: as a self-resubmitting Slurm job or on the login."""

    mode: str = "slurm"  # slurm or login
    interval_minutes: int = 30
    partition: str = ""
    account: str = ""
    qos: str = ""
    time_limit: str = "00:30:00"
    memory_gb: int = 2
    extra_sbatch: tuple[str, ...] = ()


@dataclass(frozen=True)
class Estimator:
    """An estimator name and the results category it is filed under."""

    name: str
    category: str  # empty when results go directly under results_dir


@dataclass(frozen=True)
class Task:
    """One experiment: an estimator, dataset and resample."""

    estimator: str
    category: str
    dataset: str
    resample: int

    @property
    def key(self) -> str:
        """Stable identifier used in the state file."""
        return f"{self.estimator}|{self.dataset}|{self.resample}"


@dataclass
class Experiment:
    """A validated experiment with every path resolved."""

    name: str
    config_path: Path
    cluster: str
    resource_name: str
    task: str
    user: str
    email: str
    home: Path
    repo_dir: Path
    data_dir: Path
    dataset_list: Path
    results_dir: Path
    state_dir: Path
    estimators: tuple[Estimator, ...]
    estimator_kwargs: dict
    resamples: tuple[int, ...]
    resources: Resources
    supervisor: Supervisor
    train_files: bool = False
    test_files: bool = False
    n_jobs: int = 1
    extra_args: tuple[str, ...] = ()
    max_attempts: int = 2
    priority: str = "resample"
    smallest_datasets_first: bool = False
    exclude_datasets: tuple[str, ...] = ()
    exclude_tasks: tuple[str, ...] = ()
    branch: str = ""
    pin_commit: bool = True
    require_clean: bool = True
    email_hours: float = 24.0
    numba_cache_dir: Path | None = None
    datasets: tuple[str, ...] = field(default=())

    @property
    def logs_dir(self) -> Path:
        """Directory holding this experiment's job logs."""
        return self.state_dir / "logs"

    @property
    def job_prefix(self) -> str:
        """Prefix of every Slurm job name of this experiment."""
        return f"tsml-{self.name}"

    @property
    def supervisor_job_name(self) -> str:
        """Slurm job name of the supervisor."""
        return f"tsml-supervisor-{self.name}"

    def estimator_results(self, estimator: Estimator) -> Path:
        """Results path passed to tsml-eval for an estimator."""
        if estimator.category:
            return self.results_dir / estimator.category
        return self.results_dir


def _expand(value, values):
    """Expand {placeholders} and environment variables in a string."""
    try:
        return os.path.expandvars(os.path.expanduser(str(value).format(**values)))
    except KeyError as error:
        raise ConfigError(f"Unknown placeholder {error} in {value!r}") from None


def _build(cls, raw, where):
    """Construct a dataclass from a dict, rejecting unknown keys (typos)."""
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"Unknown setting(s) in {where}: {', '.join(unknown)}")
    converted = {}
    for key, value in raw.items():
        if isinstance(value, list):
            value = tuple(value)
        converted[key] = value
    try:
        return cls(**converted)
    except TypeError as error:
        raise ConfigError(f"{where}: {error}") from None


def load_cluster(name, cluster_dir=CLUSTER_DIR):
    """Load a cluster profile by name."""
    path = Path(cluster_dir) / f"{name}.toml"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in Path(cluster_dir).glob("*.toml")))
        raise ConfigError(f"No cluster profile {path}. Available: {available}")
    with path.open("rb") as file:
        return tomllib.load(file)


def _state_dir(configured, path, values):
    """Return the state directory: as configured, else ~/tsml-slurm/<name>.

    Experiments started before the default moved from the hidden ~/.tsml-slurm keep
    using it, so a running experiment never loses track of its jobs.
    """
    if configured is not None:
        return path(configured)
    visible = path("{home}/tsml-slurm/{name}")
    hidden = path("{home}/.tsml-slurm/{name}")
    if hidden.is_dir() and not visible.exists():
        return hidden
    return visible


def load_experiment(config_path, cluster_dir=CLUSTER_DIR, user=None):
    """Load an experiment TOML, merge its cluster profile and validate."""
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as file:
        raw = tomllib.load(file)
    allowed = {"experiment", "estimators", "estimator_kwargs", "slurm", "supervisor"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown section(s) in {config_path}: {', '.join(unknown)}")
    exp = dict(raw.get("experiment", {}))
    for key in ("name", "cluster", "task", "data_dir", "datasets", "results_dir"):
        if key not in exp:
            raise ConfigError(f"[experiment] is missing required setting {key!r}")
    cluster = load_cluster(exp.pop("cluster"), cluster_dir)
    cluster_name = str(cluster.get("name", ""))
    resource_name = str(exp.pop("resources", cluster.get("default_resources", "cpu")))
    profiles = cluster.get("resources", {})
    if resource_name not in profiles:
        raise ConfigError(
            f"Cluster {cluster_name!r} has no resources.{resource_name}; "
            f"choose from {', '.join(profiles)}"
        )

    user = user or os.environ.get("USER") or getpass.getuser()
    values = {"user": user}
    values["home"] = _expand(cluster.get("home", "~"), values)
    values["repo"] = _expand(
        exp.pop("repo_dir", cluster.get("repo_dir", "{home}/Code/tsml-eval")), values
    )
    values["name"] = str(exp["name"])

    def path(value):
        resolved = _expand(value, values)
        if not _SAFE_PATH.match(resolved):
            raise ConfigError(f"Paths may not contain spaces or quotes: {resolved!r}")
        return Path(resolved)

    # Resources: cluster profile, then the experiment's [slurm] overrides.
    merged = dict(profiles[resource_name])
    merged.update(raw.get("slurm", {}))
    for key in ("container", "conda_sh"):
        if merged.get(key):
            merged[key] = _expand(merged[key], values)
    resources = _build(Resources, merged, f"resources.{resource_name}")
    sup_raw = dict(cluster.get("supervisor", {}))
    sup_raw.update(raw.get("supervisor", {}))
    supervisor = _build(Supervisor, sup_raw, "supervisor")

    # Estimators: a flat list, or a table of category = [names].
    estimators_raw = raw.get("estimators")
    if isinstance(estimators_raw, dict):
        estimators = tuple(
            Estimator(str(name), str(category))
            for category, names in estimators_raw.items()
            for name in names
        )
    elif "estimators" in exp:
        estimators = tuple(Estimator(str(name), "") for name in exp.pop("estimators"))
    else:
        raise ConfigError(
            "Give estimators either as estimators = [...] in [experiment] or as an "
            "[estimators] table of Category = [...]"
        )

    resamples = exp.pop("resamples", 30)
    start = int(exp.pop("start_resample", 0))
    if isinstance(resamples, list):
        resample_ids = tuple(int(r) for r in resamples)
    else:
        resample_ids = tuple(range(start, int(resamples)))

    email = exp.pop("email", cluster.get("email", ""))
    numba = exp.pop("numba_cache_dir", cluster.get("numba_cache_dir", ""))
    if not numba:
        env_label = resources.env_name or "container"
        numba = "{home}/.cache/numba/" + env_label
    experiment = Experiment(
        name=str(exp.pop("name")),
        config_path=config_path,
        cluster=cluster_name,
        resource_name=resource_name,
        task=str(exp.pop("task")),
        user=user,
        email=_expand(email, values) if email else "",
        home=Path(values["home"]),
        repo_dir=path(values["repo"]),
        data_dir=path(exp.pop("data_dir")),
        dataset_list=path(exp.pop("datasets")),
        results_dir=path(exp.pop("results_dir")),
        state_dir=_state_dir(exp.pop("state_dir", None), path, values),
        estimators=estimators,
        estimator_kwargs={
            str(k): dict(v) for k, v in raw.get("estimator_kwargs", {}).items()
        },
        resamples=resample_ids,
        resources=resources,
        supervisor=supervisor,
        numba_cache_dir=path(numba),
        **_simple_settings(exp),
    )
    validate(experiment)
    return experiment


def _simple_settings(exp):
    """Pop the remaining scalar [experiment] settings, rejecting unknown ones."""
    allowed = {
        "train_files": bool,
        "test_files": bool,
        "n_jobs": int,
        "extra_args": tuple,
        "max_attempts": int,
        "priority": str,
        "smallest_datasets_first": bool,
        "exclude_datasets": tuple,
        "exclude_tasks": tuple,
        "branch": str,
        "pin_commit": bool,
        "require_clean": bool,
        "email_hours": float,
    }
    unknown = sorted(set(exp) - set(allowed))
    if unknown:
        raise ConfigError(f"Unknown setting(s) in [experiment]: {', '.join(unknown)}")
    settings = {}
    for key, value in exp.items():
        kind = allowed[key]
        settings[key] = tuple(str(v) for v in value) if kind is tuple else kind(value)
    return settings


def validate(e):
    """Reject configurations that would fail on the cluster or corrupt results."""
    r = e.resources
    if not _SAFE_NAME.match(e.name):
        raise ConfigError(f"Experiment name must be letters, digits, ._+-: {e.name!r}")
    if e.task not in TASKS:
        raise ConfigError(f"task must be one of {', '.join(TASKS)}, not {e.task!r}")
    if e.train_files and e.task == "forecasting":
        raise ConfigError("forecasting experiments do not write train files")
    if e.test_files and e.task != "clustering":
        raise ConfigError("test_files only applies to clustering (test is the default)")
    if not e.estimators:
        raise ConfigError("No estimators configured")
    names = [est.name for est in e.estimators]
    if len(names) != len(set(names)):
        raise ConfigError("Each estimator may appear only once")
    for value in names + [est.category for est in e.estimators if est.category]:
        if not _SAFE_NAME.match(value):
            raise ConfigError(f"Unsafe estimator or category name: {value!r}")
    unknown_kwargs = set(e.estimator_kwargs) - set(names)
    if unknown_kwargs:
        raise ConfigError(f"estimator_kwargs for unknown: {sorted(unknown_kwargs)}")
    for estimator, kwargs in e.estimator_kwargs.items():
        for key, value in kwargs.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ConfigError(f"Bad keyword for {estimator}: {key!r}")
            if not isinstance(value, (bool, int, float, str)):
                raise ConfigError(f"{estimator}.{key} must be bool, int, float or str")
    if not e.resamples or min(e.resamples) < 0:
        raise ConfigError("resamples must select at least one non-negative resample")
    if len(e.resamples) != len(set(e.resamples)):
        raise ConfigError("resamples must be unique")
    if e.max_attempts < 1 or e.n_jobs < 1:
        raise ConfigError("max_attempts and n_jobs must be at least 1")
    if e.priority not in ("resample", "estimator", "dataset"):
        raise ConfigError("priority must be resample, estimator or dataset")
    for flag in e.extra_args:
        if flag in ("-tr", "--train_fold", "-te", "--test_fold", "-nj", "--n_jobs"):
            raise ConfigError(
                f"Use train_files/test_files/n_jobs rather than {flag} in extra_args"
            )
        if flag in ("-ow", "--overwrite"):
            raise ConfigError("--overwrite would rerun completed work on every retry")
    for task in e.exclude_tasks:
        parts = task.split("|")
        if len(parts) != 3 or parts[0] not in names or not parts[2].isdigit():
            raise ConfigError(
                f"exclude_tasks entries are 'Estimator|Dataset|Resample': {task!r}"
            )
    if r.mode not in ("array", "farm"):
        raise ConfigError("resources mode must be array or farm")
    tiers = r.memory_gb
    if not tiers or list(tiers) != sorted(set(tiers)) or tiers[0] < 1:
        raise ConfigError("memory_gb must be unique positive increasing values")
    if r.mode == "farm":
        if r.gpu:
            raise ConfigError("farm mode is for CPU work; use array mode for GPUs")
        if tiers[-1] > r.node_memory_gb:
            raise ConfigError("memory_gb tiers cannot exceed node_memory_gb")
        if e.n_jobs > r.node_cpus:
            raise ConfigError("n_jobs cannot exceed node_cpus in farm mode")
    if r.gpu_check not in ("none", "tensorflow", "torch"):
        raise ConfigError("gpu_check must be none, tensorflow or torch")
    if r.gpu_check != "none" and not r.gpu:
        raise ConfigError("gpu_check is set but no gres requests a GPU")
    if not r.container and not r.env_name:
        raise ConfigError("Set env_name (conda) or container (apptainer) in resources")
    for value in (r.gres, r.partition, r.account, r.qos):
        if any(ch.isspace() for ch in value):
            raise ConfigError(f"Whitespace in Slurm setting {value!r}")
    if e.supervisor.mode not in ("slurm", "login"):
        raise ConfigError("supervisor mode must be slurm or login")
    if e.supervisor.mode == "slurm" and not e.supervisor.partition:
        raise ConfigError("A slurm supervisor needs a partition")
    if e.branch and not re.fullmatch(r"[A-Za-z0-9._/\-]+", e.branch):
        raise ConfigError(f"Unsafe branch name {e.branch!r}")


def read_datasets(e):
    """Read the dataset list, dropping comments, blanks and excluded datasets."""
    if not e.dataset_list.is_file():
        raise ConfigError(f"Dataset list not found: {e.dataset_list}")
    names = []
    with e.dataset_list.open(encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    if len(names) != len(set(names)):
        raise ConfigError(f"Duplicate datasets in {e.dataset_list}")
    for name in names:
        if not _SAFE_NAME.match(name):
            raise ConfigError(f"Unsafe dataset name {name!r} in {e.dataset_list}")
    unknown = set(e.exclude_datasets) - set(names)
    if unknown:
        raise ConfigError(f"exclude_datasets not in the list: {sorted(unknown)}")
    names = [n for n in names if n not in e.exclude_datasets]
    if not names:
        raise ConfigError(f"No datasets in {e.dataset_list}")
    e.datasets = tuple(names)
    return e.datasets


def fingerprints(e):
    """Per-estimator hash of the settings that change what its results mean.

    Adding estimators, datasets or resamples to a running experiment is fine; changing
    the keyword arguments or flags of an estimator that already has results is not,
    because old and new results would sit side by side in one directory.
    """
    prints = {}
    for estimator in e.estimators:
        content = json.dumps(
            {
                "task": e.task,
                "results": str(e.estimator_results(estimator)),
                "kwargs": e.estimator_kwargs.get(estimator.name, {}),
                "train": e.train_files,
                "test": e.test_files,
                "extra_args": e.extra_args,
                "data_dir": str(e.data_dir),
            },
            sort_keys=True,
        )
        prints[estimator.name] = hashlib.sha256(content.encode()).hexdigest()[:12]
    return prints


# ======================================================================================
# Tasks and results
# ======================================================================================


def all_tasks(e):
    """Yield every task the experiment asks for, minus exclusions."""
    excluded = set(e.exclude_tasks)
    for estimator in e.estimators:
        for dataset in e.datasets:
            for resample in e.resamples:
                task = Task(estimator.name, estimator.category, dataset, resample)
                if task.key not in excluded:
                    yield task


def result_files(e, task):
    """Files that must exist for a task to count as complete."""
    folder = (
        e.results_dir / task.category / task.estimator / "Predictions" / task.dataset
    )
    test = folder / f"testResample{task.resample}.csv"
    train = folder / f"trainResample{task.resample}.csv"
    if e.task == "clustering":
        return [train, test] if e.test_files else [train]
    return [test, train] if e.train_files else [test]


def is_complete(e, task):
    """Whether every required result file exists and is non-empty."""
    for path in result_files(e, task):
        try:
            if path.stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def dataset_sizes(e):
    """On-disk size of each dataset directory, for smallest-first ordering."""
    sizes = {}
    for dataset in e.datasets:
        total = 0
        folder = e.data_dir / dataset
        try:
            for path in folder.iterdir():
                if path.is_file():
                    total += path.stat().st_size
        except OSError:
            total = 0
        sizes[dataset] = total
    return sizes


def order_tasks(e, tasks, sizes=None):
    """Sort tasks into submission order according to priority."""
    estimator_rank = {est.name: i for i, est in enumerate(e.estimators)}
    if e.smallest_datasets_first:
        sizes = sizes if sizes is not None else dataset_sizes(e)
        dataset_rank = {d: (sizes.get(d, 0), d) for d in e.datasets}
    else:
        dataset_rank = {d: i for i, d in enumerate(e.datasets)}

    def key(task):
        parts = {
            "resample": task.resample,
            "estimator": estimator_rank[task.estimator],
            "dataset": dataset_rank[task.dataset],
        }
        order = {
            "resample": ("resample", "estimator", "dataset"),
            "estimator": ("estimator", "resample", "dataset"),
            "dataset": ("dataset", "resample", "estimator"),
        }[e.priority]
        return tuple(parts[name] for name in order)

    return sorted(tasks, key=key)


# ======================================================================================
# Slurm
# ======================================================================================


@dataclass(frozen=True)
class QueueEntry:
    """One running or pending job, or array element, from squeue."""

    job_id: str  # array job ID for arrays, job ID otherwise
    index: int | None  # array index, or None
    state: str
    name: str
    partition: str


class Slurm:
    """Thin wrapper over the Slurm commands, replaced by a fake in the tests."""

    def available(self):
        """Whether Slurm commands exist on this host."""
        return shutil.which("squeue") is not None

    def queue(self, user):
        """Return the user's running and pending jobs."""
        out = subprocess.run(
            [
                "squeue",
                "--noheader",
                "--array",
                f"--user={user}",
                "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED",
                "--format=%A|%K|%T|%P|%j",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        entries = []
        for line in out.splitlines():
            parts = line.strip().split("|", 4)
            if len(parts) != 5:
                continue
            job_id, index, state, partition, name = (p.strip() for p in parts)
            entries.append(
                QueueEntry(
                    job_id,
                    int(index) if index.isdigit() else None,
                    state,
                    name,
                    partition,
                )
            )
        return entries

    def accounting(self, job_ids):
        """Return {(job_id, index or None): STATE} for finished jobs, best effort."""
        if not job_ids or shutil.which("sacct") is None:
            return {}
        result = subprocess.run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                f"--jobs={','.join(sorted(job_ids))}",
                "--format=JobID,State",
            ],
            capture_output=True,
            text=True,
        )
        states = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            job, state = parts[0], parts[1].split()[0] if parts[1] else ""
            match = re.fullmatch(r"(\d+)_(\d+)", job)
            if match:
                states[(match.group(1), int(match.group(2)))] = state
            elif job.isdigit():
                states[(job, None)] = state
        return states

    def submit(self, script):
        """Submit a job script and return its job ID."""
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().split(";")[0]

    def cancel(self, job_ids):
        """Cancel the given jobs."""
        if job_ids:
            subprocess.run(["scancel", *sorted(job_ids)], check=False)


# ======================================================================================
# Job scripts
# ======================================================================================

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
)


def _sbatch_lines(e, job_name, cpus, memory_gb, output, error, array=None):
    r = e.resources
    lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={r.partition}",
        f"#SBATCH --time={r.time_limit}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory_gb}G",
        f"#SBATCH --output={output}",
        f"#SBATCH --error={error}",
        "#SBATCH --mail-type=NONE",
    ]
    if r.account:
        lines.append(f"#SBATCH --account={r.account}")
    if r.qos:
        lines.append(f"#SBATCH --qos={r.qos}")
    if r.gres:
        lines.append(f"#SBATCH --gres={r.gres}")
    if array:
        lines.append(f"#SBATCH --array={array}")
    lines.extend(f"#SBATCH {extra}" for extra in r.extra_sbatch)
    return lines


def _environment_block(e, commit, threads):
    """Shell that prepares the environment and records provenance in the log."""
    r = e.resources
    q = shlex.quote
    lines = [
        "set -eo pipefail",
        'export DISPLAY="${DISPLAY:-}"  # some Iridis profile hooks read it',
        "source /etc/profile",
        "# Drop conda state inherited from the submitting shell, so activation below",
        "# cannot silently leave the job on the wrong interpreter.",
        "unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER "
        "PYTHONPATH",
    ]
    if r.module:
        lines += ["module purge", f"module load {r.module}"]
    if r.container:
        binds = sorted(
            {str(p) for p in (e.repo_dir, e.data_dir, e.results_dir, e.state_dir)}
        )
        nv = "--nv " if r.gpu else ""
        lines += [
            f"PY=(apptainer exec {nv}--bind {q(','.join(binds))} {q(r.container)} "
            "python)",
        ]
    else:
        if r.conda_sh:
            lines.append(f"conda_sh={q(r.conda_sh)}")
        else:
            lines.append(
                'conda_sh="$(dirname "$(dirname "$(command -v conda)")")'
                '/etc/profile.d/conda.sh"'
            )
        lines += [
            'if [[ ! -f "$conda_sh" ]]; then echo "ERROR: no conda.sh at $conda_sh";'
            " exit 1; fi",
            'source "$conda_sh"',
            f"conda activate {q(r.env_name)}",
            'env_dir="$(basename "${CONDA_PREFIX:-none}")"',
            f'if [[ "$env_dir" != {q(r.env_name)} ]]; then',
            f'    echo "ERROR: expected conda env {r.env_name}, got $env_dir"',
            "    exit 1",
            "fi",
            "PY=(python)",
        ]
    lines += [
        f"export PYTHONPATH={q(str(e.repo_dir))}",
        f"export NUMBA_CACHE_DIR={q(str(e.numba_cache_dir))}",
        'mkdir -p "$NUMBA_CACHE_DIR"',
        "export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2",
        f"export LOKY_MAX_CPU_COUNT={threads}",
    ]
    lines += [f"export {var}=1" for var in _THREAD_VARS]
    if r.gpu:
        if not r.container:
            lines += [
                "# pip-installed CUDA libraries (tensorflow[and-cuda], torch wheels)",
                'cuda_libs=$(find "$CONDA_PREFIX/lib" -type d -path '
                "'*/site-packages/nvidia/*/lib' 2>/dev/null | paste -sd: || true)",
                'if [[ -n "$cuda_libs" ]]; then export '
                'LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi',
            ]
        lines += [
            'nvidia-smi || { echo "ERROR: no usable GPU in this allocation"; exit 1; }',
        ]
        if r.gpu_check == "tensorflow":
            lines.append(
                '"${PY[@]}" -c "import tensorflow as tf, sys; g = '
                "tf.config.list_physical_devices('GPU'); print('TensorFlow', "
                "tf.__version__, g); sys.exit(0 if g else 'ERROR: TensorFlow sees no "
                "GPU')\""
            )
        elif r.gpu_check == "torch":
            lines.append(
                '"${PY[@]}" -c "import torch, sys; print(\'PyTorch\', '
                "torch.__version__, torch.cuda.is_available()); sys.exit(0 if "
                "torch.cuda.is_available() else 'ERROR: PyTorch sees no GPU')\""
            )
    else:
        lines.append('export CUDA_VISIBLE_DEVICES=""')
    lines += list(r.setup)
    lines += [
        f"cd {q(str(e.repo_dir))}",
        "commit=$(git rev-parse HEAD)",
        f'if [[ "$commit" != {q(commit)} ]]; then',
        f'    echo "ERROR: repository moved from {commit} to $commit after submission"',
        "    exit 1",
        "fi",
        'echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$commit"',
        '"${PY[@]}" -c "import sys, aeon, tsml_eval; print(\'python\', '
        "sys.version.split()[0], sys.executable); print('aeon', aeon.__version__, "
        "aeon.__file__); print('tsml_eval', tsml_eval.__file__)\"",
    ]
    return lines


def _experiment_arguments(e, task):
    """Arguments after data_dir for one experiment, as a list of strings."""
    estimator = Estimator(task.estimator, task.category)
    args = [
        str(e.estimator_results(estimator)),
        task.estimator,
        task.dataset,
        str(task.resample),
    ]
    if e.train_files:
        args.append("-tr")
    if e.test_files:
        args.append("-te")
    if e.n_jobs > 1:
        args += ["-nj", str(e.n_jobs)]
    args += list(e.extra_args)
    for key, value in e.estimator_kwargs.get(task.estimator, {}).items():
        if isinstance(value, bool):
            args += ["-kw", key, "true" if value else "false", "bool"]
        elif isinstance(value, int):
            args += ["-kw", key, str(value), "int"]
        elif isinstance(value, float):
            args += ["-kw", key, repr(value), "float"]
        else:
            args += ["-kw", key, value, "str"]
    return args


def _entry_module(e):
    single, threaded, _, _ = TASKS[e.task]
    return f"tsml_eval.experiments.{threaded if e.n_jobs > 1 else single}"


def task_log(e, task, job_id):
    """Log path prefix of one task attempt: <logs>/<est>/<dataset>/<job>_<resample>."""
    return e.logs_dir / task.estimator / task.dataset / f"{job_id}_{task.resample}"


CANCELLED_BY_USER = "cancelled by user"


def cancel_marker(e, task):
    """File whose existence tells a farm to skip or kill one task."""
    return e.state_dir / "cancel" / task.estimator / task.dataset / str(task.resample)


def array_script(e, tasks, memory_gb, commit):
    """One Slurm array for one estimator and dataset; index = resample."""
    first = tasks[0]
    q = shlex.quote
    log_dir = e.logs_dir / first.estimator / first.dataset
    indices = ",".join(str(t.resample) for t in tasks)
    cpus = max(e.n_jobs, e.resources.cpus_per_task)
    args = _experiment_arguments(e, first)
    args[3] = "$SLURM_ARRAY_TASK_ID"
    rendered = " ".join(
        q(a) if a != "$SLURM_ARRAY_TASK_ID" else '"$SLURM_ARRAY_TASK_ID"' for a in args
    )
    lines = ["#!/bin/bash"]
    lines += _sbatch_lines(
        e,
        f"{e.job_prefix}-{first.estimator}-{first.dataset}",
        cpus,
        memory_gb,
        log_dir / "%A_%a.out",
        log_dir / "%A_%a.err",
        array=indices,
    )
    lines += [""]
    lines += _environment_block(e, commit, e.n_jobs)
    lines += [
        f'echo "experiment={e.name} estimator={first.estimator} '
        f'dataset={first.dataset} resample=$SLURM_ARRAY_TASK_ID memory={memory_gb}G"',
        'echo "started $(date -Iseconds) $(hostname)" > '
        f'{q(str(log_dir))}/"${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.status"',
        f'"${{PY[@]}}" -u -m {_entry_module(e)} {q(str(e.data_dir))} {rendered}',
        "",
    ]
    return "\n".join(lines)


def farm_shape(e, memory_gb, n_tasks):
    """Parallel slots and CPUs for one farm job at a memory tier."""
    r = e.resources
    slots = min(r.node_cpus // e.n_jobs, r.node_memory_gb // memory_gb, n_tasks)
    return max(slots, 1)


def farm_script(e, tasks, memory_gb, commit, slots):
    """One node job running many experiments, `slots` at a time.

    Every task writes <log>.out, <log>.err and <log>.status, where the status file
    records a start line and then the exit code. A task with no status file never
    started, and one with no exit line was still running when the job ended.
    """
    q = shlex.quote
    farm_dir = e.state_dir / "farms"
    lines = ["#!/bin/bash"]
    lines += _sbatch_lines(
        e,
        f"{e.job_prefix}-farm-{memory_gb}G",
        slots * e.n_jobs,
        slots * memory_gb,
        farm_dir / "%j.out",
        farm_dir / "%j.err",
    )
    lines += [""]
    lines += _environment_block(e, commit, e.n_jobs)
    lines += ["", "task_args() {", '    case "$1" in']
    for i, task in enumerate(tasks):
        args = " ".join(q(a) for a in _experiment_arguments(e, task))
        log = q(str(task_log(e, task, "JOBID")).replace("JOBID", "@@"))
        cancel = q(str(cancel_marker(e, task)))
        lines.append(f"        {i}) ARGS=({args}); LOG={log}; CANCEL={cancel} ;;")
    lines += [
        "    esac",
        '    LOG="${LOG//@@/$SLURM_JOB_ID}"',
        "}",
        "",
        "# A task whose cancel marker exists is skipped, or killed within 5 s if it",
        "# is running (`tsml_slurm.py cancel`). setsid gives each task its own",
        "# process group, so joblib workers are killed with it.",
        "SETSID=(); command -v setsid > /dev/null && SETSID=(setsid)",
        "run_task() {",
        '    task_args "$1"',
        '    if [[ -e "$CANCEL" ]]; then',
        '        echo "cancelled $(date -Iseconds)" > "$LOG.status"',
        "        return",
        "    fi",
        '    echo "started $(date -Iseconds) $(hostname)" > "$LOG.status"',
        f'    "${{SETSID[@]}}" "${{PY[@]}}" -u -m {_entry_module(e)} '
        f'{q(str(e.data_dir))} "${{ARGS[@]}}" > "$LOG.out" 2> "$LOG.err" &',
        "    local pid=$!",
        '    while kill -0 "$pid" 2>/dev/null; do',
        '        if [[ -e "$CANCEL" ]]; then',
        '            kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null',
        '            echo "cancelled $(date -Iseconds)" >> "$LOG.status"',
        "            break",
        "        fi",
        "        sleep 5",
        "    done",
        '    wait "$pid"',
        "    local code=$?",
        '    echo "exit $code" >> "$LOG.status"',
        "}",
        "",
        "set +e",
        f"slots={slots}",
        f'echo "experiment={e.name} farm of {len(tasks)} tasks, $slots at a time, '
        f'{memory_gb}G each"',
        f"for i in $(seq 0 {len(tasks) - 1}); do",
        "    while (( $(jobs -rp | wc -l) >= slots )); do wait -n; done",
        '    run_task "$i" &',
        "done",
        "wait",
        'echo "farm finished $(date -Iseconds)"',
        "",
    ]
    return "\n".join(lines)


def supervisor_script(e, python):
    """Slurm job that runs one cycle and then arms the next."""
    s = e.supervisor
    q = shlex.quote
    out = e.state_dir / "supervisor"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={e.supervisor_job_name}",
        f"#SBATCH --partition={s.partition}",
        f"#SBATCH --time={s.time_limit}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={s.memory_gb}G",
        f"#SBATCH --begin=now+{s.interval_minutes}minutes",
        f"#SBATCH --output={out}/%j.out",
        f"#SBATCH --error={out}/%j.out",
        "#SBATCH --mail-type=FAIL",
    ]
    if e.email:
        lines.append(f"#SBATCH --mail-user={e.email}")
    if s.account:
        lines.append(f"#SBATCH --account={s.account}")
    if s.qos:
        lines.append(f"#SBATCH --qos={s.qos}")
    lines += [f"#SBATCH {extra}" for extra in s.extra_sbatch]
    lines += [
        "",
        "unset PYTHONPATH",
        f"cd {q(str(e.repo_dir))}",
        f"{q(python)} -u {q(str(SCRIPT_PATH))} submit {q(str(e.config_path))} --chain",
        "",
    ]
    return "\n".join(lines)


# ======================================================================================
# State
# ======================================================================================


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_state(e):
    """Load the experiment's state file, or a fresh state."""
    path = e.state_dir / "state.json"
    if not path.is_file():
        return {
            "version": STATE_VERSION,
            "experiment": e.name,
            "fingerprints": {},
            "commit": None,
            "tasks": {},
            "submissions": [],
            "last_email": 0,
        }
    with path.open(encoding="utf-8") as file:
        state = json.load(file)
    if state.get("version") != STATE_VERSION:
        raise RuntimeError(f"Unsupported state version in {path}")
    return state


def save_state(e, state):
    """Write the state file atomically."""
    e.state_dir.mkdir(parents=True, exist_ok=True)
    path = e.state_dir / "state.json"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(state, file, indent=1, sort_keys=True)
    tmp.replace(path)


class Lock:
    """Non-blocking advisory lock, so two cycles never submit the same work."""

    def __init__(self, path):
        self.path = path
        self.file = None

    def __enter__(self):
        """Acquire the lock or raise RuntimeError."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+")
        try:
            import fcntl

            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # Windows: inspection only, never submits
            pass
        except BlockingIOError:
            self.file.close()
            raise RuntimeError(f"Another cycle holds {self.path}") from None
        return self

    def __exit__(self, *exc):
        """Release the lock."""
        self.file.close()


def git_revision(repo):
    """Return the branch, commit and dirty flag of a repository."""

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current") or "DETACHED"
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return branch, commit, dirty


# ======================================================================================
# Diagnosis
# ======================================================================================


def _read_tail(path, limit=20000):
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - limit))
            return file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _classify_text(text):
    lower = text.lower()
    if re.search(
        r"cuda out of memory|resourceexhaustederror|cuda_error_out_of_memory", lower
    ):
        return GPU_OOM
    if re.search(
        r"oom[_ -]?kill|out[ -]of[ -]memory|memoryerror|unable to allocate", lower
    ):
        return OOM
    if "due to time limit" in lower or "time limit exceeded" in lower:
        return TIMEOUT
    if "traceback (most recent call last)" in lower:
        return ERROR
    if "cancelled" in lower:
        return CANCELLED
    return None


def _last_error_line(text):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line and not line.startswith(("File ", "^", "~")):
            return line[:200]
    return ""


def diagnose(e, task, record, slurm_state):
    """Return (reason, detail) for a task whose job ended without a result."""
    log = task_log(e, task, record["job"])
    text = _read_tail(Path(f"{log}.err")) + _read_tail(Path(f"{log}.out"))
    detail = _last_error_line(_read_tail(Path(f"{log}.err")))
    state = (slurm_state or "").upper()
    if record.get("kind") == "farm":
        status = _read_tail(Path(f"{log}.status"))
        if not status:
            return NOT_STARTED, ""
        match = re.search(r"^exit (-?\d+)", status, re.MULTILINE)
        if match is None:  # running when the farm job ended
            return {
                "TIMEOUT": TIMEOUT,
                "OUT_OF_MEMORY": OOM,
                "CANCELLED": CANCELLED,
                "NODE_FAIL": INFRA,
                "PREEMPTED": INFRA,
            }.get(state, KILLED), ""
        code = int(match.group(1))
        if code == 0:
            return NO_RESULT, detail
        reason = _classify_text(text)
        if code in (137, -9):  # SIGKILL: how the cgroup OOM killer ends a process
            return (GPU_OOM if reason == GPU_OOM else OOM), detail
        return reason or UNKNOWN, detail
    if state == "OUT_OF_MEMORY":
        return OOM, detail
    if state == "TIMEOUT":
        return TIMEOUT, detail
    if state in ("NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "REQUEUED"):
        return INFRA, ""
    if state == "CANCELLED":
        return CANCELLED, ""
    reason = _classify_text(text)
    if reason:
        return reason, detail
    if state == "COMPLETED":
        return NO_RESULT, detail
    if not text and not state:
        return NOT_STARTED, ""
    return UNKNOWN, detail


def apply_outcome(e, record, reason, detail):
    """Update a task record after a failure; decide retry, escalate or stop."""
    tiers = e.resources.memory_gb
    record.setdefault("events", []).append(
        {
            "at": _now(),
            "job": record.get("job"),
            "reason": reason,
            "memory_gb": tiers[record.get("tier", 0)],
            "detail": detail,
        }
    )
    record["job"] = None
    if reason in (NOT_STARTED, INFRA):
        record["attempts"] = max(0, record.get("attempts", 1) - 1)  # not its fault
    elif reason == OOM:
        if record.get("tier", 0) + 1 < len(tiers):
            record["tier"] = record.get("tier", 0) + 1
            record["attempts"] = max(0, record.get("attempts", 1) - 1)
        else:
            record["terminal"] = f"OOM at {tiers[-1]}G"
    elif reason == TIMEOUT:
        record["terminal"] = TIMEOUT
    elif record.get("attempts", 0) >= e.max_attempts:
        record["terminal"] = reason


# ======================================================================================
# The cycle
# ======================================================================================


@dataclass
class CycleResult:
    """Outcome of one controller cycle."""

    report: str
    settled: bool
    submitted: int = 0
    errors: list = field(default_factory=list)


def _reconcile(e, state, queue, slurm, live_grace_seconds=600):
    """Find this experiment's live tasks and diagnose the ones that ended."""
    live_arrays = {(q.job_id, q.index): q.state for q in queue}
    live_jobs = {q.job_id: q.state for q in queue}
    active = {}
    ended = []
    for key, record in state["tasks"].items():
        job = record.get("job")
        if not job:
            continue
        resample = int(key.rsplit("|", 1)[1])
        if record.get("kind") == "farm":
            live = live_jobs.get(job)
        else:
            # (job, None): an array squeue has not expanded into elements yet
            live = live_arrays.get((job, resample)) or live_arrays.get((job, None))
        if live is not None:
            active[key] = "RUNNING" if live == "RUNNING" else "PENDING"
        else:
            ended.append(key)
    accounting = slurm.accounting({state["tasks"][k]["job"] for k in ended})
    now = time.time()
    for key in ended:
        record = state["tasks"][key]
        estimator, dataset, resample = key.split("|")
        category = next(x.category for x in e.estimators if x.name == estimator)
        task = Task(estimator, category, dataset, int(resample))
        if is_complete(e, task):
            record["job"] = None
            continue
        index = None if record.get("kind") == "farm" else int(resample)
        slurm_state = accounting.get((record["job"], index))
        if slurm_state is None and record.get("kind") == "farm":
            slurm_state = accounting.get((record["job"], None))
        if slurm_state in ("PENDING", "RUNNING", "REQUEUED", "COMPLETING") or (
            slurm_state is None
            and now - record.get("submitted_at", 0) < live_grace_seconds
        ):
            active[key] = "PENDING"  # not visible to squeue yet
            continue
        reason, detail = diagnose(e, task, record, slurm_state)
        apply_outcome(e, record, reason, detail)
    return active


def _plan(e, state, missing, active, queue):
    """Choose which missing tasks to submit this cycle and group them into jobs."""
    r = e.resources
    eligible = [
        t
        for t in missing
        if t.key not in active and not state["tasks"].get(t.key, {}).get("terminal")
    ]
    eligible = order_tasks(e, eligible)
    tier_of = {t.key: state["tasks"].get(t.key, {}).get("tier", 0) for t in eligible}
    jobs = []
    if r.mode == "array":
        capacity = r.max_active - len(active)
        if r.max_user_tasks:
            in_partition = sum(1 for q in queue if q.partition == r.partition)
            capacity = min(capacity, r.max_user_tasks - in_partition)
        grouped = OrderedDict()
        for task in eligible[: max(0, capacity)]:
            grouped.setdefault(
                (task.estimator, task.dataset, tier_of[task.key]), []
            ).append(task)
        for (_, _, tier), tasks in grouped.items():
            jobs.append(("array", tier, tasks, None))
    else:
        live_farms = {
            state["tasks"][k]["job"]
            for k in active
            if state["tasks"][k].get("kind") == "farm"
        }
        free_jobs = r.max_jobs - len(live_farms)
        remaining = list(eligible)
        while free_jobs > 0 and remaining:
            tier = tier_of[remaining[0].key]
            same = [t for t in remaining if tier_of[t.key] == tier]
            memory = r.memory_gb[tier]
            slots = farm_shape(e, memory, len(same))
            chosen = same[: slots * r.waves]
            slots = farm_shape(e, memory, len(chosen))
            chosen_keys = {t.key for t in chosen}
            remaining = [t for t in remaining if t.key not in chosen_keys]
            jobs.append(("farm", tier, chosen, slots))
            free_jobs -= 1
    return jobs


def _submit_jobs(e, state, jobs, commit, slurm, dry_run):
    submitted, errors, scripts = 0, [], []
    for kind, tier, tasks, slots in jobs:
        memory = e.resources.memory_gb[tier]
        if kind == "array":
            script = array_script(e, tasks, memory, commit)
        else:
            script = farm_script(e, tasks, memory, commit, slots)
        scripts.append(script)
        if dry_run:
            continue
        for task in tasks:
            (e.logs_dir / task.estimator / task.dataset).mkdir(
                parents=True, exist_ok=True
            )
        (e.state_dir / "farms").mkdir(parents=True, exist_ok=True)
        e.numba_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            job_id = slurm.submit(script)
        except (OSError, subprocess.CalledProcessError) as error:
            stderr = getattr(error, "stderr", "") or ""
            errors.append(f"{tasks[0].estimator}/{tasks[0].dataset}: {error} {stderr}")
            continue
        script_dir = e.state_dir / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        (script_dir / f"{job_id}.sh").write_text(script, encoding="utf-8")
        state["submissions"].append(
            {
                "job": job_id,
                "kind": kind,
                "tasks": len(tasks),
                "memory_gb": memory,
                "commit": commit,
                "at": _now(),
            }
        )
        for task in tasks:
            record = state["tasks"].setdefault(task.key, {"attempts": 0, "tier": 0})
            record.update(job=job_id, kind=kind, submitted_at=time.time())
            record["attempts"] = record.get("attempts", 0) + 1
            submitted += 1
        save_state(e, state)
    return submitted, errors, scripts


def run_cycle(
    e, slurm=None, submit=True, dry_run=False, accept_commit=False, email=None
):
    """Reconcile results and queue, optionally submit, and report."""
    slurm = slurm or Slurm()
    read_datasets(e)
    stop_file = e.state_dir / "STOP"
    if submit and not dry_run and stop_file.exists():
        submit = False
        stopped = True
    else:
        stopped = stop_file.exists()
    state = load_state(e)
    errors = []

    prints = fingerprints(e)
    changed = sorted(
        name
        for name, value in prints.items()
        if state["fingerprints"].get(name, value) != value
    )
    if changed and submit:
        raise RuntimeError(
            f"Settings for {', '.join(changed)} (keyword arguments, flags or paths) "
            "changed since their first submission, so new results would be mixed "
            "with old ones. Give the changed estimator a new name or results_dir, "
            "or start a new experiment."
        )
    branch, commit, dirty = git_revision(e.repo_dir)
    if submit:
        if e.branch and branch != e.branch:
            raise RuntimeError(
                f"{e.repo_dir} is on {branch}, experiment needs {e.branch}"
            )
        if e.require_clean and dirty:
            raise RuntimeError(
                f"{e.repo_dir} has uncommitted changes; commit them so results are "
                "traceable (or set require_clean = false)"
            )
        if e.pin_commit and state["commit"] not in (None, commit) and not accept_commit:
            raise RuntimeError(
                f"The experiment started on commit {state['commit'][:10]} but the "
                f"repository is now at {commit[:10]}. Check out the original commit, "
                "or pass --accept-commit to mix results from both."
            )

    queue, queue_error = [], None
    if slurm.available():
        try:
            queue = slurm.queue(e.user)
        except (OSError, subprocess.CalledProcessError) as error:
            queue_error = f"squeue failed: {error}"
    else:
        queue_error = "squeue not found: Slurm state unknown, nothing submitted"
    if queue_error:
        errors.append(queue_error)
        submit = False
        active = {k: "UNKNOWN" for k, r in state["tasks"].items() if r.get("job")}
    else:
        active = _reconcile(e, state, queue, slurm)

    tasks = list(all_tasks(e))
    missing = [t for t in tasks if not is_complete(e, t)]
    submitted, scripts = 0, []
    if submit or dry_run:
        jobs = _plan(e, state, missing, active, queue)
        if dry_run:
            _, _, scripts = _submit_jobs(e, state, jobs, commit, slurm, dry_run=True)
            submitted = sum(len(j[2]) for j in jobs)
        elif jobs:
            for job in jobs:
                for task in job[2]:
                    state["fingerprints"][task.estimator] = prints[task.estimator]
            if state["commit"] is None or accept_commit or not e.pin_commit:
                state["commit"] = commit
            submitted, submit_errors, _ = _submit_jobs(
                e, state, jobs, commit, slurm, dry_run=False
            )
            errors += submit_errors
            for job in jobs:
                for task in job[2]:
                    if state["tasks"].get(task.key, {}).get("job"):
                        active[task.key] = "PENDING"

    terminal = {
        t.key: state["tasks"][t.key]["terminal"]
        for t in missing
        if state["tasks"].get(t.key, {}).get("terminal")
    }
    settled = (
        not queue_error
        and all(t.key in terminal for t in missing if t.key not in active)
        and not any(t.key in active for t in missing)
    )
    report = compose_report(
        e,
        tasks,
        missing,
        active,
        terminal,
        state,
        branch,
        commit,
        dirty,
        submitted,
        errors,
        stopped,
        dry_run,
    )
    if not dry_run and not queue_error:
        save_state(e, state)
    if not dry_run:
        e.state_dir.mkdir(parents=True, exist_ok=True)
        (e.state_dir / "report.txt").write_text(report, encoding="utf-8")
    if dry_run and scripts:
        report += "\n----- first job script that would be submitted -----\n"
        report += scripts[0]
    if email is not None and not dry_run:
        due = time.time() - state.get("last_email", 0) >= e.email_hours * 3600
        if email == "always" or (email == "due" and (due or settled)):
            status = send_email(e, report, settled)
            report += f"\n{status}\n"
            if status.startswith("Email sent"):
                state["last_email"] = time.time()
                save_state(e, state)
    return CycleResult(report, settled, submitted, errors)


def compose_report(
    e,
    tasks,
    missing,
    active,
    terminal,
    state,
    branch,
    commit,
    dirty,
    submitted,
    errors,
    stopped,
    dry_run,
):
    """Format the progress report."""
    total = len(tasks)
    done = total - len(missing)
    missing_keys = {t.key for t in missing}
    running = sum(1 for k, s in active.items() if s == "RUNNING" and k in missing_keys)
    pending = sum(1 for k, s in active.items() if s != "RUNNING" and k in missing_keys)
    waiting = len(missing) - running - pending - len(terminal)
    lines = [
        f"Experiment {e.name}: {e.task} on {e.cluster}/{e.resource_name} "
        f"({e.resources.mode} mode, partition {e.resources.partition})",
        f"Updated {_now()} on {socket.gethostname()}",
        f"Repository {e.repo_dir} {branch} {commit[:10]}{' (dirty)' if dirty else ''}"
        + (
            f", experiment pinned to {state['commit'][:10]}"
            if state.get("commit")
            else ""
        ),
        f"Results {e.results_dir}",
        "",
        f"Complete {done}/{total} ({100 * done / max(total, 1):.1f}%)  "
        f"running {running}  pending {pending}  failed {len(terminal)}  "
        f"not yet submitted {max(waiting, 0)}",
    ]
    if stopped:
        lines.append(f"STOP file present: no new submissions ({e.state_dir / 'STOP'})")
    if dry_run:
        lines.append(f"Dry run: would submit {submitted} tasks")
    elif submitted:
        lines.append(f"Submitted {submitted} tasks this cycle")
    lines += ["", f"{'estimator':<36}{'done':>12}{'run':>7}{'pend':>7}{'fail':>7}"]
    per = defaultdict(lambda: [0, 0, 0, 0, 0])
    for task in tasks:
        label = f"{task.category}/{task.estimator}" if task.category else task.estimator
        row = per[label]
        row[1] += 1
        if task.key not in missing_keys:
            row[0] += 1
        elif active.get(task.key) == "RUNNING":
            row[2] += 1
        elif task.key in active:
            row[3] += 1
        elif task.key in terminal:
            row[4] += 1
    for label, (d, t, r, p, f) in per.items():
        lines.append(f"{label:<36}{f'{d}/{t}':>12}{r:>7}{p:>7}{f:>7}")
    complete_datasets = sum(
        1 for dataset in e.datasets if not any(t.dataset == dataset for t in missing)
    )
    lines.append(
        f"\nDatasets complete for every estimator: "
        f"{complete_datasets}/{len(e.datasets)}"
    )
    if terminal:
        lines += [
            "",
            f"Failed tasks (will not be retried; see `retry`): {len(terminal)}",
        ]
        by_reason = defaultdict(int)
        for reason in terminal.values():
            by_reason[reason] += 1
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
            lines.append(f"  {count:>6}  {reason}")
        for key in list(terminal)[:15]:
            events = state["tasks"][key].get("events", [])
            last = events[-1] if events else {}
            estimator, dataset, resample = key.split("|")
            log = e.logs_dir / estimator / dataset / f"{last.get('job')}_{resample}"
            lines.append(f"  {key}: {terminal[key]} {last.get('detail', '')}".rstrip())
            lines.append(f"      {log}.err")
        if len(terminal) > 15:
            lines.append(f"  ... and {len(terminal) - 15} more")
    if errors:
        lines += ["", "Errors:"] + [f"  {error}" for error in errors]
    return "\n".join(lines) + "\n"


def send_email(e, report, settled):
    """Email a report with mail, mailx or sendmail."""
    if not e.email:
        return "Email not sent: no address configured"
    subject = f"[tsml-slurm] {e.name}: {'settled' if settled else 'progress'}"
    for program in ("mail", "mailx"):
        if shutil.which(program):
            try:
                subprocess.run(
                    [program, "-s", subject, e.email],
                    input=report,
                    text=True,
                    check=True,
                    capture_output=True,
                )
                return f"Email sent to {e.email}"
            except (OSError, subprocess.CalledProcessError) as error:
                return f"Email failed: {error}"
    if shutil.which("sendmail"):
        message = f"To: {e.email}\nSubject: {subject}\n\n{report}"
        try:
            subprocess.run(
                ["sendmail", "-t"],
                input=message,
                text=True,
                check=True,
                capture_output=True,
            )
            return f"Email sent to {e.email}"
        except (OSError, subprocess.CalledProcessError) as error:
            return f"Email failed: {error}"
    return "Email not sent: no mail, mailx or sendmail on this host"


# ======================================================================================
# Supervision
# ======================================================================================


def _live_supervisors(e, slurm):
    if not slurm.available():
        return []
    return [q.job_id for q in slurm.queue(e.user) if q.name == e.supervisor_job_name]


def arm_supervisor(e, slurm, python=None):
    """Submit the next self-resubmitting supervisor job unless one is queued."""
    own = os.environ.get("SLURM_JOB_ID")
    others = [j for j in _live_supervisors(e, slurm) if j != own]
    if others:
        return f"Supervisor already queued: job {others[0]}"
    (e.state_dir / "supervisor").mkdir(parents=True, exist_ok=True)
    job = slurm.submit(supervisor_script(e, python or sys.executable))
    return (
        f"Next supervisor cycle is job {job}, in "
        f"{e.supervisor.interval_minutes} minutes"
    )


def start_login_supervisor(e):
    """Run `loop` detached on the login node, in screen when available."""
    try:
        with Lock(e.state_dir / "supervisor.lock"):
            pass
    except RuntimeError:
        return f"Supervisor already running (screen -r tsml-{e.name})"
    command = [sys.executable, "-u", str(SCRIPT_PATH), "loop", str(e.config_path)]
    log = e.state_dir / "supervisor" / "loop.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    session = f"tsml-{e.name}"
    if shutil.which("screen"):
        shell = f"{shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
        subprocess.run(["screen", "-dmS", session, "bash", "-c", shell], check=True)
        return f"Supervisor running in screen session {session}; log {log}"
    with log.open("a") as handle:
        subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
    return f"Supervisor running detached (no screen found); log {log}"


def loop(e, slurm=None):
    """Login-node supervisor: cycle every interval until settled or stopped."""
    slurm = slurm or Slurm()
    with Lock(e.state_dir / "supervisor.lock"):
        while True:
            if (e.state_dir / "STOP").exists():
                print(f"{_now()} STOP file found; supervisor exiting")
                return 0
            try:
                result = run_cycle(e, slurm, email="due")
                print(result.report, flush=True)
                if result.settled:
                    print(f"{_now()} all work settled; supervisor exiting")
                    return 0
            except Exception as error:  # keep supervising through transient faults
                print(
                    f"{_now()} cycle failed: {type(error).__name__}: {error}",
                    flush=True,
                )
            time.sleep(e.supervisor.interval_minutes * 60)


# ======================================================================================
# Commands
# ======================================================================================


def cmd_check(e, args):
    """Validate an experiment and print an example job script."""
    read_datasets(e)
    print(f"Experiment {e.name} ({e.config_path})")
    print(f"  cluster {e.cluster}, resources {e.resource_name}:")
    for item in fields(Resources):
        value = getattr(e.resources, item.name)
        if value not in ("", (), None):
            print(f"    {item.name} = {value}")
    print(
        f"  supervisor: {e.supervisor.mode}, every "
        f"{e.supervisor.interval_minutes} minutes"
    )
    print(f"  state and logs: {e.state_dir}")
    print(f"  task {e.task}, entry module {_entry_module(e)}")
    print(
        f"  {len(e.estimators)} estimators x {len(e.datasets)} datasets x "
        f"{len(e.resamples)} resamples"
    )
    problems = []
    for label, path in (("repo_dir", e.repo_dir), ("data_dir", e.data_dir)):
        if not path.is_dir():
            problems.append(f"{label} does not exist: {path}")
    if e.data_dir.is_dir():
        absent = [d for d in e.datasets if not (e.data_dir / d).is_dir()]
        if absent:
            problems.append(
                f"{len(absent)} datasets missing from {e.data_dir}: "
                + ", ".join(absent[:10])
                + (" ..." if len(absent) > 10 else "")
            )
    if e.repo_dir.is_dir():
        try:
            branch, commit, dirty = git_revision(e.repo_dir)
            print(f"  repository {branch} {commit[:10]}{' (dirty)' if dirty else ''}")
            if e.branch and branch != e.branch:
                problems.append(f"repository on {branch}, experiment needs {e.branch}")
            if dirty and e.require_clean:
                problems.append("repository has uncommitted changes")
        except (OSError, subprocess.CalledProcessError) as error:
            problems.append(f"git failed in {e.repo_dir}: {error}")
    report, failed = construct_estimators(e)
    print(report)
    if failed:
        problems.append(f"estimator(s) could not be constructed: {', '.join(failed)}")
    tasks = list(all_tasks(e))
    missing = [t for t in tasks if not is_complete(e, t)]
    print(f"  {len(tasks) - len(missing)}/{len(tasks)} results already on disk")
    if missing:
        sample = order_tasks(e, missing)[:1]
        tier = e.resources.memory_gb[0]
        if e.resources.mode == "array":
            script = array_script(e, sample, tier, "COMMIT")
        else:
            script = farm_script(
                e, sample, tier, "COMMIT", farm_shape(e, tier, len(missing))
            )
        print("\n----- example job script -----\n" + script)
    if problems:
        print("PROBLEMS:\n" + "\n".join(f"  - {p}" for p in problems))
        return 1
    print("No problems found.")
    return 0


def construct_estimators(e):
    """Build each estimator with the tsml-eval factory, if tsml_eval imports."""
    _, _, module, function = TASKS[e.task]
    sys.path.insert(0, str(e.repo_dir))
    try:
        factory = getattr(
            __import__(f"tsml_eval.experiments.{module}", fromlist=[function]), function
        )
    except Exception as error:
        return (
            f"  estimator construction skipped (tsml_eval not importable with "
            f"{sys.executable}: {type(error).__name__}). Run check with the "
            "environment's python to test the estimator names."
        ), []
    lines, failed = [], []
    for estimator in e.estimators:
        try:
            built = factory(
                estimator.name,
                random_state=0,
                **e.estimator_kwargs.get(estimator.name, {}),
            )
            lines.append(f"  ok   {estimator.name} -> {type(built).__name__}")
        except Exception as error:
            lines.append(f"  FAIL {estimator.name}: {type(error).__name__}: {error}")
            failed.append(estimator.name)
    return "\n".join(lines), failed


def cmd_submit(e, args):
    """Run one cycle, optionally arming the next supervisor."""
    slurm = Slurm()
    with Lock(e.state_dir / "controller.lock"):
        result = run_cycle(
            e,
            slurm,
            submit=True,
            dry_run=args.dry_run,
            accept_commit=args.accept_commit,
            email=None if args.dry_run or args.no_email else "due",
        )
    print(result.report)
    if args.chain and not args.dry_run:
        if result.settled:
            print("All work settled; supervisor chain ends.")
        elif (e.state_dir / "STOP").exists():
            print("STOP file present; supervisor chain ends.")
        else:
            print(arm_supervisor(e, slurm))
    return 1 if result.errors else 0


def cmd_start(e, args):
    """Submit now and start the supervisor."""
    if (e.state_dir / "STOP").exists():
        (e.state_dir / "STOP").unlink()
        print("Removed STOP file.")
    args.chain = False
    args.no_email = True
    status = cmd_submit(e, args)
    if args.dry_run:
        return status
    if e.supervisor.mode == "slurm":
        print(arm_supervisor(e, Slurm()))
    else:
        print(start_login_supervisor(e))
    return status


def cmd_status(e, args):
    """Print the progress report without submitting."""
    slurm = Slurm()
    try:
        with Lock(e.state_dir / "controller.lock"):
            result = run_cycle(
                e, slurm, submit=False, email="always" if args.email else None
            )
        print(result.report)
    except RuntimeError as error:  # a cycle is running: show its last report
        print(f"{error}; showing the last saved report")
        print((e.state_dir / "report.txt").read_text(encoding="utf-8"))
    if args.running:
        print()
        print(running_report(e, load_state(e), slurm.queue(e.user)))
    if args.details:
        state = load_state(e)
        missing = [t for t in all_tasks(e) if not is_complete(e, t)]
        for task in missing:
            record = state["tasks"].get(task.key, {})
            print(
                f"{task.key}: attempts={record.get('attempts', 0)} "
                f"memory={e.resources.memory_gb[record.get('tier', 0)]}G "
                f"job={record.get('job')} {record.get('terminal', '')}"
            )
    return 0


def _matches(task, args):
    """Whether a task passes the --estimator/--dataset/--resample filters."""
    return (
        (not args.estimator or task.estimator in args.estimator)
        and (not args.dataset or task.dataset in args.dataset)
        and (args.resample is None or task.resample in args.resample)
    )


def _started(log):
    """(start time, host) from a task's .status file, or (None, "")."""
    text = _read_tail(Path(f"{log}.status"))
    match = re.search(r"^started (\S+) ?(\S*)", text, re.MULTILINE)
    if not match:
        return None, ""
    try:
        return datetime.fromisoformat(match.group(1)), match.group(2)
    except ValueError:
        return None, match.group(2)


def _elapsed(start):
    seconds = int((datetime.now(start.tzinfo) - start).total_seconds())
    hours, rest = divmod(max(seconds, 0), 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}"


def running_report(e, state, queue):
    """Table of tasks whose job is running, longest running first."""
    live = {(q.job_id, q.index): q.state for q in queue}
    live.update({(q.job_id, None): q.state for q in queue if q.index is None})
    rows = []
    for key, record in state["tasks"].items():
        job = record.get("job")
        estimator, dataset, resample = key.split("|")
        farm = record.get("kind") == "farm"
        if not job or live.get((job, None if farm else int(resample))) != "RUNNING":
            continue
        category = next(x.category for x in e.estimators if x.name == estimator)
        task = Task(estimator, category, dataset, int(resample))
        if is_complete(e, task):
            continue
        log = task_log(e, task, job)
        start, host = _started(log)
        status = _read_tail(Path(f"{log}.status"))
        if farm and start is None:
            note = "waiting in farm"
        elif re.search(r"^(exit|cancelled)", status, re.MULTILINE):
            note = "ended without a result; diagnosed when the farm ends"
        else:
            note = ""
        memory = e.resources.memory_gb[record.get("tier", 0)]
        rows.append((start, key, job, host, memory, note))
    if not rows:
        return "No tasks running."
    rows.sort(key=lambda r: (r[0] is None, r[0].timestamp() if r[0] else 0))
    lines = [f"{'task':<44}{'job':>10}{'memory':>8}{'elapsed':>11}  node"]
    for start, key, job, host, memory, note in rows:
        elapsed = _elapsed(start) if start else "-"
        lines.append(
            f"{key:<44}{job:>10}{f'{memory}G':>8}{elapsed:>11}  {host} {note}".rstrip()
        )
    return "\n".join(lines)


def cmd_cancel(e, args):
    """Cancel matching tasks and keep them from being submitted again."""
    if not (args.estimator or args.dataset or args.resample is not None):
        raise ConfigError(
            "Give --estimator, --dataset and/or --resample; to cancel everything "
            "use `stop --cancel`"
        )
    read_datasets(e)
    unknown = sorted(set(args.estimator or []) - {x.name for x in e.estimators})
    unknown += sorted(set(args.dataset or []) - set(e.datasets))
    if unknown:
        raise ConfigError(f"Not in this experiment: {', '.join(unknown)}")
    slurm = Slurm()
    queue = slurm.queue(e.user) if slurm.available() else []
    live = {(q.job_id, q.index) for q in queue}
    to_scancel, in_farms, held = set(), 0, 0
    with Lock(e.state_dir / "controller.lock"):
        state = load_state(e)
        for task in all_tasks(e):
            if not _matches(task, args) or is_complete(e, task):
                continue
            record = state["tasks"].setdefault(task.key, {"attempts": 0, "tier": 0})
            job = record.get("job")
            if job and record.get("kind") == "farm" and (job, None) in live:
                marker = cancel_marker(e, task)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(f"cancelled {_now()}\n", encoding="utf-8")
                in_farms += 1
            elif job and ((job, task.resample) in live or (job, None) in live):
                to_scancel.add(f"{job}_{task.resample}")
            else:
                held += 1
            record["job"] = None
            record["terminal"] = CANCELLED_BY_USER
        save_state(e, state)
    slurm.cancel(to_scancel)
    print(
        f"Cancelled {len(to_scancel)} array task(s) and {in_farms} farm task(s) "
        "(a running farm task is killed within seconds, a waiting one is skipped). "
        f"{held} task(s) not in the queue will not be submitted."
    )
    print("Undo with: retry --reason cancelled")
    return 0


def cmd_stop(e, args):
    """Stop submissions, and optionally cancel jobs."""
    e.state_dir.mkdir(parents=True, exist_ok=True)
    (e.state_dir / "STOP").write_text(f"stopped {_now()}\n", encoding="utf-8")
    print(f"Created {e.state_dir / 'STOP'}: no further submissions.")
    slurm = Slurm()
    if slurm.available():
        supervisors = _live_supervisors(e, slurm)
        slurm.cancel(supervisors)
        if supervisors:
            print(f"Cancelled supervisor job(s) {', '.join(supervisors)}")
        if args.cancel:
            state = load_state(e)
            jobs = {r["job"] for r in state["tasks"].values() if r.get("job")}
            live = {q.job_id for q in slurm.queue(e.user)}
            slurm.cancel(jobs & live)
            print(f"Cancelled {len(jobs & live)} experiment job(s).")
    if e.supervisor.mode == "login" and shutil.which("screen"):
        subprocess.run(["screen", "-S", f"tsml-{e.name}", "-X", "quit"], check=False)
    print(
        "Running jobs finish normally unless --cancel was given. "
        "`start` removes the STOP file."
    )
    return 0


def cmd_retry(e, args):
    """Clear failure records so tasks are submitted again."""
    with Lock(e.state_dir / "controller.lock"):
        state = load_state(e)
        cleared = 0
        for key, record in state["tasks"].items():
            terminal = record.get("terminal")
            if not terminal or (
                args.reason and args.reason.lower() not in terminal.lower()
            ):
                continue
            record.pop("terminal")
            record["attempts"] = 0
            estimator, dataset, resample = key.split("|")
            marker = e.state_dir / "cancel" / estimator / dataset / resample
            if marker.is_file():
                marker.unlink()
            if args.memory_tier is not None:
                record["tier"] = min(args.memory_tier, len(e.resources.memory_gb) - 1)
            cleared += 1
        save_state(e, state)
    print(f"{cleared} failed task(s) will be submitted again on the next cycle.")
    return 0


def cmd_loop(e, args):
    """Run the login-node supervisor loop."""
    return loop(e)


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, function, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", type=Path, help="experiment TOML file")
        p.set_defaults(function=function)
        return p

    add("check", cmd_check, "validate an experiment and show an example job")
    p = add("submit", cmd_submit, "run one reconcile-and-submit cycle")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be submitted, change nothing",
    )
    p.add_argument(
        "--accept-commit",
        action="store_true",
        help="allow a repository commit different from the pinned one",
    )
    p.add_argument("--no-email", action="store_true")
    p.add_argument("--chain", action="store_true", help=argparse.SUPPRESS)
    p = add("start", cmd_start, "submit now, then keep refilling until settled")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--accept-commit", action="store_true")
    p = add("status", cmd_status, "progress report (no submission)")
    p.add_argument("--details", action="store_true", help="list every missing task")
    p.add_argument(
        "--running",
        action="store_true",
        help="list running tasks with elapsed time, longest first",
    )
    p.add_argument("--email", action="store_true", help="email the report now")
    p = add("stop", cmd_stop, "stop refilling; running jobs continue")
    p.add_argument(
        "--cancel",
        action="store_true",
        help="also cancel this experiment's running and pending jobs",
    )
    p = add("cancel", cmd_cancel, "cancel matching tasks; they are not resubmitted")
    p.add_argument("--estimator", nargs="+", metavar="NAME")
    p.add_argument("--dataset", nargs="+", metavar="NAME")
    p.add_argument("--resample", nargs="+", type=int, metavar="N")
    p = add("retry", cmd_retry, "make failed tasks eligible for submission again")
    p.add_argument("--reason", help="only failures whose reason contains this text")
    p.add_argument(
        "--memory-tier",
        type=int,
        help="restart retried tasks at this memory tier (0 = first)",
    )
    add("loop", cmd_loop, argparse.SUPPRESS)

    args = parser.parse_args(argv)
    try:
        experiment = load_experiment(args.config)
        return args.function(experiment, args)
    except (ConfigError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
