# Running tsml-eval experiments on Hali, Iridis 6 and Iridis X

This directory holds the standard way to run batches of tsml-eval experiments on the
group's Slurm clusters. One controller, `tsml_slurm.py`, handles every cluster,
learning task (classification, regression, clustering, forecasting) and device (CPU or
GPU). Each experiment is a short TOML file.

The older scripts in `../soton/iridis/`, `../uea/` and on various branches still work
and are listed under [Migrating from the old scripts](#migrating-from-the-old-scripts).
Use this directory for new work.

## Contents

- [The idea in one paragraph](#the-idea-in-one-paragraph)
- [Which cluster for what](#which-cluster-for-what)
- [Directory layout](#directory-layout)
- [One-time setup](#one-time-setup)
- [Quick start](#quick-start)
- [Writing an experiment file](#writing-an-experiment-file)
- [Examples](#examples)
- [Monitoring and control](#monitoring-and-control)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Migrating from the old scripts](#migrating-from-the-old-scripts)
- [Adding a cluster or partition](#adding-a-cluster-or-partition)

## The idea in one paragraph

You state the results you want: estimators × datasets × resamples. The controller
works out which results are already on disk and submits only the missing ones,
staying within a queue limit. It then keeps checking. When a job ends without a
result, the controller asks Slurm why. A job killed for running out of memory is
resubmitted with more memory. Other failures are retried a limited number of times,
and time-limit failures are reported rather than repeated. Once every result exists
or every remaining task has failed for good, the controller emails a final report and
stops. Every result can be traced to the git commit that produced it.

## Which cluster for what

| Cluster | Use it for | CPU jobs | GPU jobs | Supervisor |
| --- | --- | --- | --- | --- |
| **Hali** (UEA) | CPU and GPU; long jobs (7 days) | `cpu`: one array task per experiment | `gpu`: conda env | login node, in `screen` |
| **Iridis 6** (Soton) | Large CPU runs | `cpu`: task farm, up to 4 node jobs. `cpu-array`: small runs | `gpu`: Apptainer sandbox | Slurm job that resubmits itself |
| **Iridis X** (Soton) | GPU only | – (use Iridis 6) | `gpu` (H200), `gpu-a100`, `gpu-swarm` | login node, in `screen` |

The names in code font are the resource sections defined in `clusters/<cluster>.toml`.

## Directory layout

```
slurm/
├── tsml_slurm.py            the controller (Python standard library only, 3.11+)
├── clusters/                facts about each cluster, written once
│   ├── hali.toml
│   ├── iridis6.toml
│   └── iridisx.toml
├── experiments/             example experiments: copy one and edit it
│   ├── hali_ucr_cpu.toml
│   ├── hali_hc2_components_train.toml
│   ├── hali_multiverse_gpu.toml
│   ├── hali_clustering.toml
│   ├── iridis6_quick_test.toml
│   ├── iridis6_tser_regression.toml
│   ├── iridis6_ucr_gpu.toml
│   └── iridisx_convtran.toml
└── tests/test_tsml_slurm.py offline tests (no Slurm needed)
```

Configuration has two layers, so that cluster details are not copied into every
experiment:

- **Cluster profile** (`clusters/*.toml`): home directory, partitions, accounts, QoS,
  time limits, memory tiers, modules, conda environment names, GPU `gres` strings and
  supervisor settings. Edit these when the cluster changes, not for an experiment.
- **Experiment** (`experiments/*.toml`, or anywhere you like): what to run and where
  results go. It names a cluster and one of its resource sections, and may override
  any resource setting in its own `[slurm]` table.

Results are written in the standard tsml-eval layout:

```
<results_dir>/<Category>/<Estimator>/Predictions/<Dataset>/testResample<N>.csv
```

`<Category>` is left out if you give the estimators as a flat list.

Each experiment keeps its own state directory, by default `~/.tsml-slurm/<name>/`:

```
~/.tsml-slurm/<name>/
├── state.json      attempts, memory tier, job ID and failure history per task
├── report.txt      the latest progress report
├── STOP            present = no new submissions
├── logs/<Estimator>/<Dataset>/<jobid>_<resample>.{out,err,status}
├── scripts/<jobid>.sh       every job script exactly as submitted
├── farms/<jobid>.out        task-farm job logs (Iridis 6)
└── supervisor/              supervisor logs
```

Logs are kept per experiment, not in the results tree, so old runs cannot confuse the
failure diagnosis of a new one.

## One-time setup

Do this once per cluster. The detailed environment guides are still the existing ones:
[Iridis 6](../soton/iridis/iridis_python.md) and Iridis X (`iridisx_python.md` on the
`ajb/gpu` branch). The short version:

1. **Clone tsml-eval** to the profile's `repo_dir`: `~/Code/tsml-eval` on Hali and
   Iridis 6. On Iridis X it is `~/Code/tsml-eval-gpu`, a separate GPU checkout. On
   Hali the GPU examples also use `~/Code/tsml-eval-gpu`.
2. **Create the conda environment** named in the profile and install the checkout:
   `tsml-eval` for CPU and `tsml-eval-gpu` for GPU (Hali, Iridis X).
   ```bash
   conda create -n tsml-eval python=3.12
   conda activate tsml-eval
   pip install --editable ~/Code/tsml-eval
   ```
   For GPUs, install a CUDA build, e.g. `pip install "tensorflow[and-cuda]"` and/or
   `pip install --editable ".[deep_learning]"` for PyTorch learners. On Iridis 6, GPU
   jobs use the Apptainer sandbox described in `iridis_python.md`. Its default
   location is `~/scratch/tensorflow_sandbox`.
3. **Copy the data** to the cluster, e.g. `~/Data/UCR/<Dataset>/<Dataset>_TRAIN.ts`.
   Dataset lists can come from `_tsml_research_resources/dataset_lists/` in the
   checkout, so they do not need copying.
4. **Pick a Python to run the controller.** Use the environment's interpreter, e.g.
   `~/.conda/envs/tsml-eval/bin/python`. The controller only needs the standard
   library, but with the environment's Python `check` can also construct every
   estimator and catch misspelt names. The supervisor reuses whichever interpreter
   you start it with. A handy alias:
   ```bash
   alias tsml-slurm='~/.conda/envs/tsml-eval/bin/python ~/Code/tsml-eval/_tsml_research_resources/slurm/tsml_slurm.py'
   ```
   The rest of this guide writes `tsml-slurm` for that command.

You do **not** need to `conda deactivate` before submitting. Generated jobs clear any
inherited conda state and confirm that they are running the intended environment.

## Quick start

```bash
cd ~/Code/tsml-eval/_tsml_research_resources/slurm
cp experiments/iridis6_quick_test.toml ~/my_test.toml      # edit name, data, estimators
tsml-slurm check  ~/my_test.toml            # validate paths, data, estimator names
tsml-slurm submit ~/my_test.toml --dry-run  # what would be submitted, plus one script
tsml-slurm start  ~/my_test.toml            # submit, then keep refilling until done
tsml-slurm status ~/my_test.toml            # progress at any time
```

`check` exits non-zero and lists the problems if anything is wrong: a missing data
directory, datasets absent from it, an estimator name the factory does not know, or
uncommitted changes. **Run it before every new experiment.**

## Writing an experiment file

A minimal experiment:

```toml
[experiment]
name = "ucr-rocket"                  # letters, digits, . _ + - ; names state and jobs
cluster = "hali"                     # a file in clusters/
resources = "cpu"                    # a [resources.*] section of that profile
task = "classification"              # classification | regression | clustering | forecasting
data_dir = "{home}/Data/UCR"
datasets = "{repo}/_tsml_research_resources/dataset_lists/UnivariateClassification112-UCR2018Clean.txt"
results_dir = "{home}/Results/UCR"
resamples = 30                       # resamples 0..29

[estimators]
ConvolutionBased = ["ROCKET", "MiniRocket"]
```

Paths may use `{home}` (the cluster's home directory), `{user}`, `{repo}` (the
checkout) and `{name}` (the experiment name), as well as environment variables and `~`.

### `[experiment]` settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `name` | required | Unique per experiment. Changing it starts a fresh state. |
| `cluster` | required | Profile in `clusters/`. |
| `resources` | profile's `default_resources` | Resource section, e.g. `cpu`, `gpu`, `cpu-array`. |
| `task` | required | Picks the tsml-eval entry point. |
| `data_dir`, `datasets`, `results_dir` | required | Data root, dataset list file, results root. |
| `estimators` | – | Flat list, results go straight under `results_dir`. Alternative to `[estimators]`. |
| `resamples` | `30` | A count (`30` → 0..29) or an explicit list (`[0, 5, 9]`). |
| `start_resample` | `0` | With a count: `start_resample = 10`, `resamples = 30` → 10..29. |
| `train_files` | `false` | Also write train predictions (`-tr`); a task then needs both files. |
| `test_files` | `false` | Clustering only: also write test assignments (`-te`). |
| `n_jobs` | `1` | Threads per experiment. Above 1, the threaded entry point is used with `-nj`, and more CPUs are requested. |
| `extra_args` | `[]` | Any other tsml-eval flag, e.g. `["-rn"]`, `["-pr"]`, `["-dtn", "Padder"]`. |
| `priority` | `"resample"` | Submission order: `resample` (all of resample 0 first), `estimator` (finish one estimator first), `dataset`. |
| `smallest_datasets_first` | `false` | Within the order above, smallest data on disk first. |
| `exclude_datasets` | `[]` | Datasets in the list to skip. |
| `exclude_tasks` | `[]` | Single tasks to skip, `"Estimator\|Dataset\|Resample"`. |
| `max_attempts` | `2` | Submissions allowed per task for failures other than out-of-memory. |
| `branch` | `""` | Refuse to submit unless the checkout is on this branch. |
| `require_clean` | `true` | Refuse to submit with uncommitted changes in the checkout. |
| `pin_commit` | `true` | Refuse to submit from a commit other than the experiment's first (see below). |
| `email` | profile's `email` | Report recipient; `""` disables email. |
| `email_hours` | `24` | Minimum time between progress emails. A final email is always sent. |
| `repo_dir`, `state_dir`, `numba_cache_dir` | profile / `~/.tsml-slurm/{name}` | Rarely needed overrides. |

### Other tables

```toml
[estimators]                      # category = [estimator names]
IntervalBased = ["DrCIF", "CIF"]
DeepLearning = ["H-InceptionTime"]

[estimator_kwargs.DrCIF]          # passed as -kw key value type
n_estimators = 500
[estimator_kwargs."H-InceptionTime"]   # quote names containing "-"
verbose = true

[slurm]                           # override the cluster's resource section
memory_gb = [16, 32, 64]
time_limit = "2-00:00:00"
max_active = 500

[supervisor]                      # override the cluster's supervisor
interval_minutes = 60
```

### Resource settings (`[resources.*]` in a profile, or `[slurm]` in an experiment)

| Setting | Meaning |
| --- | --- |
| `mode` | `array` (one Slurm array task per experiment) or `farm` (many experiments packed into one node job). |
| `partition`, `account`, `qos`, `time_limit` | Slurm basics. Empty `account`/`qos` are left out. |
| `memory_gb` | Memory tiers in GB. Each task starts at the first; an out-of-memory kill moves it up one tier. In farm mode the tier is per experiment. |
| `cpus_per_task` | CPUs per array task (e.g. 2 for GPU data loading). |
| `gres` | GPU request, e.g. `gpu:1` or `gpu:h200:1`. Setting it makes the job a GPU job. |
| `gpu_check` | `tensorflow`, `torch` or `none`: fail fast if the framework cannot see the GPU. |
| `max_active` | Array mode: maximum running plus pending tasks for this experiment. |
| `max_user_tasks` | Array mode: also cap the user's total tasks in the partition (0 = off). |
| `max_jobs`, `node_cpus`, `node_memory_gb`, `waves` | Farm mode: node jobs at once, node size, and how many rounds of tasks each job is given. |
| `module`, `conda_sh`, `env_name` | Environment: `module load`, then `conda activate`. An empty `conda_sh` is found from the loaded `conda`. |
| `container` | Apptainer image or sandbox; replaces conda (Iridis 6 GPU). |
| `setup` | Extra shell lines run before the experiment, e.g. `["module load cuda/12.8.0"]`. |
| `extra_sbatch` | Extra `#SBATCH` options, e.g. `["--exclude=node123"]`. |

Unknown settings are errors, so a typo such as `memroy_gb` is caught at once rather
than silently ignored.

## Examples

All of these are in `experiments/`. Copy one, change `name` and the paths, then
`check`, `submit --dry-run` and `start`.

### 1. Try a new estimator on a few problems (Iridis 6)

`iridis6_quick_test.toml` runs ROCKET and DrCIF on the 30 Bakeoff problems with 5
resamples. It uses `cpu-array`, one array task per experiment, which is the simplest
route for small runs. `require_clean = false` and `pin_commit = false` let you iterate
on uncommitted code. Do not use those two settings for results you intend to publish.

```bash
tsml-slurm check experiments/iridis6_quick_test.toml
tsml-slurm start experiments/iridis6_quick_test.toml
```

### 2. A full UCR run on Hali

`hali_ucr_cpu.toml`: four convolution classifiers × 112 problems × 30 resamples =
13,440 experiments. With `priority = "resample"` and `smallest_datasets_first = true`,
resample 0 of every problem arrives first, so a usable first table exists early.

### 3. Ensemble components with train files and parameters (Hali)

`hali_hc2_components_train.toml` shows `train_files = true` (a task is complete only
when both train and test files exist), `[estimator_kwargs.DrCIF] n_estimators = 500`,
and a larger starting memory via `[slurm] memory_gb`.

### 4. A big CPU run on Iridis 6 (task farm)

`iridis6_tser_regression.toml`: five interval regressors × 63 TSER problems × 30
resamples. The Iridis 6 `cpu` profile is a task farm. Up to `max_jobs = 4` node jobs
run at once. Each runs as many experiments in parallel as its memory tier allows
(157 at 4 GB, 9 at 64 GB) and is given `waves = 4` rounds of work. As node jobs finish,
the supervisor submits new ones. Tasks that run out of memory come back in a farm at
the next tier.

Farms waste cores when experiment times vary widely, because the job holds its cores
until its slowest task ends. `smallest_datasets_first` helps by keeping similar sizes
together. If a few huge datasets dominate, run them as a separate experiment with
`resources = "cpu-array"` and list them in `exclude_datasets` of the farm experiment.

### 5. GPU deep learners

- **Hali**, `hali_multiverse_gpu.toml`: Keras and PyTorch learners in one
  experiment. The profile's TensorFlow check fails a job immediately if the GPU is
  not visible. PyTorch learners get `device = "cuda"` so they fail rather than fall
  back to the CPU.
- **Iridis X**, `iridisx_convtran.toml`: ConvTran on the H200 queue with
  `gpu_check = "torch"`. Switch queue with `resources = "gpu-a100"` or `"gpu-swarm"`.
  Each queue has its own `gres` string, already set in the profile.
- **Iridis 6**, `iridis6_ucr_gpu.toml`: Keras learners in the Apptainer sandbox; set
  `container` to your sandbox path.

Deep-learning estimators such as ConvTran exist only on the `ajb/gpu` branch, which
these examples request with `branch = "ajb/gpu"`.

### 6. Clustering

`hali_clustering.toml`. For clustering the train file is the main output, and
`test_files = true` also writes test assignments (`-te`). Completion requires both.
It also passes `-rn` through `extra_args`.

### 7. Growing an experiment

You can add estimators, datasets or resamples to an existing experiment at any time.
Edit the TOML and run `start` again (or let the supervisor pick it up). Completed work
is never repeated.

You **cannot** change the keyword arguments, flags or results path of an estimator
that has already been submitted. The controller refuses, because old and new results
would be mixed in one directory. Give the changed variant a new identity instead:
a different `results_dir`, or a new estimator name if the factory has one (e.g.
`DrCIF-500`).

### 8. Splitting one study across clusters

Completion is judged from the files a cluster can see. To share a study between, for
example, Hali (CPU) and Iridis X (GPU), give each cluster its own experiment with a
disjoint set of estimators, then copy the results together afterwards. If two
clusters must share an estimator, split the dataset list between them, or copy the
finished results across first so the second cluster sees them as complete.

## Monitoring and control

```bash
tsml-slurm status EXP.toml             # summary: done / running / pending / failed per estimator
tsml-slurm status EXP.toml --details   # every missing task with attempts, memory and job
tsml-slurm status EXP.toml --email     # email the report now
tsml-slurm stop   EXP.toml             # no new submissions; running jobs continue
tsml-slurm stop   EXP.toml --cancel    # ... and cancel this experiment's jobs
tsml-slurm start  EXP.toml             # resume (removes STOP)
tsml-slurm retry  EXP.toml             # give failed tasks another chance
tsml-slurm retry  EXP.toml --reason OOM --memory-tier 3   # only OOM failures, from tier 3
```

A report looks like this:

```
Experiment tser-interval: regression on iridis6/cpu (farm mode, partition batch)
Complete 7123/9450 (75.4%)  running 520  pending 0  failed 6  not yet submitted 1801

estimator                                   done    run   pend   fail
IntervalBased/DrCIF                    1712/1890    104      0      2
...
Failed tasks (will not be retried; see `retry`): 6
       4  Time limit
       2  OOM at 620G
  DrCIF|PPGDalia_eq|3: Time limit
      ~/.tsml-slurm/tser-interval/logs/DrCIF/PPGDalia_eq/123456_3.err
```

Each failed task points at its log. `scripts/<jobid>.sh` holds the exact job script.

### What happens after a failure

| Reason (from `sacct`, the task's exit code or its log) | What the controller does |
| --- | --- |
| Out of memory (host RAM) | Resubmits at the next memory tier without using an attempt. At the top tier the task fails for good. |
| Time limit | Fails for good: a rerun would hit the same limit. Raise `time_limit` or move the task to a cluster with longer jobs (Hali allows 7 days), then `retry`. |
| Python exception, GPU out of memory, cancelled, finished without a result | Retried until `max_attempts` is reached. The last error line is shown in the report. |
| Never started, node failure, preemption | Resubmitted without using an attempt. |

Failed tasks keep their history in `state.json`. After fixing the cause, use `retry`.
Do not delete the state directory: it is how the controller knows which jobs are its
own.

## How it works

**The cycle.** Each `submit` (and each supervisor run) is one cycle under a lock:

1. Read the experiment and its dataset list; list every task.
2. Ask `squeue` which of this experiment's jobs are still live. Jobs are matched by
   the job IDs recorded at submission, not by job name, so other runs (including
   older scripts writing to the same results) cannot be mistaken for this one.
3. For each recorded job that has left the queue without writing its result, ask
   `sacct` why, falling back to the logs, and apply the failure rules above.
4. Choose missing, non-failed tasks in priority order, up to the free capacity.
   Group them into one array per estimator, dataset and memory tier (array mode), or
   into node-sized farms per memory tier (farm mode). Then submit.
5. Save state, write `report.txt`, and send email if it is due.

**The supervisor** repeats the cycle every `interval_minutes` until everything is
settled (complete, or failed for good) or a `STOP` file exists. It then sends a final
email and exits.

- `mode = "slurm"` (Iridis 6): a small job that starts after the interval, runs one
  cycle, and submits its successor. Nothing runs on the login node, and it survives
  login-node reboots.
- `mode = "login"` (Hali, Iridis X): a loop in a detached `screen` session named
  `tsml-<name>`. Its log is `supervisor/loop.log`. Reattach with
  `screen -r tsml-<name>`. If the login node reboots, run `start` again; nothing is
  lost.

**Reproducibility.** Every job checks, before running, that the checkout is still at
the commit it was submitted from. A `git pull` or branch switch therefore makes queued
jobs fail loudly instead of silently running different code. The controller also
records the experiment's first commit. With `pin_commit = true` it will not submit
from a different commit unless you pass `--accept-commit`, which says you knowingly
accept results from two code versions in one experiment. Job logs record the host,
Python, aeon version and paths. Jobs set `PYTHONPATH` to the checkout, so they always
import the code they were pinned to.

**Threads.** Jobs set the BLAS/OpenMP thread variables to 1 and hide GPUs from CPU
jobs, so an estimator cannot use more cores than were requested. `n_jobs > 1` asks
for that many CPUs and passes `-nj` to the threaded entry point.

**Task farms.** A farm job requests `slots × memory` and runs its task list `slots`
at a time with a bash worker pool. Each task writes `<log>.status`: a start line, then
`exit <code>`. A missing status file means the task never started, so it is
resubmitted free. Exit code 137 (SIGKILL, which is how the memory limiter kills a
process) counts as out of memory.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `check` says *estimator construction skipped* | You ran it with a Python that cannot import tsml-eval. Use the environment's interpreter. |
| `repository moved from … to …` in a job log | The checkout changed after submission. Check out the commit (or `--accept-commit` and resubmit). Avoid working in the checkout that jobs run from; use a second clone for development. |
| `has uncommitted changes` | Commit, or set `require_clean = false` for throwaway runs. |
| `Settings for X … changed since their first submission` | See [Growing an experiment](#7-growing-an-experiment). |
| Jobs fail in seconds with `no conda.sh` or the wrong environment | Check `module`, `conda_sh` and `env_name` in the profile against `conda info --envs`. |
| `TensorFlow sees no GPU` / `PyTorch sees no GPU` | The environment has a CPU-only build, or CUDA libraries are missing. See the GPU setup notes; test with an interactive GPU job. |
| `sbatch: error: Invalid account or account/partition combination` | Iridis X `swarm_a100` needs `account = "ecs"`. Hali needs `cmp`. |
| `Invalid generic resource (gres) specification` | Iridis X: the `gres` type must match the partition (`gpu:h200:1`, `gpu:a100:1`, `gpu:a100swarm:1`). |
| Nothing is submitted, report says `STOP file present` | Run `start`, or delete `~/.tsml-slurm/<name>/STOP`. |
| `Another cycle holds …controller.lock` | A cycle is running; `status` shows the last saved report. |
| Farm jobs pending for a long time on Iridis 6 | A near-full-node request waits for an empty node. Lower `node_memory_gb` or `node_cpus` in `[slurm]` to fit partly used nodes. |
| No emails | The host running the cycle has no `mail`/`sendmail` (possible on compute nodes). `report.txt` is always written; `status --email` from a login node sends it. |
| Script fails with `$'\r': command not found` | Windows line endings in a file you copied: `dos2unix file`. |

## Migrating from the old scripts

| Old | New |
| --- | --- |
| `soton/iridis/serial_scripts/*_experiments.sh` | `resources = "cpu-array"` on Iridis 6 |
| `soton/iridis/threaded_scripts/threaded_*.sh` | `n_jobs = N` |
| `soton/iridis/batch_scripts/taskfarm_*.sh` and the `run_*` controllers (`run_ucr_completion.sh`, `run_tser_interval_regressors.sh`, `run_multiverse_interval_classifier.sh`, …) | `resources = "cpu"` on Iridis 6 (farm mode with memory tiers) |
| `soton/iridis/gpu_scripts/gpu_*.sh` | `resources = "gpu"` on Iridis 6 |
| `soton/iridis/ucr_reference/` | an Iridis 6 experiment per device; `exclude_tasks` for known-bad cases |
| `uea/ada/*.sh`, `uea/hali/*.sh` | `cluster = "hali"` |
| `multiverse_controller.py` + its TOMLs, `run_multiverse_controller.sh`, `start_*.sh` | an experiment TOML. `[controller]` paths become `[experiment]` settings, `[[categories]]` become `[estimators]`, `[classifier_kwargs.X]` becomes `[estimator_kwargs.X]`, and `[slurm]` keeps only what differs from the profile (memory in GB, not MB). |
| `range_scancel.sh`, `stop_multiverse_jobs.sh`, `cancel_*` | `tsml-slurm stop EXP.toml --cancel` |
| `monitor_*.sh`, `mail_*_progress.sh` | `tsml-slurm status`, emails from the supervisor |
| `clear_controller_failures.sh` | `tsml-slurm retry` |

Results written by the old scripts are in the same layout, so a new experiment
pointing at the same `results_dir` treats them as done and fills only the gaps. Do not
run an old script and a new experiment on the same results at the same time: neither
can see the other's jobs.

## Adding a cluster or partition

To use another partition, add a `[resources.<name>]` section to the cluster's profile,
e.g. Iridis X's preemptible A100s:

```toml
[resources.gpu-scavenger]
mode = "array"
partition = "scavenger_4a100"
gres = "gpu:a100swarm:1"
time_limit = "12:00:00"
memory_gb = [32, 64]
cpus_per_task = 2
max_active = 8
module = "conda/python3"
env_name = "tsml-eval-gpu"
gpu_check = "tensorflow"
```

Preemption is safe: a preempted task writes no result and is resubmitted without using
an attempt.

For a new cluster, copy the closest profile to `clusters/<name>.toml` and set `name`,
`home`, `repo_dir`, `email`, `[supervisor]` and its resource sections. Then run
`check` on a small experiment, and `start` it, before using it for a large run.

## Testing the controller

```bash
python -m pytest _tsml_research_resources/slurm/tests -q
```

The tests use a fake Slurm. They cover configuration and validation (including every
file in `experiments/`), generated scripts (checked with `bash -n`), the queue and
retry state machine, farm packing and status-file diagnosis, commit pinning and
dry runs.
