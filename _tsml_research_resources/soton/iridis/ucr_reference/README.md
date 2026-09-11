# UCR reference results on Iridis 6

This workflow fills the clean **112 UCR datasets × 30 resamples (0–29)** for
all 54 concrete classifiers in aeon's eight main classification families, plus
Dummy and Rotation Forest. CPU and GPU runs share one manifest and monitor.

Only the explicitly configured collections under `Results/UCR` count. The
PreVal ZIP, `TestOnly`, HC2 composition experiments and experimental variants
are outside this run. Existing nonempty results are skipped. Existing train-file
conventions are preserved; new collections are test-only. No extra normalisation
or predefined resamples are applied. Resample 0 uses the original archive split.

The 10 September 2026 local inventory gives:

| Scope | Classifiers | Required experiments | Complete | Missing |
|---|---:|---:|---:|---:|
| CPU | 46 | 154,560 | 54,600 | 99,960 |
| GPU / deep learning | 10 | 33,600 | 0 | 33,600 |
| Total | 56 | 188,160 | 54,600 | 133,560 |

Thirteen classifiers are complete, five partial, and 38 have no results in their
configured collections. BOSS, cBOSS, WEASEL_V2 and MrSEQL also need 5,850 train
files between them. These are the same experiments as their missing tests, not
an extra 5,850 submitted commands. RIST needs all 30 StarLightCurves resamples.

## Configuration and prerequisites

Keep this directory inside the tsml-eval checkout on Iridis. Read and edit
`ucr_reference.json` before launch. Defaults follow the existing Iridis scripts:

- User `ajb2u23`; repos, data and results under `/iridisfs/home/ajb2u23`.
- CPU Python `/home/ajb2u23/.conda/envs/tsml-eval/bin/python` (Python 3.10+).
- CPU partition `batch`: up to four one-node task farms, 60 hours, single-threaded
  experiments, at most 192 tasks and 630 GiB requested per node. Memory tiers are
  4, 8, 16, 32, 64, 128, 256 and 620 GiB per experiment. At the first tier this
  allows 157 concurrent experiments per allocation. Each batch queues up to eight
  waves of work. Dataset size gives large problems a higher starting tier.
- GPU partition `gpu`: up to two allocations, each requesting one GPU, four CPUs
  and initially 16 GiB host RAM. Each runs up to eight experiments sequentially.
  The configured Apptainer module is `apptainer/1.5.0`; the sandbox defaults to
  `/iridisfs/home/ajb2u23/scratch/tensorflow_sandbox`. **Check this path.** It must
  contain TensorFlow and the dependencies of the ten deep classifiers. Jobs bind
  both checkouts, the data directory and the results root and use `--nv`.
- A short supervisor on `batch` requests one CPU and 2 GiB, becoming eligible
  every 15 minutes to refill free allocation slots. The job limits above apply
  to this workflow's worker allocations; the supervisor is additional. Site
  limits and other workloads can keep jobs pending. Adjust limits to available
  capacity in the config.

The wrappers select the CPU Python with `UCR_REFERENCE_PYTHON`; changing the
Python path in the JSON alone does not change the wrapper interpreter. For another
installation, export this variable or invoke `controller.py` with that Python.

The checkout must include the new `IntervalForest`, `TDMVDC`, `stc-aeon`,
`pf-aeon` and `proximitytree-aeon`
factory keys in `tsml_eval/experiments/_get_classifier.py`. The existing `STC`
alias on this branch uses an experimental class, as do `PF` and `ProximityTree`;
this workflow resolves explicit aeon aliases while retaining the normal result
directory names, including `ShapeletBased/STC` and `DistanceBased/PF`.
Construction checks verify both class name and the `aeon` module namespace.

Both checkouts must be accessible at their configured paths. The controller
pins the aeon commit and tracked diff plus hashes of the experiment sources.
Generated jobs verify these before executing. Existing historical results are
accepted by location and nonempty-file coverage; this does not establish that
all historical versions and hyperparameters match the newly pinned checkout.

## Check and launch

From this directory on Iridis:

```bash
bash run_ucr_reference.sh --dry-run
bash run_ucr_reference.sh --check
bash run_ucr_reference.sh
bash monitor_ucr_reference.sh --watch 60
bash mail_ucr_reference_progress.sh --once
```

The dry run is read-only and does not require Slurm, aeon imports or the GPU
container. It displays counts from the configured UCR root. `--check` verifies
all data files and constructs every selected classifier in the actual CPU or
container environment, without fitting anything or submitting jobs. The first
real launch also performs these checks before its first submission. Constructor
checks run on the login node; GPU availability is checked inside worker
allocations. Check output identifies missing packages or unsupported factory
keys; they are not silently removed from the target list.

Optional soft dependencies include packages used by MrSEQL, MrSQM, signatures,
TSFresh, TDMVDC and RSAST. Install the packages reported by preflight in the
appropriate environment using the repository's Iridis setup guide. This workflow
does not install packages or change either checkout during a run.

To begin with only one resource type:

```bash
bash run_ucr_reference.sh --device cpu --check
bash run_ucr_reference.sh --device cpu
bash run_ucr_reference.sh --device gpu --check
bash run_ucr_reference.sh --device gpu
```

To stage the run while `DictionaryBased` results are still being copied, start
only that category on the CPU queue:

```bash
bash run_ucr_reference.sh --device cpu --category DictionaryBased --check
bash run_ucr_reference.sh --device cpu --category DictionaryBased
```

This category filter also applies to its automatic successor. Once copying has
finished, launch the full run with:

```bash
bash run_ucr_reference.sh
```

The full launch keeps the dictionary results already present and adds the other
categories. Repeat `--category` to stage several categories together, for example
`--category DictionaryBased --category IntervalBased`.

The second launch adds GPU work to the same state; subsequent supervisors retain
both devices. A later CPU-only refill does not silently narrow an existing full
run. `--no-chain` submits one cycle without arming a new supervisor; it does not
cancel an already scheduled supervisor. The monitor always shows the full
configured classifier list.

`mail_ucr_reference_progress.sh` sends the report immediately and schedules the
next report 24 hours later. It reports active jobs, completed/total experiments,
percentage, missing work and blocked failures. The recipient is `email` in the
JSON configuration, or `UCR_REFERENCE_EMAIL`. Reports are saved under the state
directory as well as emailed. The cycle stops at 100% or with `--once`; use
`--stop` to stop future reports without cancelling worker jobs.

Do not overlap this workflow with `run_ucr_completion.sh`. The runner refuses to
start while `ucr-completion-*` jobs are live. It does not cancel those jobs.

## Monitoring and recovery

```bash
bash monitor_ucr_reference.sh
bash monitor_ucr_reference.sh --details
```

The table reports test files, required train files, completed experiments and
datasets with all 30 resamples. The footer distinguishes missing, reserved and
blocked work, followed by live Slurm jobs and worker activity. `--details` adds
the exact missing resamples and blocked log paths. Monitoring is read-only.

State is under `Results/UCR/.ucr-reference-state`:

- `run-config.json`: the resolved configuration used by workers and successors.
- `provenance.json`: the pinned source identities.
- `state.json`: submitted batches and retry records.
- `batches/<token>/`: exact commands, Slurm script, per-experiment logs and
  start/finish markers. Allocation stdout/stderr also goes here.
- `<jobid>-supervisor.out` and `.err`: reconciliation logs.

Live and completing jobs reserve their entire unfinished task lists. Jobs absent
from `squeue` stay reserved until `sacct` supplies a terminal state. An experiment
that never started in a timed-out task farm consumes no retry. Confirmed host
OOM advances its memory tier; a GPU VRAM OOM does not increase host RAM. Other
failures and timeouts are retried at the same memory tier, with at most three
failures or ten executed attempts by default. Repeated failures remain **blocked,
not complete**, and retain their logs. A farm that fails before any worker starts
halts refills for investigation rather than resubmitting indefinitely.

Each submission is journalled before calling `sbatch`. If the command is
interrupted, the next cycle recovers its ID from a unique live job name. If no
live match is available, it stops rather than guessing that submission failed.
Look up the full job name with `squeue`/`sacct`, then set that batch's `id` and
`stage: "SUBMITTED"` in `state.json`. Only remove a prepared batch after confirming
Slurm never accepted it. Preserve the batch logs. Do this with refills stopped.

To pause automatic submissions while allowing existing workers to finish:

```bash
python controller.py stop
```

This creates `.ucr-reference-state/STOP`; it sends no cancellation or mail.
After correcting a problem, remove that exact STOP file and launch the runner
again. For a retry-exhausted experiment, with refills stopped, remove its
`blocked` field and reset its `failures`/`attempts` after addressing the cause.
Do not delete the entire state directory to retry failures: it contains the
records needed to avoid duplicating live jobs. To release a confirmed startup
failure after fixing its environment, mark that batch `RECONCILED`; its unstarted
experiments will become eligible on the next launch.

If a supervisor fails, the monitor will show the stopped chain; inspect its log
and rerun the launcher. Configuration changes during an established run are
rejected. Restore the saved configuration/source versions before restarting,
or explicitly plan a new experiment version rather than mixing changed settings.
The default chain limit is 5,000 reconciliation cycles.

## Local verification

The same controller can inspect D: without importing aeon or writing results:

```powershell
python controller.py monitor --offline --results-root D:\Results\UCR `
  --dataset-list ..\..\..\dataset_lists\UnivariateClassification112-UCR2018Clean.txt
```

The relative dataset-list path above is from this directory. The runner also
accepts `--data-dir` and `--dataset-list` overrides; actual launches preserve the
resolved settings in their snapshot for all successors.

Offline controller tests:

```bash
python -m unittest discover -s . -p test_controller.py -v
```

The workflow follows `batch_scripts/run_ucr_completion.sh` for CPU task-farm and
memory conventions, `gpu_scripts/gpu_classification_experiments.sh` for GPU
resources/Apptainer, and `iridis_python.md` for environment setup. The new monitor
uses structured batch manifests and status markers; it does not inherit the old
monitor's regression-script regular expression.

Validation on 10 September 2026: 19 offline controller tests and five aeon factory
tests passed. A real Dummy experiment on MinimalChinatown verified that an empty
train result is repaired while the existing test result remains byte-for-byte
unchanged. Bash syntax checks passed for both wrappers and generated CPU/GPU
scripts. The CPU constructor check passed for 43/46 classifiers locally; the local
environment lacked `mrsqm`, `mrseql` and `roughpy`. Iridis dependency and GPU
execution checks remain to be run on the cluster. No Slurm jobs were submitted
during local validation.
