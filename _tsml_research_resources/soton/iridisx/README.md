# IridisX

Scripts for running `tsml-eval` GPU experiments on IridisX.

These are ports of the Iridis 6 `gpu_scripts`, using a conda environment rather than
an apptainer container. IridisX is the university's heterogeneous AI/ML cluster and
is *not* optimised for large multinode CPU jobs, so it is the right home for the deep
learning classifiers and the wrong home for the CPU classifier runs.

Partition names and hardware below are confirmed from the HPC wiki. Settings still
marked `TODO-IRIDISX` in the scripts need checking on the cluster itself.

## Hardware

Login node: `loginX003.iridis.soton.ac.uk` (`loginX001` also has GPUs, `loginX002` is
CPU-only). `/home`, `/scratch` and the module shares are global Storage Scale mounts,
visible identically from login and compute nodes.

Confirmed from `sinfo` on 14/08/2026. The `gres` type string differs per partition and
must be requested exactly — `gpu:a100swarm:1` on `swarm_a100`, but `gpu:a100:1` on the
open `a100` partition. A wrong type is a submission error, not a silent fallback.

| Partition | Nodes | gres | Timelimit | Notes |
| --- | --- | --- | --- | --- |
| `swarm_a100` | 5 | `gpu:a100swarm:4` | 5-00:00:00 | ECS/ORC staff and PGR only |
| `swarm_h100` | 2 | `gpu:h100swarm:8` | 5-00:00:00 | ECS/ORC staff and PGR only |
| `a100` | 12 | `gpu:a100:2` | 2-12:00:00 | open to all |
| `swarm_l4` | 3 | `gpu:l4swarm:7` | 5-00:00:00 | ECS/ORC |
| `scavenger_4a100` | 5 | `gpu:a100swarm:4` | 12:00:00 | idle `swarm_a100`, **preemptible** |
| `scavenger_8h100` | 2 | `gpu:h100swarm:8` | 12:00:00 | idle `swarm_h100`, **preemptible** |
| `quad_h200` / `i7_h200` | 4 / 25 | `gpu:h200:4` | 2-12:00:00 | H200 |
| `mi300x` | 1 | `gpu:mi300x:8` | 2-12:00:00 | **AMD**, will not run CUDA TensorFlow |
| `amd` / `amd_serial` | 90 / 14 | none | 2-12:00:00 | general CPU, no GPUs |

**Everything defaults to the open `a100` partition**, because account `normal` is the
only association this user holds and `swarm_a100` has `AllowAccounts=ecs,orc`:

    $ srun --account=ecs --partition=swarm_a100 ...
    srun: error: Unable to allocate resources: Invalid account or account/partition
    combination specified

`a100` has `DenyAccounts=student,ecsstudents` only, so `normal` is allowed there.

The cost is concurrency. The `a100` QoS caps `gres/gpu=2`, so only two GPU jobs run at
once however many are submitted; `ecsa100` would allow eight. Note also that `a100`
has a **2-12:00:00** time limit against SWARM's five days, so a configuration copied
from a SWARM setup is rejected at submission until `time_limit` is lowered.

To move to SWARM once HPC grant the `ecs` account, change four settings per
configuration: `partition = "swarm_a100"`, `gres = "gpu:a100swarm:1"`,
`account = "ecs"`, and `time_limit` up to `5-00:00:00`. The controller resumes from
whatever is already on disk, so nothing is lost or repeated by switching mid-pass.

`mi300x` is the only AMD GPU partition. A CUDA TensorFlow build cannot use it.

**`batch` does not exist on IridisX** and will fail with `Invalid partition
specified`. `serial` no longer exists on either cluster.

**`#SBATCH --nodes=1` is mandatory on IridisX** — Slurm rejects submissions without a
node count. The generated scripts already include it.

Preemption on the scavenger partitions is safe for this workload: a preempted
resample writes no results file, so rerunning the submission script resubmits exactly
the work that was lost. The scavenger queues are a good choice for test runs during
busy periods (September/October and April are the peak months).

## Order of work

1. Build the `tsml-eval-gpu` conda environment against its own checkout, see
   `iridisx_python.md`.
2. Copy the archive data to `~/Data/Multiverse` and `~/Data/UCR`. Dataset *lists* do
   not need copying, the controller configurations read them from the checkout.
3. Run the smoke test (below) and confirm TensorFlow reports an A100.
4. Start the controller.

`iridisx_probe.sh` can be rerun at any point to check the cluster configuration has
not changed, and is worth rerunning if a submission starts being rejected.

## The supported route: the Multiverse controller

The controller in `_tsml_research_resources/multiverse_controller.py` is the supported
way to run these passes, and is the same code used on Hali. It reconciles what is
already on disk against what is wanted, submits only missing work, escalates memory
after an OOM, pins the repository commit, and emails progress. The IridisX
configurations are:

| Configuration | Pass |
| --- | --- |
| `multiverse_core_resample0_hinception_gpu_iridisx.toml` | H-InceptionTime over the 66 problem Multiverse core |
| `multiverse_core_resample0_litemv_gpu_iridisx.toml` | LITETime-MV over the 66 problem Multiverse core |
| `multiverse_litemv_missing_gpu_iridisx.toml` | LITETime-MV over a hand-listed set of outstanding problems |
| `ucr_resample0_hinception_gpu_iridisx.toml` | H-InceptionTime over the 112 problem UCR clean list |

The first two mirror their Hali counterparts and differ only where the cluster forces
it. The third exists because completeness is judged per cluster: the controller skips
any dataset and resample it can see on disk under `results_root`, so work already done
on Hali looks missing on IridisX. Either copy those results across, or list the
genuinely outstanding problems in `~/DataSetLists/LITETimeMV-Missing.txt` and use the
missing-list configuration.

Always inspect a generated script before the first submission of a pass:

>python _tsml_research_resources/multiverse_controller.py --config _tsml_research_resources/multiverse_core_resample0_hinception_gpu_iridisx.toml --dry-run

Then start the supervisor, which reruns one controller cycle every 30 minutes:

>sh _tsml_research_resources/run_multiverse_controller.sh _tsml_research_resources/multiverse_core_resample0_hinception_gpu_iridisx.toml

The supervisor writes its log to `~/Results/Multiverse/.controller/supervisor.log`.
Set `MULTIVERSE_LOG_DIR` if the results live elsewhere:

>MULTIVERSE_LOG_DIR=~/Results/UCR/.controller sh _tsml_research_resources/run_multiverse_controller.sh _tsml_research_resources/ucr_resample0_hinception_gpu_iridisx.toml

**Do not start the supervisor from inside the `tsml-eval-gpu` environment.** Generated
jobs drop inherited Conda state and verify the interpreter before running, so a stale
activation now fails the job loudly rather than silently using base Python, but
`conda deactivate` first anyway.

## Smoke test before an archive run

>sh _tsml_research_resources/run_hinception_gpu_test_iridisx.sh

Runs one H-InceptionTime experiment on `AtrialFibrillation` in a real `swarm_a100`
allocation and fails loudly if TensorFlow cannot see the GPU. `AtrialFibrillation`
is in the 66 problem core list and is small (15 train, 15 test), so the check is
quick. It also prints the resolved `conda.sh` path. Pass a different problem and
resample as arguments.

`run_litemv_gpu_test_iridisx.sh` is the same check with LITETime-MV.

Both write to `~/Results/GPUTest`, never the paper tree. This matters in both
directions: an existing paper result would make the experiment skip training, so the
check would pass without testing anything, and a check that did train would add a
non-paper result to completeness reporting.

**Do not use STEW as a smoke test.** Despite its short series it has 28,512 cases
over 14 channels, and is a genuine long-running paper task rather than a quick
environment check.

A deep learner that silently trains on CPU is the main failure mode to watch for,
which is why both tests assert on the device rather than just printing it.

## Contents

`iridisx_probe.sh`
: Read-only login-node script. Prints partitions, GPU `gres` strings, walltime
limits, account/QoS associations, filesystem roots and the available conda/CUDA
modules. Submits nothing.

`iridisx_python.md`
: Setup guide for the `tsml-eval-gpu` conda environment.

`gpu_scripts/gpu_classification_experiments_ucr.sh`
`gpu_scripts/gpu_classification_experiments_multiverse.sh`
: Standalone serial submission scripts, the Iridis 6 pattern: one Slurm array per
classifier/dataset pair, a `max_num_submitted` queue-limit loop that polls `squeue`,
and a check that skips resamples already on disk. These are for ad-hoc single
classifier runs, **not** the supported route for a paper pass. They write to the same
result tree as the controller and default to the same classifier, so the two cannot
fork the results: whichever runs second skips what the first produced.

`max_num_submitted` defaults to 12. The `ecsa100` QoS caps concurrent GPUs at 8, so
higher values only lengthen the pending queue.

## Line endings

If a script fails immediately with a syntax error after being copied from Windows:

>dos2unix gpu_classification_experiments_ucr.sh

## Monitoring

As Iridis 6, see `../iridis/README.md`:

>squeue -u USERNAME --format="%12i %15P %20j %10u %10t %10M %10D %20R" -r
