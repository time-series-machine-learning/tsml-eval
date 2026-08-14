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

The scripts default to `swarm_a100`. As ECS, we are eligible, and single-GPU deep
learner jobs do not need an H100.

The relevant QoS caps concurrent GPUs, not just jobs: `ecsa100` and `ecsh100` allow
`gres/gpu=8`, the open `a100` QoS only `gres/gpu=2`. So at most 8 array tasks run at
once on `swarm_a100` however many are submitted. `max_num_submitted=12` in the scripts
leaves a few queued behind the running 8.

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
2. Copy the data and dataset lists to `~/Data` and `~/DataSetLists`.
3. Smoke test one small dataset with one classifier before submitting an archive.

`iridisx_probe.sh` can be rerun at any point to check the cluster configuration has
not changed, and is worth rerunning if a submission starts being rejected.

## Contents

`iridisx_probe.sh`
: Read-only login-node script. Prints partitions, GPU `gres` strings, walltime
limits, account/QoS associations, filesystem roots and the available conda/CUDA
modules. Submits nothing.

`iridisx_python.md`
: Setup guide for the `tsml-eval-gpu` conda environment.

`gpu_scripts/gpu_classification_experiments_ucr.sh`
: UCR univariate archive. Defaults to the 112 problem clean list.

`gpu_scripts/gpu_classification_experiments_multiverse.sh`
: Multiverse multivariate archive. Defaults to the 133 problem clean list.

Both scripts are the standard serial submission pattern from Iridis 6: one Slurm
array per classifier/dataset pair, a `max_num_submitted` queue-limit loop that polls
`squeue` and sleeps until the queue drops below the limit, and a completed-resample
check that skips resamples whose `testResample<i>.csv` already exists. Rerunning a
script after a partial run picks up only the missing work.

`max_num_submitted` defaults to 12 in both. There are only 20 A100s across
`swarm_a100`, so this should stay low.

## Dataset lists

Copy the relevant list from `_tsml_research_resources/dataset_lists/` to
`$local_path/DataSetLists/` on the cluster:

- `UnivariateClassification112-UCR2018Clean.txt` (or the full 128 archive,
  `UnivariateClassification128-UCR2018.txt`)
- `Multivariate133Classification-MultiverseClean.txt` (or the 66 problem subset,
  `MultivariateClassification66-MultiverseMini.txt`)

## Smoke test before an archive run

Set `datasets` to a file containing one small problem (`ItalyPowerDemand` for UCR,
`BasicMotions` for multivariate), `max_folds=1`, and one classifier. Check the `.out`
file: `nvidia-smi` should list the allocated A100, and TensorFlow should log a
physical GPU device rather than falling back to CPU. A deep learner that silently
runs on CPU is the main failure mode to watch for.

## Line endings

If a script fails immediately with a syntax error after being copied from Windows:

>dos2unix gpu_classification_experiments_ucr.sh

## Monitoring

As Iridis 6, see `../iridis/README.md`:

>squeue -u USERNAME --format="%12i %15P %20j %10u %10t %10M %10D %20R" -r
