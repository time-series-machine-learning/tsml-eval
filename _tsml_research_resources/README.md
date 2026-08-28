This directory contains instructions and utilities for running experiments using
tsml-eval on University of Southampton (Soton) and
University of Bradford hardware.

For non-Soton/Bradford users, the directory contents are likely irrelevant.
Some of the contents could be adapted for other purposes or could be generalised
for other Slurm/Linux devices, but running the scripts or following the hardware
instructions will likely achieve nothing without alterations.

# Long-running Core TDE experiments on a non-Slurm Unix host

`run_tde_core_missing_unix.sh` runs resample 0 for TDE on the four outstanding
Core datasets (`STEW`, `USCActivity`, `Tiselac`, and
`AustraliaRainfall_disc`). It requests TDE's own train estimate with `-tr`,
enables per-candidate progress and time estimates, and never overwrites an
existing test or train result file.

Activate the CPU `tsml-eval` environment from the `ajb/hc2` checkout, then run:

```bash
bash _tsml_research_resources/run_tde_core_missing_unix.sh start
```

The default paths are `$HOME/Data/Multiverse` and
`$HOME/Results/Multiverse`. Override them with `MULTIVERSE_DATA_DIR` and
`MULTIVERSE_RESULTS_ROOT`. The safe default runs one large experiment at a
time. A sufficiently large-memory machine can set `TDE_PARALLEL_JOBS` before
starting to run more concurrently.

Use the same script with `status` or `stop` to inspect or stop the detached
runner. The combined log is
`$HOME/Results/Multiverse/.tde-core-missing-unix/runner.log`, with a separate
log for each dataset below its `logs` directory.
