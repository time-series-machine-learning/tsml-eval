This directory contains instructions and utilities for running experiments using
tsml-eval on University of Southampton (Soton) and
University of Bradford hardware.

For non-Soton/Bradford users, the directory contents are likely irrelevant.
Some of the contents could be adapted for other purposes or could be generalised
for other Slurm/Linux devices, but running the scripts or following the hardware
instructions will likely achieve nothing without alterations.

## Recurring Hali Multiverse controller

`multiverse_controller.py` performs one restart-safe reconciliation of the 30-fold
Multiverse classification experiments. It:

- processes categories in the configured order: `IntervalBased`, `DictionaryBased`,
  `ShapeletBased`, `ConvolutionBased`, then the remaining categories;
- recognises existing `Classifier_Dataset` Slurm arrays and completed result files;
- fills the configured running/pending task ceiling without duplicating active work;
- submits one-CPU jobs and disables CUDA and numerical-library worker threads;
- starts tasks at 16 GB and retries confirmed OOMs at 32, 64, then 128 GB;
- records every observed OOM, timeout, and other failure in saved and emailed reports;
- saves and emails a progress table after every cycle.

Paths, Slurm limits, category order, and the 43 multivariate-compatible aeon
classifiers are configured in `multiverse_controller.toml`. Results are expected at
`/gpfs/home/ajb/Results/Multiverse/<Category>`.

Stop any older queue-feeding script before starting this controller. Already queued
experiment arrays can remain: the controller detects them by job name and array index.
First inspect the TOML file and run a read-only cycle:

```bash
cd /gpfs/home/ajb/Code/tsml-eval
conda activate tsml-eval
python _tsml_research_resources/multiverse_controller.py --dry-run
```

Run a report without submitting jobs with `--report-only`. To start the recurring
three-hour supervisor in `screen`:

```bash
screen -S multiverse-controller
bash _tsml_research_resources/run_multiverse_controller.sh
```

Detach with `Ctrl-a d`. The shell supervisor runs one Python cycle, lets it exit,
sleeps for three hours, and starts a fresh cycle even when the previous one failed.
Pass a different interval in seconds as the second argument if required:

```bash
bash _tsml_research_resources/run_multiverse_controller.sh \
    _tsml_research_resources/multiverse_controller.toml 7200
```

Runtime state and reports are stored under
`~/Results/Multiverse/.controller/`. In particular, inspect `latest_report.txt` and
`supervisor.log`. Timeouts are terminal outcomes. Other non-OOM failures stop after
`max_attempts`; OOMs instead move through the configured memory tiers. Once every
remaining task in a category has a terminal reported outcome, the controller advances
so permanent failures do not leave the allocation idle. Delete only that task's entries
from `.controller/state.json` after correcting its cause if it should be retried again.

Periodic email requires one of `mail`, `mailx`, or `sendmail` on the login node. Test
mail delivery before leaving the controller unattended:

```bash
echo "Multiverse controller email test" | mail -s "Multiverse test" ajb@uea.ac.uk
```
