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

- runs a breadth-first first pass, interleaving work across all configured categories;
- skips explicitly deferred slow datasets (initially `AustraliaRainfall_disc`);
- recognises existing `Classifier_Dataset` Slurm arrays and completed result files;
- records active jobs and their memory across all categories, even before that category
  becomes eligible for new submissions;
- fills the configured 8,000 running/pending task ceiling without duplicating active
  work;
- submits one-CPU jobs and disables CUDA and numerical-library worker threads;
- makes one 8 GB attempt per missing result, deferring failures to a later
  large-memory completion pass;
- records every observed OOM, timeout, and other failure in saved and emailed reports;
- saves a progress table hourly and emails it no more than once every four hours.

Paths, Slurm limits, category order, and the 43 multivariate-compatible aeon
classifiers are configured in `multiverse_controller.toml`. Results are expected at
`/gpfs/home/ajb/Results/Multiverse/<Category>`.

For a clean first-pass restart, stop every old queue feeder before cancelling jobs;
otherwise it can immediately refill the queue. Archive the old state so the new pass
does not inherit its attempt counters. Existing result files remain in place and are
skipped:

```bash
screen -S multiverse-controller -X quit
pkill -f '[r]un_multiverse_controller.sh' || true
pkill -f '[_]tsml_research_resources/multiverse_controller.py' || true
pgrep -af 'multiverse_controller.py|run_multiverse_controller.sh' || true
squeue --array --user="$USER" -o '%.20i %.10T %.50j'
keep_file="/gpfs/home/${USER}/Code/tsml-eval/_tsml_research_resources/multiverse_keep_jobs.txt"
mapfile -t cancel_ids < <(
    awk 'NR==FNR {keep[$1]=1; next} !($1 in keep) {print $1}' \
        "$keep_file" \
        <(squeue --noheader --array --user="$USER" \
            --states=RUNNING,PENDING --format='%i')
)
printf 'Cancelling %s jobs:\n' "${#cancel_ids[@]}"
printf '  %s\n' "${cancel_ids[@]}"
if ((${#cancel_ids[@]})); then
    scancel "${cancel_ids[@]}"
fi
mkdir -p ~/Results/Multiverse/.controller
if [[ -f ~/Results/Multiverse/.controller/state.json ]]; then
    mv ~/Results/Multiverse/.controller/state.json \
       ~/Results/Multiverse/.controller/state.before-8gb.$(date +%Y%m%d-%H%M%S).json
fi
squeue --array --user="$USER" -o '%.20i %.10T %.50j'
```

The cancellation block retains the exact array elements listed in
`multiverse_keep_jobs.txt` and cancels all other running or pending jobs owned by the
user, including unrelated work. Inspect the preceding `squeue` output and the keep
file before running it. Then inspect the TOML file and run a read-only cycle:

```bash
cd /gpfs/home/ajb/Code/tsml-eval
conda activate tsml-eval
python _tsml_research_resources/multiverse_controller.py --dry-run
```

Run a report without submitting jobs with `--report-only`. To start the recurring
hourly supervisor in `screen`:

```bash
screen -S multiverse-controller
bash _tsml_research_resources/run_multiverse_controller.sh
```

Detach with `Ctrl-a d`. The shell supervisor runs one Python cycle, lets it exit,
sleeps for one hour, and starts a fresh cycle even when the previous one failed. A
persistent timestamp limits successful report emails to one every four hours, including
across supervisor restarts. Pass different cycle and email intervals in seconds as the
second and third arguments if required:

```bash
bash _tsml_research_resources/run_multiverse_controller.sh \
    _tsml_research_resources/multiverse_controller.toml 7200 28800
```

Runtime state and reports are stored under
`~/Results/Multiverse/.controller/`. In particular, inspect `latest_report.txt` and
`supervisor.log`. In the supplied first-pass configuration every failure is terminal
after the single 8 GB attempt, but remains recorded for planning the later high-memory
completion pass. To enable escalation later, configure multiple `memory_mb_levels`,
increase `max_attempts`, disable `all_categories_first_pass` if ordered completion is
preferred, and archive the first-pass state before starting that distinct phase.

Periodic email requires one of `mail`, `mailx`, or `sendmail` on the login node. Test
mail delivery before leaving the controller unattended:

```bash
echo "Multiverse controller email test" | mail -s "Multiverse test" ajb@uea.ac.uk
```
