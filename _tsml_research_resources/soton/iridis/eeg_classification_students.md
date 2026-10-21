# EEG classification experiments on Iridis 6

This guide takes you from a fresh Iridis account to tsml-eval prediction files for
classifiers from aeon and aeon-neuro. Commands below are for Bash on Iridis 6. The
[general Iridis Python guide](iridis_python.md) covers cluster access, storage, Conda,
and the supplied submission scripts in more detail. Check the [HPC Community Wiki](https://sotonac.sharepoint.com/teams/HPCCommunityWiki)
for current queue rules and account instructions.

## Before you start

You need an Iridis account, access to the Southampton network or VPN, and an
aeon-neuro release containing the EEG dataset downloader and classifiers. This guide
expects aeon-neuro **0.3.0 or later** and a tsml-eval checkout containing its `neuro`
extra. If the tsml-eval changes have not been merged into `main`, use the branch
provided by your supervisor. The code below uses Python 3.13 because the aeon-neuro
checkout inspected for this guide does not declare Python 3.14 support.

The EEG dataset downloader depends on published download records. At the time this
guide was written, the local aeon-neuro dataset map listed 31 problems but marked
every record `49` (unavailable). The download step will report these names and write
an empty dataset list until the records are provided. **Do not submit experiments
against an empty list.** Ask your supervisor for the updated aeon-neuro release or
staged `.ts` files if this happens.

## 1. Connect and install the software

Connect to `iridis6.soton.ac.uk` over SSH. Run these commands on the login node,
where Git and package downloads are available:

```bash
module load conda/python3
conda create -n tsml-neuro python=3.13
conda activate tsml-neuro
git clone https://github.com/time-series-machine-learning/tsml-eval.git "$HOME/tsml-eval"
cd "$HOME/tsml-eval"
# If the aeon-neuro changes are not yet on main, checkout the supplied branch here.
python -m pip install --editable '.[neuro]'
python -c 'import aeon, aeon_neuro, tsml_eval; print(aeon.__version__, aeon_neuro.__version__, tsml_eval.__version__)'
```

If you intend to run aeon's TensorFlow classifiers such as
`InceptionTimeClassifier`, also install the `deep_learning` extra in the same
environment (or use a separate GPU environment):

```bash
python -m pip install --editable '.[neuro,deep_learning]'
```

The editable install makes the Python files in your clone the ones that run. Keep the
clone at a known commit while collecting results; record `git rev-parse HEAD` and
`python -m pip freeze` alongside the results.

## 2. Stage the EEG problems once

Put datasets and results on your scratch storage, not inside the Git checkout. The
supplied Python script calls `load_eeg_classification` for each listed EEG problem,
checks that both `.ts` splits exist, and writes only usable dataset names to a text
file. Each line in that file is consumed by the Iridis submission scripts.

```bash
EEG_WORK="/iridisfs/scratch/$USER/eeg"
mkdir -p "$EEG_WORK/data" "$EEG_WORK/results" "$EEG_WORK/output"
python "$HOME/tsml-eval/_tsml_research_resources/soton/iridis/download_eeg_classification.py" \
    --data-dir "$EEG_WORK/data" \
    --dataset-list "$EEG_WORK/datasets.txt"
cat "$EEG_WORK/datasets.txt"
```

Run this download step on a node that has internet access. The [Iridis Python
guide](iridis_python.md) notes that compute nodes may not have internet access. The
script processes one problem at a time but does load its training split into memory;
for very large problems, ask the cluster team or your supervisor for a suitable
transfer method rather than running a heavy download on the login node. You can
check a single problem first with `--only SelfRegulationSCP1`.

The resulting layout must be:

```text
eeg/data/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts
eeg/data/SelfRegulationSCP1/SelfRegulationSCP1_TEST.ts
eeg/datasets.txt
```

If some names are unavailable or a download fails, the script exits with an error
and prints a summary. It still writes the successfully staged names to
`datasets.txt`, so you can rerun it after the source problem is fixed. Existing
complete train/test pairs are reused. Inspect the list before submitting jobs.

## 3. Run one smoke test

First run one small experiment in an interactive compute session (`sinteractive`
on Iridis). Do this after the dataset files have been staged; compute nodes may not
be able to download them. In the new session, reactivate the environment:

```bash
module load conda/python3
conda activate tsml-neuro
EEG_WORK="/iridisfs/scratch/$USER/eeg"
python -u "$HOME/tsml-eval/tsml_eval/experiments/classification_experiments.py" \
    "$EEG_WORK/data" "$EEG_WORK/results" \
    ROCKET SelfRegulationSCP1 0
```

Replace `SelfRegulationSCP1` with a name actually present in your
`datasets.txt`. `ROCKET` is an aeon classifier. For an aeon-neuro CPU classifier,
replace it with `RiemannianMDMClassifier` or `CSPSVMClassifier`. The classifier
names and aliases implemented in this checkout are in
[`tsml_eval/experiments/_get_classifier.py`](../../../tsml_eval/experiments/_get_classifier.py).
New aeon-neuro classes exported by `aeon_neuro.classification` can be requested as
`neuro:ClassName`.

The final positional argument `0` uses the supplied train/test split and is also
the result-file suffix. The file should appear at:

```text
eeg/results/ROCKET/Predictions/SelfRegulationSCP1/testResample0.csv
```

The script skips a completed result by default. Add `-ow` only when you intend
to overwrite it. Add `-tr` if you also need a training results file; that can be
substantially slower because it may perform cross-validation. For nonzero resample
IDs, tsml-eval stratifies a new train/test split and writes, for example,
`testResample1.csv`.

## 4. Submit CPU experiments across the dataset list

The existing [serial classification submission script](serial_scripts/classification_experiments.sh)
submits Slurm array jobs for each dataset and classifier. Make a personal copy so
your paths and scheduler choices do not affect other students:

```bash
cp "$HOME/tsml-eval/_tsml_research_resources/soton/iridis/serial_scripts/classification_experiments.sh" \
   "$HOME/eeg_classification_cpu.sh"
```

Edit the configuration block near the top of your copy. For a first pass, use one
fold (the original split), a modest submission limit, and your own paths:

```bash
max_folds=1
start_fold=1
max_num_submitted=10
queue="batch"
username="$USER"
local_path="$HOME/"
data_dir="/iridisfs/scratch/$USER/eeg/data/"
datasets="/iridisfs/scratch/$USER/eeg/datasets.txt"
results_dir="/iridisfs/scratch/$USER/eeg/results/"
out_dir="/iridisfs/scratch/$USER/eeg/output/"
script_file_path="$HOME/tsml-eval/tsml_eval/experiments/classification_experiments.py"
env_name="tsml-neuro"
classifiers_to_run="ROCKET RiemannianMDMClassifier"
generate_train_files="false"
predefined_folds="false"
normalise_data="false"
```

Keep the trailing `/` on directory settings; the script concatenates names onto
these paths. Check `max_memory` and `max_time` in the same block. Increase them
only if jobs show they need more resources. The script requests one CPU per task,
and the Python experiment script limits common thread pools to one.

After checking that `datasets.txt` contains the intended names, leave the Conda
environment and run the **Bash** submission script on the login node:

```bash
conda deactivate
cd "/iridisfs/scratch/$USER/eeg"
bash "$HOME/eeg_classification_cpu.sh"
squeue -u "$USER"
```

The script uses Bash-specific syntax, so run it with `bash`, not `sh`. It skips
result files that already exist and writes Slurm logs under `eeg/output/`. If
you need more resamples later, increase `max_folds`; fold 1 maps to result
`Resample0`, fold 2 to `Resample1`, and so on.

## 5. GPU classifiers

For aeon-neuro deep learning models such as `EEGNetClassifier` and
`DeepConvNetClassifier`, follow the [GPU environment instructions](iridis_python.md#51-tsml-eval-gpu)
to build an Apptainer sandbox with the required packages. Install this tsml-eval
checkout and its `neuro` extra **inside the sandbox**. Copy the
[GPU classification submission script](gpu_scripts/gpu_classification_experiments.sh)
and set its `data_dir`, `datasets`, `results_dir`, `out_dir`,
`script_file_path`, `container_path`, `username`, `classifiers_to_run`, and
fold and resource settings for your account. Set `classifiers_to_run` initially
to one model and one dataset. That script requests one GPU per array task on the
`gpu` partition and calls the same tsml-eval Python experiment file via
`apptainer exec --nv`.

Confirm in a GPU job that the container can read the cloned source and scratch
data paths, imports `aeon_neuro`, and sees the GPU before submitting the full
dataset list. Do not assume that packages installed in your host Conda environment
are available inside the container.

## 6. Check and keep the results

Each successful test experiment writes a CSV under
`<results_dir>/<classifier>/Predictions/<dataset>/testResample<N>.csv`.
If `-tr` was requested, there is also a `trainResample<N>.csv`. The file begins
with experiment metadata and then stores predictions and probabilities. The
[results format notebook](../../../examples/results_format.ipynb) explains the
format and the [evaluation notebook](../../../examples/evaluation.ipynb) shows
how to compare classifiers.

For a failed job, inspect the corresponding `.err` and `.out` files in
`eeg/output/<classifier>/<dataset>/`. If the submission script prints that a job
was skipped, check whether the expected result file already exists. Keep the
dataset list, the tsml-eval Git commit, package versions, and job settings with
the CSV files so experiments can be reproduced.
