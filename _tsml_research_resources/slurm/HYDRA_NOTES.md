# Notes for Claude on Hydra: EEG classification runs

Written 2026-11-05 from the Windows machine, where these runs have been set up and
run on Iridis 6 (CPU) and Iridis X (GPU). Read this, then `README.md` in this
directory, before doing anything.

## What the user wants

Many classifiers (around 85) on the EEG archive (around 50 datasets, some very
large). Hydra is for the **smaller datasets**, so most of the grid finishes quickly.
Iridis 6 keeps the large datasets and the slow classifiers. The user will name the
classifiers. Do not guess the list.

## Working with this user

- **The user cannot paste multi-line commands.** Give one-line commands, chaining
  with `&&` or `;` where needed. Put anything longer in a script in this directory.
- **Ask before editing a config they did not ask about.** For example, don't change
  a `_cv` toml when only train/test was asked for.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  Never add a `Claude-Session:` trailer or a claude.ai session URL anywhere.
- Commit or push only when asked. The working branch is `ajb/slurm-runner`.
- Chat replies don't need the AGENTS.md AI-disclosure footer. That footer is for PR
  bodies only.

## The runner

`tsml_slurm.py` is the whole system, so no other monitoring program is needed. Each
experiment is a toml in `experiments/`, naming a cluster profile in `clusters/`.
`start` submits jobs and runs a supervisor that refills the queue, diagnoses
failures, retries out-of-memory tasks at the next memory tier, writes `report.txt`
and emails progress.

- Commands: `check`, `submit --dry-run`, `start`, `status [--progress]`,
  `retry --reason "..."`, `stop`.
- `python farms.py` gives a one-screen view of every running farm.
- A task counts as complete when its results file exists, so experiments can be
  restarted safely and nothing is run twice.
- Two modes:
  - `array` mode: one Slurm array task per experiment (Hali style).
  - `farm` mode: whole-node jobs packing many tasks (Iridis 6 style), with `top_up`
    letting a running farm take waiting tasks into free slots.
- The supervisor runs either as a Slurm job (`mode = "slurm"`) or as a loop in
  `screen` on the login node (`mode = "login"`).
- Tests: `python -m pytest tests -q`, 70 tests. On a machine whose default tmp
  directory is not writable, add `--basetemp=<writable dir>`.

### First job on Hydra: a cluster profile

There is no `clusters/hydra.toml` yet. Write one, modelled on `clusters/hali.toml`
(array mode) or `clusters/iridis6.toml` (farm mode), using facts you look up:

- `sinfo -s`, `sinfo -o "%P %c %m %l %G"` for partitions, CPUs per node, memory per
  node, time limits and GPUs.
- `sacctmgr show assoc user=$USER format=account,partition,qos` and
  `sacctmgr show qos format=name,maxwall,maxtrespu,maxjobspu` for accounts, QOS and
  per-user caps.
- `module avail` for the conda or python module, and where `conda.sh` lives.
- `echo $HOME` for `home`.

Choose array or farm mode based on Hydra's rules:
- If whole nodes are cheap to get and there is a per-user node cap, use farm mode.
- If many single-core jobs are allowed, use array mode, which is simpler.

Then confirm the settings with the user.

## Splitting the work with Iridis 6

Completion is judged by the files each cluster can see (README section 8, "Splitting
one study across clusters"). If Hydra and Iridis 6 run the same classifier, they
need **disjoint dataset lists** or they will duplicate work:

- **Hydra:** a dataset list of the small problems, for example
  `~/DataSetLists/eeg_small.txt`.
- **Iridis 6:** the rest.
- Agree the size cut-off with the user. A simple rule is total `.ts` size per
  dataset: `du -sm ~/Data/EEG/*`.

Iridis 6 already has results for many classifier and dataset pairs. Before starting
an experiment that overlaps with them, copy the existing results for those
classifiers into Hydra's `results_dir` so they count as complete, or leave the
overlapping datasets out of Hydra's list.

## Layout and setup on Hydra

Use the same layout as Iridis, so results can be merged by copying:

- data: `~/Data/EEG/<Dataset>/<Dataset>_{TRAIN,TEST}.ts` (the user is copying it now)
- dataset list: `~/DataSetLists/eeg.txt`, one name per line
- results: `~/Results/EEG/<aeon category>/<Name>/Predictions/<Dataset>/testResample0.csv`
  - the categories are ConvolutionBased, IntervalBased, DictionaryBased,
    ShapeletBased, DistanceBased, FeatureBased, Hybrid, Tabular and DeepLearning,
    grouped as aeon and aeon-neuro group them
  - these are the `[estimators]` table keys
- repo: `~/Code/tsml-eval`, branch `ajb/slurm-runner`

Settings for every EEG experiment:
- `resamples = 1`: resample 0 only, the supplied train/test splits. Random resamples
  could put one subject's trials on both sides.
- `smallest_datasets_first = true`.

The CPU environment is `tsml-eval`:
```
conda create -n tsml-eval python=3.12 && conda activate tsml-eval && pip install -e ~/Code/tsml-eval[neuro]
```
- aeon-neuro must be >= 0.1.1 (XDAWN, CSP).
- For checkpointing of the HC2 components (Arsenal, DrCIF, STC-aeon, TDE), install
  aeon from the `ajb/hc2-components-checkpoint` branch. Check it with
  `python test_checkpointing.py`. The installed aeon is 1.5.0.
- **GPU environments:** keep TensorFlow and PyTorch in separate envs. Two CUDA stacks
  in one env break both:
  - `make_tensorflow_only.sh` builds the TensorFlow env.
  - `make_torch_env.sh` builds the PyTorch env, for EEGNet, DeepConvNet and
    EEGConformer.
  - Keep `PYTHONNOUSERSITE=1` set when installing. A package in `~/.local` is shared
    by every env of the same Python version, so an uninstall in one env removed
    Keras from the other on Iridis X.

Existing experiment files show the estimator names and settings to copy:
- `iridis6_eeg_cpu.toml`: aeon-neuro Riemannian, CSP and XDAWN
- `iridis6_eeg_aeon.toml`: QUANT and tabular
- `iridis6_eeg_hc2_components.toml`: train files, checkpoints, retry_timeouts
- `iridis6_eeg_mrhydra.toml`
- `iridis6_eeg_hc2.toml`
- `iridisx_eeg_aeon_gpu.toml`: CNN, IT and LITETime-MV
- `iridisx_eeg_gpu.toml`: torch
- `hali_eeg_cpu.toml` and `hali_eeg_gpu.toml`: the same experiments in array mode

Run `check` on each new toml. It catches estimator names the factory doesn't know.

## Known data and classifier problems

- **ButtonPress is dropped** (too few channels). Leave it out of every list.
- **MatchWords:** every value is below 1e-7, so aeon 1.5.0's
  `check_collection_variance` rejects it for the aeon classifiers. **VideoEmotions:**
  channel 6 (0-based) is flat for 156 consecutive cases starting at case 4684, which
  fails the same check. The fix (rescale the data, or turn the check off) is not
  decided yet. Ask the user.
- **CovidSurvivors:** the TRAIN file was missing on Iridis X. Check both files exist
  for every dataset. `check` reports missing ones.
- **RiemannianMDM and RiemannianKNN** fail on FaceDetection and MindReading
  (singular covariance). The `-OAS` variants cover these datasets, so the failures
  are expected and need no retry.
- **HC2 from the components:** `build_hc2_from_files.py` builds HC2 from the
  components' train and test files, with no data and no refit. It writes to
  `Hybrid/HC2FromFile`. The components need `train_files = true`.

## Good practice for a big grid

- Before queueing the whole grid, run a pilot: every classifier on the two or three
  largest datasets in Hydra's list. This sets the memory tiers and time limit for
  each group and catches broken classifiers early.
- Split the classifiers into a few experiments by cost, so each gets its own memory
  tiers and time limit:
  - fast CPU classifiers;
  - slow or memory-hungry CPU classifiers;
  - TensorFlow;
  - PyTorch.
- Nothing coordinates jobs across experiments. `max_jobs` / `max_active` are per
  experiment, so keep their total within the per-user limits.
