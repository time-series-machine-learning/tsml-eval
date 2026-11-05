# Notes for Claude on Hydra: EEG classification runs

Written 2026-11-05 on the Windows machine, where these runs were set up for Iridis 6
(CPU) and Iridis X (GPU). Read this, then `README.md` in this directory, before
doing anything.

## Hydra

Hydra is the user's own workstation: 32 CPUs, 4 GPUs, and **no Slurm**. Use
`run_local.py` to run experiments on it. It reads the usual experiment tomls (with
`cluster = "hydra"`, see `clusters/hydra.toml`) and writes the usual results layout.
`tsml_slurm.py start/submit` will not work here, because it needs Slurm.

```
python run_local.py experiments/hydra_*.toml --dry-run   # what would run, and the python
python run_local.py experiments/hydra_*.toml             # run until done (in tmux/screen)
python run_local.py experiments/hydra_*.toml --status    # progress, failed tasks
python run_local.py experiments/hydra_*.toml --retry     # clear failed-for-good, then run
```

How `run_local.py` behaves:
- All the experiments given share one budget of CPUs, memory and GPUs. The defaults
  are the whole machine less 8 GB; set them with `--cpus`, `--memory-gb` and `--gpus`.
- A task takes `max(n_jobs, cpus_per_task)` CPUs and its memory tier.
- GPU tasks take one whole GPU each and see only that GPU through
  `CUDA_VISIBLE_DEVICES`. CPU tasks see no GPU.
- **The memory budget is a reservation, not a limit.** No cgroup enforces it. A task
  that uses more than its tier can push the machine into swap or wake the kernel OOM
  killer.
  - Set the tiers in `clusters/hydra.toml` from `free -g` and the pilot runs.
  - Run the biggest datasets with fewer concurrent tasks.
- A task counts as complete when its results file exists, so restarting is safe.
- Failures are handled as follows:
  - An out-of-memory failure moves the task up a memory tier.
  - A task that goes past `time_limit` is killed and recorded as a timeout.
  - Other failures are retried up to `max_attempts`.
- The dataset list is re-read every `--rescan` minutes.
- Before an env's first GPU task, the runner checks that the env can see a GPU
  (`gpu_check`).
- State is kept in `~/tsml-slurm/<name>/local.json`. Logs are in
  `~/tsml-slurm/<name>/logs/<Estimator>/<Dataset>/local-<time>_<resample>.{out,err,status}`.
- **Differences from the cluster runner:**
  - It does not refuse changed estimator kwargs. If you change an estimator's
    settings, give it a new name or a new `results_dir` yourself.
  - It does not check that the repo is clean.
- Tests: `python -m pytest tests -q`. `tests/test_run_local.py` covers this runner.

## The plan across the three resources

The user has Hali (UEA), Iridis (6 for CPU, X for GPU) and Hydra. They want to run:
- the aeon classifiers;
- 50+ braindecode models ported into aeon-neuro;
- aeon-neuro feature pipelines.

All of these have CPU and GPU versions. The plan as discussed, which the user may
change:

| Resource | Runs |
|---|---|
| Iridis 6 CPU | HC2 components and HC2, then CV, then more aeon CPU classifiers; large datasets |
| Iridis X GPU | braindecode on the medium and large datasets |
| Hali | anything that times out at Iridis 6's 60 hours (Hali allows 7 days); aeon deep learners (TensorFlow); extra GPU capacity |
| Hydra | smoke tests of every new port; feature pipelines and small datasets; merging all results; analysis |

The rules:
- **One owner per (estimator, dataset).** Give each estimator family to one
  resource. If a family is shared, share it by disjoint dataset lists, for example
  `~/DataSetLists/eeg_small.txt` on Hydra and the rest elsewhere. Completion is judged
  from the files a machine can see, so overlapping runs duplicate work.
- **One results layout,** with Hydra holding the master copy. `rsync` each cluster's
  `~/Results/EEG` into Hydra's.
- **The same package versions everywhere** (aeon, aeon-neuro, braindecode, torch).
  When they change, upgrade all three machines together.
- **Every new model goes through these stages before a cluster gets it:**
  1. A smoke test on Hydra: every model on 2–3 tiny datasets, with few epochs, for
     example through `[estimator_kwargs]`. Use a separate `results_dir` so smoke
     results never mix with real ones.
  2. A pilot on the 2–3 largest datasets, to set the memory tiers and time limits.
  3. The full grid.
- **CV is about 10× the cost of train/test.** Start it with the cheap classifiers.
- **Use one toml per (resource, family, cost class),** for example
  `hydra_eeg_braindecode_smoke.toml` and `hydra_eeg_features.toml`.

## Working with this user

- **The user cannot paste multi-line commands.** Give one-line commands. Put
  anything longer in a script.
- **Ask before editing a config they did not ask about.** For example, don't change
  a `_cv` toml when only train/test was asked for.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  Never add a `Claude-Session:` trailer or a claude.ai session URL anywhere.
- Commit or push only when asked. The branch is `ajb/slurm-runner`.
- Chat replies don't need the AGENTS.md AI-disclosure footer. That footer is for PR
  bodies only.

## Layout and environments

The layout is the same as on the clusters:

- data: `~/Data/EEG/<Dataset>/<Dataset>_{TRAIN,TEST}.ts`
- lists: `~/DataSetLists/eeg.txt`, one name per line
- results: `~/Results/EEG/<Category>/<Name>/Predictions/<Dataset>/testResample0.csv`
  - the categories are ConvolutionBased, IntervalBased, DictionaryBased,
    ShapeletBased, DistanceBased, FeatureBased, Hybrid, Tabular and DeepLearning,
    grouped as aeon and aeon-neuro group them
  - these are the `[estimators]` table keys
- repo: `~/Code/tsml-eval`

Settings for every EEG experiment:
- `resamples = 1`: resample 0 only, the supplied train/test splits. Random resamples
  could put one subject's trials on both sides.
- `smallest_datasets_first = true`.

Environments (conda; `run_local.py` finds `~/.conda/envs`, `~/miniforge3`,
`~/miniconda3` and `~/anaconda3`):

- `tsml-eval` for CPU:
  `conda create -n tsml-eval python=3.12 && conda activate tsml-eval && pip install -e ~/Code/tsml-eval[neuro]`
  - aeon-neuro must be >= 0.1.1.
  - For the checkpointed HC2 components, use aeon from the
    `ajb/hc2-components-checkpoint` branch.
- `tsml-eval-torch` for braindecode and the aeon-neuro deep models: `bash make_torch_env.sh`.
- A separate TensorFlow env for the aeon deep learners (CNN, IT, LITETime-MV):
  `make_tensorflow_only.sh`.
- Never put TensorFlow and PyTorch in one env: two CUDA stacks break both.
- Install with `PYTHONNOUSERSITE=1`. A package in `~/.local` is shared by every env of
  the same Python, and an uninstall in one env removed Keras from another on
  Iridis X.

Existing tomls to copy estimator names and settings from:
- `iridis6_eeg_cpu.toml`: Riemannian, CSP and XDAWN
- `iridis6_eeg_aeon.toml`: QUANT and tabular
- `iridis6_eeg_hc2_components.toml`
- `iridis6_eeg_mrhydra.toml`
- `iridisx_eeg_aeon_gpu.toml`
- `iridisx_eeg_gpu.toml`: EEGNet, DeepConvNet and EEGConformer, torch, `device = "cuda"`

Change `cluster` to `"hydra"` and drop the farm-only `[slurm]` keys (`max_jobs`,
`top_up`, `waves`), because the local profile is not farm mode. Then run
`python tsml_slurm.py check <toml>` to catch unknown estimator names. Its
Slurm-specific parts don't matter here.

## Known data and classifier problems

- **ButtonPress is dropped** (too few channels). Leave it out of every list.
- **MatchWords:** every value is below 1e-7, so aeon 1.5.0's
  `check_collection_variance` rejects it. **VideoEmotions:** channel 6 (0-based) is
  flat for 156 consecutive cases starting at case 4684, which fails the same check.
  The fix (rescale the data, or turn the check off) is not decided. Ask the user.
- **CovidSurvivors:** its TRAIN file was missing on Iridis X. Check every dataset has
  both files.
- **RiemannianMDM and RiemannianKNN** fail on FaceDetection and MindReading
  (singular covariance). The `-OAS` variants cover these datasets, so the failures
  are expected.
- **HC2 from components:** `build_hc2_from_files.py` builds HC2 from the
  components' train and test files, with no data and no refit, into
  `Hybrid/HC2FromFile`.
