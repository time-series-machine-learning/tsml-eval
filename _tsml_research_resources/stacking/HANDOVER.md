# Stacking HC2 components with tabular foundation models: handover

## What this is

HIVE-COTE 2 combines its components (Arsenal, DrCIF, STC, TDE) with CAWPE: a fixed
weighted sum of their class probabilities. This experiment replaces that combination
with a learned one (stacking). The question is whether tabular foundation models,
used as the combiner, beat both CAWPE and ordinary stackers.

Each dataset and resample is turned into a standard tabular classification problem.

- **Rows:** one per case.
- **Features:** each component's class probabilities in turn, so there are
  `n_components * n_classes` features. The component order is listed in each
  archive's `components.txt`, and classes are in index order.
- **Train file:** the components' train-file probabilities. These are the components'
  own out-of-bag or cross-validation estimates, which is what CAWPE uses for its
  weights.
- **Test file:** the components' test predictions.
- **Labels:** the class indices from the results files.

| Archive | Components | Datasets | Resamples |
|---|---|---|---|
| UCR | Arsenal, DrCIF, STC, TDE | 112 | 30 |
| EEG | Arsenal, DrCIF, STC-aeon, TDE | 51 | 1 (resample 0) |
| Multiverse | Arsenal, DrCIF-500, STC (no TDE train files exist) | 66 | 1 (resample 0) |

**UCR is the primary target, then EEG.** Multiverse is optional.

### Problem sizes

- **UCR:**
  - Train size: median 190 cases, largest ElectricDevices (8,926 train, 7,711 test).
  - Crop has 16,800 test cases.
  - Up to 240 features.
  - 20 datasets have more than 10 classes, up to 60.
- **EEG:**
  - Train size: median 1,260 cases, largest ALSAttention (14,400 train, 19,200 test)
    and SpellerFlashes (12,600 train).
  - Up to 12 classes and 48 features.

## Files

Zips are on the laptop in `C:/Results/Stacking/`:

| Zip | Size | Contents |
|---|---|---|
| `stacking_data_UCR.zip` | 265 MB | `UCR/<dataset>/<dataset><r>_TRAIN.ts` and `_TEST.ts` for r = 0..29 |
| `stacking_data_EEG.zip` | 9 MB | `EEG/<dataset>/<dataset>0_TRAIN.ts` and `_TEST.ts` |
| `stacking_data_Multiverse.zip` | 19 MB | the same layout for Multiverse |

Unzip all three into one folder, for example `~/Data/Stacking`, so that you have
`~/Data/Stacking/UCR/...`. Each archive folder has `datasets.txt` and
`components.txt`.

**File format.** The files are `.ts` files with a single channel; that is only how the
`.ts` format stores a flat feature vector. They are in tsml-eval's
predefined-resample layout, so they load with:

```python
from tsml_eval.utils.datasets import load_experiment_data
X_train, y_train, X_test, y_test, _ = load_experiment_data(
    "~/Data/Stacking/UCR", "Beef", 3, True
)
X_train = X_train[:, 0, :]  # (n_cases, n_components * n_classes)
```

The code is in this folder, on tsml-eval branch `ajb/stacking`:

| File | What it does |
|---|---|
| `extract_stacking_data.py` | How the problems were made from the results files. Already run; the paths in it are the laptop's |
| `stacking_models.py` | The classical stackers, and `CAWPE`, a reimplementation of the HC2 combination used as the reference |
| `foundation_models.py` | Wrappers for TabPFN (version pinned, with many-class handling) and TabICL |
| `run_stacking.py` | Runs the models and writes tsml-eval results files. Existing results are skipped, so a stopped run can be restarted |

## Models

### Reference

All three are fitted on the same train file.

| Name | What it is |
|---|---|
| `Mean` | Unweighted mean of the component probabilities |
| `CAWPE` | Weights are component train accuracy to the power 4: this is HC2 |
| `CAWPE-p0.5` | HC2 with prior correction 0.5, which is the default proposed for aeon |

`CAWPE` reproduces the HC2 built from files (checked on three EEG datasets), so it is
the baseline to beat.

### Classical stackers (running on the laptop)

- `LR`: logistic regression on the probabilities.
- `LR-log`: logistic regression on the log probabilities.
- `LDA`: shrinkage LDA.
- `NB`: Gaussian naive Bayes.
- `RF`, `ET`: random forest and extra trees, 500 trees each.
- `HGB`: histogram gradient boosting.

These are cheap, so they could equally be rerun on Hydra.

### Foundation models (`--foundation`)

| Name | Model | Notes |
|---|---|---|
| `TabPFN-v2` | TabPFN v2 weights, pinned with `create_default_for_version` | More than 10 classes goes through `tabpfn_extensions.many_class.ManyClassClassifier` (output codes over ≤10-class sub-problems). `ignore_pretraining_limits=True` lets it run above 10k train cases |
| `TabPFN-v2.5` | TabPFN v2.5 weights | **Opt-in.** The weights need a Prior Labs account: accept the licence at https://ux.priorlabs.ai/account/licenses, then `export TABPFN_TOKEN=<API key>`. Without the token, the model is left out, because otherwise the download waits for a browser login and hangs a headless job |
| `TabICL` | TabICL v2 checkpoint, 8 estimators | Handles more than 10 classes itself |

**Not used:**
- **The TabPFN cloud client:** it sends the data to Prior Labs.
- **TabPFN v3 and later:** these can be added in `foundation_stackers`.

**Telemetry.** TabPFN sends usage telemetry unless `TABPFN_DISABLE_TELEMETRY=1`;
`foundation_models.py` sets it.

## Running on Hydra

### Set-up

```bash
cd ~/Code/tsml-eval && git fetch && git checkout ajb/stacking
pip install tabpfn tabpfn-extensions tabicl   # into the tsml-eval environment
```

On the laptop this installed tabpfn 9.2.0, tabicl 2.2.0 and tabpfn-extensions 0.6.3,
and changed none of torch, scikit-learn or numpy. On Hydra, check that torch is a CUDA
build.

The weights download on first use:
- TabPFN to `~/.cache/tabpfn` (on Windows, `AppData/Roaming/tabpfn`).
- TabICL to the HuggingFace cache.

If compute nodes have no internet, run one small problem on the login node first so
the weights are cached:

```bash
python run_stacking.py UCR --foundation --datasets Beef --jobs 1 \
    --data ~/Data/Stacking --results /tmp/fm_check
```

### Commands

Run from `_tsml_research_resources/stacking`:

```bash
# foundation models, UCR first; one process per GPU
python run_stacking.py UCR --foundation --jobs 1 --device cuda \
    --data ~/Data/Stacking --results ~/Results/Stacking
python run_stacking.py EEG --foundation --jobs 1 --device cuda \
    --data ~/Data/Stacking --results ~/Results/Stacking

# one model or a few datasets, e.g. to time it first
python run_stacking.py UCR --foundation --models TabICL --datasets Beef Adiac Crop \
    --device cuda --data ~/Data/Stacking --results ~/Results/Stacking

# classical stackers and references (CPU), if not copying them from the laptop
python run_stacking.py UCR --jobs 16 --data ~/Data/Stacking --results ~/Results/Stacking
```

`--jobs` is the number of datasets in parallel. With several GPUs, run one process per
GPU with `CUDA_VISIBLE_DEVICES` and a split of `--datasets`. The runner skips
finished results, so overlapping runs only waste time; they don't corrupt anything.

Results go to `<results>/<archive>/<Reference|Linear|Trees|Other|Foundation>/<model>/
Predictions/<dataset>/testResample<r>.csv`.

## Comparing

Use `_tsml_research_resources/slurm/head_to_head.py`, which is on this branch. It
gives pairwise W/D/L, the mean difference and a Wilcoxon p value, on accuracy and on
balanced accuracy, averaged over resamples:

```bash
python ../slurm/head_to_head.py Reference/CAWPE --results ~/Results/Stacking/UCR
python ../slurm/head_to_head.py Foundation/TabPFN-v2 --results ~/Results/Stacking/UCR
python ../slurm/head_to_head.py Reference/CAWPE-p0.5 --results ~/Results/Stacking/EEG
```

## What to look out for

- **The train features are out-of-bag or cross-validation estimates, but the test
  features come from models fitted on all the training data.** The test probabilities
  are therefore usually more confident. This shift is the classic stacking risk, and
  it penalises learned combiners that read the exact probability values.
  - TDE's train estimate is known to be optimistic.
  - Arsenal's estimates are often one-hot.
- **Small-n datasets are where stacking is most likely to overfit.** UCR has many
  datasets with fewer than 100 train cases, and 60 classes against 240 features.
  Foundation models are designed for small n, which is the reason to try them.
- **Prior correction gives HC2 +0.17 accuracy and +0.74 balanced accuracy (pp) on UCR.**
  Compare the stackers against both `CAWPE` and `CAWPE-p0.5`. A learned combiner
  absorbs the class prior on its own.
- **Time for more than 10 classes.** TabPFN's many-class wrapper fits several TabPFN
  sub-models, so those 20 UCR datasets take a large part of TabPFN's time.
  Laptop smoke test, on CPU and sharing it with other jobs:

  | Dataset | TabPFN-v2 | TabICL |
  |---|---|---|
  | Beef (5 classes, 30 train cases) | 16 s | 6 s |
  | Adiac (37 classes, 390 train cases) | 1,021 s | 41 s |

  On a GPU it should be far faster, but time Adiac and Crop first.
