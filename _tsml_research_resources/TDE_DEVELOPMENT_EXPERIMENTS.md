# TDE development experiments

This document records controlled TDE variants evaluated on resample 0 of the
66-problem Multiverse-Core archive. CPU experiments run from the `ajb/hc2` branch on
HALI and write to:

```text
/gpfs/home/ajb/Results/Multiverse/DictionaryBased/<classifier>/Predictions
```

Each development name has a separate result directory, controller state directory,
and screen session. Existing prediction files are never overwritten.

## Experiment register

| Name | Parent | Deliberate change | Channel filtering | Bigrams | Status |
|---|---|---|---|---|---|
| `TDE` | Aeon TDE | Baseline | `dim_threshold=0.85`, `max_dims=20` | `None`, resolving to `False` for multivariate data | Existing comparison results |
| `TDE_Dev` | Full WIP clone of Aeon TDE | Retain every channel | `dim_threshold=0`, `max_dims=n_channels` | Standard behavior (`False` for multivariate data) | Resample-0 Core run in progress; 59/66 results inspected on 2026-09-02 |
| `TDE_Dev2` | Aeon TDE subclass | Force bigrams on | Standard TDE filtering | `True` | Resample-0 Core run in progress; 57/66 results inspected on 2026-09-01 |
| `TDE_Dev3` | Full WIP clone of Aeon TDE | Normalised per-dimension late fusion with accuracy weights | Standard TDE filtering | Standard behavior | Resample-0 Core run in progress; 59/66 accuracy-weighted and 58/66 uniform results inspected on 2026-09-02 |

The variants are independent ablations: `TDE_Dev2` and `TDE_Dev3` do **not**
inherit the all-channel behavior of `TDE_Dev`.

## Baseline: TDE

The baseline is `aeon.classification.dictionary_based.TemporalDictionaryEnsemble`
with its standard parameters. Relevant multivariate defaults are:

```python
TemporalDictionaryEnsemble(
    bigrams=None,
    dim_threshold=0.85,
    max_dims=20,
)
```

For multivariate input, `bigrams=None` is resolved internally to `False`.

## Variant 1: TDE_Dev — all channels

Hypothesis: marginal channel filtering may discard channels that are weak in
isolation but useful when combined with other channels.

Equivalent configuration:

```python
TemporalDictionaryEnsemble(
    dim_threshold=0,
    max_dims=n_channels,
)
```

Because `n_channels` is unknown until fitting, the implementation exposes
`max_dims=None` and resolves it to the observed number of channels during fit.
`dim_threshold=0` makes every channel pass the marginal-accuracy threshold.

Implementation:

- `tsml_eval/_wip/tde_dev/_tde_dev.py`
- Classifier aliases: `TDE_Dev`, `tde-dev`, `tdedev`
- Initial source: Aeon `_tde.py` at commit
  `ed21ac50acc9c80c5ff2827a374a81a0d69debbc`

HALI files:

- Configuration: `multiverse_core_resample0_tde_dev.toml`
- Launcher: `start_multiverse_core_resample0_tde_dev.sh`
- Screen: `multiverse-core-tde-dev`
- Controller state: `.controller-core-resample0-tde-dev`
- Results: `DictionaryBased/TDE_Dev`

The controller requires both test and train result files. If a test file already
exists, tsml-eval reruns the fit to produce only the missing train estimate and does
not overwrite the existing test file. TDE's own leave-one-out train-estimate support
is used rather than an external cross-validation wrapper.

Launch from the `ajb/hc2` checkout:

```bash
bash _tsml_research_resources/start_multiverse_core_resample0_tde_dev.sh
```

### Preliminary comparison with TDE

Snapshot taken on 2026-08-31 using the 50 Core datasets with valid resample-0 files
for both classifiers:

| Measure | TDE | TDE_Dev |
|---|---:|---:|
| Mean accuracy | 0.7085 | 0.7140 |
| Mean balanced accuracy | 0.6711 | 0.6777 |
| Accuracy wins/draws/losses for TDE_Dev |  | 20/13/17 |
| Median fit time | 103.7 s | 117.0 s |
| Median prediction time | 6.0 s | 20.2 s |
| Median memory | 0.564 GiB | 0.854 GiB |

The mean accuracy difference was +0.0054 for `TDE_Dev`, but was not significant
under a paired Wilcoxon test (`p=0.158`). Across 28,852 test cases, `TDE_Dev` made
50 additional correct predictions. Aggregate fitting time was 46% higher, median
prediction time was 3.4 times higher, and median memory was 51% higher. These results
are preliminary because 16 `TDE_Dev` Core results were still missing and the
completed subset is biased toward quicker datasets.

Mean log loss was worse for `TDE_Dev` because of several large outliers, notably
IRDS-SFL, DuckDuckGeese, Handwriting, and Alzheimers. Median log-loss change was
slightly favorable, so this was not a uniform degradation.

## Variant 2: TDE_Dev2 — multivariate bigrams

Hypothesis: cross-window transition information from bigrams may improve TDE on
multivariate data, where standard TDE disables bigrams.

Equivalent configuration:

```python
TemporalDictionaryEnsemble(bigrams=True)
```

All other settings remain standard, including `dim_threshold=0.85` and
`max_dims=20`.

Implementation:

- `tsml_eval/_wip/tde_dev/_tde_dev2.py`
- Classifier aliases: `TDE_Dev2`, `tde-dev2`, `tdedev2`

HALI files:

- Configuration: `multiverse_core_resample0_tde_dev2.toml`
- Launcher: `start_multiverse_core_resample0_tde_dev2.sh`
- Screen: `multiverse-core-tde-dev2`
- Controller state: `.controller-core-resample0-tde-dev2`
- Results: `DictionaryBased/TDE_Dev2`

Launch from the `ajb/hc2` checkout:

```bash
bash _tsml_research_resources/start_multiverse_core_resample0_tde_dev2.sh
```

### Preliminary comparison with TDE

Snapshot taken on 2026-08-31 using the 46 Core datasets with valid resample-0 files
for baseline TDE, `TDE_Dev`, and `TDE_Dev2`:

| Measure | TDE | TDE_Dev | TDE_Dev2 |
|---|---:|---:|---:|
| Mean accuracy | 0.7199 | 0.7262 | 0.7224 |
| Mean balanced accuracy | 0.6821 | 0.6896 | 0.6886 |
| Mean log loss | 0.8271 | 1.0167 | 0.8540 |
| Median fit time | 88.7 s | 89.3 s | 92.5 s |
| Median prediction time | 4.5 s | 16.4 s | 6.0 s |
| Median memory | 0.470 GiB | 0.573 GiB | 0.589 GiB |

Against TDE on this common subset, `TDE_Dev2` had accuracy wins/draws/losses of
17/12/17, a mean accuracy change of +0.0025, and a paired Wilcoxon p-value of 0.739.
It made 20 fewer correct predictions across all 18,327 test cases despite improving
the unweighted mean dataset accuracy, showing that its gains were concentrated in
smaller datasets. Aggregate fitting time was 24% higher than TDE.

`TDE_Dev` had the stronger signal on the same subset: +0.0063 mean accuracy,
20/12/14 wins/draws/losses, and `p=0.099`, with 89 additional correct predictions.
However, a few expensive high-dimensional datasets made its aggregate fit time 2.45
times baseline. `PEMS-SF`, `UCDHE-Rowing-MC`, and `MindReading` were the largest fit
time multipliers.

Neither result is yet statistically significant. Dev1 currently looks more promising
for accuracy but is substantially more expensive; Dev2 is cheaper than Dev1 and has
better log loss, but offers little evidence of an accuracy improvement over baseline.

## Variant 3: TDE_Dev3 - normalised per-dimension late fusion

### Current TDE behavior

For every selected channel, `IndividualTDE` creates an independent SFA bag. During
`combine_dim_bags`, the channel identifier is encoded in each word key before the
sorted bags are merged. Words from different channels therefore cannot match. A
single histogram intersection over the merged bag is exactly:

```text
sum over selected dimensions of raw histogram intersection in that dimension
```

This is late fusion by an unweighted sum. A channel with a larger shared count mass
can contribute more than another channel even if its relative match is weaker.

### Experimental formulation

`TDE_Dev3` adds two backwards-compatible controls to both the ensemble and
individual estimator:

```python
TDE_Dev3(
    multivariate_similarity="merged",
    dimension_weighting="uniform",
)
```

Those defaults preserve standard TDE. The experiment-factory name `TDE_Dev3`
selects normalised similarity with accuracy weighting. `TDE_Dev3-Uniform` selects
normalised similarity with uniform weights. These separate names allow baseline TDE,
normalisation alone, and normalisation plus weighting to be compared without changing
experiment code.

For selected dimension `d`, Dev3 uses:

```text
intersection_d(A, B) / min(mass_d(A), mass_d(B))
```

An empty bag contributes zero. The minimum-mass denominator is symmetric, bounded in
`[0, 1]` for non-negative counts, and answers whether the smaller bag is contained in
the larger. Dividing by maximum or union mass would additionally penalise unequal
bag sizes; dividing by mean mass has the same extra penalty in a softer form. Those
alternatives conflate removal of count scale with a separate bag-mass mismatch
penalty, so minimum mass is the simplest first experiment.

Uniform weights are `1 / n_selected_dimensions`. Accuracy weights retain the
training-only univariate leave-one-out scores already computed by `_select_dims` and
use:

```text
weight_d = accuracy_d / sum_selected_dimensions(accuracy)
```

If all selected scores are zero, the implementation falls back to uniform weights.
No test labels are used.

### Implementation

- `tsml_eval/_wip/tde_dev/_tde_dev3.py`
- `TDE_Dev3` and `IndividualTDE_Dev3`
- Classifier aliases: `TDE_Dev3`, `tde-dev3`, `tdedev3`
- Uniform aliases: `TDE_Dev3-Uniform`, `tde-dev3-uniform`,
  `tdedev3-uniform`
- Diagnostic: `_tsml_research_resources/tde_dev3_diagnostics.py`
- Focused tests: `tsml_eval/_wip/tde_dev/tests/test_tde_dev3.py`

HALI files:

- Accuracy-weighted configuration:
  `multiverse_core_resample0_tde_dev3.toml`
- Uniform-weight configuration:
  `multiverse_core_resample0_tde_dev3_uniform.toml`
- Launcher: `start_multiverse_core_resample0_tde_dev3.sh`
- Results: `DictionaryBased/TDE_Dev3` and
  `DictionaryBased/TDE_Dev3-Uniform`

After pulling `ajb/hc2` on HALI, launch the accuracy-weighted experiment with:

```bash
bash _tsml_research_resources/start_multiverse_core_resample0_tde_dev3.sh
```

Launch the uniform-weight ablation separately with:

```bash
bash _tsml_research_resources/start_multiverse_core_resample0_tde_dev3.sh --uniform
```

Both commands first print a dry-run report, then start a detached controller. Add
`--reset-state` only when previous terminal failures should deliberately be retried.

Normalised mode stores the already-produced dimension bags consecutively, plus a
two-dimensional case-offset array, dimension starts, and precomputed bag masses.
Prediction and leave-one-out searches are Numba kernels; they do not reconstruct
dictionaries or loop over dimensions in Python. SFA, dimension selection, ensemble
parameter search, and nearest-neighbour classification are unchanged.

### Small diagnostic results

The exact unequal-mass bag example had two equally weighted channels. Raw merged
intersection scored the informative match `3` and a large-mass match `100`, choosing
the latter. Per-dimension normalisation scored them `1.0` and `0.5`, choosing the
informative match.

The strong-plus-weak synthetic problem used one strong smooth channel, one moderate
channel, and three noisy channels, with every channel retained. With identical SFA
settings and seed 17:

| Fusion | Test accuracy | Warm fit | Warm prediction | Stored representation |
|---|---:|---:|---:|---:|
| Merged | 0.58 | 0.007 s | 0.010 s | 123,168 bytes |
| Normalised, uniform | 0.60 | 0.004 s | 0.010 s | 126,120 bytes |
| Normalised, accuracy | 0.57 | 0.004 s | 0.009 s | 126,120 bytes |

Mean bag masses by dimension were `[38.05, 46.925, 47.825, 44.325, 47.075]`.
Training-only dimension accuracies were `[0.75, 0.725, 0.625, 0.725, 0.525]`, giving
accuracy weights `[0.224, 0.216, 0.187, 0.216, 0.157]`.

The exact diagnostic establishes that the formulation removes pure bag-count
dominance. The end-to-end result is only one small synthetic split: uniform
normalisation improved by two test cases, while accuracy weighting lost one relative
to merged TDE. It supports a controlled archive benchmark of the uniform-normalised
ablation, but does not yet support treating accuracy weighting as an improvement.
Both variants should be benchmarked separately before any change to Aeon's defaults.

The normalised representation used 2.4% more memory in this diagnostic. Once Numba
was warmed, prediction time was effectively unchanged at this scale. Larger archive
problems are needed to measure the cost of per-dimension division and float64
similarity matrices.

## Latest incomplete Core comparison

Snapshot taken on 2026-09-02. Completion was 59/66 for `TDE_Dev`, 57/66 for
`TDE_Dev2`, 59/66 for accuracy-weighted `TDE_Dev3`, and 58/66 for
`TDE_Dev3-Uniform`. There were 57 datasets common to baseline TDE and all four
variants.

| Measure on 57 common datasets | TDE | TDE_Dev | TDE_Dev2 | TDE_Dev3 | Dev3 uniform |
|---|---:|---:|---:|---:|---:|
| Mean accuracy | 0.7204 | **0.7272** | 0.7217 | 0.7184 | 0.7235 |
| Mean balanced accuracy | 0.6711 | **0.6794** | 0.6761 | 0.6710 | 0.6753 |
| Mean log loss | **0.8848** | 1.0364 | 0.9191 | 0.9595 | 0.9440 |
| Median fit time | 158.4 s | 207.9 s | 139.3 s | 107.6 s | **107.4 s** |
| Median prediction time | 12.5 s | 25.8 s | 17.8 s | **10.3 s** | 14.3 s |
| Median memory | 0.850 GiB | 1.284 GiB | 1.045 GiB | **0.708 GiB** | 0.865 GiB |
| Aggregate fit-time ratio to TDE | 1.00 | 1.95 | 1.41 | **0.99** | 1.01 |

Paired accuracy outcomes against TDE were:

- `TDE_Dev`: 27/13/17 wins/draws/losses, mean difference +0.0067,
  Wilcoxon `p=0.0447`, and 414 additional correct predictions out of 55,548.
- `TDE_Dev2`: 19/12/26, mean difference +0.0013, `p=0.870`, and 67 fewer
  correct predictions.
- `TDE_Dev3`: 17/11/29, mean difference -0.0020, `p=0.164`, and 565 fewer
  correct predictions.
- `TDE_Dev3-Uniform`: 18/11/28, mean difference +0.0031, `p=0.589`, and 489
  fewer correct predictions.

On this incomplete subset, retaining all channels is the only variant showing a
meaningful positive accuracy signal, although it costs more and worsens log loss.
Its nominal Wilcoxon result is now below 0.05, but this is an incomplete, repeatedly
inspected subset and four variants are being compared, so it is not yet confirmatory.
Bigrams still provide no convincing accuracy improvement.

Uniform Dev3 outperformed accuracy-weighted Dev3 by +0.0051 mean accuracy,
25/13/19 wins/draws/losses, 76 additional correct predictions, and better log loss;
the paired difference was not significant (`p=0.119`). This suggests the dimension
accuracy weights are harmful. Normalisation alone is closer to baseline, but its
positive unweighted mean is driven by smaller datasets: it made 489 fewer total
correct predictions. Accuracy-weighted Dev3 remains the fastest and lightest option.

Missing results were:

- `TDE_Dev` and `TDE_Dev3`: `AustraliaRainfall_disc`, `CrowdSourced`,
  `FordChallenge`, `Skoda`, `STEW`, `Tiselac`, and `USCActivity`.
- `TDE_Dev2`: the same seven plus `BIDMC32HR_disc` and `BIDMC32SpO2_disc`.
- `TDE_Dev3-Uniform`: the same seven plus `BIDMC32HR_disc`.

## Common HALI controller behavior

The currently configured development launchers (`TDE_Dev`, `TDE_Dev2`, and
`TDE_Dev3`):

- run only resample 0 from `MultiverseCore.txt`;
- submit CPU jobs to the `compute` partition;
- start at 32 GB and escalate confirmed OOM failures to 64 GB and 128 GB;
- schedule smaller datasets first;
- enable TDE progress output with `verbose=1`;
- check hourly and email every four hours;
- stop once every configured result is complete; and
- leave unrelated CPU and GPU jobs and controllers running.

Use `--reset-state` only when failed attempts should deliberately be made eligible
again. It archives the previous controller state rather than deleting results:

```bash
bash _tsml_research_resources/start_multiverse_core_resample0_tde_dev2.sh --reset-state
```

## Recording subsequent variants

For each new variant, record:

1. the single intended change relative to its stated parent;
2. parameters deliberately held constant;
3. implementation, configuration, launcher, screen, state, and result names;
4. completion count and comparison date;
5. paired accuracy wins/draws/losses and a significance test;
6. fit time, prediction time, and memory changes; and
7. notable dataset-level improvements, regressions, and failures.

Avoid combining experimental changes unless the variant is explicitly intended to
test an interaction between mechanisms.
