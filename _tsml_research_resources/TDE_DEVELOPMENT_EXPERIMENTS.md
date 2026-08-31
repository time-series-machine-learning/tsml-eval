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
| `TDE_Dev` | Full WIP clone of Aeon TDE | Retain every channel | `dim_threshold=0`, `max_dims=n_channels` | Standard behavior (`False` for multivariate data) | Resample-0 Core run in progress; 48/66 results inspected on 2026-08-31 |
| `TDE_Dev2` | Aeon TDE subclass | Force bigrams on | Standard TDE filtering | `True` | Configured for resample-0 Core run |
| `TDE_Dev3` | Full WIP clone of Aeon TDE | Normalised per-dimension late fusion with accuracy weights | Standard TDE filtering | Standard behavior | Implemented and tested; synthetic diagnostics complete |

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

Launch from the `ajb/hc2` checkout:

```bash
bash _tsml_research_resources/start_multiverse_core_resample0_tde_dev.sh
```

### Preliminary comparison with TDE

Snapshot taken on 2026-08-31 using the 48 Core datasets with valid resample-0 files
for both classifiers:

| Measure | TDE | TDE_Dev |
|---|---:|---:|
| Mean accuracy | 0.7067 | 0.7121 |
| Mean balanced accuracy | 0.6681 | 0.6745 |
| Accuracy wins/draws/losses for TDE_Dev |  | 19/12/17 |
| Median fit time | 98.3 s | 98.6 s |
| Median memory | 0.516 GiB | 0.654 GiB |

The mean accuracy difference was +0.0054 for `TDE_Dev`, but was not significant
under a paired Wilcoxon test (`p=0.198`). Across 28,026 test cases, `TDE_Dev` made
42 additional correct predictions. Aggregate fitting time was 11% higher and
aggregate memory was 50% higher. These results are preliminary because 18
`TDE_Dev` Core results were still missing and the completed subset is biased toward
quicker datasets.

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

## Common HALI controller behavior

The currently configured development launchers (`TDE_Dev` and `TDE_Dev2`):

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
