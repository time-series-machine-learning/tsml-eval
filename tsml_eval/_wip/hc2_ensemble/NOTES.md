# Improving HC2 — running log

Started 2026-09-11. Project space, not the wiki: per `tsml-wiki/AGENTS.md`, agenda and
"what we tried" stay here. Only settled, attributable field facts graduate to the wiki.

## Three levels of intervention

1. **Ensemble scheme** — how the four component outputs are combined. Cheapest: needs no
   refitting, so every variant is built from stored component results files. **Current.**
2. **How existing components work** — internals of STC, DrCIF, Arsenal, TDE.
3. **Which components are used** — swap or add representation families.

## Infrastructure

`tsml_eval/estimators/classification/hybrid/hivecote_from_results.py` builds HC2 from
component results files with no data loaded and nothing refit. It drives the existing
`FromFileHIVECOTE`, taking case count and class labels from the files themselves.

Verified bit-exact: with components and full `HIVECOTEV2` built in one session
(SmoothSubspace r0/r1), the from-file build reproduces HC2 with max probability
difference 0.0 and identical predictions.

**Three HC2 builds, anchored to published reference results** (fetched via
`aeon.benchmarking.results_loaders.get_estimator_results`, 112 datasets, 30 resamples):

| build | accuracy | vs published | p |
|---|---|---|---|
| stored (`D:/Results/UCR/Hybrid/HC2`) | 89.060 | −0.081 pp | 0.147 (n.s.) |
| **published (timeseriesclassification.com)** | **89.142** | — | — |
| rebuilt from stored components | 89.301 | +0.159 pp | **0.0015** |

The stored HC2 is consistent with published; it is the **rebuild** that is the outlier,
significantly better than canonical HC2 because it inherits component runs postdating the
reference HC2 run (component files are dated later than the HC2 files). Our MrHydra
results match published to within 0.002 pp, so the discrepancy is HC2-specific.

Canonical published comparison, HC2 vs MR-Hydra: accuracy +0.742 pp (59/112, p=0.101),
BA +0.474 pp (55/112, p=0.782) — neither significant. The accuracy p closely reproduces
the p=0.1068 the wiki records from the Hydra paper, validating the pipeline.

Estimated effect of beta=0.5 transferred additively onto published HC2: accuracy +0.911
pp (p=0.010), BA +1.211 pp (p=0.0002). **An estimate, not a measurement** — published
results are aggregate accuracies with no probabilities to correct. A definitive external
comparison needs all four components plus the corrected ensemble rebuilt in one run.

**Caveat on stored results.** Against the stored `D:/Results/UCR/Hybrid/HC2`, the
rebuild differs (mean accuracy +0.27 pp, identical on 253/560). This is drift in the
stored component files, not the method: rerunning DrCIF/Arsenal/TDE today with the same
seed does not reproduce their stored files (only STC does). The stored HC2 cannot be
reproduced from the stored components by any method. All variant comparisons below are
therefore internally consistent (same stored components, same session) rather than
compared to the stored HC2 number.

## Component provenance — the rebuild is NOT HC2

Deep dive comparing each stored component's parameter line against exactly what
`HIVECOTEV2._fit` constructs:

| component | difference | substantive |
|---|---|---|
| STC | none (`verbose` absent from older dump) | **no — shapelets identical** |
| **DrCIF** | **`n_estimators` 200 (stored) vs 500 (HC2)** | **YES** |
| DrCIF | `use_pycatch22` False vs 'deprecated'; `time_limit` 0 vs None | no, aeon API drift |
| Arsenal | `time_limit_in_minutes` 0 vs 0.0 | no |
| TDE | `dim_threshold`→`channel_threshold`, `max_dims`→`max_channels`, same values | no, renames |

`n_estimators` on DrCIF is the **only** substantive parameter difference. HC2 sets
`_DEFAULT_N_TREES = 500` (`_hivecote_v2.py:128,195`); the standalone UCR run used the
class default 200 (`_drcif.py:173`), confirmed 200 on all 112 datasets.

Train estimates are comparable: all four stored train files record
`train_estimate_method="Custom"`, which at `experiments.py:240` means
`fit_predict_proba`, the same mechanism HC2 uses internally. Weights are derived
identically.

Two independent deviations from real HC2 therefore exist in the rebuild:

1. **Weaker DrCIF** — 200 vs 500 trees. Our DrCIF is −0.093 pp below published (p=0.035).
2. **Stronger Arsenal** — +0.345 pp above published (p=0.004) despite config-identical
   parameters, so this is aeon version drift, not configuration. The deprecation renames
   above confirm the stored results predate the installed aeon.

Arsenal is the component whose removal costs most (−1.10 pp BA) and whose relative
advantage drives the MrHydra gap, so (2) outweighs (1) and the rebuild ends up
significantly better than published HC2.

**DrCIF-500 does not exist for UCR.** Every DrCIF under `D:/Results/UCR` is 200 trees
(exhaustively checked). `Multiverse/IntervalBased/DrCIF-500` and
`OldMultiverse/DrCIF-500` are 500-tree but cover a different collection with **zero**
overlap with the UCR 112. Generating it costs ~353 core-hours for 30 resamples (~12 for
resample 0), extrapolated from DrCIF-200's measured 141 core-hours over 3360 runs.
`_tsml_research_resources/uea/hali/drcif500_ucr_experiments.sh` submits this on Hali.

## The target: HC2 vs MrHydra on balanced accuracy

112 UCR datasets, 30 resamples. Archive-wide HC2 leads MrHydra by 0.395 pp BA but wins
on only 53/112 (Wilcoxon p=0.948) — a tie. That tie is an artefact of aggregating two
real opposing effects:

| group | n | HC2 BA % | MrHydra BA % | diff | p |
|---|---|---|---|---|---|
| balanced (ratio=1) | 27 | 90.751 | 88.637 | **+2.115** | 0.058 |
| mild (1–2) | 52 | 89.977 | 89.824 | +0.153 | 0.964 |
| **imb ≥2:1** | 33 | 79.152 | 79.784 | **−0.632** | 0.066 |
| imb ≥5:1 | 11 | 70.021 | 70.510 | −0.489 | 0.365 |

HC2's advantage is a balanced-data phenomenon; on skewed data it is behind. The
BA-vs-accuracy shift against HC2 correlates with imbalance ratio (Spearman ρ=+0.236,
p=0.012).

HC2 still beats all four of its own components on BA (86.974 vs best component Arsenal
85.129), so the ensemble is doing its job.

## Level 1 experiments tried

All at resample 0 over 112 datasets unless noted, ΔBA/Δacc in percentage points vs HC2
built from the same stored components (BA 85.705%, acc 87.845%).

| variant | ΔBA | Δacc | ΔBA imb≥2 | verdict |
|---|---|---|---|---|
| equal weights | −0.38 | −0.37 | — | worse |
| **train-BA weights** (`mean recall^4`) | +0.04 | −0.01 | **+0.24** | promising, free, not yet significant |
| oracle **test-accuracy** weights | +0.05 | +0.07 | +0.00 | ceiling for better accuracy estimates |
| oracle **test-BA** weights | +0.23 | +0.14 | +0.49 | ceiling for balanced criteria |
| oracle test-acc weight, TDE only | +0.06 | +0.03 | — | negligible |
| TDE weight debias, offsets 0.02–0.10 | −0.03 to −0.09 | — | — | all negative |
| leave out STC | −0.40 | −0.38 | −0.18 | worse |
| leave out DrCIF | −0.30 | −0.33 | +0.09 | worse |
| leave out Arsenal | −1.10 | −0.87 | −1.94 | much worse |
| leave out TDE | −0.20 | −0.24 | −0.17 | worse |
| per-class **recall**^1/2/4 | −1.56/−2.52/−4.37 | −0.95/−1.64/−3.23 | −2.96/−4.58/−6.84 | fails badly |
| per-class **precision**^1/2/4 | −0.74/−0.85/−1.40 | −0.57/−0.70/−1.40 | −1.27/−1.42/−2.07 | fails |
| oracle per-class recall^4 | −4.02 | −2.92 | −6.36 | fails even with oracle |
| **prior correction** (÷ train prior) | **+0.95** | −0.44 | **+2.77** | large BA gain, costs accuracy |
| train-BA weights + prior correction | +0.94 | −0.41 | +2.68 | same as prior alone |
| oracle simplex reweighting (cheating) | +1.90 | — | +2.33 | selection-biased, see honest row |
| **honest split-half tuned weights** | **+0.33** | — | **+0.31** | **the real reweighting ceiling, n.s.** |

## Confirmed over 30 resamples

**Partial prior correction is the best level-1 result so far.** Divide the combined
posterior by `train_prior ** beta` before the argmax. 112 datasets, 30 resamples:

| beta | acc % | Δacc pp | p | BA % | ΔBA pp | p |
|---|---|---|---|---|---|---|
| 0 (HC2) | 89.301 | — | — | 87.244 | — | — |
| 0.25 | 89.429 | +0.129 | 0.0001 | 87.640 | +0.396 | <0.0001 |
| **0.50** | **89.469** | **+0.169** | **0.0017** | 87.982 | +0.738 | <0.0001 |
| 0.75 | 89.341 | +0.041 | 0.045 | 88.227 | +0.982 | <0.0001 |
| 1.00 | 88.991 | −0.310 | 0.181 | 88.380 | +1.135 | <0.0001 |

Contradicts the earlier expectation that beta=0 would be accuracy-optimal: HC2's
posterior *is* miscalibrated toward the majority class, and mild correction improves
**both** metrics at once. Only full correction (beta=1) trades accuracy for BA.

By group, accuracy gain at beta=0.5: balanced +0.000, mild 1–2 +0.258, imb≥2 +0.166,
imb≥5 −0.149. Accuracy-optimal beta shifts down as skew rises (beta=0.25 is best at
imb≥2), so a single global beta is a compromise. BA gain at beta=0.5: +1.871 pp at imb≥2,
+3.309 pp at imb≥5.

**vs MrHydra, with the drift confound separated.** The rebuilt HC2 baseline is +0.240 pp
accuracy above the stored HC2 purely from component drift, which is larger than the
correction itself, so the two effects must be separated:

| variant | acc vs MrHydra | p | BA vs MrHydra | p |
|---|---|---|---|---|
| stored HC2, beta=0 | +0.663 pp | 0.127 | +0.395 pp | 0.948 |
| rebuilt HC2, beta=0 | +0.903 pp | **0.0119** | +0.665 pp | 0.298 |
| rebuilt HC2, beta=0.25 | +1.032 pp | 0.0013 | +1.060 pp | 0.0006 |
| rebuilt HC2, beta=0.5 | +1.071 pp | 0.0005 | **+1.403 pp** | **<0.0001** |

**Balanced accuracy: the correction genuinely confers significance** (p=0.298 → <0.0001
on identical components, so unconfounded). **Accuracy: it does not** — the rebuilt
baseline was already significant at p=0.012 before any correction, and the stored
baseline is not significant at all (p=0.127). Do not claim the correction makes HC2
significantly more accurate than MrHydra; claim only the paired gain (+0.169 pp,
p=0.0017).

**The comparison is like-for-like.** Prior correction applied to MrHydra changes nothing
at all: +0.000 pp on both metrics at every beta, because 100.0% of its probability
vectors are one-hot and dividing a one-hot vector by a prior cannot move the argmax. So
this is a structural advantage — HC2's CAWPE sum yields soft, recalibratable posteriors,
MrHydra's ridge head yields hard decisions. Caveat: the one-hot output is a property of
MrHydra's implementation (ridge classifier, no probability estimates), not proof that a
probabilistic convolution pipeline could not also be calibrated.

**Tuning beta is worse than a constant.** Grid 0..1 step 0.125, selected on the train
posterior only (realisable), 30 resamples:

| variant | acc % | Δacc pp | p | BA % | ΔBA pp | p |
|---|---|---|---|---|---|---|
| HC2 | 89.301 | — | — | 87.244 | — | — |
| **fixed beta=0.5** | **89.469** | **+0.169** | 0.0017 | 87.982 | +0.738 | <0.0001 |
| tuned on train acc | 89.423 | +0.122 | 0.0021 | 87.664 | +0.419 | <0.0001 |
| tuned on train BA | 89.155 | −0.146 | 0.122 | **88.166** | **+0.921** | <0.0001 |
| oracle beta (test) | 89.831 | +0.530 | <0.0001 | 88.148 | +0.903 | <0.0001 |

Paired, tuned vs fixed: −0.046 pp accuracy, tuned better on only 25/112 (p=0.059).
Cause: train tuning picks beta mean 0.167, **median 0.019** — it mostly declines to
correct — against an oracle mean of 0.223. The train posterior uses the components' own
OOB/CV estimates evaluated against the class distribution they were fitted on, so the
uncorrected posterior looks better on train than on test.

**Recommendation: ship fixed beta=0.5** when accuracy is the target. Tune on train BA
only if BA is the target. Real headroom remains (oracle +0.530 pp accuracy, 3x the
constant) but needs a better selection signal, not more tuning.

Fixed beta=0.5 over-corrects at extreme skew (imb>=5 accuracy −0.149 pp, where tuning
gives +0.111). A skew-capped rule is plausible but choosing the threshold on these 112
datasets would be archive-level overfitting — validate elsewhere first.

**Train-BA weighting: significant but negligible.** 30 resamples: ΔBA +0.031 pp
(p=0.012), Δacc −0.002 pp (p=0.107). Identical by construction on balanced data, changes
a prediction on only 25/112 datasets. The apparent harm in the 1–2:1 band seen at
resample 0 did **not** replicate and was noise.

## Established findings

**The weights have almost no dynamic range.** At α=4 over train accuracies in a narrow
band, CAWPE assigns weight shares of 22.6–28.9% against an equal-weight 25%. This is the
structural reason every weight intervention is capped: a perfect test-accuracy oracle
buys only +0.05 pp BA.

**TDE's train accuracy estimate is systematically optimistic.**

| component | train acc | test acc | optimism | positive on |
|---|---|---|---|---|
| DrCIF | 0.8412 | 0.8499 | −0.0087 | 43/112 |
| Arsenal | 0.8568 | 0.8592 | −0.0024 | 48/112 |
| STC | 0.8690 | 0.8475 | +0.0215 | 65/112 |
| TDE | 0.8971 | 0.8413 | **+0.0558** | **81/112** |

Cause is visible in `aeon/classification/dictionary_based/_tde.py:437-466`: TDE draws 250
parameter samples, keeps the 50 with highest train accuracy, weights each by that same
`accuracy^4`, then estimates its own train accuracy from those retained members —
selection on the statistic being estimated, a winner's curse. Consequently TDE takes the
largest CAWPE weight share (28.9%) while being third of four on test accuracy.

This is a real correctness defect but **not** a lever for BA: correcting the weight is
worth at most +0.05 pp (the oracle), and every fixed debias offset tested made things
slightly worse.

**TDE has no probability estimates.** `_tde.py:607-611` accumulates member weights into
the winning class bin only — its "probabilities" are a hard weighted vote share. It has
the worst minority-class recall (0.6435) and largest majority/minority recall gap
(0.2549) of any classifier measured, and the largest acc−BA degradation (2.444 pp).

**Reweighting is bounded at ~+0.33 pp and is not significant.** Tuning weights on a
stratified half of the *test set* (286-point simplex grid) and scoring on the other half
gains +0.33 pp BA archive-wide and +0.31 pp on imb≥2, winning on 51/112 (p=0.211). The
cheating single-split version gains +1.90 pp, so 83% of that is selection noise. This is
the ceiling on any weight-selection scheme at level 1, given far better tuning data than
is actually available.

**Prior correction beats that ceiling by 9x on imbalanced data.** Free prior correction
gains +2.77 pp on imb≥2 against the honest reweighting ceiling of +0.31 pp (and against
the *cheating* ceiling of +2.33 pp). No weighting scheme — internal, external or oracle
— reaches what changing the decision rule gives. Prior correction scales with skew:
−0.07 pp on balanced data, +2.77 pp at ≥2:1, +4.13 pp at ≥5:1.

## Why MrHydra beats HC2 — data reasons

The driver is **representation fit**, not any dataset property we started with. Grouping
the 112 datasets by which HC2 component is most accurate (30-resample means):

| best component | n | mean acc gap (MrHydra − HC2) | MrHydra wins |
|---|---|---|---|
| Arsenal (convolution) | 51 | +0.58 pp | 30/51 |
| DrCIF (interval) | 27 | −0.93 pp | 8/27 |
| TDE (dictionary) | 23 | −1.78 pp | 6/23 |
| STC (shapelet) | 11 | −3.46 pp | 1/11 |

`arsenal_adv` = Arsenal accuracy minus the best of the other three. Spearman with the
accuracy gap ρ=+0.557 (p<1e-9); monotone across its quartiles (−3.69, −0.02, +0.37,
+0.69 pp). OLS `gap ~ arsenal_adv + log n_train + log length + log imbalance` gives
R²=0.695 with arsenal_adv t=14.4 and **every other predictor non-significant**, imbalance
included (t=−0.24, p=0.82).

So MrHydra wins essentially only where convolution is the right representation; where a
non-convolution representation wins, HC2 wins comfortably because MrHydra has no access
to it. This subsumes the imbalance story above: imbalance predicts the gap only until
representation fit is controlled for.

Signal characteristics measured from the train series (`experiments/signal_chars.py`)
separate the groups: datasets where Arsenal wins are smooth and low-frequency
(hf_energy 0.065) with high class separability (sep_ratio 1.20); where STC or DrCIF win,
high-frequency energy is 2.5–3x higher (0.181, 0.155) and separability lower. TDE wins on
the smoothest data of all (hf_energy 0.019) but with small training sets (mean n=168).

Caveat: `order_gain` in that script is broken (identically zero) — it shuffles every
series with the same permutation, which preserves class-mean distances. Fix with a
per-series permutation before using it.

## Negative results worth not repeating

- **Per-class weighting by class recall fails, and fails with an oracle too** (−4.02 pp).
  Not an estimation problem, the scheme is wrong: minority classes have low recall for
  every component, so recall weighting shrinks probability mass on exactly the classes BA
  rewards. The per-class degree of freedom that pays is shared across components (the
  prior), not component-specific.
- **Equal weighting is worse than CAWPE** (−0.38 pp), so the weights are not useless,
  just low-dynamic-range.
- **No component is expendable.** Every leave-one-out is worse on BA, including TDE.
  An earlier result suggesting removing TDE gained +1.18 pp was measured on 15 datasets
  selected for the largest MrHydra−HC2 BA gap — selection bias. Archive-wide the sign
  flips.

## Caveats on interpretation

- **Prior correction is not HC2-specific.** It is a post-hoc decision rule applicable to
  any classifier, MrHydra included. "HC2_prior beats MrHydra on imbalanced BA" is not
  like-for-like; the corrected-vs-corrected comparison has not been run.
- **If accuracy is the target metric, prior correction is the wrong move**: −0.44 pp
  archive-wide, −1.42 pp at ≥2:1, −3.77 pp at ≥5:1. It substitutes a different loss.
- The oracle simplex ceiling (+1.90 pp) picks the best of 286 weight vectors using test
  labels and is inflated by selection. A split-half honest version is in progress.

## Shipped

`build_hivecote_from_results(..., prior_correction=beta)` implements the decision-rule
change. `beta=0` is the standard CAWPE rule (default, no behaviour change); `beta=0.5` is
the recommended setting. Covered by `test_prior_correction`.

## Open / next

Resolved: train-BA weighting (significant, negligible), partial prior correction
(improves both metrics, beta=0.5), corrected-vs-corrected MrHydra (correction has zero
effect on it), beta tuning (worse than a constant).

- **Validate beta=0.5 outside these 112 datasets** (UEA multivariate, or the UCR112
  extension). Everything above was measured on the same archive used to pick beta, so
  the +0.17 pp accuracy figure is mildly optimistic. This is the main outstanding risk to
  the headline result.
- Better selection signal for beta: the oracle gets +0.530 pp accuracy vs +0.169 for the
  constant, so 3x headroom exists, but train-posterior selection captures under a quarter
  of it and loses to a constant. Needs a signal other than the train posterior.
- Why is the combined posterior majority-biased at all? Plausibly because three of four
  components produce near-degenerate estimates (Arsenal 80% one-hot, TDE a hard vote
  tally), so the CAWPE sum is sharper than a calibrated posterior. Calibrating the
  components before combining is a level-2 route that may subsume the beta hack.
- Fix `order_gain` in `experiments/signal_chars.py` (per-series permutation) before using
  the order-sensitivity feature.
- Level 2 candidate arising from level 1: TDE's optimism is a genuine defect. A CV
  estimate after transform would fix the estimate, but the oracle bounds its BA value at
  +0.05 pp — motivate it as estimate correctness, not as a BA fix.

## Write-up

`paper/prior_correction.tex` — the prior-correction method and results as a paper
section, ready to paste into Overleaf. `paper/refs.bib` has the four citations;
`paper/main.tex` is a minimal wrapper used only to check it compiles standalone
(verified: 5 pages, no warnings, no undefined citations). The section needs
`amsmath, amssymb, booktabs, graphicx` and a `proposition` theorem environment.

## Files

- `experiments/` — analysis scripts, each self-contained and reading only stored results.
- `results/` — collected CSVs, so the tables above can be regenerated without re-reading
  the archive.
