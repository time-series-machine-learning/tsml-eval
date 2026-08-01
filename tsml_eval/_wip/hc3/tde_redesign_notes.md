# TDE redesign notes for HC3

## Status and scope

These are working notes for the wider project to improve and accelerate
HIVE-COTE towards HC3. They are not a proposed contribution to the current EEG
channel-selection paper. That paper can mention scalable multivariate TDE as
future work, but changing TDE's representation search and classifier deserves a
separate evaluation.

The immediate motivation is an empirical one. TDE can have a similar runtime to
Arsenal on univariate problems, but it becomes much slower on large
multivariate collections. Preliminary OpenCloseFist LOSO runs make the
difference visible:

- 4,680 training cases per fold;
- 64 channels;
- 640 time points;
- mean Arsenal fit and estimate time of about 1.85 hours per completed fold;
- mean TDE fit and estimate time of about 4.32 hours per completed fold.

These figures are provisional and should be regenerated from the final result
files before being cited.

## What current multivariate TDE does

The current aeon implementation searches up to 250 parameter configurations and
retains up to 50 ensemble members. Each candidate `IndividualTDE` is fitted to a
70% training subsample.

For univariate data, a candidate:

1. fits one SFA transformation;
2. constructs a word-count histogram for every case;
3. estimates candidate accuracy using leave-one-out histogram-intersection
   nearest-neighbour classification.

For multivariate data, a candidate additionally:

1. fits a reduced SFA representation independently to every input channel;
2. obtains a leave-one-out accuracy estimate independently for every channel;
3. retains channels sufficiently close to the best channel, subject to
   `max_channels`;
4. constructs full bags for the retained channels;
5. merges the channel bags, encoding channel identity in the words;
6. performs another leave-one-out nearest-neighbour assessment on the combined
   representation.

This channel assessment is repeated for every candidate representation. With 64
channels and 250 candidates, the search can require up to 16,000 channel-level
SFA fits and channel accuracy assessments, before considering the combined
candidate assessments.

`max_channels` controls the size of the final representation, but it does not
avoid the initial assessment of every channel. Reducing it therefore cannot
remove the main channel-screening cost.

The relevant implementation is currently in:

- `aeon/classification/dictionary_based/_tde.py`;
- `IndividualTDE._select_channels`;
- `TemporalDictionaryEnsemble._individual_train_acc`;
- `aeon/classification/dictionary_based/_tde_sfa.py`.

## Why the scaling is unfavourable

There are three interacting costs.

### Repeated channel representation learning

Channel relevance is relearned for every window length, word length,
normalisation, pyramid and binning configuration. Some dependence between the
representation and useful channels is desirable, but doing a full supervised
search for every combination is expensive and probably redundant.

### Nearest-neighbour model selection

Histogram-intersection nearest neighbour has no conventional model-fitting
cost, but leave-one-out assessment compares many pairs of training cases. Its
cost grows approximately quadratically in the number of cases and also depends
on bag size.

The implementation has an optimised symmetric LOOCV route for at most 4,096
cases. TDE normally evaluates candidates on a 70% subsample, so the
OpenCloseFist LOSO candidate size is approximately 3,276 and remains on this
optimised route. The poor runtime on this problem is therefore not simply a
4,096-case implementation cliff.

### Larger combined bags

Retaining multiple channels increases the number of symbolic words, the cost of
merging bags, and the cost of each histogram comparison. Sparse representations
reduce storage, but irregular sparse matching is less hardware-friendly than
the regular numerical operations used by convolution-based classifiers.

Arsenal also has to process all channels, but it does not repeat supervised
per-channel LOOCV for hundreds of representation candidates. Its multivariate
scaling is consequently more favourable.

## First work package: profile before redesigning

The first TDE-specific HC3 task should be instrumentation rather than a new
classifier. Record the following separately for every candidate:

- parameter configuration;
- subsample size;
- number of input and retained channels;
- per-channel SFA fitting time;
- per-channel LOOCV time;
- selected-channel full transform time;
- channel-bag merge time;
- combined LOOCV time;
- bag vocabulary size and total non-zero counts;
- whether the candidate was retained;
- peak process memory if it can be measured reliably.

Aggregate these into:

- time by phase;
- time by parameter;
- time by cases, channels and time points;
- correlation between the channel rankings produced by different candidates;
- frequency with which each channel is retained;
- accuracy contribution of late parameter candidates;
- marginal value of ensemble members beyond the first 10, 20 and 30.

An important empirical question is whether candidate-specific channel selection
actually produces materially different useful channel sets. If the rankings
are stable, learning them hundreds of times is unnecessary. If rankings depend
strongly on representation parameters, a small number of channel-selection
families may retain most of the benefit.

The baseline suite should include:

- representative univariate archive problems;
- low-, medium- and high-channel multivariate problems;
- problems dominated separately by cases, channels and series length;
- the larger EEG problems already used in the channel-selection study.

Accuracy, fit time, train-estimate time, prediction time and peak memory must all
be retained. A faster TDE that merely moves cost into prediction is not an
unqualified improvement.

## Candidate redesigns

The following designs should be evaluated incrementally. Combining all of them
at once would make it difficult to determine where any improvement came from.

### A. Learn a global channel screen once

Use a cheap supervised channel filter once at the start of TDE and run the
existing TDE search on the reduced collection.

Possible filters include DetachRocket, TSelect, a cheap SFA configuration, or a
small ensemble of several cheap SFA configurations. DetachRocket is especially
relevant because it has performed well in the EEG pipeline experiments, but the
choice must be retested across general multivariate archive data.

This is the lowest-risk change and provides a clear baseline. It preserves the
existing TDE representation and NN classifier while removing repeated work on
obviously unhelpful channels.

Questions:

- How many channels should be retained?
- Is a fixed fraction sufficient, or should retention be learned?
- Does global screening remove complementary channels that are useful only for
  particular SFA configurations?
- Should the selected set be shared by every ensemble member or supplemented
  with member-specific exploration?

### B. Cache or share channel representations

Several candidate configurations share work, particularly when only pyramid
levels or related parameters differ. Investigate whether Fourier coefficients,
breakpoints, words or channel accuracy information can be reused safely across
candidates.

This route could accelerate TDE without altering its predictions, making it
particularly attractive for an implementation-focused HC3 improvement.

### C. Two-stage candidate evaluation

Use a cheap proxy to reject poor parameter configurations, then run the full
combined LOOCV only on promising candidates.

Possible proxies:

- a smaller case subsample;
- a small fixed validation split;
- fewer channels;
- shorter or truncated series;
- a lightweight linear classifier on the sparse bags;
- approximate or blocked nearest-neighbour evaluation.

The proxy must be assessed by its ability to rank candidates, not only by its
own classification accuracy.

### C1. Alternative parameter search, including Nelder-Mead

The present TDE search first evaluates 50 randomly selected configurations and
then uses kernel ridge predictions over previous configurations to choose the
remaining candidates. Alternative searches should be investigated independently
of changes to the representation and classifier.

Nelder-Mead is feasible as an experimental search strategy, and aeon already
contains implementations for forecasting. The forecasting utility cannot be
used directly, however. It assumes continuous, unconstrained parameters and a
fixed set of Numba-dispatched forecasting losses. TDE has a bounded,
mixed-discrete search space:

- window size: ordered integer selected from a data-dependent grid;
- word length: one of 8, 10, 12, 14 and 16;
- normalisation: Boolean;
- pyramid levels: one of 1, 2 and 3;
- information-gain versus equi-depth binning: Boolean.

Alphabet size is currently fixed at four and is not a search dimension.

Simply rounding a five-dimensional continuous simplex to these values is likely
to produce large plateaus and repeated configurations. Small simplex movements
could decode to the same TDE model, while treating the Boolean settings as
continuous would give the optimiser a false notion of distance.

A practical mixed-discrete prototype should:

1. run separate searches for the four
   `(normalisation, binning_method)` combinations;
2. optimise three normalised ordinal coordinates for window size, word length
   and pyramid level;
3. snap each proposed point to the nearest legal configuration;
4. cache every evaluated configuration and never refit a duplicate;
5. use an initial simplex whose vertices decode to distinct configurations;
6. use a fixed stratified subsample or validation scheme so the objective is
   deterministic across candidates;
7. stop by a budget of unique model evaluations rather than only by continuous
   simplex tolerance;
8. restart from diverse unevaluated configurations when a simplex collapses
   onto an already explored discrete region.

The fixed evaluation sample is important. Current TDE draws a new 70% subsample
for every candidate. That is acceptable for ensemble diversity, but the
resulting objective noise can mislead a local optimiser. Search quality and
ensemble diversity should be separated: optimise on common data, then refit
selected ensemble members on deterministic or independently drawn subsamples.

Nelder-Mead is a local search, so one run should not be expected to cover the
whole TDE space. A reasonable first design is four categorical regimes with
several short restarts, all sharing one global budget of 250 unique candidate
evaluations. It should be compared fairly with:

- the current random plus kernel-ridge strategy;
- uniform random search;
- a space-filling initial design such as Sobol sampling;
- a discrete coordinate or pattern search;
- optionally TPE or another mixed-parameter optimiser.

All methods must receive the same number of unique expensive evaluations and
the same evaluation data. Report best candidate accuracy as a function of
evaluation count, final TDE ensemble accuracy, runtime and diversity. A search
that finds one strong candidate quickly may not necessarily produce the best
ensemble of complementary candidates.

Nelder-Mead is particularly worth trying after the objective has been made
cheaper, for example by global channel screening or proxy evaluation. On the
current multivariate objective, an unsuccessful 250-evaluation search remains
very expensive regardless of how intelligently the next configuration is
chosen.

### D. Replace leave-one-out nearest neighbour

Represent each case as a sparse vector whose feature keys include:

- SFA word;
- channel identity;
- spatial-pyramid level;
- optionally the representation/candidate identity.

Then compare learned classifiers:

#### Ridge or logistic regression

These are natural first baselines for high-dimensional sparse word-count data.
They should fit and predict much more efficiently than quadratic LOOCV and can
combine evidence across words. Existing symbolic classifiers such as the WEASEL
family mean that simply attaching a linear classifier is not itself a
sufficiently novel contribution; the novelty would have to lie in scalable
multivariate representation and channel learning.

#### ExtraTrees

ExtraTrees could model nonlinear word and channel interactions and expose
feature or channel importances. It is appealing if different combinations of
patterns discriminate different classes or subjects.

Potential disadvantages are:

- sparse symbolic vocabularies can be very large;
- fitting a forest for every one of 250 candidates may replace one bottleneck
  with another;
- an ensemble containing many candidate-specific forests could be too large;
- dense conversion must be avoided;
- probability calibration and HC weighting would need checking.

A more promising ExtraTrees design may be:

1. cheaply score many SFA candidates;
2. retain a small number of representations;
3. concatenate their sparse feature matrices;
4. fit one final ExtraTrees model.

This should be compared with one final linear model on the same concatenated
representation.

#### Keep histogram-intersection NN as the reference

The existing classifier is essential as an ablation baseline. It may remain
competitive on small data, and its inductive bias may contribute diversity to
HIVE-COTE even if another classifier has better average standalone accuracy.

### E. Change the ensemble structure

Current TDE retains many complete `IndividualTDE` models. An alternative is a
shared symbolic feature bank followed by one classifier, or a much smaller
ensemble of classifiers over complementary feature groups.

This may reduce both runtime and model size but is the most substantial departure
from TDE. It should follow, rather than precede, the profiling and simpler
ablations.

## Recommended experimental sequence

1. Reproduce current TDE and Arsenal timing on selected univariate and
   multivariate problems.
2. Add phase-level TDE profiling without changing predictions.
3. Measure the stability of channel rankings across candidates.
4. Add global channel screening while retaining histogram-intersection NN.
5. Test candidate budgets of 25, 50, 100 and 250.
6. Compare LOOCV with a fixed validation split and cheaper candidate proxies.
7. Export candidate bags as sparse matrices.
8. Compare NN, ridge/logistic regression and a small ExtraTrees model on the
   same fixed representations.
9. Compare candidate-specific classifiers with one classifier over a
   concatenated representation.
10. Put the strongest scalable TDE variant back into HIVE-COTE and measure both
    standalone accuracy and ensemble contribution.

Each stage should be an ablation. The principal comparisons are:

- accuracy versus runtime;
- accuracy versus peak memory;
- model size and prediction time;
- contribution to HC accuracy and diversity;
- univariate versus multivariate behaviour.

## Relationship to HC3

The goal is not simply to make standalone TDE faster. An HC3 component should
provide useful predictions under a shared computational budget.

Possible outcomes include:

- a faster drop-in TDE with identical or near-identical predictions;
- a scalable multivariate TDE that uses shared channel selection;
- a learned symbolic classifier that replaces NN;
- a smaller symbolic component that is less accurate alone but contributes
  equivalent diversity to HC;
- removal of TDE from some data regimes through component gating.

Component gating is worth treating explicitly. Dataset characteristics could
determine whether HC3 uses full TDE, a reduced TDE, or no TDE. Relevant
characteristics include case count, channel count, length, estimated TDE cost,
and the diversity of quick component predictions. Dropping TDE everywhere
should not be the default conclusion merely because its current multivariate
implementation is slow.

## Immediate hypotheses

1. Most multivariate TDE time is spent in repeated channel-level representation
   fitting and LOOCV, rather than in the final retained ensemble.
2. Channel rankings are sufficiently correlated across candidates that a
   shared or grouped channel screen will retain most accuracy.
3. External channel reduction will help TDE more than Arsenal because it avoids
   TDE's repeated internal channel assessment.
4. A sparse linear classifier will provide the clearest initial speed baseline
   against NN.
5. ExtraTrees may improve nonlinear discrimination, but one final forest over
   selected representations will be more practical than one forest per
   candidate.
6. TDE may still be valuable to HC even when it is not the strongest standalone
   classifier, so ensemble diversity must be measured before it is removed.

## Open decisions

- Should the first implementation remain prediction-compatible with current
  TDE, or prioritise a new scalable classifier?
- Should channel screening live inside TDE or be a reusable HC3 preprocessing
  module?
- Is the target a fixed-resource classifier, a contractable classifier, or
  both?
- Should representation search optimise validation accuracy, HC ensemble
  contribution, or a joint accuracy-cost objective?
- Can a shared symbolic feature bank serve both classification and train
  probability estimation without leakage?
