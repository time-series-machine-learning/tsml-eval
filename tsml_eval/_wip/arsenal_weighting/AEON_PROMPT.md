# Prompt for the aeon repo: fix Arsenal's binary member weighting

Copy everything below into a session on the `ajb/arsenal` branch of aeon.

---

In `aeon/classification/convolution_based/_arsenal.py`, `_fit_ridge_classifier` fits each
ensemble member with

```python
RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight=class_weight,
                  scoring="accuracy")
```

and the caller reads `ridge.best_score_` as that member's accuracy, which becomes its
CAWPE weight in `_fit_ensemble_estimator` and `_train_probas_for_estimator`.

**For two-class problems `best_score_` is exactly 1.0 whatever the data.** With
`cv=None` the efficient generalised cross-validation path reconstructs predicted labels
with `argmax(axis=1)` over the single column that `LabelBinarizer` produces for two
classes, which always returns the first class. This is an open scikit-learn defect,
https://github.com/scikit-learn/scikit-learn/issues/34942, reproduced on 1.7.2:

```python
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
rng = np.random.RandomState(0)
X = rng.normal(size=(200, 50)); y = (X[:, 0] + 0.6 * rng.normal(size=200) > 0).astype(int)
AL = np.logspace(-3, 3, 10)
print(RidgeClassifierCV(alphas=AL, scoring="accuracy").fit(X, y).best_score_)          # 1.0
print(RidgeClassifierCV(alphas=AL, scoring="accuracy", cv=5).fit(X, y).best_score_)    # 0.78
```

There are two consequences, and the second is the more serious.

1. Every member of a binary Arsenal receives weight `1 ** alpha == 1`, so the ensemble is
   unweighted. Verified: on ItalyPowerDemand every `weights_` entry is exactly 1.0, and
   an Arsenal that ignores its weights entirely produces byte-identical probabilities on
   100% of runs across the 40 binary UCR datasets and 30 resamples.
2. The same score selects the ridge penalty. Since all ten alphas tie at 1.0, `argmax`
   takes the first, so **every binary member is fitted at `alpha=1e-3`**, essentially
   unregularised, chosen by no criterion. Verified: member alphas on a binary problem are
   `[0.001] * n_estimators`, against a mix of 46.4 and 0.001 once corrected. This is a
   regression that this branch introduced: with the previous default scorer the penalty
   was selected by leave-one-out squared error, which is at least a legitimate criterion.

36% of the UCR archive is binary.

## What to change

Route **binary problems only** through a corrected estimate. Multiclass must stay on the
existing path and remain bit-identical, because scikit-learn's reconstruction is already
correct when `LabelBinarizer` produces more than one column.

Do not use explicit `cv=`, which costs `n_splits * n_alphas` ridge fits. The leave-one-out
predictions already exist in closed form: with a scorer set, `cv_results_` holds per-point
predictions, and only the reconstruction of labels from them is wrong. Redo the
reconstruction, which costs one extra ridge fit.

Reference implementation, verified against explicit 5-fold cross-validation:

```python
def _binary_loo_accuracies(ridge, y):
    """Per-alpha leave-one-out accuracy from the stored predictions.

    The stored column for two classes is a signed decision value, so the predicted class
    is its sign; scikit-learn takes argmax over that single column and always returns the
    first class.
    """
    cv_results = getattr(ridge, "cv_results_", None)
    if cv_results is None or cv_results.ndim != 3 or cv_results.shape[1] != 1:
        return None
    positive = y == ridge.classes_[1]
    decisions = cv_results[:, 0, :]
    return np.array(
        [(np.sign(decisions[:, i]) > 0) == positive for i in range(decisions.shape[1])]
    ).mean(axis=1)


def _fit_ridge_binary_loo(X, y, class_weight, alphas):
    search = RidgeClassifierCV(
        alphas=alphas, class_weight=class_weight,
        scoring="accuracy", store_cv_results=True,
    ).fit(X, y)
    accuracies = _binary_loo_accuracies(search, y)
    if accuracies is None:
        return None
    best = int(np.argmax(accuracies))
    ridge = RidgeClassifier(alpha=alphas[best], class_weight=class_weight).fit(X, y)
    ridge.best_score_ = float(accuracies[best])
    ridge.alpha_ = alphas[best]
    return ridge
```

Fallbacks, in order: if the reconstruction returns None or raises `LinAlgError`, use the
existing explicit stratified cross-validation path; if no fold split is possible because a
class has fewer than two cases, keep the current behaviour rather than failing. Record
which path each member took on the fitted estimator so a run can be audited.

## Tests

- A binary problem must not give every member the same weight, and weights must lie in
  `(0, 1]`.
- A multiclass problem must be **bit-identical** to the current implementation for
  `weights_`, `predict_proba` and `fit_predict_proba`.
- The closed-form estimate must agree with explicit `cv=5` on a binary problem to within
  a few hundredths, since leave-one-out and five-fold estimate the same quantity.
- A guard that fails loudly if `cv_results_` stops holding per-point predictions when a
  scorer is set. That semantic is an implementation detail and the fix is silently wrong
  without it.
- `n_jobs` invariance and pickle round-trip must still hold.

Expected classifier results for Arsenal on binary problems will change, so regenerate the
stored expected values.

## What not to claim

This is a correctness fix, not an accuracy improvement.

- Standalone Arsenal on the 40 binary UCR datasets over 30 resamples: 91.311% to 91.813%,
  **+0.502 pp but p=0.22**, not significant.
- Multiclass: unchanged, exactly.
- Inside HC2: **−0.040 pp, p=0.41**. No measurable benefit at the ensemble level.
- Runtime: neutral, 1.47s against 1.37s on a binary problem, within noise. Explicit
  cross-validation would be roughly four times slower.

Note also that discarding Arsenal's member weights altogether costs 0.003 pp over the
whole archive, so a reviewer may reasonably ask whether the internal weighting earns its
place at all. Better to state that number up front than to be asked.

A working implementation with tests, targeting the same aeon structure, is in tsml-eval at
`tsml_eval/_wip/arsenal_weighting/`.
