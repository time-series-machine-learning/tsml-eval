"""DrCIF variants for forecasting experiments: other series representations,
interval features and base trees, plus an R-STSF style "transform once" regressor.

DrCIF builds each tree on random intervals of three representations of the input
(the series, its first order differences and its periodogram), summarised by
catch22 plus seven summary statistics. ``DrCIFVariantRegressor`` exposes those
choices as parameters so alternatives can be compared:

representations (any subset, in order)
    ``"raw"``    the series
    ``"d1"``     first order differences
    ``"d2"``     second order differences (QUANT uses these)
    ``"pgram"``  periodogram
    ``"res"``    one-step-ahead residuals of simple exponential smoothing
                 (alpha=0.3) run over each series, an exponential smoothing
                 residual representation
    ``"ar"``     autoregressive coefficients (R-STSF uses these)
features
    ``"drcif"``     catch22 + mean, std, slope, median, IQR, min, max (DrCIF)
    ``"stats"``     the seven summary statistics only (no catch22, much faster)
    ``"drcif+q"`` / ``"stats+q"``  the above plus the 10/25/75/90% quantiles
                 (quantiles are QUANT's features)
base
    ``"tree"``  CART regression tree per ensemble member (DrCIF)
    ``"extra"`` extremely randomised tree per member (QUANT and R-STSF use extra
                trees)

With the defaults (``representations=("raw", "d1", "pgram")``, ``features="drcif"``,
``base="tree"``) it is the same model as aeon's ``DrCIFRegressor``.

``RSTSFRegressor`` follows R-STSF instead: interval features are extracted once
(random intervals per representation, rather than new intervals for every tree as
in DrCIF) and a single ``ExtraTreesRegressor`` is fit on all of them. R-STSF picks
its intervals in a supervised way using a classification score (Fisher), which does
not apply to regression, so the intervals are random here.
"""

__all__ = ["DrCIFVariantRegressor", "RSTSFRegressor"]

import numpy as np
from aeon.base._estimators.interval_based import BaseIntervalForest
from aeon.regression import BaseRegressor
from aeon.transformations.collection import (
    ARCoefficientTransformer,
    PeriodogramTransformer,
)
from aeon.transformations.collection.feature_based._catch22 import Catch22
from aeon.transformations.collection.interval_based import RandomIntervals
from aeon.utils.numba.general import first_order_differences_3d
from aeon.utils.numba.stats import (
    row_count_above_mean,
    row_count_mean_crossing,
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_quantile,
    row_quantile25,
    row_quantile75,
    row_slope,
    row_std,
)
from numba import njit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor


@njit(cache=True, fastmath=True)
def second_order_differences_3d(X):
    """Second order differences along the last axis of a 3D array."""
    return X[:, :, 2:] - 2 * X[:, :, 1:-1] + X[:, :, :-2]


@njit(cache=True, fastmath=True)
def ses_residuals_3d(X):
    """One-step-ahead residuals of simple exponential smoothing (alpha=0.3).

    The level starts at the first value, so the first residual is 0.
    """
    alpha = 0.3
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            level = X[i, c, 0]
            for t in range(1, X.shape[2]):
                e = X[i, c, t] - level
                out[i, c, t] = e
                level += alpha * e
    return out


@njit(cache=True, fastmath=True)
def row_quantile10(X):
    """10% quantile of each row."""
    return row_quantile(X, 0.1)


@njit(cache=True, fastmath=True)
def row_quantile90(X):
    """90% quantile of each row."""
    return row_quantile(X, 0.9)


def _ar_order(X):
    # R-STSF's lag rule, capped so short series still have a few coefficients
    n = X.shape[2]
    return max(1, min(int(12 * (n / 100.0) ** 0.25), n // 2 - 1))


def _representation(name):
    if name == "raw":
        return None
    if name == "d1":
        return FunctionTransformer(func=first_order_differences_3d, validate=False)
    if name == "d2":
        return FunctionTransformer(func=second_order_differences_3d, validate=False)
    if name == "pgram":
        return PeriodogramTransformer()
    if name == "res":
        return FunctionTransformer(func=ses_residuals_3d, validate=False)
    if name == "ar":
        return ARCoefficientTransformer(order=_ar_order, replace_nan=True)
    raise ValueError(f"unknown representation {name!r}")


_STATS = [
    row_mean,
    row_std,
    row_slope,
    row_median,
    row_iqr,
    row_numba_min,
    row_numba_max,
]
_QUANTILES = [row_quantile10, row_quantile25, row_quantile75, row_quantile90]


def _features(name, catch22=True):
    feats = list(_STATS)
    base, _, q = name.partition("+")
    if base == "drcif" and catch22:
        feats = [Catch22(outlier_norm=True)] + feats
    elif base not in ("drcif", "stats"):
        raise ValueError(f"unknown features {name!r}")
    if q == "q":
        feats += _QUANTILES
    elif q:
        raise ValueError(f"unknown features {name!r}")
    return feats


class DrCIFVariantRegressor(BaseIntervalForest, BaseRegressor):
    """DrCIF with configurable representations, interval features and base tree.

    See the module docstring for the options. The remaining parameters are those
    of ``DrCIFRegressor``.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
    }

    def __init__(
        self,
        representations=("raw", "d1", "pgram"),
        features="drcif",
        base="tree",
        n_estimators=200,
        n_intervals=(4, "sqrt-div"),
        min_interval_length=3,
        max_interval_length=0.5,
        att_subsample_size=10,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.representations = representations
        self.features = features
        self.base = base
        if base == "tree":
            base_estimator = DecisionTreeRegressor(criterion="squared_error")
        elif base == "extra":
            base_estimator = ExtraTreeRegressor(criterion="squared_error")
        else:
            raise ValueError(f"unknown base {base!r}")
        reps = tuple(representations)
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=_features(features),
            series_transformers=[_representation(r) for r in reps],
            att_subsample_size=att_subsample_size,
            replace_nan=0,
            time_limit_in_minutes=None,
            contract_max_n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

    def _fit(self, X, y):
        # catch22 fails on very short series, so drop any representation whose
        # output is shorter than 3 points for this data (e.g. AR coefficients of
        # a 5 point window); the raw series is always kept.
        names, keep = [], []
        for r in self.representations:
            t = _representation(r)
            n = X.shape[2] if t is None else np.asarray(t.fit_transform(X[:1])).shape[2]
            if r == "raw" or n >= 3:
                names.append(r)
                keep.append(t)
        self.series_transformers = keep
        self.representations_used_ = names
        return super()._fit(X, y)

    def _predict(self, X):
        return super()._predict(X)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"n_estimators": 2, "n_intervals": 2, "att_subsample_size": 2}


class RSTSFRegressor(BaseRegressor):
    """R-STSF style regressor: interval features extracted once, one extra trees.

    Parameters
    ----------
    representations : tuple of str, default=("raw", "d1", "pgram", "ar")
        Representations as in ``DrCIFVariantRegressor`` (R-STSF's four by default).
    n_intervals : int, default=50
        Random intervals per representation.
    min_interval_length : int, default=3
    quantiles : bool, default=False
        Add the 10/25/75/90% quantiles to R-STSF's interval statistics (mean,
        median, std, slope, IQR, min, max, mean crossings, count above mean).
    catch22 : bool, default=False
        Add catch22 to the interval features (as DrCIF uses), on intervals of at
        least 5 points (catch22 fails on shorter ones).
    n_estimators : int, default=200
    max_features : float or str, default="sqrt"
    random_state, n_jobs : as usual.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
    }

    def __init__(
        self,
        representations=("raw", "d1", "pgram", "ar"),
        n_intervals=50,
        min_interval_length=3,
        quantiles=False,
        catch22=False,
        n_estimators=200,
        max_features="sqrt",
        random_state=None,
        n_jobs=1,
    ):
        self.representations = representations
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.quantiles = quantiles
        self.catch22 = catch22
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

    def _features(self, catch22=False):
        f = list(_STATS) + [row_count_mean_crossing, row_count_above_mean]
        if catch22:
            f = [Catch22(outlier_norm=True)] + f
        return f + (list(_QUANTILES) if self.quantiles else [])

    def _transforms(self, X, fit):
        if fit:
            self.series_transformers_ = [_representation(r) for r in self.representations]
            self.interval_transformers_ = []
        out = []
        for i, t in enumerate(self.series_transformers_):
            Xt = X if t is None else (t.fit_transform(X) if fit else t.transform(X))
            Xt = np.nan_to_num(np.asarray(Xt, dtype=float))
            if fit:
                c22 = self.catch22 and Xt.shape[2] >= 5
                mil = min(self.min_interval_length, Xt.shape[2])
                ri = RandomIntervals(
                    n_intervals=self.n_intervals,
                    min_interval_length=max(mil, 5) if c22 else mil,
                    features=self._features(c22),
                    random_state=None
                    if self.random_state is None
                    else self.random_state + i,
                    n_jobs=self.n_jobs,
                )
                out.append(ri.fit_transform(Xt))
                self.interval_transformers_.append(ri)
            else:
                out.append(self.interval_transformers_[i].transform(Xt))
        return np.nan_to_num(np.hstack(out))

    def _fit(self, X, y):
        Xt = self._transforms(X, fit=True)
        self.estimator_ = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(Xt, y)
        return self

    def _predict(self, X):
        return self.estimator_.predict(self._transforms(X, fit=False))

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"n_estimators": 2, "n_intervals": 2}
