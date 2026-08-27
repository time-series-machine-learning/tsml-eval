"""PULSAR interval regressor.

Regression counterpart of :class:`PULSARClassifier`. The feature-generation
pipeline (four representations, local statistics over dilated intervals,
hierarchical pooling, and per-channel concatenation for multivariate series) is
reused verbatim from the classifier, since the transform is target agnostic.
Only the supervised parts differ: candidate features are ranked by their
absolute correlation with the target instead of a Fisher score, and the selected,
scaled features are passed to Ridge and Extra-Trees regression heads whose
predictions are averaged.
"""

__maintainer__ = []
__all__ = ["PULSARRegressor"]

from time import perf_counter

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from aeon.regression.base import BaseRegressor
from aeon.utils.validation import check_n_jobs
from tsml_eval._wip.classification._pulsar import (
    PULSARClassifier,
    _DEFAULT_LOCAL_STATISTICS,
    _DEFAULT_POOLING_OPERATORS,
    _DEFAULT_REPRESENTATIONS,
    _VALID_LOCAL_STATISTICS,
    _VALID_POOLING_OPERATORS,
    _VALID_REPRESENTATIONS,
)


def _correlation_scores(X, y):
    """Absolute Pearson correlation of each candidate feature with the target.

    Monotonic with the univariate ``f_regression`` F-statistic, so it ranks
    features identically while staying dependency free. Constant columns score
    zero.
    """
    y = np.asarray(y, dtype=np.float64)
    y_centered = y - y.mean()
    y_norm = np.sqrt(np.sum(y_centered**2))
    if y_norm <= 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    X_centered = X - X.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(X_centered**2, axis=0)) * y_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.abs(X_centered.T @ y_centered) / denominator
    scores[~np.isfinite(scores)] = 0.0
    return scores


class PULSARRegressor(BaseRegressor):
    """PULSAR interval regressor.

    Parameters
    ----------
    representations : tuple of str or None, default=None
        Representations to use. ``None`` selects ``("original", "periodogram",
        "derivative", "autoregressive")``.
    interval_lengths : tuple of int, default=(7, 9, 11)
        Base interval lengths.
    max_dilation : int, default=16
        Maximum power-of-two dilation.
    local_statistics : tuple of str or None, default=None
        Local statistics. ``None`` selects the seven published statistics.
    pooling_operators : tuple of str or None, default=None
        Pooling operators. ``None`` selects the nine published operators.
    hierarchical_depth : int, default=4
        Number of hierarchy levels, including the global level.
    n_random_pooling_operators : int, default=6
        Number of pooling operators randomly retained per finer partition.
    feature_selection_percentage : float, default=40
        Percentage of finer pooled and raw candidate features retained by
        correlation with the target.
    heads : tuple of str, default=("ridge", "extra_trees")
        Regression heads to average. Valid values are ``"ridge"`` and
        ``"extra_trees"``.
    n_estimators : int, default=50
        Number of trees in the Extra-Trees head.
    n_jobs : int, default=1
        Number of jobs used by supported regression heads.
    random_state : int, RandomState instance or None, default=None
        Controls pooling-operator selection and randomized regression heads.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:multithreading": True,
        "algorithm_type": "interval",
        "X_inner_type": "numpy3D",
        "python_dependencies": "statsmodels",
    }

    # The transform is target agnostic, so it is reused verbatim from the
    # classifier. These are plain functions bound as methods here.
    _get_all_representations = PULSARClassifier._get_all_representations
    _new_interval_states = PULSARClassifier._new_interval_states
    _pool_response_map = PULSARClassifier._pool_response_map
    _feature_transform = PULSARClassifier._feature_transform

    def __init__(
        self,
        representations=None,
        interval_lengths=(7, 9, 11),
        max_dilation=16,
        local_statistics=None,
        pooling_operators=None,
        hierarchical_depth=4,
        n_random_pooling_operators=6,
        feature_selection_percentage=40,
        heads=("ridge", "extra_trees"),
        n_estimators=50,
        n_jobs=1,
        random_state=None,
    ):
        self.representations = representations
        self.interval_lengths = interval_lengths
        self.max_dilation = max_dilation
        self.local_statistics = local_statistics
        self.pooling_operators = pooling_operators
        self.hierarchical_depth = hierarchical_depth
        self.n_random_pooling_operators = n_random_pooling_operators
        self.feature_selection_percentage = feature_selection_percentage
        self.heads = heads
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _validate_parameters(self):
        """Validate constructor values used by feature generation and heads."""
        representations = (
            _DEFAULT_REPRESENTATIONS
            if self.representations is None
            else self.representations
        )
        local_statistics = (
            _DEFAULT_LOCAL_STATISTICS
            if self.local_statistics is None
            else self.local_statistics
        )
        pooling_operators = (
            _DEFAULT_POOLING_OPERATORS
            if self.pooling_operators is None
            else self.pooling_operators
        )
        if any(r not in _VALID_REPRESENTATIONS for r in representations):
            raise ValueError("representations contains an unknown value")
        if any(s not in _VALID_LOCAL_STATISTICS for s in local_statistics):
            raise ValueError("local_statistics contains an unknown value")
        if any(o not in _VALID_POOLING_OPERATORS for o in pooling_operators):
            raise ValueError("pooling_operators contains an unknown value")
        if (
            not isinstance(self.n_random_pooling_operators, (int, np.integer))
            or self.n_random_pooling_operators < 1
        ):
            raise ValueError("n_random_pooling_operators must be >= 1")
        if not 0 <= self.feature_selection_percentage <= 100:
            raise ValueError("feature_selection_percentage must be in [0, 100]")
        if (
            isinstance(self.heads, str)
            or not self.heads
            or any(head not in ("ridge", "extra_trees") for head in self.heads)
        ):
            raise ValueError("heads must contain only 'ridge' and 'extra_trees'")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self._n_jobs = check_n_jobs(self.n_jobs)
        self._representations = tuple(representations)
        self._local_statistics = tuple(local_statistics)
        self._pooling_operators = tuple(pooling_operators)

    def _fit_head(self, name, X, y):
        """Fit one regression head."""
        if name == "ridge":
            estimator = RidgeCV(alphas=np.logspace(-3, 3, 10))
        else:
            estimator = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_features=0.10,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        estimator.fit(X, y)
        return estimator

    def _fit(self, X, y):
        """Fit feature generation, selection, scaling, and regression heads."""
        self._validate_parameters()
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        y = np.asarray(y, dtype=np.float64)

        start = perf_counter()
        global_features, candidates, stage_times = self._feature_transform(X, True)
        if candidates.shape[1] == 0:
            raise ValueError("PULSAR generated no candidate features")

        selection_start = perf_counter()
        scores = _correlation_scores(candidates, y)
        n_selected = max(
            1,
            min(
                candidates.shape[1],
                int(candidates.shape[1] * self.feature_selection_percentage / 100),
            ),
        )
        selected = np.argsort(scores, kind="stable")[-n_selected:]
        stage_times["correlation_scoring_and_selection"] = (
            perf_counter() - selection_start
        )
        self.selected_feature_indices_ = selected.astype(np.intp, copy=False)
        self.feature_selection_scores_ = scores
        self.selected_feature_metadata_ = tuple(
            self.candidate_feature_metadata_[index] for index in selected
        )

        combined = np.hstack((global_features, candidates[:, selected]))
        scaling_start = perf_counter()
        self.scaler_ = StandardScaler()
        combined = self.scaler_.fit_transform(combined)
        stage_times["scaling"] = perf_counter() - scaling_start
        self.n_features_in_ = combined.shape[1]
        self.n_global_features_ = global_features.shape[1]
        self.n_candidate_features_ = candidates.shape[1]
        self.n_selected_features_ = selected.shape[0]

        head_start = perf_counter()
        self.heads_ = {name: self._fit_head(name, combined, y) for name in self.heads}
        stage_times["head_fitting"] = perf_counter() - head_start
        stage_times["total"] = perf_counter() - start
        self.fit_stage_times_ = stage_times
        return self

    def _transform_for_prediction(self, X):
        """Generate, select, and scale features for new series."""
        global_features, candidates, stage_times = self._feature_transform(X, False)
        start = perf_counter()
        combined = np.hstack(
            (global_features, candidates[:, self.selected_feature_indices_])
        )
        combined = self.scaler_.transform(combined)
        stage_times["scaling"] = perf_counter() - start
        self.predict_stage_times_ = stage_times
        return combined

    def _predict(self, X):
        """Predict targets as the mean of the regression heads."""
        features = self._transform_for_prediction(X)
        predictions = [head.predict(features) for head in self.heads_.values()]
        result = np.mean(predictions, axis=0)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return a small parameter set for estimator checks."""
        return {
            "representations": ("original", "derivative"),
            "interval_lengths": (3,),
            "max_dilation": 2,
            "local_statistics": ("mean", "stdev"),
            "pooling_operators": ("max", "mean"),
            "hierarchical_depth": 2,
            "n_random_pooling_operators": 1,
            "feature_selection_percentage": 40,
            "heads": ("ridge",),
            "n_jobs": 1,
            "random_state": 0,
        }
