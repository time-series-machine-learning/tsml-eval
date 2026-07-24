"""Tests for guarded multiaxis reduction."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tsml_eval.experiments._guarded_multiaxis import GuardedMultiAxisReducer


class _SignalClassifier(ClassifierMixin, BaseEstimator):
    """Classify from the sign of the collection mean."""

    def fit(self, X, y):
        """Return the fitted proxy."""
        return self

    def predict(self, X):
        """Predict using the retained signal."""
        return (X.mean(axis=(1, 2)) > 0).astype(int)


class _ConstantClassifier(ClassifierMixin, BaseEstimator):
    """Predict the majority observed class."""

    def fit(self, X, y):
        """Store the majority class."""
        values, counts = np.unique(y, return_counts=True)
        self.class_ = values[np.argmax(counts)]
        return self

    def predict(self, X):
        """Return constant predictions."""
        return np.repeat(self.class_, len(X))


def _make_data(n_cases=40, n_channels=3, n_timepoints=40):
    rng = np.random.RandomState(7)
    y = np.asarray([0, 1] * (n_cases // 2))
    X = rng.normal(scale=0.1, size=(n_cases, n_channels, n_timepoints))
    X[y == 1] += 1
    X[y == 0] -= 1
    return X, y


def test_fit_resample_keeps_train_and_test_aligned():
    """Cases are removed only from train; channel/time changes apply to both."""
    X, y = _make_data()
    reducer = GuardedMultiAxisReducer(
        channel_selector="none",
        proxy_estimator=_SignalClassifier(),
        strategy="time",
        time_fractions=(0.5, 1.0),
        slice_fractions=(0.5, 1.0),
        slice_positions=(0.5,),
        min_timepoints=1,
        max_score_loss=0,
        random_state=0,
    )

    Xt, yt = reducer.fit_resample(X, y)
    Xtest = reducer.transform(X[:5])

    assert Xt.shape[0] == len(yt)
    assert Xt.shape[1:] == Xtest.shape[1:]
    assert Xtest.shape[0] == 5
    assert reducer.n_timepoints_selected_ <= X.shape[2]


def test_aggressive_guard_rejects_score_loss():
    """An aggressive candidate cannot be selected when it loses to full data."""
    X, y = _make_data(n_timepoints=20)
    reducer = GuardedMultiAxisReducer(
        channel_selector="none",
        proxy_estimator=_ConstantClassifier(),
        strategy="time",
        time_fractions=(0.125, 1.0),
        slice_fractions=(1.0,),
        min_timepoints=1,
        aggressive_fraction=0.25,
        aggressive_margin=0.01,
        random_state=0,
    ).fit(X, y)

    aggressive = reducer.candidate_results_.query("family == 'downsample'").iloc[0]
    assert not aggressive["eligible"]
    assert reducer.route_ == "full"


def test_case_sampling_is_nested_and_test_cases_are_retained():
    """Nested deterministic case samples reduce train but never test cases."""
    X, y = _make_data(n_cases=40, n_timepoints=10)
    reducer = GuardedMultiAxisReducer(
        channel_selector="none",
        proxy_estimator=_SignalClassifier(),
        strategy="case",
        case_fractions=(0.25, 0.5, 1.0),
        min_cases_per_class=1,
        random_state=3,
    )
    available = np.arange(len(y))
    small = reducer._sample_case_indices(y, 0.25, available)
    large = reducer._sample_case_indices(y, 0.5, available)
    assert set(small).issubset(large)

    Xt, yt = reducer.fit_resample(X, y)
    Xtest = reducer.transform(X[:7])
    assert len(Xt) == len(yt) == reducer.n_cases_selected_
    assert len(Xtest) == 7


def test_slice_indices_are_contiguous():
    """A selected slice stores and applies contiguous time indices."""
    X, y = _make_data()
    reducer = GuardedMultiAxisReducer(
        channel_selector="none",
        proxy_estimator=_SignalClassifier(),
        strategy="time",
        time_fractions=(1.0,),
        slice_fractions=(0.5, 1.0),
        slice_positions=(0.5,),
        min_timepoints=1,
        max_score_loss=0,
        random_state=0,
    ).fit(X, y)

    assert reducer.route_ == "slice"
    assert np.all(np.diff(reducer.time_indices_) == 1)
