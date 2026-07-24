"""Experimental guarded reduction over channels, cases, and the time axis."""

from __future__ import annotations

import copy
import inspect
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

__all__ = ["GuardedMultiAxisReducer"]


class GuardedMultiAxisReducer(BaseEstimator):
    """Learn a guarded reduction of channels, training cases, and time points.

    The reducer creates one fixed stratified training/validation split and fits its
    tuning channel selector only on the internal training fold. A lightweight proxy
    for an HC2 component is selected on that fold, unless a proxy estimator or
    component is supplied explicitly. Monotone candidate sequences are evaluated
    from strong to weak reduction. A candidate is eligible only when its validation
    score is close to the full-data proxy score. More aggressive candidates are
    required to match or improve the full-data score. The chosen channel selector is
    then refitted on all available training data.

    Case sampling is applied only to training data. Channel selection and temporal
    reduction are learned on the training collection and applied unchanged to test
    collections.

    Parameters
    ----------
    channel_selector : {"tselect", "none"}, estimator, or None, default="tselect"
        Channel selector fitted before the other reductions. An estimator must
        implement ``fit`` and ``transform``.
    proxy_estimator : estimator or None, default=None
        Explicit lightweight classifier used for candidate evaluation. When supplied,
        ``proxy_component`` is ignored.
    proxy_component : {"auto", "tde", "arsenal", "drcif", "stc"}, default="auto"
        Lightweight HC2 component proxy. ``"auto"`` evaluates all four proxies on
        the full internal split and retains the most accurate one.
    strategy : {"auto", "case", "time", "all"}, default="auto"
        Candidate families. ``"auto"`` evaluates case sampling when cases outnumber
        time points after channel selection, otherwise temporal downsampling and
        slicing. ``"time"`` evaluates both temporal families. ``"all"`` evaluates
        every family.
    case_fractions : tuple of float, default=(0.25, 0.5, 1.0)
        Candidate fractions of training cases.
    time_fractions : tuple of float, default=(0.125, 0.25, 0.5, 1.0)
        Candidate retained lengths for temporal downsampling.
    slice_fractions : tuple of float, default=(0.25, 0.5, 0.75, 1.0)
        Candidate retained lengths for contiguous slicing.
    slice_positions : tuple of float, default=(0.0, 0.5, 1.0)
        Relative start positions for slices, where zero is the start and one is the
        latest valid start.
    validation_size : float, default=0.33
        Fraction of training cases assigned to the fixed internal validation split.
    scoring : {"balanced_accuracy", "accuracy"}, default="balanced_accuracy"
        Validation metric.
    max_score_loss : float, default=0.01
        Maximum score loss from the full-data proxy allowed for ordinary candidates.
    aggressive_fraction : float, default=0.25
        Fractions below this threshold receive the aggressive guard.
    aggressive_margin : float, default=0.0
        Minimum improvement over the full-data score required for aggressive
        candidates.
    min_improvement : float, default=0.0025
        Improvement required to continue a candidate family toward weaker reduction.
    min_cases_per_class : int, default=2
        Minimum number of retained training cases per class where possible.
    min_timepoints : int, default=10
        Minimum retained time length.
    time_reduction : {"resample", "subsample"}, default="resample"
        Temporal downsampling operation. Slicing always retains original contiguous
        observations.
    fail_fast : bool, default=False
        If True, re-raise proxy errors. Otherwise failed candidates are recorded.
    random_state : int or None, default=None
        Random seed used for splitting, nested case sampling, TSelect, and proxies.
    n_jobs : int, default=1
        Thread count passed to lightweight proxies that support it.

    Attributes
    ----------
    channels_selected_ : np.ndarray
        Selected original channel indices.
    case_indices_ : np.ndarray
        Selected training-case indices.
    time_indices_ : np.ndarray
        Selected original time indices. For Fourier resampling these describe the
        regular target grid; the transform itself uses ``scipy.signal.resample``.
    route_ : str
        Selected family: ``"full"``, ``"case"``, ``"downsample"``, or ``"slice"``.
    proxy_component_ : str
        Selected lightweight proxy.
    proxy_component_scores_ : dict
        Full-split scores used for automatic proxy selection.
    candidate_results_ : pandas.DataFrame
        Complete evaluated candidate trace.
    candidate_levels_ : list
        Candidate fractions evaluated in order.
    validation_scores_ : list
        Candidate validation scores in matching order.
    full_score_ : float
        Full-data proxy validation score after channel selection.
    best_score_ : float
        Best finite candidate score.
    selection_score_ : float
        Score for the selected candidate.
    """

    def __init__(
        self,
        channel_selector: str | Any | None = "tselect",
        proxy_estimator: Any | None = None,
        proxy_component: str = "auto",
        strategy: str = "auto",
        case_fractions: tuple[float, ...] = (0.25, 0.5, 1.0),
        time_fractions: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0),
        slice_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
        slice_positions: tuple[float, ...] = (0.0, 0.5, 1.0),
        validation_size: float = 0.33,
        scoring: str = "balanced_accuracy",
        max_score_loss: float = 0.01,
        aggressive_fraction: float = 0.25,
        aggressive_margin: float = 0.0,
        min_improvement: float = 0.0025,
        min_cases_per_class: int = 2,
        min_timepoints: int = 10,
        time_reduction: str = "resample",
        fail_fast: bool = False,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        self.channel_selector = channel_selector
        self.proxy_estimator = proxy_estimator
        self.proxy_component = proxy_component
        self.strategy = strategy
        self.case_fractions = case_fractions
        self.time_fractions = time_fractions
        self.slice_fractions = slice_fractions
        self.slice_positions = slice_positions
        self.validation_size = validation_size
        self.scoring = scoring
        self.max_score_loss = max_score_loss
        self.aggressive_fraction = aggressive_fraction
        self.aggressive_margin = aggressive_margin
        self.min_improvement = min_improvement
        self.min_cases_per_class = min_cases_per_class
        self.min_timepoints = min_timepoints
        self.time_reduction = time_reduction
        self.fail_fast = fail_fast
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the guarded reducer on a training collection."""
        X = np.asarray(X)
        y = np.asarray(y)
        self._validate_inputs(X, y)

        self.n_cases_in_, self.n_channels_in_, self.n_timepoints_in_ = X.shape
        train_idx, val_idx = self._make_train_val_indices(y)
        self.train_indices_ = train_idx
        self.validation_indices_ = val_idx

        self.tuning_channel_selector_ = self._make_channel_selector()
        X_channels, self.tuning_channels_selected_ = self._fit_channel_selector(
            self.tuning_channel_selector_, X, y, train_idx
        )
        self.n_channels_tuning_ = X_channels.shape[1]
        self.n_channels_selected_ = self.n_channels_tuning_

        self.proxy_component_, self.proxy_component_scores_ = (
            self._select_proxy_component(X_channels, y, train_idx, val_idx)
        )

        full_candidate = self._candidate(
            family="full",
            fraction=1.0,
            case_fraction=1.0,
            time_indices=np.arange(self.n_timepoints_in_, dtype=int),
            slice_start=None,
        )
        rows = [
            self._evaluate_candidate(X_channels, y, train_idx, val_idx, full_candidate)
        ]
        self.full_score_ = float(rows[0]["score"])
        if not np.isfinite(self.full_score_):
            raise RuntimeError("The full-data proxy evaluation failed.")

        families = self._resolve_families()
        if "case" in families:
            rows.extend(
                self._evaluate_monotone_family(
                    X_channels,
                    y,
                    train_idx,
                    val_idx,
                    family="case",
                    fractions=self._normalise_fractions(
                        self.case_fractions, "case_fractions"
                    ),
                )
            )
        if "downsample" in families:
            rows.extend(
                self._evaluate_monotone_family(
                    X_channels,
                    y,
                    train_idx,
                    val_idx,
                    family="downsample",
                    fractions=self._normalise_fractions(
                        self.time_fractions, "time_fractions"
                    ),
                )
            )
        if "slice" in families:
            rows.extend(
                self._evaluate_monotone_family(
                    X_channels,
                    y,
                    train_idx,
                    val_idx,
                    family="slice",
                    fractions=self._normalise_fractions(
                        self.slice_fractions, "slice_fractions"
                    ),
                )
            )

        self.channel_selector_ = self._make_channel_selector()
        _, self.channels_selected_ = self._fit_channel_selector(
            self.channel_selector_, X, y
        )
        self.n_channels_selected_ = len(self.channels_selected_)

        self.candidate_results_ = pd.DataFrame(rows)
        self.candidate_results_["n_channels_final"] = self.n_channels_selected_
        self.candidate_results_["input_size"] = (
            self.candidate_results_["n_cases_final_train"]
            * self.n_channels_selected_
            * self.candidate_results_["n_timepoints_final"]
        )
        selected = self._select_candidate(self.candidate_results_)
        self.selected_candidate_ = selected
        self.route_ = str(selected["family"])
        self.case_fraction_ = float(selected["case_fraction"])
        self.time_indices_ = np.asarray(selected["time_indices"], dtype=int)
        self.slice_start_ = (
            None if pd.isna(selected["slice_start"]) else int(selected["slice_start"])
        )
        self.selection_score_ = float(selected["score"])
        finite_scores = self.candidate_results_.loc[
            np.isfinite(self.candidate_results_["score"]), "score"
        ]
        self.best_score_ = float(finite_scores.max())

        self.case_indices_ = self._sample_case_indices(
            y, self.case_fraction_, np.arange(len(y), dtype=int)
        )
        self.n_cases_selected_ = len(self.case_indices_)
        self.n_timepoints_selected_ = len(self.time_indices_)
        self.candidate_levels_ = self.candidate_results_["fraction"].tolist()
        self.validation_scores_ = self.candidate_results_["score"].tolist()
        return self

    def fit_resample(self, X, y):
        """Fit the reducer and return aligned reduced training data and labels."""
        return self.fit(X, y).resample_train(X, y)

    def resample_train(self, X, y):
        """Apply learned channel, case, and time reductions to training data."""
        check_is_fitted(self, "case_indices_")
        X = np.asarray(X)
        y = np.asarray(y)
        self._validate_transform_input(X)
        if len(y) != len(X):
            raise ValueError("X and y have inconsistent numbers of cases.")

        X = self._apply_channels(X)
        X = X[self.case_indices_]
        return self._apply_time(X), y[self.case_indices_]

    def transform(self, X):
        """Apply learned channel and time reductions without removing test cases."""
        check_is_fitted(self, "time_indices_")
        X = np.asarray(X)
        self._validate_transform_input(X)
        return self._apply_time(self._apply_channels(X))

    def transform_test(self, X):
        """Alias for :meth:`transform`."""
        return self.transform(X)

    def get_reduction_summary(self):
        """Return the fitted reduction configuration and validation trace."""
        check_is_fitted(self, "selected_candidate_")
        return {
            "route": self.route_,
            "proxy_component": self.proxy_component_,
            "proxy_component_scores": dict(self.proxy_component_scores_),
            "n_cases_in": self.n_cases_in_,
            "n_cases_selected": self.n_cases_selected_,
            "n_channels_in": self.n_channels_in_,
            "n_channels_selected": self.n_channels_selected_,
            "n_channels_tuning": self.n_channels_tuning_,
            "n_timepoints_in": self.n_timepoints_in_,
            "n_timepoints_selected": self.n_timepoints_selected_,
            "case_fraction": self.case_fraction_,
            "channels_selected": self.channels_selected_.tolist(),
            "tuning_channels_selected": self.tuning_channels_selected_.tolist(),
            "case_indices": self.case_indices_.tolist(),
            "time_indices": self.time_indices_.tolist(),
            "slice_start": self.slice_start_,
            "full_score": self.full_score_,
            "best_score": self.best_score_,
            "selection_score": self.selection_score_,
            "score_is_tuning_score": True,
            "max_score_loss": self.max_score_loss,
            "aggressive_fraction": self.aggressive_fraction,
            "aggressive_margin": self.aggressive_margin,
        }

    def _validate_inputs(self, X, y):
        if X.ndim != 3:
            raise ValueError(
                "GuardedMultiAxisReducer expects shape "
                "(n_cases, n_channels, n_timepoints)."
            )
        if len(y) != len(X):
            raise ValueError("X and y have inconsistent numbers of cases.")
        if self.proxy_component not in {"auto", "tde", "arsenal", "drcif", "stc"}:
            raise ValueError(
                "proxy_component must be one of "
                "{'auto', 'tde', 'arsenal', 'drcif', 'stc'}."
            )
        if self.strategy not in {"auto", "case", "time", "all"}:
            raise ValueError("strategy must be one of {'auto', 'case', 'time', 'all'}.")
        if self.scoring not in {"balanced_accuracy", "accuracy"}:
            raise ValueError("scoring must be 'balanced_accuracy' or 'accuracy'.")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be in the interval (0, 1).")
        if self.max_score_loss < 0 or self.aggressive_margin < 0:
            raise ValueError("score guards must be non-negative.")
        if not 0 < self.aggressive_fraction <= 1:
            raise ValueError("aggressive_fraction must be in the interval (0, 1].")
        if self.min_improvement < 0:
            raise ValueError("min_improvement must be non-negative.")
        if self.min_cases_per_class < 1 or self.min_timepoints < 1:
            raise ValueError("minimum retained sizes must be positive.")
        if self.time_reduction not in {"resample", "subsample"}:
            raise ValueError("time_reduction must be 'resample' or 'subsample'.")
        for position in self.slice_positions:
            if not 0 <= position <= 1:
                raise ValueError("slice_positions values must lie in [0, 1].")

    def _validate_transform_input(self, X):
        if X.ndim != 3:
            raise ValueError(
                "GuardedMultiAxisReducer expects shape "
                "(n_cases, n_channels, n_timepoints)."
            )
        if X.shape[1:] != (self.n_channels_in_, self.n_timepoints_in_):
            raise ValueError(
                "X channel/time shape does not match the fitted training collection."
            )

    def _make_channel_selector(self):
        if self.channel_selector is None:
            return None
        if isinstance(self.channel_selector, str):
            key = self.channel_selector.casefold()
            if key == "none":
                return None
            if key == "tselect":
                from aeon.transformations.collection.channel_selection import TSelect

                return TSelect(random_state=self.random_state)
            raise ValueError("channel_selector string must be 'tselect' or 'none'.")
        return self._safe_clone(self.channel_selector)

    def _fit_channel_selector(self, selector, X, y, fit_indices=None):
        if selector is None:
            return X, np.arange(self.n_channels_in_, dtype=int)
        if fit_indices is None:
            selector.fit(X, y)
        else:
            selector.fit(X[fit_indices], y[fit_indices])
        transformed = np.asarray(selector.transform(X))
        selected = getattr(selector, "channels_selected_", None)
        if selected is None:
            raise AttributeError(
                "channel_selector must expose channels_selected_ after fit."
            )
        return transformed, np.asarray(selected, dtype=int)

    def _make_train_val_indices(self, y):
        indices = np.arange(len(y), dtype=int)
        _, counts = np.unique(y, return_counts=True)
        stratify = y if np.min(counts) >= 2 else None
        try:
            train, validation = train_test_split(
                indices,
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            train, validation = train_test_split(
                indices,
                test_size=self.validation_size,
                random_state=self.random_state,
            )
        return np.asarray(train, dtype=int), np.asarray(validation, dtype=int)

    def _select_proxy_component(self, X, y, train_idx, val_idx):
        if self.proxy_estimator is not None:
            return "custom", {}
        components = (
            ("tde", "arsenal", "drcif", "stc")
            if self.proxy_component == "auto"
            else (self.proxy_component,)
        )
        scores = {}
        for component in components:
            estimator = self._make_component_proxy(component)
            try:
                estimator.fit(X[train_idx], y[train_idx])
                scores[component] = self._score(
                    y[val_idx], estimator.predict(X[val_idx])
                )
            except Exception:
                if self.fail_fast:
                    raise
                scores[component] = -np.inf
        finite = {key: value for key, value in scores.items() if np.isfinite(value)}
        if not finite:
            raise RuntimeError("All lightweight HC2 component proxies failed.")
        selected = max(components, key=lambda key: scores[key])
        return selected, scores

    def _make_proxy(self):
        if self.proxy_estimator is not None:
            return self._safe_clone(self.proxy_estimator)
        return self._make_component_proxy(self.proxy_component_)

    def _make_component_proxy(self, component):
        if component == "tde":
            from aeon.classification.dictionary_based import IndividualTDE

            kwargs = {
                "window_size": 16,
                "word_length": 8,
                "norm": False,
                "levels": 1,
                "bigrams": False,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }
            parameters = inspect.signature(IndividualTDE).parameters
            if "max_channels" in parameters:
                kwargs["max_channels"] = 10
            elif "max_dims" in parameters:
                kwargs["max_dims"] = 10
            return IndividualTDE(**kwargs)
        if component == "arsenal":
            from aeon.classification.convolution_based import MiniRocketClassifier

            return MiniRocketClassifier(
                n_kernels=336,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        if component == "drcif":
            from aeon.classification.interval_based import DrCIFClassifier

            return DrCIFClassifier(
                n_estimators=10,
                n_intervals=2,
                att_subsample_size=3,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        if component == "stc":
            from aeon.classification.shapelet_based import ShapeletTransformClassifier

            return ShapeletTransformClassifier(
                n_shapelet_samples=100,
                max_shapelets=10,
                estimator=RidgeClassifierCV(alphas=np.logspace(-3, 3, 7)),
                batch_size=20,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        raise RuntimeError(f"Unknown proxy component: {component}")

    def _resolve_families(self):
        if self.strategy == "case":
            return ("case",)
        if self.strategy == "time":
            return ("downsample", "slice")
        if self.strategy == "all":
            return ("case", "downsample", "slice")
        if self.n_cases_in_ > self.n_timepoints_in_:
            return ("case",)
        return ("downsample", "slice")

    def _evaluate_monotone_family(self, X, y, train_idx, val_idx, family, fractions):
        rows = []
        previous_score = -np.inf
        for fraction in fractions:
            if np.isclose(fraction, 1.0):
                continue
            candidates = self._candidates_for_level(family, fraction)
            level_rows = [
                self._evaluate_candidate(X, y, train_idx, val_idx, candidate)
                for candidate in candidates
            ]
            rows.extend(level_rows)
            finite_scores = [
                row["score"] for row in level_rows if np.isfinite(row["score"])
            ]
            if not finite_scores:
                continue
            level_score = max(finite_scores)
            if np.isfinite(previous_score):
                if level_score <= previous_score + self.min_improvement:
                    break
            previous_score = level_score
        return rows

    def _candidates_for_level(self, family, fraction):
        full_time = np.arange(self.n_timepoints_in_, dtype=int)
        if family == "case":
            return [
                self._candidate(
                    family="case",
                    fraction=fraction,
                    case_fraction=fraction,
                    time_indices=full_time,
                    slice_start=None,
                )
            ]

        length = self._length_from_fraction(fraction)
        if family == "downsample":
            return [
                self._candidate(
                    family="downsample",
                    fraction=fraction,
                    case_fraction=1.0,
                    time_indices=self._regular_time_indices(length),
                    slice_start=None,
                )
            ]

        max_start = self.n_timepoints_in_ - length
        starts = sorted(
            {int(round(position * max_start)) for position in self.slice_positions}
        )
        return [
            self._candidate(
                family="slice",
                fraction=fraction,
                case_fraction=1.0,
                time_indices=np.arange(start, start + length, dtype=int),
                slice_start=start,
            )
            for start in starts
        ]

    @staticmethod
    def _candidate(family, fraction, case_fraction, time_indices, slice_start):
        suffix = "" if slice_start is None else f"_start_{slice_start}"
        return {
            "family": family,
            "candidate": f"{family}_{fraction:g}{suffix}",
            "fraction": float(fraction),
            "case_fraction": float(case_fraction),
            "time_indices": np.asarray(time_indices, dtype=int),
            "slice_start": slice_start,
        }

    def _evaluate_candidate(self, X, y, train_idx, val_idx, candidate):
        proxy_train = self._sample_case_indices(
            y, candidate["case_fraction"], train_idx
        )
        start = time.perf_counter()
        error = None
        try:
            estimator = self._make_proxy()
            X_train = self._apply_candidate_time(
                X[proxy_train], candidate["family"], candidate["time_indices"]
            )
            X_validation = self._apply_candidate_time(
                X[val_idx], candidate["family"], candidate["time_indices"]
            )
            estimator.fit(X_train, y[proxy_train])
            fit_time = time.perf_counter() - start
            predict_start = time.perf_counter()
            predictions = estimator.predict(X_validation)
            predict_time = time.perf_counter() - predict_start
            score = self._score(y[val_idx], predictions)
        except Exception as exc:
            if self.fail_fast:
                raise
            fit_time = time.perf_counter() - start
            predict_time = np.nan
            score = -np.inf
            error = repr(exc)

        final_cases = len(
            self._sample_case_indices(
                y,
                candidate["case_fraction"],
                np.arange(len(y), dtype=int),
            )
        )
        final_time = len(candidate["time_indices"])
        fraction = candidate["fraction"]
        aggressive = fraction < self.aggressive_fraction
        if candidate["family"] == "full":
            guard_threshold = score
            eligible = bool(np.isfinite(score))
        else:
            guard_threshold = (
                self.full_score_ + self.aggressive_margin
                if aggressive
                else self.full_score_ - self.max_score_loss
            )
            eligible = bool(np.isfinite(score) and score >= guard_threshold)
        recorded_full_score = (
            score if candidate["family"] == "full" else self.full_score_
        )
        return {
            **candidate,
            "time_indices": candidate["time_indices"].tolist(),
            "n_cases_proxy_train": len(proxy_train),
            "n_cases_final_train": final_cases,
            "n_channels_final": self.n_channels_selected_,
            "n_timepoints_final": final_time,
            "score": score,
            "full_score": recorded_full_score,
            "guard_threshold": guard_threshold,
            "aggressive": aggressive,
            "eligible": eligible,
            "input_size": final_cases * self.n_channels_selected_ * final_time,
            "fit_time": fit_time,
            "predict_time": predict_time,
            "total_time": fit_time + (0.0 if np.isnan(predict_time) else predict_time),
            "selected": False,
            "error": error,
        }

    def _select_candidate(self, results):
        finite = results[np.isfinite(results["score"])].copy()
        eligible = finite[finite["eligible"]].copy()
        if eligible.empty:
            eligible = finite[finite["family"] == "full"].copy()
        eligible = eligible.sort_values(
            by=["input_size", "score", "total_time"],
            ascending=[True, False, True],
        )
        selected_index = eligible.index[0]
        self.candidate_results_.loc[selected_index, "selected"] = True
        return self.candidate_results_.loc[selected_index].to_dict()

    def _sample_case_indices(self, y, fraction, available_indices):
        available_indices = np.asarray(available_indices, dtype=int)
        if fraction >= 1:
            return np.sort(available_indices)
        rng = np.random.default_rng(self.random_state)
        selected = []
        for cls in np.unique(y[available_indices]):
            cls_indices = available_indices[y[available_indices] == cls]
            n_select = max(
                self.min_cases_per_class,
                int(math.ceil(fraction * len(cls_indices))),
            )
            n_select = min(n_select, len(cls_indices))
            ordering = rng.permutation(cls_indices)
            selected.append(ordering[:n_select])
        return np.sort(np.concatenate(selected).astype(int))

    def _apply_channels(self, X):
        if self.channel_selector_ is None:
            return X
        return np.asarray(self.channel_selector_.transform(X))

    def _apply_time(self, X):
        return self._apply_candidate_time(X, self.route_, self.time_indices_)

    def _apply_candidate_time(self, X, family, time_indices):
        time_indices = np.asarray(time_indices, dtype=int)
        if len(time_indices) >= X.shape[2]:
            return X
        if family == "slice" or self.time_reduction == "subsample":
            return X[:, :, time_indices]
        return resample(X, num=len(time_indices), axis=2)

    def _length_from_fraction(self, fraction):
        length = int(math.ceil(fraction * self.n_timepoints_in_))
        return min(
            self.n_timepoints_in_,
            max(self.min_timepoints, length),
        )

    def _regular_time_indices(self, length):
        if length >= self.n_timepoints_in_:
            return np.arange(self.n_timepoints_in_, dtype=int)
        indices = np.linspace(0, self.n_timepoints_in_ - 1, length)
        indices = np.unique(np.round(indices).astype(int))
        if len(indices) < length:
            missing = np.setdiff1d(
                np.arange(self.n_timepoints_in_, dtype=int),
                indices,
                assume_unique=True,
            )
            indices = np.sort(
                np.concatenate([indices, missing[: length - len(indices)]])
            )
        return indices

    @staticmethod
    def _normalise_fractions(fractions, name):
        if fractions is None or len(fractions) == 0:
            raise ValueError(f"{name} must contain at least one fraction.")
        clean = []
        for fraction in fractions:
            if not 0 < fraction <= 1:
                raise ValueError(f"All {name} values must lie in (0, 1].")
            clean.append(float(fraction))
        if 1.0 not in clean:
            clean.append(1.0)
        return tuple(sorted(set(clean)))

    def _score(self, y_true, y_pred):
        if self.scoring == "balanced_accuracy":
            return balanced_accuracy_score(y_true, y_pred)
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def _safe_clone(estimator):
        try:
            return clone(estimator)
        except Exception:
            return copy.deepcopy(estimator)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return a cheap deterministic parameter set for estimator testing."""
        from sklearn.dummy import DummyClassifier

        return {
            "channel_selector": "none",
            "proxy_estimator": DummyClassifier(strategy="most_frequent"),
            "strategy": "time",
            "time_fractions": (0.5, 1.0),
            "slice_fractions": (0.5, 1.0),
            "slice_positions": (0.5,),
            "min_timepoints": 1,
            "random_state": 0,
        }
