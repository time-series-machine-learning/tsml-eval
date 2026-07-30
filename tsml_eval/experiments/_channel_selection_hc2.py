"""Channel-selection classifier pipelines used by EEG experiments."""

__maintainer__ = ["TonyBagnall"]

import time
from math import ceil

import numpy as np
from aeon.classification.base import BaseClassifier
from sklearn.base import clone


class ChannelSelectionClassifierPipeline(BaseClassifier):
    """Fit a learned collection transform followed by a classifier.

    Unlike aeon's standard ``ClassifierPipeline``, this pipeline also supports
    resampling transforms whose ``fit_resample`` method returns aligned ``X`` and
    ``y``. This is required for ``CaseTimeReducer`` when it selects case
    subsampling. All other configured transforms retain ``y`` unchanged.

    Parameters
    ----------
    selector : str
        Channel-selection, channel-creation, or case/time-reduction method.
    classifier : classifier
        Classifier fitted after the learned transform.
    proportion : float, default=0.25
        Proportion of channels/components retained by methods with an explicit
        output-size parameter.
    random_state : int or None, default=None
        Random seed passed to stochastic transforms.
    n_jobs : int, default=1
        Number of jobs passed to transforms which support parallelism.
    proxy_component : str or None, default=None
        Downstream HC2 component used by component-aware reducers.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        selector,
        classifier,
        proportion=0.25,
        random_state=None,
        n_jobs=1,
        proxy_component=None,
    ):
        self.selector = selector
        self.classifier = classifier
        self.proportion = proportion
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.proxy_component = proxy_component
        super().__init__()

    def _fit(self, X, y):
        """Fit the transform and classifier, retaining aligned training labels."""
        self.train_input_shape_ = tuple(int(v) for v in X.shape)
        self.transformer_ = _make_channel_transformer(
            self.selector,
            n_channels=X.shape[1],
            proportion=self.proportion,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            proxy_component=self.proxy_component,
        )

        start = time.perf_counter_ns()
        if hasattr(self.transformer_, "fit_resample"):
            Xt, yt = self.transformer_.fit_resample(X, y)
        else:
            Xt = self.transformer_.fit_transform(X, y)
            yt = y
        self.transform_fit_time_millis_ = (
            time.perf_counter_ns() - start
        ) / 1_000_000
        self.train_output_shape_ = tuple(int(v) for v in Xt.shape)
        self.n_train_labels_in_ = int(len(y))
        self.n_train_labels_out_ = int(len(yt))
        self.train_class_counts_in_ = _class_counts(y)
        self.train_class_counts_out_ = _class_counts(yt)

        self.classifier_ = clone(self.classifier)
        start = time.perf_counter_ns()
        self.classifier_.fit(Xt, yt)
        self.hc2_fit_time_millis_ = (time.perf_counter_ns() - start) / 1_000_000
        return self

    def _predict(self, X):
        """Transform test cases without case subsampling, then predict."""
        Xt = self._time_test_transform(X)
        start = time.perf_counter_ns()
        predictions = self.classifier_.predict(Xt)
        self.hc2_predict_time_millis_ = (
            time.perf_counter_ns() - start
        ) / 1_000_000
        return predictions

    def _predict_proba(self, X):
        """Transform test cases without case subsampling, then predict probabilities."""
        Xt = self._time_test_transform(X)
        start = time.perf_counter_ns()
        probabilities = self.classifier_.predict_proba(Xt)
        self.hc2_predict_time_millis_ = (
            time.perf_counter_ns() - start
        ) / 1_000_000
        return probabilities

    def _time_test_transform(self, X):
        """Transform test data while recording the transform-only wall time."""
        self.test_input_shape_ = tuple(int(v) for v in X.shape)
        start = time.perf_counter_ns()
        Xt = self.transformer_.transform(X)
        self.transform_predict_time_millis_ = (
            time.perf_counter_ns() - start
        ) / 1_000_000
        self.test_output_shape_ = tuple(int(v) for v in Xt.shape)
        return Xt

    def get_experiment_metadata(self):
        """Return compact fitted metadata for inclusion in experiment result files."""
        if not hasattr(self, "transformer_"):
            return {}

        metadata = {
            "transformer_class": type(self.transformer_).__name__,
            "classifier_class": type(self.classifier_).__name__,
            "timings_ms": {
                "transform_fit": getattr(
                    self, "transform_fit_time_millis_", None
                ),
                "classifier_fit": getattr(
                    self, "hc2_fit_time_millis_", None
                ),
                "hc2_fit": getattr(self, "hc2_fit_time_millis_", None),
                "transform_predict": getattr(
                    self, "transform_predict_time_millis_", None
                ),
                "classifier_predict": getattr(
                    self, "hc2_predict_time_millis_", None
                ),
                "hc2_predict": getattr(
                    self, "hc2_predict_time_millis_", None
                ),
            },
            "train_input_shape": getattr(self, "train_input_shape_", None),
            "train_output_shape": getattr(self, "train_output_shape_", None),
            "test_input_shape": getattr(self, "test_input_shape_", None),
            "test_output_shape": getattr(self, "test_output_shape_", None),
            "n_train_labels_in": getattr(self, "n_train_labels_in_", None),
            "n_train_labels_out": getattr(self, "n_train_labels_out_", None),
            "train_class_counts_in": getattr(
                self, "train_class_counts_in_", None
            ),
            "train_class_counts_out": getattr(
                self, "train_class_counts_out_", None
            ),
        }

        transformer = getattr(self, "transformer_", None)
        selector_metadata = _selector_metadata(transformer)
        if selector_metadata:
            metadata["selector"] = selector_metadata
            if "channels_selected" in selector_metadata:
                metadata["channels_selected"] = selector_metadata[
                    "channels_selected"
                ]

        hc2_metadata = _hc2_metadata(getattr(self, "classifier_", None))
        if hc2_metadata:
            metadata["hc2"] = hc2_metadata

        if transformer is not None and hasattr(
            transformer, "get_reduction_summary"
        ):
            summary = transformer.get_reduction_summary()
            large_index_fields = {
                "case_indices",
                "time_indices",
            }
            metadata["reduction_summary"] = {
                key: value
                for key, value in summary.items()
                if key not in large_index_fields
            }

        candidate_results = getattr(transformer, "candidate_results_", None)
        if candidate_results is not None:
            trace_columns = [
                "candidate",
                "family",
                "fraction",
                "case_fraction",
                "use_channels",
                "n_cases_final_train",
                "n_channels_final",
                "n_timepoints_final",
                "input_size",
                "score",
                "full_score",
                "guard_threshold",
                "aggressive",
                "eligible",
                "selected",
                "fit_time",
                "predict_time",
                "error",
            ]
            available = [
                column
                for column in trace_columns
                if column in candidate_results.columns
            ]
            records = candidate_results[available].to_dict("records")
            for record in records:
                if "fit_time" in record:
                    record["fit_time_seconds"] = record.pop("fit_time")
                if "predict_time" in record:
                    record["predict_time_seconds"] = record.pop(
                        "predict_time"
                    )
            metadata["reduction_candidates"] = records

        return _metadata_to_builtin(metadata)


def _metadata_to_builtin(value):
    """Convert NumPy/pandas scalar containers to compact built-in values."""
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if hasattr(key, "item"):
                key = key.item()
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                key = str(key)
            converted[key] = _metadata_to_builtin(item)
        return converted
    if isinstance(value, (set, frozenset)):
        try:
            value = sorted(value)
        except TypeError:
            value = list(value)
        return [_metadata_to_builtin(item) for item in value]
    if isinstance(value, (list, tuple)):
        return [_metadata_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _class_counts(y):
    """Return compact class frequencies for resampling diagnostics."""
    labels, counts = np.unique(y, return_counts=True)
    return {
        str(label.item() if hasattr(label, "item") else label): int(count)
        for label, count in zip(labels, counts)
    }


def _selector_metadata(transformer):
    """Return selected channels and compact fitted selector diagnostics."""
    if transformer is None:
        return {}

    selector = getattr(transformer, "channel_selector_", None)
    if selector is None or isinstance(selector, str):
        selector = transformer
    selector_metadata = {"class": type(selector).__name__}
    attribute_names = (
        "channels_selected_",
        "channel_scores_",
        "scores_",
        "channel_ranking_",
        "channel_feature_proportions_",
        "removed_series_auc_",
        "removed_series_corr_",
        "clusters_",
        "cluster_labels_",
        "best_score_",
    )
    for attribute_name in attribute_names:
        value = getattr(selector, attribute_name, None)
        if value is None and selector is not transformer:
            value = getattr(transformer, attribute_name, None)
        if value is not None:
            key = attribute_name.removesuffix("_")
            selector_metadata[key] = _metadata_to_builtin(value)

    if len(selector_metadata) == 1:
        return {}
    return selector_metadata


def _hc2_metadata(classifier):
    """Return fitted HC2 weights, estimates and actual component settings."""
    if classifier is None:
        return {}

    component_names = ("stc", "drcif", "arsenal", "tde")
    weights = {}
    estimates = {}
    components = {}
    for name in component_names:
        weight = getattr(classifier, f"{name}_weight_", None)
        if weight is not None:
            weight = float(weight)
            weights[name] = weight
            estimates[name] = weight**0.25

        fitted_component = getattr(classifier, f"_{name}", None)
        component_params = getattr(classifier, f"_{name}_params", None)
        if fitted_component is not None:
            component_info = {"class": type(fitted_component).__name__}
            if component_params is not None:
                component_info["params"] = component_params
            fit_time = getattr(fitted_component, "fit_time_millis_", None)
            if fit_time is not None:
                component_info["fit_time_millis"] = fit_time
            components[name] = component_info

    if not weights and not components:
        return {}

    return _metadata_to_builtin(
        {
            "weights": weights,
            "training_accuracy_estimates": estimates,
            "components": components,
        }
    )


def _make_channel_transformer(
    selector,
    n_channels,
    proportion=0.25,
    random_state=None,
    n_jobs=1,
    proxy_component=None,
):
    """Construct a channel transform after the input channel count is known."""
    if not 0 < proportion <= 1:
        raise ValueError("proportion must be in the range (0, 1].")

    selector_key = selector.casefold()
    n_components = max(1, ceil(proportion * n_channels))

    if selector_key == "ecs":
        from aeon.transformations.collection.channel_selection import ElbowClassSum

        return ElbowClassSum()
    if selector_key == "ecp":
        from aeon.transformations.collection.channel_selection import (
            ElbowClassPairwise,
        )

        return ElbowClassPairwise()
    if selector_key == "random":
        from aeon.transformations.collection.channel_selection import (
            RandomChannelSelector,
        )

        return RandomChannelSelector(p=proportion, random_state=random_state)
    if selector_key == "tselect":
        from aeon.transformations.collection.channel_selection import TSelect

        return TSelect(random_state=random_state)
    if selector_key == "csp":
        from aeon_neuro.transformations.collection.channel_creation import (
            CommonSpacialPatterns,
        )

        return CommonSpacialPatterns(
            n_components=n_components,
            log=None,
            transform_into="csp_space",
            random_state=random_state,
        )
    if selector_key == "riemannian":
        from aeon_neuro.transformations.collection.channel_selection import Riemannian

        return Riemannian(
            proportion=proportion,
            regularization=1e-6,
            n_jobs=n_jobs,
        )
    if selector_key == "detachrocket":
        from aeon_neuro.transformations.collection.channel_selection import (
            DetachRocketChannelSelector,
        )

        return DetachRocketChannelSelector(
            proportion=proportion,
            n_kernels=2000,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if selector_key == "casetimereducer":
        from aeon_neuro.transformations.collection.channel_selection import (
            CaseTimeReducer,
        )

        return CaseTimeReducer(
            strategy="auto",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if selector_key == "guardedmultiaxis":
        from tsml_eval.experiments._guarded_multiaxis import GuardedMultiAxisReducer

        return GuardedMultiAxisReducer(
            channel_selector="tselect",
            proxy_component="auto",
            strategy="auto",
            max_score_loss=0.01,
            aggressive_fraction=0.25,
            aggressive_margin=0.0,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if selector_key == "guardedmultiaxisv2":
        from tsml_eval.experiments._guarded_multiaxis import GuardedMultiAxisReducer

        component = (
            "auto"
            if proxy_component is None or proxy_component.casefold() == "hc2"
            else proxy_component.casefold()
        )
        return GuardedMultiAxisReducer(
            channel_selector="tselect",
            proxy_component=component,
            strategy="all",
            reference="raw",
            separate_proxy_selection=True,
            evaluate_combinations=True,
            refit_channel_selector=False,
            max_score_loss=0.01,
            aggressive_fraction=0.25,
            aggressive_margin=0.0,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if selector_key == "guardedtemporalv3":
        from tsml_eval.experiments._guarded_multiaxis import GuardedMultiAxisReducer

        component = (
            "auto"
            if proxy_component is None or proxy_component.casefold() == "hc2"
            else proxy_component.casefold()
        )
        return GuardedMultiAxisReducer(
            channel_selector="tselect",
            proxy_component=component,
            strategy="time",
            reference="channel",
            raw_fallback=True,
            separate_proxy_selection=True,
            evaluate_combinations=False,
            refit_channel_selector=True,
            min_slice_timepoints=1000,
            max_score_loss=0.01,
            aggressive_fraction=0.25,
            aggressive_margin=0.0,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if selector_key == "guardedtemporalv5":
        from tsml_eval.experiments._guarded_multiaxis import GuardedMultiAxisReducer

        component = (
            "auto"
            if proxy_component is None or proxy_component.casefold() == "hc2"
            else proxy_component.casefold()
        )
        return GuardedMultiAxisReducer(
            channel_selector="detachrocket",
            proxy_component=component,
            strategy="time",
            reference="channel",
            raw_fallback=True,
            separate_proxy_selection=True,
            evaluate_combinations=False,
            refit_channel_selector=True,
            min_slice_timepoints=1000,
            max_score_loss=0.01,
            aggressive_fraction=0.25,
            aggressive_margin=0.0,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if selector_key == "guardedtemporalv4":
        component = (
            "arsenal"
            if proxy_component is None or proxy_component.casefold() == "hc2"
            else proxy_component.casefold()
        )
        return _make_gmarv4_transformer(
            component=component,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if selector_key == "cleverrank":
        from aeon_neuro.transformations.collection.channel_selection import CLeVerRank

        return CLeVerRank(n_channels=n_components)
    if selector_key == "clevercluster":
        from aeon_neuro.transformations.collection.channel_selection import (
            CLeVerCluster,
        )

        return CLeVerCluster(
            n_channels=n_components,
            random_state=random_state,
        )
    if selector_key == "cleverhybrid":
        from aeon_neuro.transformations.collection.channel_selection import (
            CLeVerHybrid,
        )

        return CLeVerHybrid(
            n_channels=n_components,
            random_state=random_state,
        )

    raise ValueError(f"Unknown channel selector: {selector}")


def _make_gmarv4_transformer(component, random_state=None, n_jobs=1):
    """Construct the component-specific guarded temporal reducer for GMARv4."""
    from tsml_eval.experiments._guarded_multiaxis import GuardedMultiAxisReducer

    component = component.casefold()
    if component not in {"arsenal", "drcif", "stc", "tde"}:
        raise ValueError(
            "GMARv4 component must be one of "
            "{'arsenal', 'drcif', 'stc', 'tde'}."
        )

    parameters = {
        "channel_selector": "tselect",
        "proxy_component": component,
        "strategy": "time",
        "reference": "channel",
        "raw_fallback": True,
        "separate_proxy_selection": True,
        "evaluate_combinations": False,
        "refit_channel_selector": True,
        "min_slice_timepoints": 1000,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }

    if component in {"arsenal", "drcif"}:
        # The empirical GMARv3 results show that both components benefit from
        # strong temporal reduction, particularly downsampling.
        parameters.update(
            time_fractions=(0.125, 0.25, 0.5, 1.0),
            slice_fractions=(0.25, 0.5, 0.75, 1.0),
            max_score_loss=0.01,
            aggressive_fraction=0.25,
            aggressive_margin=0.0,
        )
    elif component == "stc":
        # STC receives a smaller accuracy allowance and does not evaluate the
        # most aggressive temporal candidates.
        parameters.update(
            time_fractions=(0.5, 0.75, 1.0),
            slice_fractions=(0.5, 0.75, 1.0),
            max_score_loss=0.005,
            aggressive_fraction=0.5,
            aggressive_margin=0.0,
        )
    else:
        # TDE was the only component for which GMARv3 temporal reduction
        # reduced accuracy overall. Retain it for ensemble diversity, but only
        # accept a temporal reduction when its proxy score improves.
        parameters.update(
            time_fractions=(0.5, 0.75, 1.0),
            slice_fractions=(0.5, 0.75, 1.0),
            max_score_loss=0.0,
            aggressive_fraction=1.0,
            aggressive_margin=0.0025,
        )

    return GuardedMultiAxisReducer(**parameters)
