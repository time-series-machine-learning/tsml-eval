"""Component-aware guarded temporal reduction for HIVE-COTE 2.0."""

__maintainer__ = ["TonyBagnall"]

import time

import numpy as np
from aeon.classification.convolution_based import Arsenal
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.interval_based import DrCIFClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.utils.validation import check_n_jobs
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from tsml_eval.experiments._channel_selection_hc2 import (
    _hc2_metadata,
    _make_gear_transformer,
    _make_gmarv4_transformer,
    _metadata_to_builtin,
    _selector_metadata,
)

__all__ = ["ComponentAwareGEARHIVECOTEV2", "ComponentAwareGMARHIVECOTEV2"]


class ComponentAwareGMARHIVECOTEV2(HIVECOTEV2):
    """HIVE-COTE 2.0 with one learned guarded view per component.

    Each HC2 component fits a GMARv4 reducer using a lightweight proxy for that
    same component. The component is then fitted and queried only on its own
    learned channel/time representation. Standard HC2 train-estimate weighting
    and probability combination are retained.

    ``component_estimators`` is an experimental injection point primarily used
    for lightweight testing. If supplied, it must contain estimators named
    ``"stc"``, ``"drcif"``, ``"arsenal"`` and ``"tde"``.
    """

    _reducer_name = "GMARv4"

    def __init__(
        self,
        stc_params=None,
        drcif_params=None,
        arsenal_params=None,
        tde_params=None,
        time_limit_in_minutes=0,
        save_component_probas=False,
        verbose=0,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
        component_estimators=None,
    ):
        self.component_estimators = component_estimators
        super().__init__(
            stc_params=stc_params,
            drcif_params=drcif_params,
            arsenal_params=arsenal_params,
            tde_params=tde_params,
            time_limit_in_minutes=time_limit_in_minutes,
            save_component_probas=save_component_probas,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

    def _fit(self, X, y):
        """Fit four independently reduced HC2 components."""
        self._n_jobs = check_n_jobs(self.n_jobs)
        self.component_reducers_ = {}
        self.component_timings_millis_ = {}
        self.component_train_input_shapes_ = {}
        self.component_train_output_shapes_ = {}
        self.component_test_input_shapes_ = {}
        self.component_test_output_shapes_ = {}
        self.component_weights_ = {}
        self.component_names_ = []
        self.fitted_estimators_ = []
        self.weights_ = []

        components = self._make_components()
        for name in ("stc", "drcif", "arsenal", "tde"):
            reducer = self._make_component_reducer(
                component=name,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            )
            self.component_reducers_[name] = reducer
            self.component_train_input_shapes_[name] = tuple(
                int(value) for value in X.shape
            )

            start = time.perf_counter_ns()
            Xt, yt = reducer.fit_resample(X, y)
            transform_fit = (time.perf_counter_ns() - start) / 1_000_000
            if len(yt) != len(y) or not np.array_equal(yt, y):
                raise RuntimeError(
                    f"{self._reducer_name} temporal reducers must retain all "
                    "training labels "
                    "in their original order."
                )
            self.component_train_output_shapes_[name] = tuple(
                int(value) for value in Xt.shape
            )

            component = components[name]
            start = time.perf_counter_ns()
            train_predictions = component.fit_predict(Xt, yt)
            classifier_fit = (time.perf_counter_ns() - start) / 1_000_000
            weight = accuracy_score(yt, train_predictions) ** 4

            setattr(self, f"_{name}", component)
            self._store_component_weight(name, component, weight)
            self.component_timings_millis_[name] = {
                "transform_fit": transform_fit,
                "classifier_fit": classifier_fit,
                "transform_predict": 0.0,
                "classifier_predict": 0.0,
            }

            if self.verbose > 0:
                print(f"{self._reducer_name} {name} weight = {weight}")  # noqa: T201

        return self

    def _make_component_reducer(self, component, random_state, n_jobs):
        """Construct the historical GMARv4 reducer for one HC2 component."""
        return _make_gmarv4_transformer(
            component=component,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def _predict_proba(self, X, return_component_probas=False):
        """Combine probabilities obtained from each component-specific view."""
        probabilities, component_probabilities = self._component_probabilities(X)
        if self.save_component_probas or return_component_probas:
            self.component_probas = component_probabilities
        return probabilities

    def predict_proba_with_components(self, X):
        """Return ensemble and component probabilities from their learned views."""
        self._check_is_fitted()
        return self._component_probabilities(X)

    def _component_probabilities(self, X):
        """Compute component probabilities and their weighted combination."""
        distributions = np.zeros((X.shape[0], self.n_classes_))
        component_probabilities = {}

        for name in ("stc", "drcif", "arsenal", "tde"):
            reducer = self.component_reducers_[name]
            self.component_test_input_shapes_[name] = tuple(
                int(value) for value in X.shape
            )

            start = time.perf_counter_ns()
            Xt = reducer.transform(X)
            transform_predict = (time.perf_counter_ns() - start) / 1_000_000
            self.component_test_output_shapes_[name] = tuple(
                int(value) for value in Xt.shape
            )

            component = getattr(self, f"_{name}")
            start = time.perf_counter_ns()
            probabilities = component.predict_proba(Xt)
            classifier_predict = (time.perf_counter_ns() - start) / 1_000_000
            self.component_timings_millis_[name][
                "transform_predict"
            ] += transform_predict
            self.component_timings_millis_[name][
                "classifier_predict"
            ] += classifier_predict

            weight = self.component_weights_[name]
            distributions += probabilities * weight
            component_probabilities[_COMPONENT_DISPLAY_NAMES[name]] = probabilities

        totals = distributions.sum(axis=1, keepdims=True)
        final_probabilities = np.divide(
            distributions,
            totals,
            out=np.full_like(distributions, 1 / self.n_classes_),
            where=totals != 0,
        )
        return final_probabilities, component_probabilities

    def _store_component_weight(self, name, component, weight):
        """Store weights using both current and legacy aeon representations."""
        self.component_weights_[name] = weight
        self.component_names_.append(_COMPONENT_DISPLAY_NAMES[name])
        self.fitted_estimators_.append(component)
        self.weights_.append(weight)

        attribute_name = f"{name}_weight_"
        descriptor = getattr(type(self), attribute_name, None)
        if not isinstance(descriptor, property) or descriptor.fset is not None:
            setattr(self, attribute_name, weight)

    def get_component_weights(self):
        """Return fitted weights using aeon's public HC2 component names."""
        return dict(zip(self.component_names_, self.weights_))

    def _make_components(self):
        """Construct exact HC2-budget components or supplied test components."""
        if self.component_estimators is not None:
            required = {"stc", "drcif", "arsenal", "tde"}
            missing = required.difference(self.component_estimators)
            if missing:
                raise ValueError(
                    "component_estimators is missing: "
                    + ", ".join(sorted(missing))
                )
            return {
                name: clone(self.component_estimators[name])
                for name in required
            }

        stc_params = (
            {"n_shapelet_samples": HIVECOTEV2._DEFAULT_N_SHAPELETS}
            if self.stc_params is None
            else dict(self.stc_params)
        )
        drcif_params = (
            {"n_estimators": HIVECOTEV2._DEFAULT_N_TREES}
            if self.drcif_params is None
            else dict(self.drcif_params)
        )
        arsenal_params = (
            {
                "n_kernels": HIVECOTEV2._DEFAULT_N_KERNELS,
                "n_estimators": HIVECOTEV2._DEFAULT_N_ESTIMATORS,
            }
            if self.arsenal_params is None
            else dict(self.arsenal_params)
        )
        tde_params = (
            {
                "n_parameter_samples": HIVECOTEV2._DEFAULT_N_PARA_SAMPLES,
                "max_ensemble_size": HIVECOTEV2._DEFAULT_MAX_ENSEMBLE_SIZE,
                "randomly_selected_params": HIVECOTEV2._DEFAULT_RAND_PARAMS,
            }
            if self.tde_params is None
            else dict(self.tde_params)
        )

        if self.time_limit_in_minutes > 0:
            component_contract = self.time_limit_in_minutes / 6
            for parameters in (
                stc_params,
                drcif_params,
                arsenal_params,
                tde_params,
            ):
                parameters["time_limit_in_minutes"] = component_contract

        self._stc_params = stc_params
        self._drcif_params = drcif_params
        self._arsenal_params = arsenal_params
        self._tde_params = tde_params
        return {
            "stc": ShapeletTransformClassifier(
                **stc_params,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            ),
            "drcif": DrCIFClassifier(
                **drcif_params,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            ),
            "arsenal": Arsenal(
                **arsenal_params,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            ),
            "tde": TemporalDictionaryEnsemble(
                **tde_params,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            ),
        }

    def get_experiment_metadata(self):
        """Return component reductions, shapes, weights and split timings."""
        if not hasattr(self, "component_reducers_"):
            return {}

        components = {}
        for name, reducer in self.component_reducers_.items():
            reduction_summary = reducer.get_reduction_summary()
            reduction_summary = {
                key: value
                for key, value in reduction_summary.items()
                if key not in {"case_indices", "time_indices"}
            }
            components[name] = {
                "transformer_class": type(reducer).__name__,
                "selector": _selector_metadata(reducer),
                "reduction_summary": reduction_summary,
                "reduction_candidates": _reduction_candidate_records(reducer),
                "train_input_shape": self.component_train_input_shapes_[name],
                "train_output_shape": self.component_train_output_shapes_[name],
                "test_input_shape": self.component_test_input_shapes_.get(name),
                "test_output_shape": self.component_test_output_shapes_.get(name),
                "timings_ms": self.component_timings_millis_[name],
            }

        transform_fit = sum(
            values["transform_fit"]
            for values in self.component_timings_millis_.values()
        )
        classifier_fit = sum(
            values["classifier_fit"]
            for values in self.component_timings_millis_.values()
        )
        transform_predict = sum(
            values["transform_predict"]
            for values in self.component_timings_millis_.values()
        )
        classifier_predict = sum(
            values["classifier_predict"]
            for values in self.component_timings_millis_.values()
        )
        return _metadata_to_builtin(
            {
                "classifier_class": type(self).__name__,
                "component_aware_reduction": True,
                "timings_ms": {
                    "transform_fit": transform_fit,
                    "classifier_fit": classifier_fit,
                    "hc2_fit": transform_fit + classifier_fit,
                    "transform_predict": transform_predict,
                    "classifier_predict": classifier_predict,
                    "hc2_predict": transform_predict + classifier_predict,
                },
                "components": components,
                "hc2": _hc2_metadata(self),
            }
        )


class ComponentAwareGEARHIVECOTEV2(ComponentAwareGMARHIVECOTEV2):
    """HIVE-COTE 2.0 using a tailored GEAR reduction for each component."""

    _reducer_name = "GEAR-Comp"

    def __init__(
        self,
        stc_params=None,
        drcif_params=None,
        arsenal_params=None,
        tde_params=None,
        time_limit_in_minutes=0,
        save_component_probas=False,
        verbose=0,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
        component_estimators=None,
    ):
        super().__init__(
            stc_params=stc_params,
            drcif_params=drcif_params,
            arsenal_params=arsenal_params,
            tde_params=tde_params,
            time_limit_in_minutes=time_limit_in_minutes,
            save_component_probas=save_component_probas,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            component_estimators=component_estimators,
        )

    def _make_component_reducer(self, component, random_state, n_jobs):
        """Construct the component-specific GEAR reducer."""
        return _make_gear_transformer(
            component=component,
            random_state=random_state,
            n_jobs=n_jobs,
        )


def _reduction_candidate_records(reducer):
    """Return a compact serialisable candidate trace for one component."""
    candidate_results = getattr(reducer, "candidate_results_", None)
    if candidate_results is None:
        return []

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
        column for column in trace_columns if column in candidate_results.columns
    ]
    records = candidate_results[available].to_dict("records")
    for record in records:
        if "fit_time" in record:
            record["fit_time_seconds"] = record.pop("fit_time")
        if "predict_time" in record:
            record["predict_time_seconds"] = record.pop("predict_time")
    return records


_COMPONENT_DISPLAY_NAMES = {
    "stc": "STC",
    "drcif": "DrCIF",
    "arsenal": "Arsenal",
    "tde": "TDE",
}
