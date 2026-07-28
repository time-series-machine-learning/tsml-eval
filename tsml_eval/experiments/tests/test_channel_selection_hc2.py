"""Tests for the EEG channel-selection HC2 classifier pipelines."""

import json
import sys
from types import ModuleType

import numpy as np
import pytest
from aeon.classification.base import BaseClassifier

from tsml_eval.evaluation.storage import load_classifier_results
from tsml_eval.experiments import (
    get_classifier_by_name,
    run_classification_experiment,
)
from tsml_eval.experiments._channel_selection_hc2 import (
    ChannelSelectionClassifierPipeline,
    _make_channel_transformer,
    _make_gmarv4_transformer,
)
from tsml_eval.experiments._component_aware_gmar import (
    ComponentAwareGMARHIVECOTEV2,
)
from tsml_eval.experiments._guarded_multiaxis import GuardedMultiAxisReducer
from tsml_eval.experiments.experiments import _get_estimator_parameter_info


class _RecordingClassifier(BaseClassifier):
    """Small classifier which records the number of fitted cases."""

    _tags = {"capability:multivariate": True}

    def __init__(self):
        super().__init__()

    def _fit(self, X, y):
        self.n_cases_fit_ = X.shape[0]
        return self

    def _predict(self, X):
        return np.repeat(self.classes_[0], X.shape[0])

    def _predict_proba(self, X):
        probabilities = np.zeros((X.shape[0], self.n_classes_))
        probabilities[:, 0] = 1
        return probabilities


class _HalfCaseResampler:
    """Test resampler which retains every second training case."""

    def fit_resample(self, X, y):
        self.is_fitted_ = True
        indices = np.asarray([0, 1, 4, 5, 8])
        return X[indices], y[indices]

    def transform(self, X):
        return X


class _PerfectTrainEstimateClassifier(BaseClassifier):
    """Small classifier with a deterministic perfect training estimate."""

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X, y):
        self.n_timepoints_fit_ = X.shape[2]
        return self

    def _fit_predict(self, X, y, **kwargs):
        self._fit(X, y)
        return np.asarray(y)

    def _predict(self, X):
        return np.repeat(self.classes_[0], X.shape[0])

    def _predict_proba(self, X):
        probabilities = np.zeros((X.shape[0], self.n_classes_))
        probabilities[:, 0] = 1
        return probabilities


class _ComponentTestReducer:
    """Identity-like reducer retaining a component-specific temporal length."""

    def __init__(self, component):
        self.component = component
        self.n_timepoints = {
            "stc": 5,
            "drcif": 6,
            "arsenal": 7,
            "tde": 8,
        }[component]

    def fit_resample(self, X, y):
        self.channels_selected_ = np.arange(X.shape[1])
        return X[:, :, : self.n_timepoints], y

    def transform(self, X):
        return X[:, :, : self.n_timepoints]

    def get_reduction_summary(self):
        return {
            "route": "downsample",
            "proxy_component": self.component,
            "n_timepoints_selected": self.n_timepoints,
            "case_indices": np.arange(2),
            "time_indices": np.arange(self.n_timepoints),
        }


@pytest.mark.parametrize(
    "classifier_name, selector",
    [
        ("ECS-HC2", "ECS"),
        ("ECP-HC2", "ECP"),
        ("TSelect-HC2", "TSelect"),
        ("Random-HC2", "Random"),
        ("Riemannian-HC2", "Riemannian"),
        ("DetachRocket-HC2", "DetachRocket"),
        ("CSP-HC2", "CSP"),
        ("CaseTimeReducer-HC2", "CaseTimeReducer"),
        ("GuardedMultiAxis-HC2", "GuardedMultiAxis"),
        ("GMARv2-HC2", "GuardedMultiAxisV2"),
        ("GMARv3-HC2", "GuardedTemporalV3"),
        ("CLeVerRank-HC2", "CLeVerRank"),
        ("CLeVerCluster-HC2", "CLeVerCluster"),
        ("CLeVerHybrid-HC2", "CLeVerHybrid"),
    ],
)
def test_channel_selection_hc2_factory_options(classifier_name, selector, monkeypatch):
    """Factory options construct the requested transform-plus-HC2 pipeline."""
    monkeypatch.setitem(sys.modules, "aeon_neuro", ModuleType("aeon_neuro"))

    pipeline = get_classifier_by_name(
        classifier_name,
        random_state=7,
        n_jobs=1,
    )

    assert isinstance(pipeline, ChannelSelectionClassifierPipeline)
    assert pipeline.selector == selector
    assert pipeline.proportion == 0.25
    assert pipeline.random_state == 7
    assert pipeline.classifier.random_state == 7


def test_gmarv4_hc2_factory_is_component_aware():
    """GMARv4-HC2 places a separate reducer inside every HC2 component."""
    classifier = get_classifier_by_name(
        "GMARv4-HC2",
        random_state=7,
        n_jobs=1,
    )

    assert isinstance(classifier, ComponentAwareGMARHIVECOTEV2)
    assert classifier.random_state == 7
    assert classifier.n_jobs == 1


@pytest.mark.parametrize(
    "classifier_name, expected_class, parameter, expected_value",
    [
        ("ECS-Arsenal", "Arsenal", "n_kernels", 2000),
        ("ECS-DrCIF", "DrCIFClassifier", "n_estimators", 500),
        ("ECS-STC", "ShapeletTransformClassifier", "n_shapelet_samples", 10000),
        ("ECS-TDE", "TemporalDictionaryEnsemble", "n_parameter_samples", 250),
        ("GMARv2-Arsenal", "Arsenal", "n_kernels", 2000),
        ("GMARv3-Arsenal", "Arsenal", "n_kernels", 2000),
        ("GMARv4-Arsenal", "Arsenal", "n_kernels", 2000),
    ],
)
def test_channel_selection_component_pipeline_options(
    classifier_name,
    expected_class,
    parameter,
    expected_value,
):
    """Component pipelines use the same budgets as the HC2 components."""
    pipeline = get_classifier_by_name(
        classifier_name,
        random_state=7,
        n_jobs=1,
    )

    assert isinstance(pipeline, ChannelSelectionClassifierPipeline)
    if classifier_name.startswith("GMARv2"):
        expected_selector = "GuardedMultiAxisV2"
    elif classifier_name.startswith("GMARv3"):
        expected_selector = "GuardedTemporalV3"
    elif classifier_name.startswith("GMARv4"):
        expected_selector = "GuardedTemporalV4"
    else:
        expected_selector = "ECS"
    assert pipeline.selector == expected_selector
    assert type(pipeline.classifier).__name__ == expected_class
    assert pipeline.classifier.get_params()[parameter] == expected_value
    if classifier_name.startswith(("GMARv2", "GMARv3", "GMARv4")):
        assert pipeline.proxy_component == "arsenal"


def test_full_hc2_stc_component_option():
    """The full-data STC baseline uses aeon's exact HC2 component."""
    classifier = get_classifier_by_name(
        "Full-STC",
        random_state=7,
        n_jobs=1,
    )

    assert type(classifier).__name__ == "ShapeletTransformClassifier"
    assert classifier.n_shapelet_samples == 10000


def test_resampling_pipeline_keeps_training_labels_aligned(monkeypatch):
    """A case resampler passes matching reduced X and y to its classifier."""
    monkeypatch.setattr(
        "tsml_eval.experiments._channel_selection_hc2._make_channel_transformer",
        lambda *args, **kwargs: _HalfCaseResampler(),
    )
    X = np.zeros((10, 3, 8))
    y = np.asarray([0, 1] * 5)
    pipeline = ChannelSelectionClassifierPipeline(
        selector="CaseTimeReducer",
        classifier=_RecordingClassifier(),
        random_state=0,
    )

    pipeline.fit(X, y)

    assert pipeline.classifier_.n_cases_fit_ == 5
    assert pipeline.predict_proba(X).shape == (10, 2)
    metadata = pipeline.get_experiment_metadata()
    timings = metadata["timings_ms"]
    assert timings["transform_fit"] >= 0
    assert timings["hc2_fit"] >= 0
    assert timings["transform_predict"] >= 0
    assert timings["hc2_predict"] >= 0
    assert metadata["train_input_shape"] == [10, 3, 8]
    assert metadata["train_output_shape"] == [5, 3, 8]
    assert metadata["test_input_shape"] == [10, 3, 8]
    assert metadata["test_output_shape"] == [10, 3, 8]
    parameter_info = _get_estimator_parameter_info(pipeline)
    encoded_metadata = parameter_info.split(
        " | experiment_metadata=", maxsplit=1
    )[1]
    decoded_metadata = json.loads(encoded_metadata)
    assert "transform_fit" in decoded_metadata["timings_ms"]
    assert "hc2_fit" in decoded_metadata["timings_ms"]
    assert decoded_metadata["train_class_counts_in"] == {"0": 5, "1": 5}
    assert decoded_metadata["train_class_counts_out"] == {"0": 3, "1": 2}


def test_guarded_reduction_trace_is_in_experiment_metadata(monkeypatch):
    """Guarded route, retained sizes and candidate scores are recorded compactly."""
    reducer = GuardedMultiAxisReducer(
        channel_selector="none",
        proxy_estimator=_RecordingClassifier(),
        strategy="time",
        time_fractions=(0.5, 1.0),
        slice_fractions=(1.0,),
        min_timepoints=1,
        random_state=0,
    )
    monkeypatch.setattr(
        "tsml_eval.experiments._channel_selection_hc2._make_channel_transformer",
        lambda *args, **kwargs: reducer,
    )
    X = np.zeros((10, 3, 8))
    y = np.asarray([0, 1] * 5)
    pipeline = ChannelSelectionClassifierPipeline(
        selector="GuardedMultiAxis",
        classifier=_RecordingClassifier(),
        random_state=0,
    )

    pipeline.fit(X, y)
    pipeline.predict_proba(X)
    metadata = pipeline.get_experiment_metadata()

    assert metadata["reduction_summary"]["route"] in {
        "full",
        "downsample",
        "slice",
    }
    assert metadata["reduction_summary"]["n_cases_in"] == 10
    assert metadata["channels_selected"] == [0, 1, 2]
    assert metadata["reduction_summary"]["channels_selected"] == [0, 1, 2]
    assert "case_indices" not in metadata["reduction_summary"]
    assert "time_indices" not in metadata["reduction_summary"]
    assert len(metadata["reduction_candidates"]) >= 2
    assert "fit_time_seconds" in metadata["reduction_candidates"][0]
    assert "predict_time_seconds" in metadata["reduction_candidates"][0]
    assert sum(
        candidate["selected"] for candidate in metadata["reduction_candidates"]
    ) == 1


def test_pipeline_timings_are_written_to_result_file(monkeypatch, tmp_path):
    """Fitted transform and HC2 timings are persisted in line-two metadata."""
    monkeypatch.setattr(
        "tsml_eval.experiments._channel_selection_hc2._make_channel_transformer",
        lambda *args, **kwargs: _HalfCaseResampler(),
    )
    X = np.zeros((10, 3, 8))
    y = np.asarray([0, 1] * 5)
    pipeline = ChannelSelectionClassifierPipeline(
        selector="CaseTimeReducer",
        classifier=_RecordingClassifier(),
        random_state=0,
    )

    run_classification_experiment(
        X,
        y,
        X,
        y,
        pipeline,
        str(tmp_path),
        classifier_name="TimedPipeline",
        dataset_name="Toy",
        resample_id=0,
        benchmark_time=False,
    )

    result_file = (
        tmp_path
        / "TimedPipeline"
        / "Predictions"
        / "Toy"
        / "testResample0.csv"
    )
    parameter_info = result_file.read_text().splitlines()[1]
    encoded_metadata = parameter_info.split(
        " | experiment_metadata=", maxsplit=1
    )[1]
    metadata = json.loads(encoded_metadata)
    assert metadata["run"]["experiment"]["classifier"] == "TimedPipeline"
    assert metadata["run"]["experiment"]["dataset"] == "Toy"
    assert metadata["run"]["data"]["train"]["shape"] == [10, 3, 8]
    assert metadata["run"]["data"]["train_class_counts"] == {"0": 5, "1": 5}
    assert "python" in metadata["run"]["environment"]
    assert "tsml-eval" in metadata["run"]["environment"]["packages"]
    assert "transform_fit" in metadata["timings_ms"]
    assert "hc2_fit" in metadata["timings_ms"]
    assert "transform_predict" in metadata["timings_ms"]
    assert "hc2_predict" in metadata["timings_ms"]
    loaded = load_classifier_results(str(result_file))
    assert " | experiment_metadata=" in loaded.parameter_info


def test_guarded_multiaxis_transformer_is_local_to_tsml_eval():
    """The experimental reducer does not require an aeon-neuro import."""
    transformer = _make_channel_transformer(
        selector="GuardedMultiAxis",
        n_channels=4,
        random_state=0,
        n_jobs=1,
    )

    assert isinstance(transformer, GuardedMultiAxisReducer)


def test_guarded_multiaxis_v2_uses_component_proxy():
    """GMARv2 uses the matching proxy and raw guarded integrated search."""
    transformer = _make_channel_transformer(
        selector="GuardedMultiAxisV2",
        n_channels=4,
        random_state=0,
        n_jobs=1,
        proxy_component="DrCIF",
    )

    assert isinstance(transformer, GuardedMultiAxisReducer)
    assert transformer.proxy_component == "drcif"
    assert transformer.strategy == "all"
    assert transformer.reference == "raw"
    assert transformer.separate_proxy_selection
    assert transformer.evaluate_combinations
    assert not transformer.refit_channel_selector


def test_guarded_temporal_v3_uses_channel_guard_and_long_slices():
    """GMARv3 removes case reduction and guards long-series time choices."""
    transformer = _make_channel_transformer(
        selector="GuardedTemporalV3",
        n_channels=4,
        random_state=0,
        n_jobs=1,
        proxy_component="DrCIF",
    )

    assert isinstance(transformer, GuardedMultiAxisReducer)
    assert transformer.proxy_component == "drcif"
    assert transformer.strategy == "time"
    assert transformer.reference == "channel"
    assert transformer.raw_fallback
    assert transformer.separate_proxy_selection
    assert not transformer.evaluate_combinations
    assert transformer.refit_channel_selector
    assert transformer.min_slice_timepoints == 1000


@pytest.mark.parametrize(
    "component, time_fractions, max_score_loss, aggressive_fraction",
    [
        ("Arsenal", (0.125, 0.25, 0.5, 1.0), 0.01, 0.25),
        ("DrCIF", (0.125, 0.25, 0.5, 1.0), 0.01, 0.25),
        ("STC", (0.5, 0.75, 1.0), 0.005, 0.5),
        ("TDE", (0.5, 0.75, 1.0), 0.0, 1.0),
    ],
)
def test_gmarv4_component_policies(
    component,
    time_fractions,
    max_score_loss,
    aggressive_fraction,
):
    """GMARv4 uses empirically motivated guards for each HC2 component."""
    transformer = _make_gmarv4_transformer(
        component=component,
        random_state=0,
        n_jobs=1,
    )

    assert isinstance(transformer, GuardedMultiAxisReducer)
    assert transformer.proxy_component == component.casefold()
    assert transformer.strategy == "time"
    assert transformer.time_fractions == time_fractions
    assert transformer.max_score_loss == max_score_loss
    assert transformer.aggressive_fraction == aggressive_fraction
    assert transformer.raw_fallback
    assert transformer.min_slice_timepoints == 1000


def test_component_aware_gmar_fits_distinct_views(monkeypatch):
    """Each HC2 component is fitted and queried on its own learned view."""
    monkeypatch.setattr(
        "tsml_eval.experiments._component_aware_gmar."
        "_make_gmarv4_transformer",
        lambda component, **kwargs: _ComponentTestReducer(component),
    )
    component_estimators = {
        name: _PerfectTrainEstimateClassifier()
        for name in ("stc", "drcif", "arsenal", "tde")
    }
    classifier = ComponentAwareGMARHIVECOTEV2(
        component_estimators=component_estimators,
        random_state=0,
        n_jobs=1,
    )
    X = np.zeros((12, 2, 8))
    y = np.asarray([0, 1] * 6)

    classifier.fit(X, y)
    probabilities = classifier.predict_proba(X)
    combined, component_probabilities = classifier.predict_proba_with_components(
        X
    )
    metadata = classifier.get_experiment_metadata()

    assert probabilities.shape == (12, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert np.allclose(combined, probabilities)
    assert set(component_probabilities) == {"STC", "DrCIF", "Arsenal", "TDE"}
    assert {
        name: getattr(classifier, f"_{name}").n_timepoints_fit_
        for name in ("stc", "drcif", "arsenal", "tde")
    } == {
        "stc": 5,
        "drcif": 6,
        "arsenal": 7,
        "tde": 8,
    }
    assert all(
        getattr(classifier, f"{name}_weight_") == 1
        for name in ("stc", "drcif", "arsenal", "tde")
    )
    assert metadata["component_aware_reduction"]
    assert set(metadata["components"]) == {
        "stc",
        "drcif",
        "arsenal",
        "tde",
    }
    assert metadata["components"]["stc"]["test_output_shape"] == [12, 2, 5]
    assert metadata["components"]["tde"]["test_output_shape"] == [12, 2, 8]
    assert "case_indices" not in metadata["components"]["stc"][
        "reduction_summary"
    ]
    assert metadata["timings_ms"]["hc2_fit"] >= 0
    assert metadata["timings_ms"]["hc2_predict"] >= 0


def test_component_weights_support_read_only_aeon_properties(monkeypatch):
    """New aeon HC2 weight properties are backed by names and weight lists."""
    classifier = ComponentAwareGMARHIVECOTEV2(random_state=0)
    classifier.component_weights_ = {}
    classifier.component_names_ = []
    classifier.fitted_estimators_ = []
    classifier.weights_ = []
    monkeypatch.setattr(
        ComponentAwareGMARHIVECOTEV2,
        "stc_weight_",
        property(
            lambda self: self.get_component_weights().get("STC", 0.0)
        ),
        raising=False,
    )
    component = _PerfectTrainEstimateClassifier()

    classifier._store_component_weight("stc", component, 0.625)

    assert classifier.stc_weight_ == 0.625
    assert classifier.component_names_ == ["STC"]
    assert classifier.fitted_estimators_ == [component]
    assert classifier.weights_ == [0.625]
