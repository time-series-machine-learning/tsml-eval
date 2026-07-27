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


@pytest.mark.parametrize(
    "classifier_name, expected_class, parameter, expected_value",
    [
        ("ECS-Arsenal", "Arsenal", "n_kernels", 2000),
        ("ECS-DrCIF", "DrCIFClassifier", "n_estimators", 500),
        ("ECS-STC", "ShapeletTransformClassifier", "n_shapelet_samples", 10000),
        ("ECS-TDE", "TemporalDictionaryEnsemble", "n_parameter_samples", 250),
        ("GMARv2-Arsenal", "Arsenal", "n_kernels", 2000),
        ("GMARv3-Arsenal", "Arsenal", "n_kernels", 2000),
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
    else:
        expected_selector = "ECS"
    assert pipeline.selector == expected_selector
    assert type(pipeline.classifier).__name__ == expected_class
    assert pipeline.classifier.get_params()[parameter] == expected_value
    if classifier_name.startswith(("GMARv2", "GMARv3")):
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
