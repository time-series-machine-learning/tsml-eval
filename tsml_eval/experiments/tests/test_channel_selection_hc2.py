"""Tests for the EEG channel-selection HC2 classifier pipelines."""

import sys
from types import ModuleType

import numpy as np
import pytest
from aeon.classification.base import BaseClassifier

from tsml_eval.experiments import get_classifier_by_name
from tsml_eval.experiments._channel_selection_hc2 import (
    ChannelSelectionClassifierPipeline,
    _make_channel_transformer,
)
from tsml_eval.experiments._guarded_multiaxis import GuardedMultiAxisReducer


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


def test_guarded_multiaxis_transformer_is_local_to_tsml_eval():
    """The experimental reducer does not require an aeon-neuro import."""
    transformer = _make_channel_transformer(
        selector="GuardedMultiAxis",
        n_channels=4,
        random_state=0,
        n_jobs=1,
    )

    assert isinstance(transformer, GuardedMultiAxisReducer)
