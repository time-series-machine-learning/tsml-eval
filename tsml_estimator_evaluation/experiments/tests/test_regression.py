# -*- coding: utf-8 -*-
"""Tests for regression utilities."""

from sktime.datasets import load_unit_test
from tsml_estimator_evaluation.experiments.regression_experiments import resample


def test_resample():
    trainX, trainy = load_unit_test(split="train")
    testX, testy = load_unit_test(split="train")
    train_size = trainy.size
    test_size = testy.size
    trainX, trainy, testX, testy = resample(trainX, trainy, testX, testy, 1)
    assert trainy.size == train_size and testy.size == test_size
