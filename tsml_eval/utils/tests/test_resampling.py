# -*- coding: utf-8 -*-
from sktime.datasets import load_arrow_head, load_unit_test

from tsml_eval.utils.experiments import resample


def test_resample():
    train_X, train_y = load_arrow_head(split="train")
    test_X, test_y = load_unit_test(split="train")

    train_size = train_y.size
    test_size = test_y.size

    train_X, train_y, test_X, test_y = resample(train_X, train_y, test_X, test_y, 1)

    assert train_y.size == train_size and test_y.size == test_size
