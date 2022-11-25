# -*- coding: utf-8 -*-
__author__ = ["MatthewMiddlehurst"]

import os

from tsml_estimator_evaluation.experiments.classification_experiments import (
    run_experiment,
)


def test_run_experiment():
    result_path = "../../../test_output/classification/"
    classifier = "DummyClassifier"
    dataset = "UnitTest"
    args = [
        None,
        "../../data/",
        result_path,
        classifier,
        dataset,
        "1",
        "True",
        "False",
    ]
    run_experiment(args, overwrite=True)

    test_file = (
        result_path + classifier + "/Predictions/" + dataset + "/testResample0.csv"
    )
    train_file = (
        result_path + classifier + "/Predictions/" + dataset + "/trainResample0.csv"
    )

    assert os.path.exists(test_file) and os.path.exists(train_file)

    os.remove(test_file)
    os.remove(train_file)
