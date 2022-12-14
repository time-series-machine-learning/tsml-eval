# -*- coding: utf-8 -*-
from tsml_eval.evaluation import (
    fetch_classifier_metric,
    from_metric_summary_to_dataset_format,
)
from tsml_eval.evaluation._utils import resolve_experiment_paths


def test_resolve_experiment_paths():
    result = resolve_experiment_paths(
        "../../../evaluation/tests/dummy_results/distance-results", "test"
    )
    print(result.keys())
    joe = ""


def test_from_metric_summary_to_dataset_format():
    metric = "ACC"
    datasets = ["Chinatown", "ItalyPowerDemand"]
    classifiers = ["HC2", "InceptionTime", "ROCKET"]
    res = fetch_classifier_metric("ACC", classifiers, datasets, 6)
    test = from_metric_summary_to_dataset_format(res)
    joe = ""
