"""Collating reference results."""

import os

import numpy as np
import pandas as pd
from aeon.benchmarking import get_estimator_results
from aeon.classification import DummyClassifier
from aeon.datasets import load_classification
from aeon.visualisation import plot_critical_difference
from sklearn.metrics import accuracy_score
from tsml.datasets import load_minimal_chinatown

from tsml_eval.evaluation.storage import load_classifier_results
from tsml_eval.experiments import (
    experiments,
    get_classifier_by_name,
    run_classification_experiment,
)

from tsml_eval.evaluation import evaluate_classifiers_by_problem

local_results = "C:\\ResultsWorkingArea\\"
hpc_results =  "/mainfs/lyceum/ik2g21/aeon/ReferenceResults/results/"
save_path =  "/mainfs/lyceum/ik2g21/aeon/ReferenceResults/evaluated_results/"
hpc_names = ["proximityforest","redcomets"]
shapelet_classifier = ["STC"]
distance_based = ["PF"]
resamples = 30


# evaluate_classifiers_by_problem(
#         "/mainfs/lyceum/ik2g21/aeon/ReferenceResults/results/",
#         "PF",
#         "/mainfs/lyceum/ik2g21/aeon/ReferenceResults/evaluated_results/",
#         resamples=30,
#         eval_name="proximityforest_eval",
#     )

# evaluate_classifiers_by_problem(
#         "/mainfs/lyceum/ik2g21/aeon/ReferenceResults/results/",
#         "redcomets",
#         "/mainfs/lyceum/ik2g21/aeon/ReferenceResults/evaluated_results/",
#         resamples=30,
#         eval_name="redcomets_eval",
#     )

pf_res = load_classifier_results(
    # "/mainfs/lyceum/ik2g21/aeon/ReferenceResults/evaluated_results/proximityforest_eval/Accuracy/accuracy_mean.csv"
    #TODO: provide file path of each resample
)

redcomets_res = load_classifier_results(
#    "/mainfs/lyceum/ik2g21/aeon/ReferenceResults/evaluated_results/redcomets_eval/Accuracy/accuracy_mean.csv"
    #TODO: provide file path of each resample
)

shapelet_benchmark = ["STC"]
distance_benchmark = ["PF"]

shapelet_res = get_estimator_results(
    estimators=shapelet_benchmark, task="classification", measure="accuracy"
)

shapelet_res["redcomets"] = redcomets_res

shapelet_table = pd.DataFrame(shapelet_res)
shapelet_plt, _ = plot_critical_difference(np.array(shapelet_table), list(shapelet_table.columns))
shapelet_plt.show()

distance_res = get_estimator_results(
    estimators=distance_benchmark, task="classification", measure="accuracy"
)

distance_res["proximityforest"] = pf_res

distance_table = pd.DataFrame(shapelet_res)
distance_plt, _ = plot_critical_difference(np.array(distance_table), list(distance_table.columns))
distance_plt.show()


evaluate_classifiers_by_problem(
    load_path=results,
    classifier_names=names,
    resamples=resamples,
    eval_name="pf_summary",
    save_path=results,
    verify_results=False,
    error_on_missing=False
)
    



resamples_all, data_names = get_estimator_results_as_array(
    estimators=classifiers, default_only=False
)
print("Results are averaged over 30 stratified resamples.")
print(
    f"{resamples_all[3][1]}"
)



paired_sorted = sorted(zip(resamples_all, data_names))
names, _ = zip(*paired_sorted)
sorted_rows = [row for _, row in paired_sorted]
sorted_results = np.array(sorted_rows)
print(names)
print(sorted_results)