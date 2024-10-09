"""Collating results example."""

from tsml_eval.evaluation.multiple_estimator_evaluation import \
     evaluate_clusterers_by_problem, evaluate_classifiers_by_problem, evaluate_regressors_by_problem
from aeon.datasets.tsc_data_lists import univariate_equal_length

local_results = "C:\\ResultsWorkingArea\\"
ada_results = "/gpfs/home/ajb/ResultsWorkingArea/STC_compare/STC_COMPARE/"
ada_names = ["MAIN", "STC", "REF"]
local_names =["kmeans-erp","kmedoids-edr","kmedoids-erp"]
resamples = 30


def classifier_results(results, names, resamples=30, problems=univariate_equal_length):
    evaluate_classifiers_by_problem(
        load_path=results,
        classifier_names=names,
        dataset_names=list(problems),
        resamples=resamples,
        eval_name="cls_eval",
        save_path=results,
        verify_results=False,
        error_on_missing=False
    )

def regressor_results(results, names, resamples=30, problems=univariate_equal_length):
    evaluate_regressors_by_problem(
        load_path=results,
        regressor_names=names,
        dataset_names=list(univariate_equal_length),
        resamples=resamples,
        eval_name="reg_eval",
        save_path=results,
        verify_results=False,
        error_on_missing=False
    )


def clustering_results(results, names, resamples=10, problems=univariate_equal_length):
    evaluate_clusterers_by_problem(
        load_path=results,
        clusterer_names=names,
        dataset_names=list(univariate_equal_length),
        resamples=resamples,
        eval_name="reg_eval",
        save_path=results,
        verify_results=False,
        error_on_missing=False
    )


def extract_reference_results():
     classifier_names=names
