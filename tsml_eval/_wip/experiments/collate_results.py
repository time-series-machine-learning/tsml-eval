from tsml_eval.evaluation.multiple_estimator_evaluation import \
     evaluate_clusterers_by_problem, evaluate_classifiers_by_problem, evaluate_regressors_by_problem
from aeon.datasets.tsc_data_lists import univariate_equal_length

resamples = 30
results = "/gpfs/home/ajb/ResultsWorkingArea/STC_compare/STC_COMPARE/"
names = ["MAIN", "STC", "REF"]

evaluate_classifiers_by_problem(
     load_path=results,
     classifier_names=names,
     dataset_names=list(univariate_equal_length),
     resamples=30,
     eval_name="cls_test",
     save_path=results,
     verify_results=False,
     error_on_missing=False
)
