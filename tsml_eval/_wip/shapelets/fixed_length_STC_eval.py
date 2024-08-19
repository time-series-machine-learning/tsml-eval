from tsml_eval.evaluation import evaluate_classifiers_by_problem

classifiers = ["stc", "fixedlengthshapelettransformclassifier"]
datasets = "/mainfs/home/ajb2u23/DataSetLists/TSC_112_2019.txt" 

evaluate_classifiers_by_problem(
    "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/results/",
    classifiers,
    datasets,
    "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/evaluated_results/",
    resamples=1,
    eval_name="FixedLengthEval",
)