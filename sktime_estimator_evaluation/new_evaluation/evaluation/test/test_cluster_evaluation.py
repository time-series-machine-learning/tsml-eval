from sktime_estimator_evaluation.new_evaluation.evaluation.result_evaluation import (
    evaluate_results,
    CLUSTER_METRIC_CALLABLES
)

def test_cluster_evaluation():
    result = evaluate_results(
        experiment_name='test',
        path='ignore-results/',
        output_dir='./out/',
        metrics=CLUSTER_METRIC_CALLABLES
    )


