from sktime_estimator_evaluation.new_evaluation.evaluation.result_evaluation import (
    evaluate_results,
    CLUSTER_METRIC_CALLABLES,
    read_evaluation_metric_results
)

def test_cluster_evaluation():
    """Evaluate clustering results."""
    result = evaluate_results(
        experiment_name='test',
        path='ignore-results/',
        output_dir='./out/',
        metrics=CLUSTER_METRIC_CALLABLES
    )

def test_classification_evaluation():
    """Evaluate classification results."""
    pass


def test_read_evaluation_metric_results():
    """Read evaluation metric results."""
    test = read_evaluation_metric_results('./out/')
    pass