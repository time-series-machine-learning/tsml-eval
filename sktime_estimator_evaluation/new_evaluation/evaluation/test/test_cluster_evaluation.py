from sktime_estimator_evaluation.new_evaluation.evaluation.cluster_evaluation import evaluate_cluster_results

def test_cluster_evaluation():
    result = evaluate_cluster_results(
        experiment_name='test',
        path='../../../evaluation/tests/dummy_results/distance-results',
        output_dir='./out/'
    )
