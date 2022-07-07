from sktime_estimator_evaluation.new_evaluation.evaluation.cluster_evaluation import evaluate_cluster_results

def test_cluster_evaluation():
    result = evaluate_cluster_results(
        experiment_name='test',
        path='ignore-results/',
        output_dir='./out/'
    )


