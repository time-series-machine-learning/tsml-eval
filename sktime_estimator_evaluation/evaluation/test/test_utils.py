from sktime_estimator_evaluation.evaluation._utils import resolve_experiment_paths

def test_resolve_experiment_paths():
    result = resolve_experiment_paths('../../../evaluation/tests/dummy_results/distance-results', 'test')
    print(result.keys())
    joe = ''