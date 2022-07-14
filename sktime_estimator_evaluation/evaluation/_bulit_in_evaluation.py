from typing import List, Union
import platform

import pandas as pd

from sktime_estimator_evaluation.evaluation import evaluate_metric_results

ListOrString = Union[List[str], str]

PATH_TO_CLASSIFICATION_RESULTS = '../../results/'

def _resolve_to_list(x: ListOrString) -> List[str]:
    if isinstance(x, str):
        return [x]
    return x

def fetch_classifier_metric(
        metrics: ListOrString,
        classifiers: ListOrString,
        datasets: ListOrString,
        folds = 30
) -> List[pd.DataFrame]:
    """Fetch the metric for a classifier over a dataset.

    Parameters
    ----------
    metric: str
        The metric to fetch.
    classifiers: str or list of str
        The classifier to fetch the metric for.
    datasets: str or list of str
        The dataset to fetch the metric for.
    folds: int
        The number of folds to use for the evaluation. NOTE: folds are 0 indexing
        so if you ask for '6' youll get folds 0-5 (i.e. 6 folds).
    """
    metrics = _resolve_to_list(metrics)
    datasets = _resolve_to_list(datasets)
    classifiers = _resolve_to_list(classifiers)

    def custom_classification(path: str):
        # Check os to determine split value
        if 'Windows' in platform.platform():
            split_subdir = path.split('\\')
        else:
            split_subdir = path.split('/')
        metric_name = 'ACC'
        file_name_split = split_subdir[-1].split('_')
        estimator_name = file_name_split[0]
        split = file_name_split[0].split('FOLDS')[0].lower()
        return estimator_name, metric_name, split

    classification_results = evaluate_metric_results(
        PATH_TO_CLASSIFICATION_RESULTS, custom_classification
    )

    temp = []
    for metric in metrics:
        for result in classification_results:
            if result['metric_name'] == metric:
                for estimator_result in result['test_estimator_results']:
                    uno = ''
                    for classifier in classifiers:
                        dose = ''
                        if classifier == estimator_result['estimator_name']:
                            tres = ''
                            df = estimator_result['result']
                            curr = df[df[df.columns[0]].isin(datasets)]
                            curr = curr.iloc[:, 0:folds+1]
                            temp.append(curr)
                            break

                break
    return temp

if __name__ == '__main__':
    metric = 'ACC'
    datasets = ["Chinatown", "ItalyPowerDemand"]
    classifiers = ["HC2", "InceptionTime", "ROCKET"]
    fetch_classifier_metric('ACC', classifiers, datasets, 6)







