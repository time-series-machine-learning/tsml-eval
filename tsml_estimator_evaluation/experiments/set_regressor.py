# -*- coding: utf-8 -*-
"""Set regressor function."""
__author__ = ["TonyBagnall"]


from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.regression.kernel_based import RocketRegressor
from sktime.regression.interval_based import TimeSeriesForestRegressor
from tsml_estimator_evaluation.experiments.classification_experiments import list_estimators
from sktime.registry import all_estimators

cls = all_estimators(estimator_types="regressor")
names = [i for i, _ in cls]
print(names)


def set_regressor(regressor, resample_id=None, train_file=False, n_jobs=1):
    """Construct a classifier, possibly seeded.

    Basic way of creating the classifier to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducibility for use with load_and_run_classification_experiment. You can pass a
    classifier object instead to run_classification_experiment.
    TODO: add contract and checkpoint options

    Parameters
    ----------
    regressor : str
        String indicating which Regressor you want.
    resample_id : int or None, default=None
        Classifier random seed.
    train_file : bool, default=False
        Whether a train file is being produced.

    Return
    ------
    classifier : A BaseClassifier.
        The classifier matching the input classifier name.
    """
    name = regressor.lower()
    if name == "cnn" or name == "cnnregressor":
        from sktime.regression.deep_learning.cnn import CNNRegressor
        return CNNRegressor(random_state=resample_id)
    elif name == "tapnet" or name == "tapnetregressor":
            from sktime.regression.deep_learning.tapnet import TapNetRegressor
            return TapNetRegressor(random_state=resample_id)
    elif name == "knn" or name == "kneighborstimeseriesregressor":
        return KNeighborsTimeSeriesRegressor(
            random_state=resample_id,
            n_jobs=n_jobs,
        )
    elif name == "rocket" or name == "rocketregressor":
        return RocketRegressor(
            random_state=resample_id,
            n_jobs=n_jobs,
        )
    elif name == "tsf" or name == "timeseriesforestregressor":
        return TimeSeriesForestRegressor(
            random_state=resample_id,
            n_jobs=n_jobs,
        )
