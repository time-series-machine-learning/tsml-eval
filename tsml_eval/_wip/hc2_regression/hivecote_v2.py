"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V2.

Upgraded hybrid ensemble of classifiers from 4 separate time series classification
representations, using the weighted probabilistic CAWPE as an ensemble controller.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["HIVECOTEV2"]

from datetime import datetime

import numpy as np
from aeon.regression.base import BaseRegressor
from aeon.regression.interval_based import DrCIFRegressor
from sklearn.metrics import mean_squared_error

from tsml_eval._wip.hc2_regression.arsenal import Arsenal
from tsml_eval._wip.hc2_regression.str import ShapeletTransformRegressor
from tsml_eval._wip.hc2_regression.tde import TemporalDictionaryEnsemble


class HIVECOTEV2(BaseRegressor):
    """Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V2.

    An ensemble of the STC, DrCIF, Arsenal and TDE classifiers from different feature
    representations using the CAWPE structure as described in [1].

    Parameters
    ----------
    stc_params : dict or None, default=None
        Parameters for the ShapeletTransformClassifier module. If None, uses the
        default parameters with a 2 hour transform contract.
    drcif_params : dict or None, default=None
        Parameters for the DrCIF module. If None, uses the default parameters with
        n_estimators set to 500.
    arsenal_params : dict or None, default=None
        Parameters for the Arsenal module. If None, uses the default parameters.
    tde_params : dict or None, default=None
        Parameters for the TemporalDictionaryEnsemble module. If None, uses the default
        parameters.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding
        n_estimators/n_parameter_samples for each component.
        Default of 0 means n_estimators/n_parameter_samples for each component is used.
    save_component_probas : bool, default=False
        When predict/predict_proba is called, save each HIVE-COTEV2 component
        probability predictions in component_probas.
    verbose : int, default=0
        Level of output printed to the console (for information only).
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The unique class labels.
    stc_weight_ : float
        The weight for STC probabilities.
    drcif_weight_ : float
        The weight for DrCIF probabilities.
    arsenal_weight_ : float
        The weight for Arsenal probabilities.
    tde_weight_ : float
        The weight for TDE probabilities.
    component_probas : dict
        Only used if save_component_probas is true. Saved probability predictions for
        each HIVE-COTEV2 component.

    See Also
    --------
    HIVECOTEV1, ShapeletTransformClassifier, DrCIF, Arsenal, TemporalDictionaryEnsemble

    Notes
    -----
    For the Java version, see
    `https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/hybrids/HIVE_COTE.java`_.

    References
    ----------
    .. [1] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." Machine Learning (2021).
    """

    _tags = {
        "capability:multivariate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "classifier_type": "hybrid",
    }

    def __init__(
        self,
        stc_params=None,
        drcif_params=None,
        arsenal_params=None,
        tde_params=None,
        time_limit_in_minutes=0,
        save_component_preds=False,
        verbose=0,
        n_jobs=1,
        random_state=None,
    ):
        self.stc_params = stc_params
        self.drcif_params = drcif_params
        self.arsenal_params = arsenal_params
        self.tde_params = tde_params

        self.time_limit_in_minutes = time_limit_in_minutes

        self.save_component_preds = save_component_preds
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.stc_weight_ = 0
        self.drcif_weight_ = 0
        self.arsenal_weight_ = 0
        self.tde_weight_ = 0
        self.component_probas = {}

        self._stc_params = stc_params
        self._drcif_params = drcif_params
        self._arsenal_params = arsenal_params
        self._tde_params = tde_params
        self._stc = None
        self._drcif = None
        self._arsenal = None
        self._tde = None

        super().__init__()

    def _fit(self, X, y):
        """Fit HIVE-COTE 2.0 to training data.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        # Default values from HC2 paper
        if self.stc_params is None:
            self._stc_params = {"transform_limit_in_minutes": 120}
        if self.drcif_params is None:
            self._drcif_params = {"n_estimators": 500}
        if self.arsenal_params is None:
            self._arsenal_params = {}
        if self.tde_params is None:
            self._tde_params = {}

        # If we are contracting split the contract time between each algorithm
        if self.time_limit_in_minutes > 0:
            # Leave 1/3 for train estimates
            ct = self.time_limit_in_minutes / 6
            self._stc_params["time_limit_in_minutes"] = ct
            self._drcif_params["time_limit_in_minutes"] = ct
            self._arsenal_params["time_limit_in_minutes"] = ct
            self._tde_params["time_limit_in_minutes"] = ct

        # Build STC
        self._stc = ShapeletTransformRegressor(
            **self._stc_params,
            save_transformed_data=True,
            random_state=self.random_state,
            n_jobs=self._threads_to_use,
        )
        self._stc.fit(X, y)

        if self.verbose > 0:
            print("STC ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        # Find STC weight using train set estimate
        train_preds = self._stc._get_train_preds(X, y)
        self.stc_weight_ = mean_squared_error(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "STC train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("STC weight = " + str(self.stc_weight_))  # noqa

        # Build DrCIF
        self._drcif = DrCIFRegressor(
            **self._drcif_params,
            save_transformed_data=True,
            random_state=self.random_state,
            n_jobs=self._threads_to_use,
        )
        self._drcif.fit(X, y)

        if self.verbose > 0:
            print("DrCIF ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        # Find DrCIF weight using train set estimate
        train_preds = self._drcif._get_train_preds(X, y)
        self.drcif_weight_ = mean_squared_error(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "DrCIF train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("DrCIF weight = " + str(self.drcif_weight_))  # noqa

        # Build Arsenal
        self._arsenal = Arsenal(
            **self._arsenal_params,
            save_transformed_data=True,
            random_state=self.random_state,
            n_jobs=self._threads_to_use,
        )
        self._arsenal.fit(X, y)

        if self.verbose > 0:
            print("Arsenal ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        # Find Arsenal weight using train set estimate
        train_preds = self._arsenal._get_train_preds(X, y)
        self.arsenal_weight_ = mean_squared_error(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "Arsenal train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("Arsenal weight = " + str(self.arsenal_weight_))  # noqa

        # Build TDE
        self._tde = TemporalDictionaryEnsemble(
            **self._tde_params,
            save_train_predictions=True,
            random_state=self.random_state,
            n_jobs=self._threads_to_use,
        )
        self._tde.fit(X, y)

        if self.verbose > 0:
            print("TDE ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        # Find TDE weight using train set estimate
        train_preds = self._tde._get_train_preds(X, y, train_estimate_method="loocv")
        self.tde_weight_ = mean_squared_error(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "TDE train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("TDE weight = " + str(self.tde_weight_))  # noqa

        self._weight_sum = (
            self.stc_weight_
            + self.drcif_weight_
            + self.arsenal_weight_
            + self.tde_weight_
        )

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        dists = np.zeros((X.shape[0],))

        # Call predict proba on each classifier, multiply the probabilities by the
        # classifiers weight then add them to the current HC2 probabilities
        stc_preds = self._stc.predict(X)
        dists = np.add(
            dists,
            stc_preds * self.stc_weight_,
        )
        drcif_preds = self._drcif.predict(X)
        dists = np.add(
            dists,
            drcif_preds * self.drcif_weight_,
        )
        arsenal_preds = self._arsenal.predict(X)
        dists = np.add(
            dists,
            arsenal_preds * self.arsenal_weight_,
        )
        tde_preds = self._tde.predict(X)
        dists = np.add(
            dists,
            tde_preds * self.tde_weight_,
        )

        if self.save_component_preds:
            self.component_probas = {
                "STC": stc_preds,
                "DrCIF": drcif_preds,
                "Arsenal": arsenal_preds,
                "TDE": tde_preds,
            }

        # Make each instances probability array sum to 1 and return
        return np.around(dists / self._weight_sum, 10)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from sklearn.ensemble import RandomForestClassifier

        if parameter_set == "results_comparison":
            return {
                "stc_params": {
                    "estimator": RandomForestClassifier(n_estimators=3),
                    "n_shapelet_samples": 50,
                    "max_shapelets": 5,
                    "batch_size": 10,
                },
                "drcif_params": {
                    "n_estimators": 3,
                    "n_intervals": 2,
                    "att_subsample_size": 2,
                },
                "arsenal_params": {"num_kernels": 50, "n_estimators": 3},
                "tde_params": {
                    "n_parameter_samples": 5,
                    "max_ensemble_size": 3,
                    "randomly_selected_params": 3,
                },
            }
        else:
            return {
                "stc_params": {
                    "estimator": RandomForestClassifier(n_estimators=1),
                    "n_shapelet_samples": 5,
                    "max_shapelets": 5,
                    "batch_size": 5,
                },
                "drcif_params": {
                    "n_estimators": 1,
                    "n_intervals": 2,
                    "att_subsample_size": 2,
                },
                "arsenal_params": {"num_kernels": 5, "n_estimators": 1},
                "tde_params": {
                    "n_parameter_samples": 1,
                    "max_ensemble_size": 1,
                    "randomly_selected_params": 1,
                },
            }
