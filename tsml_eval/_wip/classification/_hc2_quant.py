"""HIVE-COTE V2 variant with QUANT replacing the DrCIF component.

Clone of aeon's HIVECOTEV2 (aeon 1.4.0) where the interval-based module is
BaggedQUANT (QUANT with bootstrap sampling and OOB train estimates) instead of
DrCIF. All other components (STC, Arsenal, TDE) and the CAWPE weighting are
unchanged.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["HC2Quant"]

from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import Arsenal
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.utils.validation import check_n_jobs

from tsml_eval._wip.classification._bagged_quant import BaggedQUANT


class HC2Quant(BaseClassifier):
    """HIVE-COTE V2 with QUANT in place of DrCIF.

    An ensemble of the STC, QUANT, Arsenal and TDE classifiers from different
    feature representations using the CAWPE structure. Identical to aeon's
    HIVECOTEV2 except the interval-based module.

    Parameters
    ----------
    stc_params : dict or None, default=None
        Parameters for the ShapeletTransformClassifier module. If None, uses the
        default parameters with 10,000 shapelet samples.
    quant_params : dict or None, default=None
        Parameters for the BaggedQUANT module. If None, uses the QUANT defaults
        (interval_depth=6, quantile_divisor=4, 200 bagged extra trees). Note
        the QUANT module is not contractable and has no n_jobs parameter; its
        CAWPE weight comes from its out-of-bag train estimate.
    arsenal_params : dict or None, default=None
        Parameters for the Arsenal module. If None, uses the default parameters.
    tde_params : dict or None, default=None
        Parameters for the TemporalDictionaryEnsemble module. If None, uses the
        default parameters.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes for the contractable
        components (STC, Arsenal, TDE). QUANT is never contracted.
    save_component_probas : bool, default=False
        When predict/predict_proba is called, save each component's probability
        predictions in component_probas.
    verbose : int, default=0
        Level of output printed to the console (for information only).
    random_state : int, RandomState instance or None, default=None
        Seed or RandomState for reproducibility.
    n_jobs : int, default=1
        The number of jobs for the STC, Arsenal and TDE modules.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The unique class labels.
    stc_weight_ : float
        The weight for STC probabilities.
    quant_weight_ : float
        The weight for QUANT probabilities.
    arsenal_weight_ : float
        The weight for Arsenal probabilities.
    tde_weight_ : float
        The weight for TDE probabilities.
    component_probas : dict
        Only used if save_component_probas is true. Saved probability
        predictions for each component.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "hybrid",
    }

    def __init__(
        self,
        stc_params=None,
        quant_params=None,
        arsenal_params=None,
        tde_params=None,
        time_limit_in_minutes=0,
        save_component_probas=False,
        verbose=0,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.stc_params = stc_params
        self.quant_params = quant_params
        self.arsenal_params = arsenal_params
        self.tde_params = tde_params
        self.time_limit_in_minutes = time_limit_in_minutes
        self.save_component_probas = save_component_probas
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        self.stc_weight_ = 0
        self.quant_weight_ = 0
        self.arsenal_weight_ = 0
        self.tde_weight_ = 0
        self.component_probas = {}

        self._stc_params = stc_params
        self._quant_params = quant_params
        self._arsenal_params = arsenal_params
        self._tde_params = tde_params
        self._stc = None
        self._quant = None
        self._arsenal = None
        self._tde = None

        super().__init__()

    _DEFAULT_N_SHAPELETS = 10000
    _DEFAULT_N_KERNELS = 2000
    _DEFAULT_N_ESTIMATORS = 25
    _DEFAULT_N_PARA_SAMPLES = 250
    _DEFAULT_MAX_ENSEMBLE_SIZE = 50
    _DEFAULT_RAND_PARAMS = 50

    def _fit(self, X, y):
        """Fit HC2Quant to training data."""
        self._n_jobs = check_n_jobs(self.n_jobs)

        if self.stc_params is None:
            self._stc_params = {"n_shapelet_samples": HC2Quant._DEFAULT_N_SHAPELETS}
        if self.quant_params is None:
            self._quant_params = {}
        if self.arsenal_params is None:
            self._arsenal_params = {
                "n_kernels": HC2Quant._DEFAULT_N_KERNELS,
                "n_estimators": HC2Quant._DEFAULT_N_ESTIMATORS,
            }
        if self.tde_params is None:
            self._tde_params = {
                "n_parameter_samples": HC2Quant._DEFAULT_N_PARA_SAMPLES,
                "max_ensemble_size": HC2Quant._DEFAULT_MAX_ENSEMBLE_SIZE,
                "randomly_selected_params": HC2Quant._DEFAULT_RAND_PARAMS,
            }

        # If we are contracting split the contract time between the three
        # contractable components (QUANT is not contractable)
        if self.time_limit_in_minutes > 0:
            # Leave 1/3 for train estimates
            ct = self.time_limit_in_minutes / 6
            self._stc_params["time_limit_in_minutes"] = ct
            self._arsenal_params["time_limit_in_minutes"] = ct
            self._tde_params["time_limit_in_minutes"] = ct

        # Build STC
        self._stc = ShapeletTransformClassifier(
            **self._stc_params,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        train_preds = self._stc.fit_predict(X, y)
        self.stc_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print("STC ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa
            print("STC weight = " + str(self.stc_weight_))  # noqa

        # Build QUANT (bagged variant, so the CAWPE weight comes from an
        # out-of-bag train estimate as DrCIF's does, not 10-fold CV)
        self._quant = BaggedQUANT(
            **self._quant_params,
            random_state=self.random_state,
        )
        train_preds = self._quant.fit_predict(X, y)
        self.quant_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print("QUANT ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa
            print("QUANT weight = " + str(self.quant_weight_))  # noqa

        # Build Arsenal
        self._arsenal = Arsenal(
            **self._arsenal_params,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        train_preds = self._arsenal.fit_predict(X, y)
        self.arsenal_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print("Arsenal ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa
            print("Arsenal weight = " + str(self.arsenal_weight_))  # noqa

        # Build TDE
        self._tde = TemporalDictionaryEnsemble(
            **self._tde_params,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        train_preds = self._tde.fit_predict(X, y)
        self.tde_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print("TDE ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa
            print("TDE weight = " + str(self.tde_weight_))  # noqa

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X."""
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def _predict_proba(self, X, return_component_probas=False) -> np.ndarray:
        """Predicts label probabilities for sequences in X."""
        dists = np.zeros((X.shape[0], self.n_classes_))

        # Call predict proba on each classifier, multiply the probabilities by
        # the classifier's weight then add them to the current probabilities
        stc_probas = self._stc.predict_proba(X)
        dists = np.add(
            dists,
            stc_probas * (np.ones(self.n_classes_) * self.stc_weight_),
        )
        quant_probas = self._quant.predict_proba(X)
        dists = np.add(
            dists,
            quant_probas * (np.ones(self.n_classes_) * self.quant_weight_),
        )
        arsenal_probas = self._arsenal.predict_proba(X)
        dists = np.add(
            dists,
            arsenal_probas * (np.ones(self.n_classes_) * self.arsenal_weight_),
        )
        tde_probas = self._tde.predict_proba(X)
        dists = np.add(
            dists,
            tde_probas * (np.ones(self.n_classes_) * self.tde_weight_),
        )

        if self.save_component_probas:
            self.component_probas = {
                "STC": stc_probas,
                "QUANT": quant_probas,
                "Arsenal": arsenal_probas,
                "TDE": tde_probas,
            }

        # Make each instances probability array sum to 1 and return
        return dists / dists.sum(axis=1, keepdims=True)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sklearn.ensemble import RandomForestClassifier

        return {
            "stc_params": {
                "estimator": RandomForestClassifier(n_estimators=1),
                "n_shapelet_samples": 5,
                "max_shapelets": 5,
                "batch_size": 5,
            },
            "quant_params": {"interval_depth": 2, "quantile_divisor": 8},
            "arsenal_params": {"n_kernels": 5, "n_estimators": 1},
            "tde_params": {
                "n_parameter_samples": 1,
                "max_ensemble_size": 1,
                "randomly_selected_params": 1,
            },
        }
