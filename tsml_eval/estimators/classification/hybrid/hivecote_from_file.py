"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) from file.

Hybrid ensemble of classifiers from separate time series classification
representations, using the weighted probabilistic CAWPE as an ensemble controller.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["FromFileHIVECOTE"]

import time

import numpy as np
from aeon.classification import BaseClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state

from tsml_eval.evaluation.storage import ClassifierResults
from tsml_eval.utils.functions import time_to_milliseconds


class FromFileHIVECOTE(BaseClassifier):
    """HIVE-COTE from file.

    Builds the Hierarchical Vote Collective of Transformation-based Ensembles
    (HIVE-COTE) from results files.
    For example, HC2 is n ensemble of the STC, DrCIF, Arsenal and TDE classifiers from
    different feature representations using the CAWPE structure as described in [1].

    Parameters
    ----------
    classifiers : list
        The paths to results files used. i.e. Arsenal, DrCIF, STC and TDE files
        for HC2.
    alpha : int, default=4
        The exponent to extenuate differences in classifiers and weighting with the
        accuracy estimate.
    tune_alpha : bool, default=False
        Tests alpha [1..10] and uses the best value.
    new_weights : list, default=None
        An alternative list of weights to use. Replaces accuracy weights if not None.
    acc_filter : (str, float) tuple, default=None
        Removes the worst component based or on accuracies on
        training data if first item is "train" or test data if first item is "test".
        Be warned that using the test data in this way is cheating, and should only be
        used for exploration.
        The second item is the threshold for the filter, e.g. 0.5 will remove all
        components with accuracies less than 50% of the best component.
        Will always keep the most accurate component.
    overwrite_y : bool, default=False
        If True, the labels in the loaded files will overwrite the labels
        passed in the fit method.
    skip_y_check : bool, default=False
        If True, the labels in the loaded files will not be checked against
        the labels passed in the fit method.
    skip_shape_check : bool, default=False
        If True, the shape of the loaded files (n_cases and n_classes) will not be
        checked against the shape of the data passed in the fit method. This can cause
        breakage, so use with caution.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    weights_ : list
        The weight for Arsenal, DrCIF, STC and TDE probabilities.

    References
    ----------
    .. [1] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." Machine Learning (2021).
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "algorithm_type": "hybrid",
    }

    def __init__(
        self,
        classifiers,
        alpha=4,
        tune_alpha=False,
        new_weights=None,
        acc_filter=None,
        overwrite_y=False,
        skip_y_check=False,
        skip_shape_check=False,
        random_state=None,
    ):
        self.classifiers = classifiers
        self.alpha = alpha
        self.tune_alpha = tune_alpha
        self.new_weights = new_weights
        self.acc_filter = acc_filter
        self.overwrite_y = overwrite_y
        self.skip_y_check = skip_y_check
        self.skip_shape_check = skip_shape_check
        self.random_state = random_state

        super().__init__()

    # todo remove this when aeon stops overriding this attribute
    # todo also remove test skip in all_aeon_estimators
    def fit(self, X, y):  # type: ignore[misc]
        """Fit method."""
        super().fit(X, y)
        self.fit_time_millis_ = self._file_fit_time_millis_
        return self

    def _fit(self, X, y):
        self.weights_ = []
        self._file_fit_time_millis_ = 0

        n_instances = len(X)
        acc_list = []

        # load train file at path (trainResample.csv if random_state is None,
        # trainResample{self.random_state}.csv otherwise)
        if self.random_state is not None:
            file_name = f"trainResample{self.random_state}.csv"
        else:
            file_name = "trainResample.csv"

        if self.tune_alpha:
            X_probas = np.zeros((n_instances, len(self.classifiers), self.n_classes_))

        for i, path in enumerate(self.classifiers):
            cr = ClassifierResults().load_from_file(path + file_name)

            if cr.fit_and_estimate_time == -1:
                self._file_fit_time_millis_ = -1
            elif self._file_fit_time_millis_ != -1:
                self._file_fit_time_millis_ += time_to_milliseconds(
                    cr.fit_and_estimate_time, cr.time_unit
                )

            # verify file matches data
            if not self.skip_shape_check:
                if cr.n_cases != n_instances:
                    raise ValueError(
                        f"n_instances of {path + file_name} does not match X, "
                        f"expected {n_instances}, got {cr.n_cases}"
                    )
                if not self.skip_y_check and cr.n_classes != self.n_classes_:
                    raise ValueError(
                        f"n_classes of {path + file_name} does not match X, "
                        f"expected {self.n_classes_}, got {cr.n_classes}"
                    )

            for j in range(n_instances):
                if self.overwrite_y:
                    if i == 0:
                        y[j] = cr.class_labels[j]
                    elif not self.skip_y_check:
                        assert cr.class_labels[j] == y[j]
                elif not self.skip_y_check:
                    if i == 0:
                        le = preprocessing.LabelEncoder()
                        y = le.fit_transform(y)
                    assert cr.class_labels[j] == y[j]

                if self.tune_alpha:
                    X_probas[j][i] = cr.probabilities[j]

            acc_list.append(cr.accuracy)

        start = int(round(time.time() * 1000))

        self._alpha = self._tune_alpha(X_probas, y) if self.tune_alpha else self.alpha

        if self.new_weights:
            acc_list = self.new_weights
            assert len(acc_list) == len(self.classifiers)

        # add a weight to the weight list based on the files accuracy
        for acc in acc_list:
            self.weights_.append(acc**self._alpha)

        self._use_classifier = [True for _ in range(len(self.classifiers))]
        if self.acc_filter is not None:
            self._acc_filter = (
                ("train", self.acc_filter)
                if isinstance(self.acc_filter, float)
                else self.acc_filter
            )

            if self._acc_filter[0] != "train" and self._acc_filter[0] != "test":
                raise ValueError(
                    f"acc_filter[0] must be 'train' or 'test', got "
                    f"{self._acc_filter[0]}"
                )
            elif self._acc_filter[1] <= 0 or self._acc_filter[1] >= 1:
                raise ValueError(
                    "acc_filter[1] must be in between 0 and 1, got "
                    f"{self._acc_filter[1]}"
                )

            if self._acc_filter[0] == "test":
                # load test file at path (testResample.csv if random_state is None,
                # testResample{self.random_state}.csv otherwise)
                if self.random_state is not None:
                    file_name = f"testResample{self.random_state}.csv"
                else:
                    file_name = "testResample.csv"

                acc_list = []
                for path in self.classifiers:
                    cr = ClassifierResults().load_from_file(path + file_name)
                    acc_list.append(cr.accuracy)

            argmax = np.argmax(acc_list)

            # add a weight to the weight list based on the files accuracy
            for i, acc in enumerate(acc_list):
                if i != argmax and acc < acc_list[argmax] * self._acc_filter[1]:
                    self._use_classifier[i] = False

        if self._file_fit_time_millis_ != -1:
            self._file_fit_time_millis_ += int(round(time.time() * 1000)) - start

    def _predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def _predict_proba(self, X):
        n_instances = len(X)

        # load test file at path (testResample.csv if random_state is None,
        # testResample{self.random_state}.csv otherwise)
        if self.random_state is not None:
            file_name = f"testResample{self.random_state}.csv"
        else:
            file_name = "testResample.csv"

        dists = np.zeros((n_instances, self.n_classes_))

        if not self.skip_y_check:
            y = np.zeros(n_instances)

        for i, path in enumerate(self.classifiers):
            cr = ClassifierResults().load_from_file(path + file_name)

            # verify file matches data
            if not self.skip_shape_check:
                if cr.n_cases != n_instances:
                    raise ValueError(
                        f"n_instances of {path + file_name} does not match X, "
                        f"expected {n_instances}, got {cr.n_cases}"
                    )
                if not self.skip_y_check and cr.n_classes != self.n_classes_:
                    raise ValueError(
                        f"n_classes of {path + file_name} does not match X, "
                        f"expected {self.n_classes_}, got {cr.n_classes}"
                    )

            # apply this files weights to the probabilities in the test file
            for j in range(n_instances):
                if not self.skip_y_check:
                    if i == 0:
                        y[j] = cr.class_labels[j]
                    else:
                        assert cr.class_labels[j] == y[j]

                if self._use_classifier[i]:
                    dists[j] = np.add(
                        dists[j],
                        cr.probabilities[j]
                        * (np.ones(self.n_classes_) * self.weights_[i]),
                    )

        # Make each instances probability array sum to 1 and return
        return dists / dists.sum(axis=1, keepdims=True)

    def _tune_alpha(self, X_probas, y):
        n_instances = len(y)
        n_files = X_probas.shape[1]

        n_splits = 10
        _, counts = np.unique(y, return_counts=True)
        min_class = np.min(counts)
        if min_class < n_splits:
            n_splits = min_class

        if n_splits == 1:
            return self.alpha

        alpha_values = range(1, 10)  # tested alpha values
        avg_alpha_acc = np.zeros(len(alpha_values))  # performance of each alpha value
        for i, alpha in enumerate(alpha_values):
            kf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=(
                    np.random.randint(0, np.iinfo(np.int32).max)
                    if self.random_state is None
                    else self.random_state
                ),
            )
            preds = np.zeros(n_instances)

            for train_index, test_index in kf.split(X_probas, y):
                weight_list = []
                for n in range(n_files):
                    train_preds = np.argmax(X_probas[train_index, n], axis=1)
                    train_acc = accuracy_score(y[train_index], train_preds)
                    weight_list.append(train_acc**alpha)

                dists = np.zeros((len(test_index), self.n_classes_))
                # apply the weights to the probabilities in the test set
                for n in range(n_files):
                    for v in range(len(test_index)):
                        dists[v] = np.add(
                            dists[v],
                            [k for k in X_probas[test_index[v], n]]
                            * (np.ones(self.n_classes_) * weight_list[n]),
                        )

                # Make each instances probability array sum to 1
                dists = dists / dists.sum(axis=1, keepdims=True)
                preds[test_index] = np.argmax(dists, axis=1)

            avg_alpha_acc[i] = accuracy_score(y, preds)

        best_alpha = alpha_values[avg_alpha_acc.argmax()]

        return best_alpha

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
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
        from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH

        file_paths = [
            _TEST_RESULTS_PATH + "/classification/TestResults/Test1/",
            _TEST_RESULTS_PATH + "/classification/TestResults/Test2/",
        ]
        return {
            "classifiers": file_paths,
            "skip_y_check": True,
            "skip_shape_check": True,
            "random_state": 0,
        }
