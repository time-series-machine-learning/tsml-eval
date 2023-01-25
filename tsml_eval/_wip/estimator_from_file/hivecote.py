# -*- coding: utf-8 -*-
"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) from file.

Upgraded hybrid ensemble of classifiers from 4 separate time series classification
representations, using the weighted probabilistic CAWPE as an ensemble controller.
This version loads the ensembles predictions from file and allows to change the alfa value.
"""

__author__ = ["MatthewMiddlehurst", "ander-hg"]
__all__ = ["FromFileHIVECOTE"]

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state
from sktime.classification import BaseClassifier


class FromFileHIVECOTE(BaseClassifier):
    """Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) from file.
    An ensemble of the STC, DrCIF, Arsenal and TDE classifiers from different feature
    representations using the CAWPE structure as described in [1].

    Parameters
    ----------
    file_paths : list
        The paths for Arsenal, DrCIF, STC and TDE files.
    alpha : int, default=4
        The exponent to extenuate diferences in classifers and weighting with the accuracy estimate.
    tune_alpha : bool, default=False
        Tests alpha [1..10] and sets the best one
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    _weights : list
        The weight for Arsenal, DrCIF, STC and TDE probabilities.

    References
    ----------
    .. [1] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." Machine Learning (2021).
    """

    _tags = {
        "capability:multivariate": True,
        "classifier_type": "hybrid",
    }

    def __init__(
        self,
        file_paths,
        alpha=4,
        tune_alpha=False,
        overwrite_y=False,
        skip_y_check=False,
        random_state=None,
    ):
        self.file_paths = file_paths
        self.alpha = alpha
        self.tune_alpha = tune_alpha
        self.overwrite_y = overwrite_y
        self.skip_y_check = skip_y_check
        self.random_state = random_state

        self.predict_y = []

        self._alpha = alpha
        self._weights = []

        super(FromFileHIVECOTE, self).__init__()

    def _fit(self, X, y):
        """Load HIVE-COTE accuracies from the training file.

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
        Updates the attribute _weights with the loaded from file accuracies to the power of alfa.
        """
        n_instances, _, _ = X.shape
        self._weights = []

        # load train file at path (trainResample.csv if random_state is None,
        # trainResample{self.random_state}.csv otherwise)
        if self.random_state is not None:
            file_name = f"trainResample{self.random_state}.csv"
        else:
            file_name = "trainResample.csv"

        if self.tune_alpha:
            X_probas = np.zeros((n_instances, len(self.file_paths), self.n_classes_))

        acc_list = []
        all_lines = []
        for i, path in enumerate(self.file_paths):
            f = open(path + file_name, "r")
            lines = f.readlines()
            line2 = lines[2].split(",")

            # verify file matches data
            if len(lines) - 3 != n_instances:  # verify n_instances
                print(
                    "ERROR n_instances does not match in: ",
                    path + file_name,
                    len(lines) - 3,
                    n_instances,
                )
            if len(np.unique(y)) != int(line2[5]):  # verify n_classes
                print(
                    "ERROR n_classes does not match in: ",
                    path + file_name,
                    self.n_classes_,
                    line2[5],
                )

            for j in range(n_instances):
                line = lines[j + 3].split(",")

                if self.overwrite_y:
                    if i == 0:
                        y[j] = float(line[0])
                    elif not self.skip_y_check:
                        assert y[j] == float(line[0])
                elif not self.skip_y_check:
                    if i == 0:
                        le = preprocessing.LabelEncoder()
                        y = le.fit_transform(y)
                    assert float(line[0]) == y[j]

                if self.tune_alpha:
                    X_probas[j][i] = [float(k) for k in (line[3:])]

            acc_list.append(float(line2[0]))
            if self.tune_alpha:
                all_lines.append(lines)

        if self.tune_alpha:
            self._alpha = self._tune_alpha(X_probas, y)

        # add a weight to the weight list based on the files accuracy
        for acc in acc_list:
            self._weights.append(acc**self._alpha)

    def _tune_alpha(self, X_probas, y):
        """Finds the best alpha value if self.tune_alpha == True.

        Parameters
        ----------
        all_files_lines : list
            The content of each file as each element of the list.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Estimates through cross validation the best alpha of a range of values.
        """
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
                random_state=np.random.randint(0, np.iinfo(np.int32).max)
                if self.random_state is None
                else self.random_state,
            )
            preds = np.zeros(n_instances)

            for (train_index, test_index) in kf.split(X_probas, y):
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

    def _predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def _predict_proba(self, X):
        """Predicts labels probabilities sequences reading from files.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.

        Notes
        ----
        Predicts labels probabilities for sequences in X loading each ensemble estimated probabilities from file.
        Loads the probabilities from the test files,
        applies the weights and returns the estimated probabilities.
        """
        n_instances, _, _ = X.shape

        # for each file path input:
        #   load test file at path (testResample.csv if random_state is None,
        #   testResample0.csv otherwise)
        file_name = "testResample.csv"
        if self.random_state is not None:
            file_name = "testResample" + str(self.random_state) + ".csv"

        dists = np.zeros((n_instances, self.n_classes_))

        if not self.skip_y_check:
            y = np.zeros(n_instances)

        for i, path in enumerate(self.file_paths):
            f = open(path + file_name, "r")
            lines = f.readlines()
            line2 = lines[2].split(",")

            # verify file matches data
            if len(lines) - 3 != n_instances:  # verify n_instances
                print(
                    "ERROR n_instances does not match in: ",
                    path + file_name,
                    len(lines) - 3,
                    n_instances,
                )
            if self.n_classes_ != int(line2[5]):  # verify n_classes
                print(
                    "ERROR n_classes does not match in: ",
                    path + file_name,
                    self.n_classes_,
                    line2[5],
                )

            #   apply this files weights to the probabilities in the test file
            for j in range(n_instances):
                line = lines[j + 3].split(",")

                dists[j] = np.add(
                    dists[j],
                    [float(k) for k in (line[3:])]
                    * (np.ones(self.n_classes_) * self._weights[i]),
                )

                if not self.skip_y_check:
                    if i == 0:
                        y[j] = float(line[0])
                    else:
                        assert y[j] == float(line[0])

        if self.overwrite_y:
            self.predict_y = y

        # Make each instances probability array sum to 1 and return
        return dists / dists.sum(axis=1, keepdims=True)

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
        file_paths = [
            "test_files/Test/Test1/",
            "test_files/Test/Test2/",
        ]
        return {"file_paths": file_paths, "skip_y_check": True, "random_state": 0}
