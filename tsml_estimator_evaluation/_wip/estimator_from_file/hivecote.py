# -*- coding: utf-8 -*-
"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) from file.

Upgraded hybrid ensemble of classifiers from 4 separate time series classification
representations, using the weighted probabilistic CAWPE as an ensemble controller.
This version loads the ensembles predictions from file and allows to change the alfa value.
"""

__author__ = ["MatthewMiddlehurst", "ander-hg"]
__all__ = ["FromFileHIVECOTE"]

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sktime.classification import BaseClassifier
from sklearn.model_selection import KFold

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
        random_state=None,
    ):
        self.file_paths = file_paths
        self.alpha = alpha
        self.tune_alpha = tune_alpha
        self.random_state = random_state

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

        self._weights = []

        #   load train file at path (trainResample.csv if random_state is None, trainResample0.csv otherwise)
        file_name = 'trainResample.csv'
        if self.random_state != None:
            file_name = 'trainResample' + str(self.random_state) + '.csv'

        acc_list = []
        all_lines = []
        for path in self.file_paths:
            f = open(path + file_name, "r")
            lines = f.readlines()
            line2 = lines[2].split(",")

            #   verify file matches data, i.e. n_instances and n_classes
            if len(lines)-3 != len(X):  # verify n_instances
                print("ERROR n_instances does not match in: ", path + file_name, len(lines) - 3, len(X))
            if len(np.unique(y)) != int(line2[5]): # verify n_classes
                print("ERROR n_classes does not match in: ", path + file_name, len(np.unique(y)), line2[5])

            acc_list.append(float(line2[0]))
            if self.tune_alpha:
                all_lines.append(lines)

        if self.tune_alpha:
            self.alpha = self._tune_alpha(all_lines)

        #   add a weight to the weight list based on the files accuracy
        for acc in acc_list:
            self._weights.append(acc ** self.alpha)

    def _tune_alpha(self, all_files_lines):
        alpha = 1
        n_splits = 5
        n_samples = len(all_files_lines[0])-3
        n_files = len(all_files_lines)
        x_probas = np.zeros((n_samples, n_files, self.n_classes_))
        y_probas = np.zeros(n_samples, dtype=int)
        for i, lines in enumerate(all_files_lines):
            for j in range(n_samples):
                line = lines[j+3].split(",")
                acc_0 = float(line[3])
                acc_1 = float(line[4])
                x_probas[j][i] = [acc_0, acc_1]
                y_probas[j] = int(line[0]) # its getting y 4 times, not efficient

        alpha_values = range(1, 10)
        avg_acc_alpha = np.zeros(len(alpha_values))
        for i, alpha in enumerate(alpha_values):
            kf = KFold(n_splits=n_splits)
            # print(kf.get_n_splits(X))
            # print(kf)
            avg_acc = np.zeros(n_splits)
            for j, (train_index, test_index) in enumerate(kf.split(x_probas)):
                print(f"Fold {j}:")
                print(f"  Train: index={train_index}")
                print(f"  Test:  index={test_index}")
                x_test_set = x_probas[test_index]
                y_test_set = y_probas[test_index]
                #acc_list = Acc[train_index].sum(axis=0)/len(train_index)
                weight_list = []
                for n in range(n_files):
                    train_preds = [int(x) for x in self.classes_[np.argmax(x_probas[train_index, n], axis=1)]]
                    train_acc = accuracy_score(y_probas[train_index], train_preds)
                    weight_list.append(train_acc ** alpha)
                predictions_0 = (x_test_set[:, :, 0] * weight_list).sum(axis=1)
                predictions_1 = (x_test_set[:, :, 1] * weight_list).sum(axis=1)
                predictions = np.column_stack((predictions_0, predictions_1))
                # Make each instances probability array sum to 1
                predictions = predictions / predictions.sum(axis=1, keepdims=True)
                predicted_acc = np.zeros(len(y_test_set))
                for k, l in enumerate(y_test_set):
                    predicted_acc[k] = predictions[k, l]
                avg_acc[j] = predicted_acc.mean()
            avg_acc_alpha[i] = avg_acc.mean()
        print("AVG_ACC/ALPHA")
        print(avg_acc_alpha)
        print(avg_acc_alpha[avg_acc_alpha.argmax()])
        best_alpha = alpha_values[avg_acc_alpha.argmax()]
        print(best_alpha)

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

        # for each file path input:
        #   load test file at path (testResample.csv if random_state is None,
        #   testResample0.csv otherwise)
        file_name = 'testResample.csv'
        if self.random_state != None:
            file_name = 'testResample' + str(self.random_state) + '.csv'

        dists = np.zeros((X.shape[0], self.n_classes_))

        i = 0
        for path in self.file_paths:
            f = open(path + file_name, "r")
            lines = f.readlines()
            line2 = lines[2].split(",")

            #   verify file matches data, i.e. n_instances and n_classes
            if len(lines) - 3 != len(X):  # verify n_instances
                print("ERROR n_instances does not match in: ", path + file_name, len(lines) - 3, len(X))
            if self.n_classes_ != int(line2[5]):  # verify n_classes
                print("ERROR n_classes does not match in: ", path + file_name, self.n_classes_, line2[5])

            #   apply this files weights to the probabilities in the test file
            for j in range(X.shape[0]):
                dists[j] = np.add(
                    dists[j],
                    [float(k) for k in (lines[j+3].split(",")[3:])] * (np.ones(self.n_classes_) * self._weights[i]),
            )
            i += 1

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
            "test_files/Arsenal/",
            "test_files/DrCIF/",
            "test_files/STC/",
            "test_files/TDE/",
        ]
        return {"file_paths": file_paths, "random_state": 0}
