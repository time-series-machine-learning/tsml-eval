"""IVC consensus clustering algorithm from results files."""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import check_random_state

from tsml_eval.estimators.clustering.consensus.ivc import IterativeVotingClustering


class FromFileIterativeVotingClustering(IterativeVotingClustering):
    """
    IVC (Iterative Voting Clustering) Consensus Clusterer.

    IVC is a consensus clustering algorithm that combines the results of multiple
    base clusterers to find a consensus clustering. It iteratively refines cluster
    assignments based on a majority voting scheme.

    Parameters
    ----------
    clusterers : list of str
        A list of paths to the clusterer result files to use in the ensemble.
    init : {'plus', 'random', 'aligned'}, default='plus'
        The method used to initialize the cluster centers. 'plus' uses the
        k-means++ initialization method, 'random' uses random selection, and
        'aligned' uses a method that aligns the cluster assignmeds from the base
        clusterers.
    n_clusters : int, default=8
        The number of clusters to form.
    max_iterations : int, default=500
        The maximum number of iterations to perform.
    overwrite_y : bool, default=False
        If True, the labels in the loaded files will overwrite the labels
        passed in the fit method.
    skip_y_check : bool, default=False
        If True, the labels in the loaded files will not be checked against
        the labels passed in the fit method.
    random_state : int, default=None
        The seed for random number generation.

    Attributes
    ----------
    labels_ : ndarray of shape (n_instances,)
        Labels of each point from the last fit.
    """

    def __init__(
        self,
        clusterers,
        init="plus",
        n_clusters=8,
        max_iterations=500,
        overwrite_y=False,
        skip_y_check=False,
        random_state=None,
    ):
        self.overwrite_y = overwrite_y
        self.skip_y_check = skip_y_check

        super().__init__(
            clusterers=clusterers,
            init=init,
            n_clusters=n_clusters,
            max_iterations=max_iterations,
            random_state=random_state,
        )

    def fit(self, X, y=None):
        """Fit model to X using IVC."""
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "FromFileIterativeVotingClustering is not a time series clusterer. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, ensure_min_samples=self.n_clusters)

        # load train file at path (trainResample.csv if random_state is None,
        # trainResample{self.random_state}.csv otherwise)
        if self.random_state is not None:
            file_name = f"trainResample{self.random_state}.csv"
        else:
            file_name = "trainResample.csv"

        cluster_assignments = np.zeros(
            (len(self.clusterers), X.shape[0]), dtype=np.int32
        )
        for i, path in enumerate(self.clusterers):
            f = open(path + file_name)
            lines = f.readlines()
            line2 = lines[2].split(",")

            # verify file matches data
            if len(lines) - 3 != X.shape[0]:
                raise ValueError(
                    f"n_instances of {path + file_name} does not match X, "
                    f"expected {X.shape[0]}, got {len(lines) - 3}"
                )
            if (
                y is not None
                and not self.skip_y_check
                and len(np.unique(y)) != int(line2[5])
            ):
                raise ValueError(
                    f"n_classes of {path + file_name} does not match X, "
                    f"expected {len(np.unique(y))}, got {line2[6]}"
                )

            for j in range(X.shape[0]):
                line = lines[j + 3].split(",")

                if self.overwrite_y:
                    if i == 0:
                        y[j] = float(line[0])
                    elif not self.skip_y_check:
                        assert y[j] == float(line[0])
                elif y is not None and not self.skip_y_check:
                    if i == 0:
                        le = preprocessing.LabelEncoder()
                        y = le.fit_transform(y)
                    assert float(line[0]) == y[j]

                cluster_assignments[i][j] = int(line[1])

            uc = np.unique(cluster_assignments[i])
            if (np.sort(uc) != np.arange(self.n_clusters)).any():
                raise ValueError(
                    "Input clusterers must have cluster labels in the range "
                    "0 to  n_clusters - 1."
                )

        self._build_ensemble(cluster_assignments)

        return self

    def predict(self, X):
        """Predict cluster labels for X."""
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "FromFileIterativeVotingClustering is not a time series clusterer. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, reset=False)

        # load train file at path (trainResample.csv if random_state is None,
        # trainResample{self.random_state}.csv otherwise)
        if self.random_state is not None:
            file_name = f"testResample{self.random_state}.csv"
        else:
            file_name = "testResample.csv"

        cluster_assignments = np.zeros(
            (len(self.clusterers), X.shape[0]), dtype=np.int32
        )
        for i, path in enumerate(self.clusterers):
            f = open(path + file_name)
            lines = f.readlines()

            # verify file matches data
            if len(lines) - 3 != len(X):
                if len(lines) - 3 != X.shape[0]:
                    raise ValueError(
                        f"n_instances of {path + file_name} does not match X, "
                        f"expected {X.shape[0]}, got {len(lines) - 3}"
                    )

            for j in range(len(X)):
                line = lines[j + 3].split(",")
                cluster_assignments[i][j] = int(line[1])

        rng = check_random_state(self.random_state)
        labels = self._calculate_cluster_membership(cluster_assignments, rng)

        return labels
