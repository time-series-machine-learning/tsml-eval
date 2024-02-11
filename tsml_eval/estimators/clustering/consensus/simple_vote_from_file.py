"""Simple vote ensemble for clustering using results files."""

import numpy as np
import pandas as pd
from sklearn import preprocessing

from tsml_eval.estimators.clustering.consensus.simple_vote import SimpleVote


class FromFileSimpleVote(SimpleVote):
    """
    SimpleVote clustering ensemble.

    Parameters
    ----------
    clusterers : list of str
        A list of paths to the clusterer result files to use in the ensemble.
    n_clusters : int, default=8
        The number of clusters to form.
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
        n_clusters=8,
        overwrite_y=False,
        skip_y_check=False,
        random_state=None,
    ):
        self.overwrite_y = overwrite_y
        self.skip_y_check = skip_y_check

        super().__init__(
            clusterers=clusterers, n_clusters=n_clusters, random_state=random_state
        )

    def fit(self, X, y=None):
        """Fit model to X using a simple vote ensemble."""
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "FromFileSimpleVote is not a time series clusterer. "
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
            if uc.shape[0] != self.n_clusters:
                raise ValueError(
                    "Input clusterers must have the same number of clusters as the "
                    f"FromFileSimpleVote n_clusters ({self.n_clusters}). Found "
                    f"{uc.shape[0]} for clusterer {i}."
                )
            elif (np.sort(uc) != np.arange(self.n_clusters)).any():
                raise ValueError(
                    "Input clusterers must have cluster labels in the range "
                    "0 to  n_clusters - 1."
                )

        self._build_ensemble(cluster_assignments)

        return self

    def predict_proba(self, X):
        """Predict cluster probabilities for X."""
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "FromFileSimpleVote is not a time series clusterer. "
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

            if i == 0:
                for j in range(len(X)):
                    line = lines[j + 3].split(",")
                    cluster_assignments[i][j] = int(line[1])
            else:
                for j in range(len(X)):
                    line = lines[j + 3].split(",")
                    cluster_assignments[i][j] = self._new_labels[i - 1][int(line[1])]

        votes = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_clusters),
            axis=0,
            arr=cluster_assignments,
        ).transpose()

        return votes / len(self.clusterers)
