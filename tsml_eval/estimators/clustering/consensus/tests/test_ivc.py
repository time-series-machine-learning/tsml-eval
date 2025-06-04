"""Tests for IterativeVotingClustering."""

import numpy as np
import pytest
from aeon.datasets import load_arrow_head
from aeon.testing.data_generation import make_example_3d_numpy
from sklearn.metrics import rand_score

from tsml_eval.estimators.clustering.consensus.ivc import IterativeVotingClustering
from tsml_eval.estimators.clustering.consensus.ivc_from_file import (
    FromFileIterativeVotingClustering,
)
from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH


@pytest.mark.parametrize("init", ["plus", "random", "aligned"])
def test_ivc_init_methods(init):
    """Test IVC init methods."""
    X, y = make_example_3d_numpy(n_cases=30, n_timepoints=30, n_labels=4)

    ivc = IterativeVotingClustering(
        n_clusters=4,
        max_iterations=20,
        init=init,
    )
    ivc.fit(X, y)
    preds = ivc.predict(X)

    assert ivc.labels_.shape == (len(X),)
    assert preds.shape == (len(X),)


def test_from_file_iterative_voting_clustering():
    """Test SimpleVote from file with ArrowHead results."""
    X_train, y_train = load_arrow_head(split="train")
    X_test, y_test = load_arrow_head(split="test")

    file_paths = [
        _TEST_RESULTS_PATH + "/clustering/PAM-DTW/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/clustering/PAM-ERP/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/clustering/PAM-MSM/Predictions/ArrowHead/",
    ]

    ivc = FromFileIterativeVotingClustering(
        clusterers=file_paths, n_clusters=3, max_iterations=100, random_state=0
    )
    ivc.fit(X_train, y_train)
    preds = ivc.predict(X_test)

    assert ivc.labels_.shape == (len(X_train),)
    assert isinstance(ivc.labels_, np.ndarray)
    assert rand_score(y_train, ivc.labels_) >= 0.6
    assert preds.shape == (len(X_test),)
    assert isinstance(preds, np.ndarray)
    assert rand_score(y_test, preds) >= 0.6
