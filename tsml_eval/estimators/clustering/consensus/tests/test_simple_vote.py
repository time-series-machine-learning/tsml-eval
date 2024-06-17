"""Test the SimpleVote."""

import numpy as np
from aeon.datasets import load_arrow_head
from sklearn.metrics import rand_score

from tsml_eval.estimators.clustering.consensus.simple_vote_from_file import (
    FromFileSimpleVote,
)
from tsml_eval.testing.testing_utils import _TEST_RESULTS_PATH


def test_from_file_simple_vote():
    """Test SimpleVote from file with ArrowHead results."""
    X_train, y_train = load_arrow_head(split="train")
    X_test, y_test = load_arrow_head(split="test")

    file_paths = [
        _TEST_RESULTS_PATH + "/clustering/PAM-DTW/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/clustering/PAM-ERP/Predictions/ArrowHead/",
        _TEST_RESULTS_PATH + "/clustering/PAM-MSM/Predictions/ArrowHead/",
    ]

    sv = FromFileSimpleVote(clusterers=file_paths, n_clusters=3, random_state=0)
    sv.fit(X_train, y_train)
    preds = sv.predict(X_test)

    assert sv.labels_.shape == (len(X_train),)
    assert isinstance(sv.labels_, np.ndarray)
    assert rand_score(y_train, sv.labels_) >= 0.6
    assert preds.shape == (len(X_test),)
    assert isinstance(preds, np.ndarray)
    assert rand_score(y_test, preds) >= 0.6
