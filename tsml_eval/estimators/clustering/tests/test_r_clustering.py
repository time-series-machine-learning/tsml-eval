"""Tests for RClustering"""
import numpy as np
from aeon.testing.utils.data_gen import make_example_3d_numpy
from tsml_eval.estimators.clustering import RClustering
from sklearn.cluster import KMeans


def test_r_clustering():
    """Test RClustering."""
    train_X = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)
    test_X = make_example_3d_numpy(10, 1, 10, random_state=2, return_y=False)

    kmeans = KMeans(random_state=1, n_init=10, n_clusters=2)

    r_clustering = RClustering(estimator=kmeans, random_state=1)
    r_clustering.fit(train_X)
    labels = r_clustering.labels_
    assert labels is not None
    assert len(labels) == 10
    assert np.unique(labels).shape[0] == 2
    predictions = r_clustering.predict(test_X)
    assert predictions is not None
    assert len(predictions) == 10
    assert np.unique(predictions).shape[0] == 2
