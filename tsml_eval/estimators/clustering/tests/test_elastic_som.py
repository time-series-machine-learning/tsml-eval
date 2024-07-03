"""Test elastic som."""

import numpy as np
from aeon.testing.utils.data_gen import make_example_3d_numpy

from tsml_eval.estimators.clustering import ElasticSOM


def test_elastic_som_clustering():
    """Test Elastic Som."""
    train_X = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)
    test_X = make_example_3d_numpy(10, 1, 10, random_state=2, return_y=False)

    elastic_som = ElasticSOM(n_clusters=2, distance="euclidean", random_state=1)
    elastic_som.fit(train_X)
    labels = elastic_som.labels_
    assert labels is not None
    assert len(labels) == 10
    assert np.unique(labels).shape[0] == 2
    predictions = elastic_som.predict(test_X)
    assert predictions is not None
    assert len(predictions) == 10
    assert np.unique(predictions).shape[0] == 2


def test_elastic_som_distances():
    """Test Elastic Som with different distance and param."""
    train_X = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)
    test_X = make_example_3d_numpy(10, 1, 10, random_state=2, return_y=False)

    elastic_som_ed = ElasticSOM(n_clusters=2, distance="euclidean", random_state=1)
    elastic_som_ed.fit(train_X)
    labels_ed = elastic_som_ed.labels_
    predictions_ed = elastic_som_ed.predict(test_X)

    elastic_som_dtw = ElasticSOM(n_clusters=2, distance="dtw", random_state=1)
    elastic_som_dtw.fit(train_X)
    labels_dtw = elastic_som_dtw.labels_
    predictions_dtw = elastic_som_dtw.predict(test_X)

    elastic_som_dtw_window = ElasticSOM(
        n_clusters=2, distance="dtw", distance_params={"window": 0.2}, random_state=1
    )
    elastic_som_dtw_window.fit(train_X)
    labels_dtw_window = elastic_som_dtw_window.labels_
    predictions_dtw_window = elastic_som_dtw_window.predict(test_X)

    assert not np.array_equal(labels_ed, labels_dtw)
    assert not np.array_equal(labels_ed, labels_dtw_window)

    assert not np.array_equal(predictions_ed, predictions_dtw)

    # These should be equal as the window is causing result to essentially be euclidean
    assert np.array_equal(predictions_ed, predictions_dtw_window)
