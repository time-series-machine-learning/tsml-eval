from aeon.testing.data_generation import make_example_3d_numpy_list

from tsml_eval._wip.unequal_length._stc import ShapeletTransformClassifier


def test_stc_unequal_length():
    """Test STC with unequal length time series."""
    X, y = make_example_3d_numpy_list(n_cases=20, min_n_timepoints=15, max_n_timepoints=20)
    stc = ShapeletTransformClassifier(n_shapelet_samples=200, max_shapelets=50)
    stc.fit(X, y)

    X, y = make_example_3d_numpy_list(n_cases=20, min_n_timepoints=5, max_n_timepoints=15)
    stc.predict(X)
