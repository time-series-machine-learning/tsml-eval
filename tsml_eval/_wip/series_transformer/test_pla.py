import pytest
import numpy as np
import pandas as pd
from _pla import PiecewiseLinearApproximation 


@pytest.fixture
def X():
    return np.array([573.0,375.0,301.0,212.0,55.0,34.0,25.0,33.0,113.0,143.0,303.0,
                      615.0,1226.0,1281.0,1221.0,1081.0,866.0,1096.0,1039.0,975.0,
                      746.0,581.0,409.0,182.0])

def test_piecewise_linear_approximation_sliding_window(X):
    pla = PiecewiseLinearApproximation(PiecewiseLinearApproximation.Transformer.SlidingWindow, 100)
    result = pla.fit_transform(X)
    expected = np.array([573., 375., 301., 212., 53., 38., 23., 33., 113., 143., 303., 
                        615., 1226., 1281., 1221., 1081., 866., 1097.16666667, 
                        1036.66666667, 976.16666667, 747.16666667, 
                        578.66666667, 410.16666667, 182.])
    np.testing.assert_array_almost_equal(result, expected)

def test_piecewise_linear_approximation_top_down(X):
    pla  = PiecewiseLinearApproximation(PiecewiseLinearApproximation.Transformer.TopDown, 100)
    result = pla.fit_transform(X)
    expected = np.array([573., 375., 301., 212., 53., 38., 23., 33., 113., 143., 303., 
                         615., 1226., 1281., 1221., 1081., 866., 1097.16666667, 
                         1036.66666667, 976.16666667, 746., 581., 409., 182.])
    np.testing.assert_array_almost_equal(result, expected)

def test_piecewise_linear_approximation_bottom_up(X):
    result  = PiecewiseLinearApproximation(PiecewiseLinearApproximation.Transformer.BottomUp, 5).fit_transform(X)
    expected = np.array([538.8, 423.1, 307.4, 191.7, 48., 40.5, 33., 25.5, 43.6, 
                         210.2,376.8, 543.4, 1276.5, 1227., 1177.5, 1128., 953.5, 
                         980.5, 1007.5, 1034.5, 759.1, 572.7, 386.3, 199.9])
    np.testing.assert_array_almost_equal(result, expected)

def test_piecewise_linear_approximation_SWAB(X):
    result  = PiecewiseLinearApproximation(PiecewiseLinearApproximation.Transformer.SWAB, 5).fit_transform(X)
    expected = np.array([538.8, 423.1, 307.4, 191.7, 48., 40.5, 33., 25.5, 43.6, 210.2,
                        376.8,  543.4, 1276.5, 1227., 1177.5, 1128., 953.5, 980.5, 
                        1007.5, 1034.5, 759.1, 572.7, 386.3, 199.9])
    np.testing.assert_array_almost_equal(result, expected)

def test_piecewise_linear_approximation_check_diff_in_params(X):
    transformers = [PiecewiseLinearApproximation.Transformer.SlidingWindow, 
                    PiecewiseLinearApproximation.Transformer.TopDown,
                    PiecewiseLinearApproximation.Transformer.BottomUp,
                    PiecewiseLinearApproximation.Transformer.SWAB]
    for i in range(len(transformers)):
        low_error_pla = PiecewiseLinearApproximation(transformers[i], 1)
        high_error_pla = PiecewiseLinearApproximation(transformers[i], float("inf"))
        low_error_result = low_error_pla.fit_transform(X)
        high_error_result = high_error_pla.fit_transform(X)
        assert not np.allclose(low_error_result, high_error_result)

def test_piecewise_linear_approximation_wrong_parameters(X):
    with pytest.raises(ValueError):
        PiecewiseLinearApproximation("Fake Transformer", 100)
    with pytest.raises(ValueError):
        PiecewiseLinearApproximation(PiecewiseLinearApproximation.Transformer.SWAB, "max_error")
    with pytest.raises(ValueError):
        PiecewiseLinearApproximation(PiecewiseLinearApproximation.Transformer.SWAB, 100, "buffer_size")

def test_piecewise_linear_approximation_one_segment(X):
    X = X[:2]
    pla = PiecewiseLinearApproximation(PiecewiseLinearApproximation.Transformer.BottomUp, 10)
    result = pla.fit_transform(X)
    assert 0 == len(pla.segment_dense)
    np.testing.assert_array_almost_equal(X, result, decimal=1)