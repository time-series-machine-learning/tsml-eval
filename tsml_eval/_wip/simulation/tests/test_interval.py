"""Tests for the interval-optimal simulator."""

import numpy as np
import pytest

from tsml_eval._wip.simulation import SHAPES, simulate_interval_shape_data


def test_shape_and_labels():
    X, y = simulate_interval_shape_data(
        n_cases_per_class=(30, 20), series_length=300, random_state=0
    )
    assert X.shape == (50, 1, 300)
    assert y.shape == (50,)
    counts = dict(zip(*np.unique(y, return_counts=True)))
    assert counts == {0: 30, 1: 20}


def test_intervals_non_overlapping_and_in_range():
    _, _, p = simulate_interval_shape_data(
        series_length=400, n_intervals=4, noise_to_signal=4,
        random_state=0, return_params=True,
    )
    s, length = p["intervals"], p["interval_length"]
    assert (s >= 0).all() and (s + length <= 400).all()
    assert all(s[i + 1] >= s[i] + length for i in range(len(s) - 1))


def test_class_difference_confined_to_intervals():
    """With no noise, classes differ ONLY inside the shared intervals."""
    X, y, p = simulate_interval_shape_data(
        n_cases_per_class=(1, 1), series_length=400, noise_sigma=0.0,
        random_state=1, return_params=True,
    )
    diff = np.abs(X[y == 0][0, 0] - X[y == 1][0, 0]) > 1e-9
    inside = np.zeros(400, dtype=bool)
    for s in p["intervals"]:
        inside[s : s + p["interval_length"]] = True
    assert not (diff & ~inside).any()  # no difference outside intervals
    assert diff.any()  # but there IS a difference inside


def test_deterministic():
    a = simulate_interval_shape_data(random_state=7)
    b = simulate_interval_shape_data(random_state=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_multiclass_distinct_shapes():
    _, y, p = simulate_interval_shape_data(
        n_cases_per_class=[8] * 5, random_state=3, return_params=True
    )
    assert len(set(p["shapes"])) == 5
    assert set(np.unique(y)) == {0, 1, 2, 3, 4}


def test_noise_free_templates_are_clean():
    """Every case of a class is identical when noise_sigma=0."""
    X, y = simulate_interval_shape_data(
        n_cases_per_class=(5, 5), noise_sigma=0.0, random_state=0
    )
    for c in (0, 1):
        cls = X[y == c]
        assert np.allclose(cls, cls[0])


@pytest.mark.parametrize("name", list(SHAPES))
def test_each_shape_spans_amplitude(name):
    shape = SHAPES[name](40, amplitude=2.0, base=-1.0)
    assert shape.shape == (40,)
    assert np.isfinite(shape).all()
    # shape uses its amplitude (not flat)
    assert shape.max() - shape.min() > 0.5


def test_too_many_classes_without_shapes_raises():
    with pytest.raises(ValueError, match="distinct shapes"):
        simulate_interval_shape_data(n_cases_per_class=[5] * 6)


def test_non_distinct_shapes_raise():
    with pytest.raises(ValueError, match="distinct"):
        simulate_interval_shape_data(
            n_cases_per_class=(5, 5), shapes=["sine", "sine"]
        )


def test_intervals_too_big_raises():
    with pytest.raises(ValueError, match="do not fit"):
        simulate_interval_shape_data(
            series_length=50, n_intervals=10, interval_length=20
        )


def test_scale_discriminator_variance_in_intervals():
    """'scale': no deterministic template; classes differ in in-interval variance."""
    X, y, p = simulate_interval_shape_data(
        n_cases_per_class=(200, 200), series_length=300, discriminator="scale",
        n_intervals=3, noise_sigma=1.0, random_state=0, return_params=True,
    )
    inside = np.zeros(300, dtype=bool)
    for s in p["intervals"]:
        inside[s : s + p["interval_length"]] = True
    # per-class std inside the intervals should differ (the discriminator);
    # outside it should match the background for both classes
    std0_in = X[y == 0][:, 0, inside].std()
    std1_in = X[y == 1][:, 0, inside].std()
    std0_out = X[y == 0][:, 0, ~inside].std()
    assert std1_in > std0_in * 1.3  # class 1 clearly noisier in-interval
    assert abs(std0_out - 1.0) < 0.2  # background ~ noise_sigma
    assert p["shapes"] is None


def test_trend_discriminator_runs():
    X, y, p = simulate_interval_shape_data(
        n_cases_per_class=(10, 10), series_length=300, discriminator="trend",
        noise_sigma=0.0, n_intervals=1, interval_length=100, random_state=0,
        return_params=True,
    )
    s, L = p["intervals"][0], p["interval_length"]
    # opposite slopes: class-mean ramp increases for one class, decreases the other
    seg0 = X[y == 0][0, 0, s : s + L]
    seg1 = X[y == 1][0, 0, s : s + L]
    assert np.sign(seg0[-1] - seg0[0]) == -np.sign(seg1[-1] - seg1[0])


def test_invalid_discriminator_raises():
    with pytest.raises(ValueError, match="Unknown discriminator"):
        simulate_interval_shape_data(discriminator="nonsense")
