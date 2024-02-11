"""Tests for arguments parsing."""

import pytest

from tsml_eval.testing.testing_utils import suppress_output
from tsml_eval.utils.arguments import parse_args


def test_positional_args():
    """Test parsing of positional arguments."""
    args = [
        "D:/data/",
        "D:/results/",
        "Est",
        "Dataset",
        "0",
    ]
    args = parse_args(args)

    assert args.data_path == "D:/data/"
    assert args.results_path == "D:/results/"
    assert args.estimator_name == "Est"
    assert args.dataset_name == "Dataset"
    assert args.resample_id == 0
    assert args.random_seed is None
    assert args.n_jobs == 1
    assert args.train_fold is False
    assert args.kwargs == {}


@pytest.mark.parametrize(
    "args",
    [
        ["D:/data/", "D:/results/", "Est", "Dataset"],
        ["D:/data/", "D:/results/", "Est", "Dataset", "zero"],
        ["D:/data/", "D:/results/", "Est", "Dataset", 1],
        ["D:/data/", "D:/results/", "Est", "Dataset", "0", "1"],
        [[1, 2], "D:/results/", "Est", "Dataset", "0"],
    ],
)
@suppress_output()
def test_wrong_positional_args(args):
    """Test parsing of incorrect positional arguments."""
    with pytest.raises((SystemExit, TypeError, ValueError)):
        parse_args(args)


def test_kw_args():
    """Test parsing of keyword arguments."""
    args = [
        "D:/data/",
        "D:/results/",
        "Est",
        "Dataset",
        "1",
        "-rs",
        "10",
        "--n_jobs",
        "4",
        "-tr",
        "--kwargs",
        "key1",
        "value1",
        "str",
        "-kw",
        "key2",
        "value2",
        "str",
    ]
    args = parse_args(args)

    assert args.data_path == "D:/data/"
    assert args.results_path == "D:/results/"
    assert args.estimator_name == "Est"
    assert args.dataset_name == "Dataset"
    assert args.resample_id == 1
    assert args.random_seed == 10
    assert args.n_jobs == 4
    assert args.train_fold is True
    assert args.kwargs["key1"] == "value1"
    assert args.kwargs["key2"] == "value2"


@pytest.mark.parametrize(
    "args",
    [
        ["D:/data/", "D:/results/", "Est", "Dataset", "1", "-rs"],
        ["D:/data/", "D:/results/", "Est", "Dataset", "1", "-rs", "zero"],
        ["D:/data/", "D:/results/", "Est", "Dataset", "1", "-tr", "True"],
        ["D:/data/", "D:/results/", "Est", "Dataset", "1", "-kw", "key value str"],
        ["D:/data/", "D:/results/", "Est", "Dataset", "1", "-kw", "key", "value"],
    ],
)
@suppress_output()
def test_wrong_kw_args(args):
    """Test parsing of incorrect keyword arguments."""
    with pytest.raises((SystemExit, TypeError, ValueError)):
        parse_args(args)
