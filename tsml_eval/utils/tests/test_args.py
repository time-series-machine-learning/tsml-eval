import pytest

from tsml_eval.utils.experiments import parse_args
from tsml_eval.utils.test_utils import suppress_output


def test_positional_args():
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
    assert args.kwargs is None


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
    with pytest.raises((SystemExit, TypeError, ValueError)):
        parse_args(args)


def test_kw_args():
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
        "-kw",
        "key2",
        "value2",
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
    assert args.kwargs[0][0] == "key1"
    assert args.kwargs[0][1] == "value1"
    assert args.kwargs[1][0] == "key2"
    assert args.kwargs[1][1] == "value2"


@pytest.mark.parametrize(
    "args",
    [
        ["D:/data/", "D:/results/", "Est", "Dataset", "1", "-rs"],
        ["D:/data/", "D:/results/", "Est", "Dataset", "1", "-rs", "zero"],
        ["D:/data/", "D:/results/", "Est", "Dataset", "1", "-tr", "True"],
        ["D:/data/", "D:/results/", "Est", "Dataset", "1", "-kw", "key value"],
    ],
)
@suppress_output()
def test_wrong_kw_args(args):
    with pytest.raises((SystemExit, TypeError, ValueError)):
        parse_args(args)
