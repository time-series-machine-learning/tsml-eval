"""Tests for within-subject cross-validation with contiguous folds."""

import csv
import shutil

import numpy as np
import pytest
from aeon.classification import DummyClassifier

from tsml_eval.experiments import classification_experiments
from tsml_eval.experiments.within_subject import (
    run_within_subject_experiment,
    subject_folds,
)
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH

_FITS = []


class _RecordingDummy(DummyClassifier):
    """Dummy classifier with a train_groups parameter, recording each fit."""

    def __init__(self, train_groups=None):
        self.train_groups = train_groups
        super().__init__()

    def _fit(self, X, y):
        _FITS.append((len(y), self.train_groups))
        return super()._fit(X, y)


def _dataset(tmp_path):
    """MinimalChinatown with two subjects and case tables in a shuffled time order.

    Subject "a" holds the even train and test cases, "b" the odd ones. Starts run
    backwards within each split, and the test split is a later session, so recording
    order is: train cases last to first, then test cases last to first.
    """
    dataset = "MinimalChinatown"
    data = tmp_path / "data"
    shutil.copytree(f"{_TEST_DATA_PATH}/{dataset}", data / dataset)
    for split, session in (("TRAIN", "1"), ("TEST", "2")):
        ids = ["a" if i % 2 == 0 else "b" for i in range(20)]
        (data / dataset / f"{dataset}_id_{split}.txt").write_text("\n".join(ids))
        with open(data / dataset / f"cases_{split.lower()}.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(
                ["row", "participant_id", "session_id", "run_id", "start_sample"]
            )
            for i in range(20):
                w.writerow([i, ids[i], session, "0", str(1000 * (20 - i))])
    return data, dataset


def _read(path):
    with open(path, newline="") as file:
        lines = file.read().splitlines()
    assert lines[0].startswith("# dataset=MinimalChinatown")
    return list(csv.DictReader(lines[1:]))


def test_subject_folds_are_contiguous():
    """Folds are consecutive blocks of the subject's cases in recording order."""
    groups = np.array(["s1", "s2"] * 10)
    order = [(0, 0, (0, float(-i), "")) for i in range(20)]  # reverse time order
    subject, folds = subject_folds(groups, order, 1, n_folds=4)
    assert subject == "s2"
    assert [len(f) for f in folds] == [3, 3, 2, 2]
    flat = np.concatenate(folds)
    assert list(flat) == sorted(np.flatnonzero(groups == "s2"), reverse=True)
    with pytest.raises(ValueError, match="fewer than"):
        subject_folds(groups, order, 0, n_folds=11)


def test_within_subject_file(tmp_path):
    """One file per subject, every case once, folds in recording order."""
    data, dataset = _dataset(tmp_path)
    _FITS.clear()
    clf = _RecordingDummy(train_groups=["stale"] * 40)
    run_within_subject_experiment(
        str(data),
        str(tmp_path / "res"),
        dataset,
        clf,
        "Rec",
        subject_index=1,
        n_folds=5,
    )
    out = tmp_path / "res" / "Rec" / "WithinSubject" / dataset / "subject1.csv"
    rows = _read(out)
    assert len(rows) == 20  # subject "b": 10 train + 10 test cases
    assert [int(r["subject_position"]) for r in rows] == list(range(20))
    assert [int(r["fold"]) for r in rows] == [k for k in range(5) for _ in range(4)]
    # recording order: train rows 19, 17, ..., 1, then test rows 19, ..., 1
    expected = [("TRAIN", r) for r in range(19, 0, -2)]
    expected += [("TEST", r) for r in range(19, 0, -2)]
    assert [(r["source_split"], int(r["source_row"])) for r in rows] == expected
    assert {k for k in rows[0] if k.startswith("p_")} == {"p_1", "p_2"}
    assert all(abs(float(r["p_1"]) + float(r["p_2"]) - 1) < 1e-6 for r in rows)
    # one subject cannot be split by subject, so train_groups is dropped
    assert _FITS == [(16, None)] * 5
    assert not (out.parent / "subject1.csv.partial").exists()


def test_within_subject_resumes(tmp_path):
    """Folds already in the .partial file are not fitted again."""
    data, dataset = _dataset(tmp_path)
    res = tmp_path / "res"
    run_within_subject_experiment(
        str(data),
        str(res),
        dataset,
        _RecordingDummy(),
        "Rec",
        subject_index=0,
        n_folds=5,
    )
    out = res / "Rec" / "WithinSubject" / dataset / "subject0.csv"
    lines = out.read_text().splitlines()
    kept = [lines[1]] + [x for x in lines[2:] if x.split(",")[0] in ("0", "1")]
    (out.parent / "subject0.csv.partial").write_text("\n".join(kept) + "\n")
    out.unlink()
    _FITS.clear()
    run_within_subject_experiment(
        str(data),
        str(res),
        dataset,
        _RecordingDummy(),
        "Rec",
        subject_index=0,
        n_folds=5,
    )
    assert len(_FITS) == 3
    assert len(_read(out)) == 20


def test_within_subject_command_line(tmp_path):
    """-ws runs the within-subject experiment for subject {resample_id}."""
    data, dataset = _dataset(tmp_path)
    args = [
        str(data),
        str(tmp_path / "res"),
        "DummyClassifier",
        dataset,
        "0",
        "-ws",
        "4",
    ]
    classification_experiments.run_experiment(args)
    out = tmp_path / "res" / "DummyClassifier" / "WithinSubject" / dataset
    assert len(_read(out / "subject0.csv")) == 20
