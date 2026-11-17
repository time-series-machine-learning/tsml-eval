"""General tests for the experiments module."""

import os
import shutil

from aeon.classification import DummyClassifier

from tsml_eval.experiments import (
    classification_experiments,
    load_and_run_classification_experiment,
)
from tsml_eval.experiments.tests import _CLASSIFIER_RESULTS_PATH
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH, _TEST_OUTPUT_PATH
from tsml_eval.utils.tests.test_results_writing import _check_classification_file_format


def test_kwargs():
    """Test experiments with kwargs input."""
    dataset = "MinimalChinatown"
    classifier = "LogisticRegression"

    result_path = _TEST_OUTPUT_PATH + "/kwargs/"

    args = [
        _TEST_DATA_PATH,
        result_path,
        classifier,
        dataset,
        "0",
        "--kwargs",
        "fit_intercept",
        "False",
        "bool",
        "--kwargs",
        "C",
        "0.8",
        "float",
        "--kwargs",
        "max_iter",
        "10",
        "int",
        "-ow",
    ]

    classification_experiments.run_experiment(args)

    test_file = f"{result_path}{classifier}/Predictions/{dataset}/testResample0.csv"

    assert os.path.exists(test_file)
    os.remove(test_file)


def test_experiments_predefined_resample_data_loading():
    """Test experiments with data loading."""
    dataset = "PredefinedChinatown"

    load_and_run_classification_experiment(
        _TEST_DATA_PATH + "_test_data/",
        _CLASSIFIER_RESULTS_PATH,
        dataset,
        DummyClassifier(),
        resample_id=5,
        predefined_resample=True,
    )

    test_file = (
        f"{_CLASSIFIER_RESULTS_PATH}/DummyClassifier/Predictions/{dataset}/"
        "testResample5.csv"
    )
    assert os.path.exists(test_file)
    _check_classification_file_format(test_file)

    os.remove(test_file)


def test_experiments_cv_folds_by_subject(tmp_path):
    """Cross-validation folds keep subjects apart and use the resample suffix."""
    dataset = "MinimalChinatown"
    data = tmp_path / "data"
    shutil.copytree(f"{_TEST_DATA_PATH}/{dataset}", data / dataset)
    # 20 train and 20 test cases, 9 subjects: leave one subject out, 9 folds
    for split, start in (("TRAIN", 0), ("TEST", 20)):
        ids = [str((start + i) % 9) for i in range(20)]
        (data / dataset / f"{dataset}_id_{split}.txt").write_text("\n".join(ids))

    results = tmp_path / "results"
    args = [str(data), str(results), "DummyClassifier", dataset, "8", "-cv", "10"]
    classification_experiments.run_experiment(args)
    test_file = results / "DummyClassifier" / "Predictions" / dataset
    test_file = test_file / "testResample8.csv"
    assert test_file.exists()
    _check_classification_file_format(str(test_file))


_CHECKPOINT_CALLS = []


class _CheckpointingMock(DummyClassifier):
    """Dummy classifier with the aeon checkpoint API, recording what is called."""

    _tags = {
        **DummyClassifier._tags,
        "capability:checkpointing": True,
        "capability:train_estimate": True,
    }

    def __init__(self, checkpoint_path=None, checkpoint_interval=None, verbose=0):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        super().__init__()

    def _fit_predict_proba(self, X, y):
        _CHECKPOINT_CALLS.append("fit_predict_proba")
        self._fit(X, y)
        self._save()
        return self._predict_proba(X)

    def resume_fit_predict_proba(self, X, y):
        _CHECKPOINT_CALLS.append("resume_fit_predict_proba")
        return self._predict_proba(self._preprocess_collection(X))

    def resume_fit(self, X, y):
        _CHECKPOINT_CALLS.append("resume_fit")
        return self

    def _save(self):
        if self.checkpoint_path:
            import pickle

            with open(self.checkpoint_path, "wb") as file:
                pickle.dump(self, file)

    @classmethod
    def load_checkpoint(cls, path):
        import pickle

        _CHECKPOINT_CALLS.append("load_checkpoint")
        with open(path, "rb") as file:
            return pickle.load(file)


def test_experiments_checkpoint_resume(tmp_path):
    """A checkpoint is used to resume a run, and removed once results are written."""
    from tsml_eval.datasets._loaders import load_minimal_chinatown

    dataset = "MinimalChinatown"
    X_train, y_train = load_minimal_chinatown(split="TRAIN")
    results = tmp_path / "results"
    checkpoint = results / "Mock" / "Checkpoints" / dataset / "Resample0.ckpt"

    def run():
        load_and_run_classification_experiment(
            _TEST_DATA_PATH,
            str(results),
            dataset,
            _CheckpointingMock(checkpoint_interval=30),
            classifier_name="Mock",
            build_train_file=True,
            overwrite=True,
            checkpoint=True,
        )

    # a fresh run fits with train estimates, saving a checkpoint, then removes it
    _CHECKPOINT_CALLS.clear()
    run()
    assert _CHECKPOINT_CALLS == ["fit_predict_proba"]
    assert not checkpoint.exists()
    predictions = results / "Mock" / "Predictions" / dataset
    assert (predictions / "trainResample0.csv").exists()
    assert (predictions / "testResample0.csv").exists()

    # a run killed part way leaves a checkpoint: the next run resumes from it
    killed = _CheckpointingMock(checkpoint_path=str(checkpoint))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    killed.fit(X_train, y_train)
    killed._save()
    _CHECKPOINT_CALLS.clear()
    run()
    assert _CHECKPOINT_CALLS == ["load_checkpoint", "resume_fit_predict_proba"]
    assert not checkpoint.exists()
    _check_classification_file_format(str(predictions / "trainResample0.csv"))
