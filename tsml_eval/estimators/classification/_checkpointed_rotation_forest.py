"""Rotation Forest on concatenated channels, resumable from aeon checkpoints."""

__all__ = ["CheckpointedRotationForestClassifier"]

import numpy as np
from aeon.base import CheckpointableMixin
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.classification.sklearn import RotationForestClassifier


class CheckpointedRotationForestClassifier(CheckpointableMixin, BaseClassifier):
    """Rotation Forest on series flattened to vectors, with checkpointing.

    The model is the one tsml-eval builds for "RotationForest" on time series:
    each case's channels are concatenated into one vector, and a
    ``RotationForestClassifier`` is fitted to the vectors, its random state set by
    ``_clone_estimator`` from ``random_state`` exactly as ``SklearnToAeonClassifier``
    does. Results therefore match an uncheckpointed run with the same seed.

    tsml-eval only checkpoints aeon classifiers, and wraps scikit-learn style
    estimators in a fresh clone, so the forest's own checkpoint support is not
    reachable through "RotationForest". This classifier holds the forest as a
    checkpoint child, as ``ShapeletTransformClassifier`` does: the forest's safe
    boundaries (each batch of trees) save this classifier, and ``resume_fit``
    continues the forest from its last saved tree.

    Needs an aeon with checkpointing (``aeon.base.CheckpointableMixin`` and a
    checkpointable ``RotationForestClassifier``), such as the
    ``ajb/hc2-components-checkpoint`` branch.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees.
    time_limit_in_minutes : float, default=0.0
        Contract on total training time, across resumes; 0 means none.
    contract_max_n_estimators : int, default=500
        Maximum number of trees when contracted.
    n_jobs : int, default=1
        Threads for the forest.
    random_state : int, RandomState or None, default=None
        Seed; the forest is seeded from it as in ``SklearnToAeonClassifier``.
    checkpoint_path : str, Path or None, default=None
        Checkpoint file.
    checkpoint_interval : float or None, default=None
        Minimum minutes between periodic checkpoints.
    verbose : int, default=0
        Forest progress output.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:checkpointing": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "X_inner_type": "numpy3D",
    }

    _checkpoint_mutable_params = CheckpointableMixin._checkpoint_mutable_params + (
        "time_limit_in_minutes",
    )

    def __init__(
        self,
        n_estimators=200,
        time_limit_in_minutes=0.0,
        contract_max_n_estimators=500,
        n_jobs=1,
        random_state=None,
        checkpoint_path=None,
        checkpoint_interval=None,
        verbose=0,
    ):
        self.n_estimators = n_estimators
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        super().__init__()

    @staticmethod
    def _flatten(X):
        return X.reshape(X.shape[0], -1)

    def _apply_runtime_settings(self):
        self.estimator_.n_jobs = self.n_jobs
        self.estimator_.verbose = self.verbose
        self.estimator_.time_limit_in_minutes = self.time_limit_in_minutes

    def _fit(self, X, y):
        self.estimator_ = _clone_estimator(
            RotationForestClassifier(
                n_estimators=self.n_estimators,
                time_limit_in_minutes=self.time_limit_in_minutes,
                contract_max_n_estimators=self.contract_max_n_estimators,
                n_jobs=self.n_jobs,
            ),
            self.random_state,
        )
        self._apply_runtime_settings()
        # the forest holds no path of its own: its boundaries save this classifier
        self.estimator_._set_checkpoint_parent(self)
        # flattening is the whole preparation, so a checkpoint is valid from here
        self._checkpoint_ready = True
        self.estimator_.fit(self._flatten(X), y)
        return self

    def _resume_fit(self, X, y):
        self._apply_runtime_settings()
        self.estimator_._set_checkpoint_parent(self)
        if getattr(self.estimator_, "_checkpoint_ready", False):
            self.estimator_.resume_fit(self._flatten(X), y)
        else:
            self.estimator_.fit(self._flatten(X), y)
        return self

    def _predict(self, X):
        return self.estimator_.predict(self._flatten(X))

    def _predict_proba(self, X):
        return self.estimator_.predict_proba(self._flatten(X))

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"n_estimators": 3}


def _check_equivalence(X, y, X_test, random_state=0):  # pragma: no cover
    """Probabilities from this class and from the tsml-eval sklearn route."""
    from tsml_eval.estimators.classification._sklearn import SklearnToAeonClassifier

    a = CheckpointedRotationForestClassifier(n_estimators=5, random_state=random_state)
    b = SklearnToAeonClassifier(
        RotationForestClassifier(n_estimators=5, random_state=random_state),
        pad_unequal=True,
        concatenate_channels=True,
        random_state=random_state,
    )
    return np.allclose(a.fit(X, y).predict_proba(X_test), b.fit(X, y).predict_proba(X_test))
