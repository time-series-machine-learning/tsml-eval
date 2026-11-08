"""Tabular foundation models as level-1 (stacking) models.

Imported lazily by stacking_models.stackers(foundation=True), so the classical
models run without these packages:

    pip install tabpfn tabpfn-extensions tabicl

TabPFN collects usage telemetry unless TABPFN_DISABLE_TELEMETRY=1, which this module
sets. Weights download on first use. TabPFN v2 and TabICL need no account; TabPFN v2.5
needs a Prior Labs API key in TABPFN_TOKEN and is left out without one.
"""

import os

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa: E402

TABPFN_MAX_CLASSES = 10


class TabPFNStacker(ClassifierMixin, BaseEstimator):
    """TabPFN, pinned to a model version, with many-class handling.

    More than 10 classes goes through tabpfn_extensions' ManyClassClassifier (output
    codes over at most 10-class sub-problems).
    """

    def __init__(self, version="v2", device="auto", random_state=0):
        self.version = version
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        import numpy as np
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        est = TabPFNClassifier.create_default_for_version(
            ModelVersion(self.version),
            device=self.device,
            random_state=self.random_state,
            ignore_pretraining_limits=True,
        )
        self.classes_ = np.unique(y)
        if len(self.classes_) > TABPFN_MAX_CLASSES:
            from tabpfn_extensions.many_class import ManyClassClassifier

            est = ManyClassClassifier(
                estimator=est,
                alphabet_size=TABPFN_MAX_CLASSES,
                random_state=self.random_state,
            )
        self.model_ = clone(est).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TabICLStacker(ClassifierMixin, BaseEstimator):
    """TabICL; it handles more than 10 classes itself."""

    def __init__(self, n_estimators=8, device=None, random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        import numpy as np
        from tabicl import TabICLClassifier

        self.classes_ = np.unique(y)
        self.model_ = TabICLClassifier(
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state,
        ).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def foundation_stackers(random_state=0, device=None):
    """{name: (category, estimator)} for the foundation models."""
    pfn = dict(device=device or "auto", random_state=random_state)
    models = {
        "TabPFN-v2": ("Foundation", TabPFNStacker("v2", **pfn)),
        "TabICL": (
            "Foundation",
            TabICLStacker(device=device, random_state=random_state),
        ),
    }
    # v2.5 weights need a Prior Labs licence and API key; without TABPFN_TOKEN the
    # download waits for a browser login, so it is only included with a key
    if os.environ.get("TABPFN_TOKEN"):
        models["TabPFN-v2.5"] = ("Foundation", TabPFNStacker("v2.5", **pfn))
    return models
