"""Check that aeon checkpointing survives a killed fit (the HC2 components).

For each estimator this fits once without interruption, then fits again in a child
process with checkpointing on and kills that process as soon as the first
checkpoint file appears, as Slurm would at a time limit. It then loads the
checkpoint, calls resume_fit and checks that the resumed model predicts exactly as
the uninterrupted one. Arsenal also checks its train estimates (the train-file
predictions) survive the resume.

Needs an aeon with checkpointing for all four (ajb/hc2-components-checkpoint). Takes a few minutes:

    python test_checkpointing.py
"""

import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SEED = 0


def data():
    """Return a small random multivariate problem, train and test."""
    rng = np.random.RandomState(SEED)
    X = rng.normal(size=(60, 3, 80))
    y = np.array(["a", "b"] * 30)
    X[y == "b", :, 30:40] += 1.5
    return X[:40], y[:40], X[40:], y[40:]


def make(name, path=None):
    """Return the estimator, with checkpointing on if a path is given."""
    checkpoint = {}
    if path is not None:
        # a tiny interval writes at every safe boundary, so a kill soon after the
        # first write leaves work still to do
        checkpoint = {"checkpoint_path": path, "checkpoint_interval": 1e-6}
    if name == "Arsenal":
        from aeon.classification.convolution_based import Arsenal

        return Arsenal(n_estimators=8, n_kernels=200, random_state=SEED, **checkpoint)
    if name == "DrCIF":
        from aeon.classification.interval_based import DrCIFClassifier

        return DrCIFClassifier(n_estimators=8, random_state=SEED, **checkpoint)
    if name == "TDE":
        from aeon.classification.dictionary_based import TemporalDictionaryEnsemble

        return TemporalDictionaryEnsemble(
            n_parameter_samples=20, max_ensemble_size=5, random_state=SEED, **checkpoint
        )
    from aeon.classification.shapelet_based import ShapeletTransformClassifier
    from aeon.classification.sklearn import RotationForestClassifier

    return ShapeletTransformClassifier(
        n_shapelet_samples=200,
        max_shapelets=20,
        estimator=RotationForestClassifier(n_estimators=20, random_state=SEED),
        random_state=SEED,
        **checkpoint,
    )


def child(name, path, train_estimate):
    """Fit with checkpointing; the parent kills this process mid-fit."""
    X, y, _, _ = data()
    est = make(name, path)
    if train_estimate:
        est.fit_predict_proba(X, y)
    else:
        est.fit(X, y)


def arsenal_train_proba(est):
    """Rebuild Arsenal's train estimates from its retained continuation state."""
    from aeon.classification.convolution_based import _arsenal

    class_indices, weights, oobs = zip(*est._train_estimates)
    votes = _arsenal._aggregate_class_votes(
        class_indices, weights, est.n_cases_, est.n_classes_, oobs
    )
    return _arsenal._normalise_oob_probabilities(votes, weights, oobs, est.n_classes_)


def check(name, train_estimate):
    """Kill a checkpointed fit, resume it, and compare with an uninterrupted fit."""
    from aeon.base import CheckpointableMixin  # noqa: F401  (fails on old aeon)

    X, y, X_test, _ = data()
    reference = make(name)
    if train_estimate:
        ref_train = reference.fit_predict_proba(X, y)
    else:
        reference.fit(X, y)
    ref_test = reference.predict_proba(X_test)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.ckpt"
        proc = mp.Process(target=child, args=(name, str(path), train_estimate))
        proc.start()
        deadline = time.time() + 600
        while not path.exists() and proc.is_alive() and time.time() < deadline:
            time.sleep(0.05)
        killed = proc.is_alive()
        proc.kill()
        proc.join()
        if not path.exists():
            return f"FAIL {name}: no checkpoint was written"
        if not killed:
            return f"SKIP {name}: fit finished before it could be killed"

        est = type(reference).load_checkpoint(path)
        built = getattr(est, "n_estimators_", "?")
        est.resume_fit(X, y)
        same_test = np.allclose(est.predict_proba(X_test), ref_test)
        result = (
            f"{'PASS' if same_test else 'FAIL'} {name}: killed after first "
            f"checkpoint (members built: {built}); resumed test probabilities "
            f"{'match' if same_test else 'DIFFER from'} an uninterrupted fit"
        )
        if train_estimate:
            same_train = np.allclose(arsenal_train_proba(est), ref_train)
            result += f"; train estimates {'match' if same_train else 'DIFFER'}"
            same_test = same_test and same_train
        return result, same_test


def main():
    """Run the checks and exit non-zero if any fail."""
    import aeon

    print("aeon", aeon.__version__, os.path.dirname(aeon.__file__))
    ok = True
    checks = (("Arsenal", True), ("STC", False), ("DrCIF", False), ("TDE", False))
    for name, train_estimate in checks:
        outcome = check(name, train_estimate)
        if isinstance(outcome, str):
            print(outcome)
            ok = ok and outcome.startswith("SKIP")
        else:
            print(outcome[0])
            ok = ok and outcome[1]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
