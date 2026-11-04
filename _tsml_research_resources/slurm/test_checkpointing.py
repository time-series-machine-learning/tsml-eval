"""Check that aeon checkpointing survives a killed fit (the HC2 components).

For each estimator this fits once without interruption, then fits again in a child
process with checkpointing on and kills that process as soon as the first
checkpoint file appears, as Slurm would at a time limit. It then loads the
checkpoint, resumes it and checks that the resumed model predicts exactly as the
uninterrupted one. Fits begun with fit_predict_proba (as tsml-eval does for train
files) are resumed with resume_fit_predict_proba, and their train estimates must
match too.

Needs an aeon with checkpointing for all four (the ajb/hc2-components-checkpoint
branch). Takes a few minutes:

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


def _writes(path, previous):
    """Return how many times the checkpoint file has been (re)written so far."""
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return previous[0], previous[1]
    if stamp != previous[1]:
        return previous[0] + 1, stamp
    return previous


def check(name, train_estimate, kill_after=1):
    """Kill a checkpointed fit, resume it, and compare with an uninterrupted fit.

    The child is killed once the checkpoint has been written ``kill_after``
    times. A fit begun with fit_predict_proba is resumed with
    resume_fit_predict_proba, whose train estimates must also match.
    """
    from aeon.base import CheckpointableMixin  # noqa: F401  (fails on old aeon)

    X, y, X_test, _ = data()
    reference = make(name)
    if train_estimate:
        ref_train = reference.fit_predict_proba(X, y)
    else:
        reference.fit(X, y)
    ref_test = reference.predict_proba(X_test)
    how = "fit_predict_proba" if train_estimate else "fit"
    label = f"{name} ({how}, killed after checkpoint {kill_after})"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.ckpt"
        # spawn, not fork: after the reference fit this process has started
        # OpenMP, and GNU OpenMP aborts in a forked child (the Linux default)
        proc = mp.get_context("spawn").Process(
            target=child, args=(name, str(path), train_estimate)
        )
        proc.start()
        deadline = time.time() + 600
        writes = (0, None)
        while writes[0] < kill_after and proc.is_alive() and time.time() < deadline:
            writes = _writes(path, writes)
            time.sleep(0.01)
        killed = proc.is_alive()
        proc.kill()
        proc.join()
        if not path.exists():
            return f"FAIL {label}: no checkpoint was written", False
        if not killed:
            return f"SKIP {label}: fit finished before it could be killed", True

        est = type(reference).load_checkpoint(path)
        if train_estimate:
            if not hasattr(est, "resume_fit_predict_proba"):
                return f"FAIL {label}: aeon has no resume_fit_predict_proba", False
            train = est.resume_fit_predict_proba(X, y)
        else:
            est.resume_fit(X, y)
        ok = np.allclose(est.predict_proba(X_test), ref_test)
        result = f"test probabilities {'match' if ok else 'DIFFER'}"
        if train_estimate:
            same_train = np.allclose(train, ref_train)
            result += f", train estimates {'match' if same_train else 'DIFFER'}"
            ok = ok and same_train
        return f"{'PASS' if ok else 'FAIL'} {label}: {result}", ok


def main():
    """Run the checks and exit non-zero if any fail."""
    import aeon

    print("aeon", aeon.__version__, os.path.dirname(aeon.__file__))
    checks = [(n, False, 1) for n in ("Arsenal", "STC", "DrCIF", "TDE")]
    checks += [(n, True, 1) for n in ("Arsenal", "STC", "DrCIF", "TDE")]
    # STC's first checkpoint is the end of the shapelet transform; the second is
    # inside the rotation forest, so this resumes the forest part way through
    checks += [("STC", True, 2), ("Arsenal", True, 3)]
    ok = True
    for name, train_estimate, kill_after in checks:
        message, passed = check(name, train_estimate, kill_after)
        print(message, flush=True)
        ok = ok and passed
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
