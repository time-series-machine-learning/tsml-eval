"""Build HIVE-COTE ensembles from results files without loading the data.

``FromFileHIVECOTE`` forms the ensemble from component results files, but it is still
a classifier: ``fit`` and ``predict`` require the time series to be passed in, so an
experiment using it must load the archive. Everything the ensemble actually uses is
already in the component files, so the functions here take the case count and class
labels from those files instead and drive ``FromFileHIVECOTE`` with a placeholder X.

This makes sweeping HC2 variants over an archive cheap, as no data is read and no
component is refit.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "component_prediction_paths",
    "build_hivecote_from_results",
]

import os

import numpy as np
from sklearn.utils import check_random_state

from tsml_eval.estimators.classification.hybrid.hivecote_from_file import (
    FromFileHIVECOTE,
)
from tsml_eval.evaluation.storage import ClassifierResults
from tsml_eval.utils.functions import time_to_milliseconds


def component_prediction_paths(paths, dataset):
    """Return the ``Predictions/<dataset>/`` directory of each component.

    Parameters
    ----------
    paths : list of str
        Component results directories, each containing ``Predictions/<dataset>/``.
    dataset : str
        Name of the dataset.

    Returns
    -------
    list of str
        Paths ending in a separator, as expected by ``FromFileHIVECOTE``.
    """
    return [os.path.join(p, "Predictions", dataset, "") for p in paths]


def _results_file(path, resample_id, split):
    """Path of a component results file for a resample and split."""
    name = "trainResample" if split == "TRAIN" else "testResample"
    suffix = "" if resample_id is None else str(resample_id)
    return f"{path}{name}{suffix}.csv"


def _load_split(paths, resample_id, split, verify_values=False):
    """Load one split of every component, checking the files line up."""
    results = []
    for path in paths:
        file = _results_file(path, resample_id, split)
        results.append(
            ClassifierResults().load_from_file(file, verify_values=verify_values)
        )

    ref = results[0]
    ref_file = _results_file(paths[0], resample_id, split)
    for cr, path in zip(results[1:], paths[1:]):
        file = _results_file(path, resample_id, split)
        if cr.n_cases != ref.n_cases:
            raise ValueError(
                f"n_cases of {file} does not match {ref_file}, expected "
                f"{ref.n_cases}, got {cr.n_cases}"
            )
        if cr.n_classes != ref.n_classes:
            raise ValueError(
                f"n_classes of {file} does not match {ref_file}, expected "
                f"{ref.n_classes}, got {cr.n_classes}"
            )
        if not np.array_equal(cr.class_labels, ref.class_labels):
            raise ValueError(f"class labels of {file} do not match {ref_file}")

    return results


def _placeholder_X(n_cases):
    """A minimal valid collection standing in for data we never look at."""
    return np.zeros((n_cases, 1, 2))


def _total_fit_time(train_results):
    """Total build plus error estimate time of the components, in milliseconds."""
    total = 0
    for cr in train_results:
        if cr.fit_and_estimate_time is None or cr.fit_and_estimate_time == -1:
            return -1
        total += time_to_milliseconds(cr.fit_and_estimate_time, cr.time_unit)
    return total


def _total_predict_time(results):
    """Total prediction time of the components, in milliseconds."""
    total = 0
    for cr in results:
        if cr.predict_time is None or cr.predict_time == -1:
            return -1
        total += time_to_milliseconds(cr.predict_time, cr.time_unit)
    return total


def build_hivecote_from_results(
    paths,
    dataset,
    resample_id,
    output_path,
    classifier_name="HC2",
    write_train_file=False,
    verify_values=False,
    prior_correction=0.0,
    **hivecote_params,
):
    """Build a HIVE-COTE from component results files and write the results.

    The component train files provide the accuracy estimates used for the CAWPE
    weights, which are applied to the component test files. The data itself is never
    loaded.

    Parameters
    ----------
    paths : list of str
        Component results directories, each containing
        ``Predictions/<dataset>/<split>Resample<resample_id>.csv``.
    dataset : str
        Name of the dataset.
    resample_id : int or None
        Resample to build the ensemble for. Also used as the ``random_state`` of the
        ensemble, as ``FromFileHIVECOTE`` selects the component files by random state.
    output_path : str
        Directory to write to, using the standard
        ``<classifier_name>/Predictions/<dataset>/`` structure.
    classifier_name : str, default="HC2"
        Name written to the results file and used in the output file structure.
    write_train_file : bool, default=False
        Whether to also combine the component train files and write a train results
        file for the ensemble.
    verify_values : bool, default=False
        Whether loading a component file re-derives its stored statistics. Slow when
        sweeping an archive.
    prior_correction : float, default=0.0
        Exponent ``beta`` for dividing the combined posterior by the train class prior
        raised to ``beta`` before predicting. 0 leaves the CAWPE decision rule
        unchanged, 1 is full prior correction (the Bayes rule for a uniform prior).
        Intermediate values partially correct the majority-class bias of the combined
        posterior. On the 112 UCR datasets over 30 resamples, ``beta=0.5`` improved
        both accuracy (+0.17 pp) and balanced accuracy (+0.74 pp); see
        ``tsml_eval/_wip/hc2_ensemble/NOTES.md``.
    **hivecote_params
        Further parameters for ``FromFileHIVECOTE``, e.g. ``alpha``, ``tune_alpha``,
        ``new_weights`` or ``acc_filter``.

    Returns
    -------
    test_results : ClassifierResults
        The written test results.
    train_results : ClassifierResults or None
        The written train results, or None if ``write_train_file`` is False.
    """
    component_paths = component_prediction_paths(paths, dataset)

    train_results = _load_split(
        component_paths, resample_id, "TRAIN", verify_values=verify_values
    )
    test_results = _load_split(
        component_paths, resample_id, "TEST", verify_values=verify_values
    )

    # The labels and case counts come from the files, so no data is needed. The
    # component files store label indices, which are the class labels here.
    train_labels = train_results[0].class_labels
    hc = FromFileHIVECOTE(
        classifiers=component_paths,
        random_state=resample_id,
        skip_shape_check=True,
        **hivecote_params,
    )
    hc.fit(_placeholder_X(train_results[0].n_cases), train_labels)

    prior = None
    if prior_correction:
        _, counts = np.unique(train_labels, return_counts=True)
        prior = counts / counts.sum()

    fit_time = _total_fit_time(train_results)
    weights = [float(w) for w in hc.weights_]
    params = hc.get_params(deep=False)
    params.pop("classifiers")
    parameter_info = (
        f"Built from files by build_hivecote_from_results. weights={weights}, "
        f"prior_correction={prior_correction}, components={list(paths)}, "
        f"params={params}"
    )

    written_test = _predict_and_write(
        hc,
        test_results,
        dataset,
        resample_id,
        "TEST",
        classifier_name,
        parameter_info,
        output_path,
        fit_time,
        prior=prior,
        beta=prior_correction,
    )

    written_train = None
    if write_train_file:
        # The fitted weights, voting over the component train probabilities.
        written_train = _predict_and_write(
            hc,
            train_results,
            dataset,
            resample_id,
            "TRAIN",
            classifier_name,
            parameter_info,
            output_path,
            fit_time,
            probabilities=_weighted_vote(train_results, hc),
            prior=prior,
            beta=prior_correction,
        )

    return written_test, written_train


def _weighted_vote(results, hc):
    """Apply the fitted weights to already loaded component probabilities."""
    dists = np.zeros(results[0].probabilities.shape)
    for i, cr in enumerate(results):
        if hc._use_classifier[i]:
            dists = np.add(dists, cr.probabilities * hc.weights_[i])
    return dists / dists.sum(axis=1, keepdims=True)


def _predict_and_write(
    hc,
    results,
    dataset,
    resample_id,
    split,
    classifier_name,
    parameter_info,
    output_path,
    fit_time,
    probabilities=None,
    prior=None,
    beta=0.0,
):
    """Combine one split and write the ensemble results file."""
    if probabilities is None:
        probabilities = hc.predict_proba(_placeholder_X(results[0].n_cases))

    if beta:
        # Decide under a partially uniform class prior. The written probabilities are
        # the corrected ones, so the file's argmax matches the predictions.
        probabilities = probabilities / (prior**beta)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    # Tie breaking matches the aeon HIVE-COTE classifiers.
    rng = check_random_state(resample_id)
    predictions = np.array(
        [int(rng.choice(np.flatnonzero(prob == prob.max()))) for prob in probabilities]
    )

    cr = ClassifierResults(
        dataset_name=dataset,
        classifier_name=classifier_name,
        split=split,
        resample_id=resample_id,
        time_unit="MILLISECONDS",
        description="Generated by build_hivecote_from_results.",
        parameter_info=parameter_info,
        fit_time=fit_time,
        predict_time=_total_predict_time(results),
        n_classes=results[0].n_classes,
        # The reader parses labels with int(), so keep them as integer indices.
        class_labels=np.asarray(results[0].class_labels).astype(int),
        predictions=predictions,
        probabilities=probabilities,
    )
    cr.save_to_file(output_path, full_path=False)
    return cr
