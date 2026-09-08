"""Run the explanatory simulation protocol through tsml-eval's experiment machinery.

Each protocol condition is one "dataset" and each replicate is one "resample", so
results land in the standard hierarchy

    <results_path>/<estimator>/Predictions/<condition>/testResample<r>.csv

written by :func:`tsml_eval.experiments.run_classification_experiment`, and are
readable by the normal tsml-eval loaders. Replicates are independently generated
populations rather than resamples of one finite dataset; that distinction is
recorded in the manifest this module writes alongside the results, since the file
layout cannot express it.

Usage::

    # what would run, without running it
    python run_simulation_experiments.py --dry-run

    # the protocol's feasibility pilot: 8 anchor conditions, 5 replicates
    python run_simulation_experiments.py --pilot --results-path /path/to/results

    # a single cheap cell, to check the plumbing
    python run_simulation_experiments.py --smoke --results-path /path/to/results
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["ESTIMATORS", "pilot_conditions", "run_condition", "main"]

import argparse
import json
import os
import sys

import numpy as np

from tsml_eval._wip.simulation._protocol import (
    REFERENCE_LENGTH,
    condition_id,
    draw_anchor,
    protocol_conditions,
    simulate_protocol_problem,
)

#: The ordinary estimators, and the three diagnostics that are excluded from
#: method-family rankings. Diagnostics are named so the analysis can drop them.
ESTIMATORS = (
    "tsf",
    "drcif",
    "rstsf",
    "quant",
    "pulsar",
    "rocket",
    "rdst",
)
DIAGNOSTICS = ("dummy",)

N_TRAIN = 200
N_TEST = 1000
N_REPLICATES = 30


def pilot_conditions(conditions):
    """The feasibility pilot: the aligned and uniform anchor cells only.

    Eight conditions, one aligned and one uniform for each mechanism, at the
    reference length and strength.
    """
    wanted = []
    for condition in conditions:
        if condition["sweep"] != "alignment":
            continue
        if condition["length"] != REFERENCE_LENGTH or condition["strength"] != 1.0:
            continue
        if condition["uniform_location"] or condition["jitter"] == 0:
            wanted.append(condition)
    return wanted


def _generate(condition, replicate, seed_base):
    """Training and test splits for one condition and replicate.

    The population anchor is drawn once and shared by the two splits, and the
    splits use separate streams so test cases are independent of training cases
    given the population.
    """
    anchor_rng = np.random.RandomState(seed_base + 7919 * replicate)
    anchor = draw_anchor(
        512, condition["length"], condition["jitter"], anchor_rng
    )
    common = dict(
        mechanism=condition["mechanism"],
        length=condition["length"],
        strength=condition["strength"],
        jitter=condition["jitter"],
        uniform_location=condition["uniform_location"],
        shuffle_interval=condition["shuffle_interval"],
        match_kl=condition["match_kl"],
        anchor=anchor,
    )
    X_train, y_train = simulate_protocol_problem(
        n_cases=N_TRAIN, random_state=seed_base + 2 * replicate, **common
    )
    X_test, y_test = simulate_protocol_problem(
        n_cases=N_TEST, random_state=seed_base + 2 * replicate + 1, **common
    )
    return X_train, y_train, X_test, y_test


def run_condition(condition, estimator_name, replicate, results_path, seed_base=0):
    """Run one (condition, estimator, replicate) cell and write its result file."""
    from tsml_eval.experiments import (
        get_classifier_by_name,
        run_classification_experiment,
    )

    X_train, y_train, X_test, y_test = _generate(condition, replicate, seed_base)
    classifier = get_classifier_by_name(estimator_name, random_state=replicate)
    run_classification_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier,
        results_path,
        classifier_name=estimator_name,
        dataset_name=condition["id"],
        resample_id=replicate,
        build_test_file=True,
        build_train_file=False,
    )
    return os.path.join(
        results_path,
        estimator_name,
        "Predictions",
        condition["id"],
        "testResample%d.csv" % replicate,
    )


def write_manifest(conditions, estimators, replicates, results_path):
    """Record what the file layout cannot: replicates are populations, not resamples."""
    os.makedirs(results_path, exist_ok=True)
    path = os.path.join(results_path, "protocol-manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "protocol": "focused explanatory simulation, interval-based review",
                "replicate_semantics": (
                    "each testResample<r>.csv is an independently generated "
                    "population replicate, not a resample of one finite dataset"
                ),
                "n_conditions": len(conditions),
                "n_replicates": replicates,
                "n_train": N_TRAIN,
                "n_test": N_TEST,
                "series_length": 512,
                "estimators": list(estimators),
                "conditions": conditions,
            },
            fh,
            indent=2,
        )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed-base", type=int, default=0)
    args = parser.parse_args(argv)

    conditions = protocol_conditions()
    estimators = ESTIMATORS + DIAGNOSTICS

    if args.smoke:
        conditions = conditions[:1]
        estimators = ("dummy",)
        replicates = 1
    elif args.pilot:
        conditions = pilot_conditions(conditions)
        replicates = 5
    else:
        replicates = N_REPLICATES

    total = len(conditions) * len(estimators) * replicates
    print("conditions : %d" % len(conditions))
    print("estimators : %d %s" % (len(estimators), list(estimators)))
    print("replicates : %d" % replicates)
    print("fits       : %d" % total)

    if args.dry_run:
        for condition in conditions:
            print("  %-12s %s" % (condition["sweep"], condition["id"]))
        return 0

    if args.results_path is None:
        parser.error("--results-path is required unless --dry-run is given")

    write_manifest(conditions, estimators, replicates, args.results_path)
    done = 0
    for condition in conditions:
        for estimator_name in estimators:
            for replicate in range(replicates):
                path = run_condition(
                    condition,
                    estimator_name,
                    replicate,
                    args.results_path,
                    seed_base=args.seed_base,
                )
                done += 1
                print("[%d/%d] %s" % (done, total, path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
