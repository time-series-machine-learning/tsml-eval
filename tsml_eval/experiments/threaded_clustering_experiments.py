"""Clustering Experiments: code for experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard tsml format.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]


import sys

from tsml_eval.experiments import load_and_run_clustering_experiment
from tsml_eval.experiments.classification_experiments import _results_present
from tsml_eval.experiments.set_clusterer import set_clusterer
from tsml_eval.utils.experiments import parse_args
from tsml_eval.utils.functions import pair_list_to_dict


def run_experiment(args, overwrite=False):
    """Mechanism for testing clusterers on the UCR data format.

    This mirrors the mechanism used in the Java based tsml. Results generated using the
    method are in the same format as tsml and can be directly compared to the results
    generated in Java.
    """
    # cluster run (with args), this is fragile
    # don't run threaded jobs on ADA unless you have reserved the whole node and know
    # what you are doing
    if args is not None and args.__len__() > 1:
        print("Input args = ", args)
        args = parse_args(args)

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not args.overwrite and _results_present(
            args.results_path,
            args.estimator_name,
            args.dataset_name,
            resample_id=args.resample_id,
            split="BOTH" if args.test_fold else "TRAIN",
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_clustering_experiment(
                args.data_path,
                args.results_path,
                args.dataset_name,
                set_clusterer(
                    args.estimator_name,
                    random_state=args.resample_id
                    if args.random_seed is None
                    else args.random_seed,
                    n_jobs=args.n_jobs,
                    fit_contract=args.fit_contract,
                    checkpoint=args.checkpoint,
                    **pair_list_to_dict(args.kwargs),
                ),
                resample_id=args.resample_id,
                clusterer_name=args.estimator_name,
                overwrite=args.overwrite,
                build_test_file=args.test_fold,
                predefined_resample=args.predefined_resample,
            )
    # local run (no args)
    else:
        # These are example parameters, change as required for local runs
        # Do not include paths to your local directories here in PRs
        data_path = "../"
        results_path = "../"
        estimator_name = "KMeans-DTW"
        dataset_name = "ArrowHead"
        resample_id = 0
        n_jobs = 1
        overwrite = False
        predefined_resample = False
        test_fold = False
        fit_contract = None
        checkpoint = None
        kwargs = {}

        clusterer = set_clusterer(
            estimator_name,
            random_state=resample_id,
            n_jobs=n_jobs,
            fit_contract=fit_contract,
            checkpoint=checkpoint,
            **kwargs,
        )
        print(f"Local Run of {estimator_name} ({clusterer.__class__.__name__}).")

        load_and_run_clustering_experiment(
            data_path,
            results_path,
            dataset_name,
            clusterer,
            resample_id=resample_id,
            clusterer_name=estimator_name,
            overwrite=overwrite,
            build_test_file=test_fold,
            predefined_resample=predefined_resample,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
