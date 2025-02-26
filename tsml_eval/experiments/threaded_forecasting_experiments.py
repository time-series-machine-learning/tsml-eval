"""Regression Experiments: code for experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard tsml format.
"""

__maintainer__ = ["MatthewMiddlehurst"]

import sys

from tsml_eval.experiments._get_forecaster import get_forecaster_by_name

# todo replace when added back to init
from tsml_eval.experiments.experiments import load_and_run_forecasting_experiment
from tsml_eval.experiments.tests import _FORECASTER_RESULTS_PATH
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH
from tsml_eval.utils.arguments import parse_args
from tsml_eval.utils.experiments import _results_present


def run_experiment(args, overwrite=False):
    """Mechanism for testing forecasters using a csv data format."""
    # cluster run (with args), this is fragile
    # don't run threaded jobs on the cluster unless you have reserved the whole node
    # and know what you are doing
    if args is not None and args.__len__() > 0:
        print("Input args = ", args)
        args = parse_args(args)

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not overwrite and _results_present(
            args.results_path,
            args.estimator_name,
            args.dataset_name,
            resample_id=args.resample_id,
            split="TEST",
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_forecasting_experiment(
                args.data_path,
                args.results_path,
                args.dataset_name,
                get_forecaster_by_name(
                    args.estimator_name,
                    random_state=(
                        args.resample_id
                        if args.random_seed is None
                        else args.random_seed
                    ),
                    n_jobs=args.n_jobs,
                    **args.kwargs,
                ),
                forecaster_name=args.estimator_name,
                random_seed=(
                    args.resample_id if args.random_seed is None else args.random_seed
                ),
                write_attributes=args.write_attributes,
                att_max_shape=args.att_max_shape,
                benchmark_time=args.benchmark_time,
                overwrite=args.overwrite,
            )
    # local run (no args)
    else:
        # These are example parameters, change as required for local runs
        # Do not include paths to your local directories here in PRs
        # If threading is required, see the threaded version of this file
        data_path = _TEST_DATA_PATH
        results_path = _FORECASTER_RESULTS_PATH
        estimator_name = "DummyForecaster"
        dataset_name = "ShampooSales"
        random_seed = 0
        n_jobs = 1
        write_attributes = True
        att_max_shape = 0
        benchmark_time = True
        overwrite = False
        kwargs = {}

        forecaster = get_forecaster_by_name(
            estimator_name,
            random_state=random_seed,
            n_jobs=n_jobs,
            **kwargs,
        )
        print(f"Local Run of {estimator_name} ({forecaster.__class__.__name__}).")

        load_and_run_forecasting_experiment(
            data_path,
            results_path,
            dataset_name,
            forecaster,
            forecaster_name=estimator_name,
            random_seed=random_seed,
            write_attributes=write_attributes,
            att_max_shape=att_max_shape,
            benchmark_time=benchmark_time,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    print("Running forecasting_experiments.py main")
    run_experiment(sys.argv[1:])
