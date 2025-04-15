"""Classification Experiments: code for experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard tsml format.
"""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os
import sys

# Do these before any other imports in i.e. numpy. This includes imports from other
# files.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MPI_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"


import numba
from aeon.utils.validation._dependencies import _check_soft_dependencies

from tsml_eval.experiments import (
    get_classifier_by_name,
    get_data_transform_by_name,
    load_and_run_classification_experiment,
)
from tsml_eval.experiments.tests import _CLASSIFIER_RESULTS_PATH
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH
from tsml_eval.utils.arguments import parse_args
from tsml_eval.utils.experiments import _results_present, assign_gpu


def run_experiment(args):
    """Mechanism for testing classifiers on the UCR data format.

    This mirrors the mechanism used in the Java based tsml. Results generated using the
    method are in the same format as tsml and can be directly compared to the results
    generated in Java.

    Attempts to avoid the use of threading as much as possible.
    """
    numba.set_num_threads(1)
    if _check_soft_dependencies("torch", severity="none"):
        import torch

        torch.set_num_threads(1)

    # if multiple GPUs are available, assign the one with the least usage to the process
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        try:
            gpu = assign_gpu(set_environ=True)
            print(f"Assigned GPU {gpu} to process.")  # pragma: no cover
        except Exception:
            print("Unable to assign GPU to process.")

    # cluster run (with args), this is fragile
    if args is not None and args.__len__() > 0:
        print("Input args = ", args)
        args = parse_args(args)

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not args.overwrite and _results_present(
            args.results_path,
            args.estimator_name,
            args.dataset_name,
            resample_id=args.resample_id,
            split="BOTH" if args.train_fold else "TEST",
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_classification_experiment(
                args.data_path,
                args.results_path,
                args.dataset_name,
                get_classifier_by_name(
                    args.estimator_name,
                    random_state=(
                        args.resample_id
                        if args.random_seed is None
                        else args.random_seed
                    ),
                    n_jobs=1,
                    fit_contract=args.fit_contract,
                    checkpoint=args.checkpoint,
                    **args.kwargs,
                ),
                classifier_name=args.estimator_name,
                resample_id=args.resample_id,
                data_transforms=get_data_transform_by_name(
                    args.data_transform_name,
                    row_normalise=args.row_normalise,
                    random_state=(
                        args.resample_id
                        if args.random_seed is None
                        else args.random_seed
                    ),
                    n_jobs=1,
                ),
                transform_train_only=args.transform_train_only,
                build_train_file=args.train_fold,
                write_attributes=args.write_attributes,
                att_max_shape=args.att_max_shape,
                benchmark_time=args.benchmark_time,
                overwrite=args.overwrite,
                predefined_resample=args.predefined_resample,
            )
    # local run (no args)
    else:
        # These are example parameters, change as required for local runs
        # Do not include paths to your local directories here in PRs
        # If threading is required, see the threaded version of this file
        data_path = _TEST_DATA_PATH
        results_path = _CLASSIFIER_RESULTS_PATH
        estimator_name = "ROCKET"
        dataset_name = "MinimalChinatown"
        row_normalise = False
        transform_name = None
        transform_train_only = False
        resample_id = 0
        train_fold = False
        write_attributes = True
        att_max_shape = 0
        benchmark_time = True
        overwrite = False
        predefined_resample = False
        fit_contract = 0
        checkpoint = None
        kwargs = {}

        classifier = get_classifier_by_name(
            estimator_name,
            random_state=resample_id,
            n_jobs=1,
            fit_contract=fit_contract,
            checkpoint=checkpoint,
            **kwargs,
        )
        transform = get_data_transform_by_name(
            transform_name,
            row_normalise=row_normalise,
            random_state=resample_id,
        )
        print(f"Local Run of {estimator_name} ({classifier.__class__.__name__}).")

        load_and_run_classification_experiment(
            data_path,
            results_path,
            dataset_name,
            classifier,
            classifier_name=estimator_name,
            resample_id=resample_id,
            data_transforms=transform,
            transform_train_only=transform_train_only,
            build_train_file=train_fold,
            write_attributes=write_attributes,
            att_max_shape=att_max_shape,
            benchmark_time=benchmark_time,
            overwrite=overwrite,
            predefined_resample=predefined_resample,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    print("Running classification_experiments.py main")
    run_experiment(sys.argv[1:])
