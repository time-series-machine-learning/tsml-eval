"""Regression Experiments: code for experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard tsml format.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys

import numba
from aeon.utils.validation._dependencies import _check_soft_dependencies

from tsml_eval.experiments import load_and_run_regression_experiment
from tsml_eval.experiments.set_regressor import set_regressor
from tsml_eval.utils.experiments import _results_present, assign_gpu, parse_args
from tsml_eval.utils.functions import pair_list_to_dict


def run_experiment(args):
    """Mechanism for testing regressors on the UCR data format.

    This mirrors the mechanism used in the Java based tsml. Results generated using the
    method are in the same format as tsml and can be directly compared to the results
    generated in Java.

    Attempts to avoid the use of threading as much as possible.
    """
    numba.set_num_threads(1)
    if _check_soft_dependencies("torch", severity="none"):
        import torch

        torch.set_num_threads(1)

    # if multiple GPUs are available, assign the one with lease usage to the process
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        try:
            gpu = assign_gpu()
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"Assigned GPU {gpu} to process.")
        except Exception:
            print("Unable to assign GPU to process.")

    # cluster run (with args), this is fragile
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
            split="BOTH" if args.train_fold else "TEST",
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_regression_experiment(
                args.data_path,
                args.results_path,
                args.dataset_name,
                set_regressor(
                    args.estimator_name,
                    random_state=args.resample_id
                    if args.random_seed is None
                    else args.random_seed,
                    n_jobs=1,
                    build_train_file=args.train_fold,
                    fit_contract=args.fit_contract,
                    checkpoint=args.checkpoint,
                    **pair_list_to_dict(args.kwargs),
                ),
                resample_id=args.resample_id,
                regressor_name=args.estimator_name,
                overwrite=args.overwrite,
                build_train_file=args.train_fold,
                predefined_resample=args.predefined_resample,
            )
    # local run (no args)
    else:
        # These are example parameters, change as required for local runs
        # Do not include paths to your local directories here in PRs
        # If threading is required, see the threaded version of this file
        data_path = "../"
        results_path = "../"
        estimator_name = "LR"
        dataset_name = "Covid3Month"
        resample_id = 0
        overwrite = False
        predefined_resample = False
        train_fold = False
        fit_contract = None
        checkpoint = None
        kwargs = {}

        regressor = set_regressor(
            estimator_name,
            random_state=resample_id,
            n_jobs=1,
            build_train_file=train_fold,
            fit_contract=fit_contract,
            checkpoint=checkpoint,
            **kwargs,
        )
        print(f"Local Run of {estimator_name} ({regressor.__class__.__name__}).")

        load_and_run_regression_experiment(
            data_path,
            results_path,
            dataset_name,
            regressor,
            resample_id=resample_id,
            regressor_name=estimator_name,
            overwrite=overwrite,
            build_train_file=train_fold,
            predefined_resample=predefined_resample,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
