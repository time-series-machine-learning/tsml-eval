"""Experiment runner for TSER expansion publication."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst", "dguijo"]

import os
import sys

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from aeon.base._base import _clone_estimator

from tsml_eval.experiments import load_and_run_regression_experiment
from tsml_eval.publications.y2023.tser_archive_expansion.set_tser_exp_regressor import (
    _set_tser_exp_regressor,
)
from tsml_eval.publications.y2023.tser_archive_expansion.tests import (
    _TSER_ARCHIVE_TEST_RESULTS_PATH,
)
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH
from tsml_eval.utils.arguments import parse_args
from tsml_eval.utils.experiments import _results_present

# all regressors ran without duplicates
regressors_5A2 = [
    "1NN-DTW",
    "1NN-ED",
    "5NN-DTW",
    "5NN-ED",
    "FCN",
    "FPCR",
    "FPCR-Bs",
    "Grid-SVR",
    "Inception",
    "RandF",
    "ResNet",
    "ROCKET",
    "XGBoost",
]
regressors_5B = ["CNN", "InceptionE", "MultiROCKET", "Ridge", "RotF", "TSF"]
regressors_5C = ["DrCIF", "FreshPRINCE"]


def _run_experiment(args):
    if args is None or args.__len__() < 1:
        data_path = _TEST_DATA_PATH
        results_path = _TSER_ARCHIVE_TEST_RESULTS_PATH
        regressor = "ROCKET"
        dataset_name = "MinimalGasPrices"
        resample_id = 0
        n_jobs = 1
        kwargs = {}
        overwrite = False
    else:
        print("Input args = ", args)
        args = parse_args(args)
        data_path = args.data_path
        results_path = args.results_path
        regressor = args.estimator_name
        dataset_name = args.dataset_name
        resample_id = args.resample_id
        n_jobs = args.n_jobs
        kwargs = args.kwargs
        overwrite = args.overwrite

    # Skip if not overwrite and results already present
    # this is also checked in load_and_run, but doing a quick check here so can
    # print a message and make sure data is not loaded
    if not overwrite and _results_present(
        results_path,
        regressor,
        dataset_name,
        resample_id=resample_id,
        split="TEST",
    ):
        print("Ignoring, results already present")
    else:
        load_and_run_regression_experiment(
            data_path,
            results_path,
            dataset_name,
            (
                _set_tser_exp_regressor(
                    regressor,
                    random_state=resample_id,
                    n_jobs=n_jobs,
                    **kwargs,
                )
                if isinstance(regressor, str)
                else _clone_estimator(regressor, resample_id)
            ),
            regressor_name=regressor,
            resample_id=resample_id,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.

    1. Edit the arguments from line 48-55 to suit your experiment, The most important
       are:
         data_path: the path to the data
         results_path: the path to the results
         regressor: the name of the regressor to use (check set_tser_exp_regressor.py),
         or an estimator object
         resample_id: the data resample id and random seed to use
    2. Run the script, if the experiment runs successfully a set of folders and a
       results csv file will be created in the results path.

    For evaluation of the written results, you can use the evaluation package, see
    our examples for usage:
    https://github.com/time-series-machine-learning/tsml-eval/blob/main/examples/

    For using your own regressor, any regressor following the sklearn, aeon,
    or tsml interface should be compatible with this file.
    """
    print("Running run_experiments.py main")
    _run_experiment(sys.argv[1:])
