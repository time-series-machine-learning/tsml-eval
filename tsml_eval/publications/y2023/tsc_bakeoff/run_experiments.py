"""Experiment runner for bakeoff redux 2023 publication."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os
import sys

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from tsml.base import _clone_estimator

from tsml_eval.experiments import load_and_run_classification_experiment
from tsml_eval.publications.y2023.tsc_bakeoff.set_bakeoff_classifier import (
    _set_bakeoff_classifier,
)
from tsml_eval.publications.y2023.tsc_bakeoff.tests import _BAKEOFF_TEST_RESULTS_PATH
from tsml_eval.testing.testing_utils import _TEST_DATA_PATH
from tsml_eval.utils.arguments import parse_args
from tsml_eval.utils.experiments import _results_present

# all classifiers ran without duplicates
distance_based = ["1NN-DTW", "GRAIL"]
feature_based = ["Catch22", "FreshPRINCE", "TSFresh", "Signatures"]
shapelet_based = ["STC", "RDST", "RSF", "MrSQM"]
interval_based = ["R-STSF", "RISE", "TSF", "CIF", "STSF", "DrCIF", "QUANT"]
dictionary_based = ["BOSS", "cBOSS", "TDE", "WEASEL", "WEASEL-D"]
convolution_based = [
    "ROCKET",
    "MiniROCKET",
    "MultiROCKET",
    "Arsenal",
    "Hydra",
    "Hydra-MultiROCKET",
]
deep_learning = ["CNN", "ResNet", "InceptionTime", "H-InceptionTime", "LITETime"]
hybrid = ["HC1", "HC2", "RIST"]
# top performing classifiers
top_classifiers = [
    "FreshPRINCE",
    "RDST",
    "QUANT",
    "WEASEL-D",
    "Hydra-MultiROCKET",
    "H-InceptionTime",
    "HC2",
]


def _run_experiment(args, predefined_resample):
    if args is None or args.__len__() < 1:
        data_path = _TEST_DATA_PATH
        results_path = _BAKEOFF_TEST_RESULTS_PATH
        classifier = "ROCKET"
        dataset_name = "MinimalChinatown"
        resample_id = 0
        n_jobs = 1
        kwargs = {}
        overwrite = False
    else:
        print("Input args = ", args)
        args = parse_args(args)
        data_path = args.data_path
        results_path = args.results_path
        classifier = args.estimator_name
        dataset_name = args.dataset_name
        resample_id = args.resample_id
        n_jobs = args.n_jobs
        kwargs = args.kwargs
        overwrite = args.overwrite
        predefined_resample = predefined_resample or args.predefined_resample

    # Skip if not overwrite and results already present
    # this is also checked in load_and_run, but doing a quick check here so can
    # print a message and make sure data is not loaded
    if not overwrite and _results_present(
        results_path,
        classifier,
        dataset_name,
        resample_id=resample_id,
        split="TEST",
    ):
        print("Ignoring, results already present")
    else:
        load_and_run_classification_experiment(
            data_path,
            results_path,
            dataset_name,
            (
                _set_bakeoff_classifier(
                    classifier,
                    random_state=resample_id,
                    n_jobs=n_jobs,
                    **kwargs,
                )
                if isinstance(classifier, str)
                else _clone_estimator(classifier, resample_id)
            ),
            classifier_name=classifier,
            resample_id=resample_id,
            overwrite=overwrite,
            predefined_resample=predefined_resample,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.

    1. Edit the arguments from line 49-56 to suit your experiment, The most important
       are:
         data_path: the path to the data
         results_path: the path to the results
         classifier: the name of the classifier to use (check
         set_bakeoff_classifier.py), or an estimator object
         resample_id: the data resample id and random seed to use
    2. Run the script, if the experiment runs successfully a set of folders and a
       results csv file will be created in the results path.

    For evaluation of the written results, you can use the evaluation package, see
    our examples for usage:
    https://github.com/time-series-machine-learning/tsml-eval/blob/main/examples/

    For using your own classifier, any classifier following the sklearn, aeon,
    or tsml interface should be compatible with this file.
    """
    print("Running run_experiments.py main")
    # The for the UCR 112 datasets set this to true to exactly reproduce results in the
    # paper. This uses the randomly generated resamples from tsml-java if the file is
    # present (see notebook for link). Except for PF, the 30 new datasets use
    # tsml-eval resamples.
    predefined_resample = False
    _run_experiment(sys.argv[1:], predefined_resample)
