"""Unit test utilities."""

import os
import sys
from contextlib import contextmanager
from os import devnull
from pathlib import Path

from sklearn.base import BaseEstimator

_TEST_DATA_PATH = f"{os.path.dirname(Path(__file__).parent.parent)}/tsml_eval/datasets/"

_TEST_RESULTS_PATH = (
    f"{os.path.dirname(Path(__file__).parent.parent)}/tsml_eval/testing/"
    f"_test_results_files/"
)

_TEST_EVAL_PATH = (
    f"{os.path.dirname(Path(__file__).parent.parent)}/tsml_eval/testing/"
    f"_test_eval_files/"
)

_TEST_OUTPUT_PATH = f"{os.path.dirname(Path(__file__).parent.parent)}/test_output/"


def _check_set_method(
    set_method,
    estimator_name_list,
    estimator_list,
    estimator_dict,
    all_estimator_names,
):
    for estimator_names in estimator_name_list:
        estimator_names = (
            [estimator_names] if isinstance(estimator_names, str) else estimator_names
        )
        s_out = None

        for estimator_alias in estimator_names:
            # no duplicate names
            assert (
                estimator_alias not in all_estimator_names
            ), f"Estimator {estimator_alias} is duplicated"
            all_estimator_names.append(estimator_alias)

            # all names should pass except for not installed soft dependencies
            try:
                out = set_method(estimator_alias)
            except ModuleNotFoundError as err:
                exempt_errors = [
                    "optional dependency",
                    "soft dependency",
                    "python version",
                    "No module named 'xgboost'",
                ]
                if any(s in str(err) for s in exempt_errors):
                    continue
                else:
                    raise err

            assert out is not None, f"Estimator {estimator_alias} not found"

            # data transformers can return multiple transforms
            if not isinstance(out, list):
                out = [out]

            if s_out is None:
                # make sure this set of names returns a unique estimator
                for e in estimator_list:
                    if len(e) == len(out) and type(e[0]) is type(out[0]):
                        assert not all(
                            [
                                str(out[i].get_params()) == str(e[i].get_params())
                                for i in range(len(out))
                            ]
                        )

                s_out = out
                estimator_list.append(out)
            else:
                # make sure all names in a set return the same estimators
                assert len(out) == len(s_out)
                assert all(
                    [
                        str(out[i].get_params()) == str(s_out[i].get_params())
                        for i in range(len(out))
                    ]
                )

            # make sure output are estimators, and record if the class name matches
            # an alias name
            for e in out:
                assert isinstance(
                    e, BaseEstimator
                ), f"Estimator {estimator_alias} is not a BaseEstimator"

                e_name = e.__class__.__name__.lower()
                if e_name == estimator_alias.lower():
                    estimator_dict[e_name] = True
                elif e_name not in estimator_dict:
                    estimator_dict[e_name] = False


EXEMPT_ESTIMATOR_NAMES = [
    "channelensembleregressor",
    "gridsearchcv",
]


def _check_set_method_results(
    estimator_dict, estimator_name="Estimators", method_name="the method"
):
    for estimator in EXEMPT_ESTIMATOR_NAMES:
        if estimator in estimator_dict:
            estimator_dict.pop(estimator)

    if not all(estimator_dict.values()):
        missing_keys = [key for key, value in estimator_dict.items() if not value]
        raise ValueError(
            f"All {estimator_name.lower()} seen in {method_name} must have an entry "
            "for the full class name (usually with default parameters). "
            f"{estimator_name} with missing entries: {missing_keys}."
        )


@contextmanager
def suppress_output(suppress_stdout=True, suppress_stderr=True):
    """Redirects stdout and/or stderr to devnull."""
    with open(devnull, "w") as null:
        stdout = sys.stdout
        stderr = sys.stderr
        try:
            if suppress_stdout:
                sys.stdout = null
            if suppress_stderr:
                sys.stderr = null
            yield
        finally:
            if suppress_stdout:
                sys.stdout = stdout
            if suppress_stderr:
                sys.stderr = stderr
