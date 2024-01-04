import os
import sys
from contextlib import contextmanager
from os import devnull
from pathlib import Path

_TEST_DATA_PATH = os.path.dirname(Path(__file__).parent.parent) + "/tsml_eval/datasets/"

_TEST_RESULTS_PATH = (
    os.path.dirname(Path(__file__).parent.parent)
    + "/tsml_eval/testing/_test_result_files/"
)

_TEST_OUTPUT_PATH = os.path.dirname(Path(__file__).parent.parent) + "/test_output/"


def _check_set_method(
    set_method, estimator_sub_list, estimator_dict, all_estimator_names
):
    for estimator_names in estimator_sub_list:
        estimator_names = (
            [estimator_names] if isinstance(estimator_names, str) else estimator_names
        )

        for estimator_alias in estimator_names:
            assert (
                estimator_alias not in all_estimator_names
            ), f"Estimator {estimator_alias} is duplicated"
            all_estimator_names.append(estimator_alias)

            try:
                e = set_method(estimator_alias)
            except ModuleNotFoundError:
                continue

            assert e is not None, f"Estimator {estimator_alias} not found"

            c_name = e.__class__.__name__.lower()
            if c_name == estimator_alias.lower():
                estimator_dict[c_name] = True
            elif c_name not in estimator_dict:
                estimator_dict[c_name] = False


EXEMPT_ESTIMATOR_NAMES = [
    "channelensembleregressor",
    "gridsearchcv",
    "transformedtargetforecaster",
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
