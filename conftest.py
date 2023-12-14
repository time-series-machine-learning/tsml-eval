"""Main configuration file for pytest."""

__author__ = ["MatthewMiddlehurst"]

import shutil

from tsml_eval.experiments import experiments
from tsml_eval.testing.test_utils import _TEST_OUTPUT_PATH

KEEP_PYTEST_OUTPUT = True


def pytest_sessionfinish(session, exitstatus):
    """Call after test run is finished, before returning the exit status to system."""
    if not hasattr(session.config, "workerinput") and not KEEP_PYTEST_OUTPUT:
        shutil.rmtree(_TEST_OUTPUT_PATH)


def pytest_addoption(parser):
    """Pytest command line parser options adder."""
    parser.addoption(
        "--meminterval",
        type=float,
        default=5.0,
        help="Set the time interval in seconds for recording memory usage "
        "(default: %(default)s).",
    )
    parser.addoption(
        "--keepoutput",
        action="store_true",
        help="Keep the unit test output folder after running pytest"
        " (default: %(default)s).",
    )


def pytest_configure(config):
    """Pytest configuration preamble."""
    experiments.MEMRECORD_INTERVAL = config.getoption("--meminterval")
    global KEEP_PYTEST_OUTPUT
    KEEP_PYTEST_OUTPUT = config.getoption("--keepoutput")
