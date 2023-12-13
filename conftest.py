"""Main configuration file for pytest."""

__author__ = ["MatthewMiddlehurst"]

import shutil

from tsml_eval.experiments import experiments
from tsml_eval.testing.test_utils import _TEST_OUTPUT_PATH


def pytest_sessionfinish(session, exitstatus):
    """
    Called after test run is finished, right before returning the exit status to
    the system.
    """
    if not hasattr(session.config, "workerinput"):
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


def pytest_configure(config):
    """Pytest configuration preamble."""
    experiments.MEMRECORD_INTERVAL = config.getoption("--meminterval")
