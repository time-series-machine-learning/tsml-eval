"""Main configuration file for pytest."""

__author__ = ["MatthewMiddlehurst"]

from tsml_eval.experiments import experiments


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
