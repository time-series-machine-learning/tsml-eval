"""Structural regression checks for the dynamic Iridis task-farm runners."""

import re
from pathlib import Path


_SCRIPT_DIR = Path(__file__).parent / "soton" / "iridis" / "batch_scripts"
_DYNAMIC_RUNNERS = (
    "run_tser_interval_regressors.sh",
    "run_ucr_pulsar_classifier.sh",
    "run_multiverse_pulsar_classifier.sh",
)


def _script(name):
    return (_SCRIPT_DIR / name).read_text()


def _first_array(text, name):
    match = re.search(
        rf"(?m)^{re.escape(name)}=\(\n(?P<body>.*?)^\)", text, re.DOTALL
    )
    assert match is not None
    return re.findall(r'^\s*"([^"]+)"\s*$', match.group("body"), re.MULTILINE)


def test_tser_monitor_classifier_list_matches_runner():
    runner_regressors = _first_array(
        _script("run_tser_interval_regressors.sh"), "regressors"
    )
    monitor_regressors = _first_array(
        _script("monitor_tser_interval_regressors.sh"), "regressors"
    )

    assert runner_regressors == monitor_regressors
    assert "pulsar" in runner_regressors
    assert len(runner_regressors) == 8


def test_dynamic_runners_preserve_refill_and_failure_invariants():
    for runner in _DYNAMIC_RUNNERS:
        text = _script(runner)

        assert "wait_for_queue_slot" not in text
        assert "sacct --noheader --parsable2" in text
        assert '[[ "${reason}" == "SUBMITTED" ]]' in text
        assert "attempt_job_id" in text
        assert 'classify_failure_result="TIMEOUT"' in text
        assert "pinned_pulsar_hash" in text
