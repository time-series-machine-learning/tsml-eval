"""Focused tests for concise Multiverse controller emails."""

from _tsml_research_resources import multiverse_controller as controller


def test_email_reports_progress_by_classifier(tmp_path):
    """Email should show classifier completion and running jobs, not failures."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = controller.ControllerConfig(
        username="tester",
        email="tester@example.com",
        repo_dir=tmp_path,
        data_dir=data_dir,
        dataset_file=tmp_path / "datasets.txt",
        results_root=tmp_path / "results",
        state_dir=tmp_path / "state",
        resamples=2,
        max_attempts=1,
        all_categories_first_pass=False,
        small_datasets_first=False,
        excluded_datasets=(),
        validate_results=False,
        account="account",
        partition="compute",
        qos="qos",
        max_active_tasks=10,
        memory_mb_levels=(32000,),
        time_limit="1-00:00:00",
        module="python",
        conda_sh=tmp_path / "conda.sh",
        env_name="tsml-eval",
        numba_cache_dir=tmp_path / "numba",
        categories=(controller.Category("IntervalBased", ("CIF", "DrCIF")),),
    )
    result = (
        config.results_root
        / "IntervalBased"
        / "CIF"
        / "Predictions"
        / "ProblemA"
        / "testResample0.csv"
    )
    result.parent.mkdir(parents=True)
    result.write_text("result\n", encoding="utf-8")
    running = controller.Task("IntervalBased", "DrCIF", "ProblemA", 1)
    snapshot = controller.SlurmSnapshot(
        {running.job_key: "RUNNING"},
        1,
        nodes={running.job_key: "compute042"},
    )

    report = controller._compose_email_report(config, ("ProblemA",), snapshot)

    assert report.splitlines()[0] == "Complete: 1/4 (25.0%)"
    assert "CIF" in report and "1" in report
    assert "DrCIF" in report and "RUNNING" not in report
    assert "Running jobs: 1" in report
    assert "Machine:" in report
    assert "Running nodes: compute042 (1)" in report
    assert "OOM" not in report
    assert "Timeout" not in report
    assert "Terminal" not in report


def test_excluded_tasks_are_removed_from_work_and_denominator(tmp_path):
    """Known terminal tasks should not be scheduled or depress progress."""
    excluded = "IntervalBased|CIF|ProblemA|0"
    config = controller.ControllerConfig(
        username="tester",
        email="",
        repo_dir=tmp_path,
        data_dir=tmp_path / "data",
        dataset_file=tmp_path / "datasets.txt",
        results_root=tmp_path / "results",
        state_dir=tmp_path / "state",
        resamples=1,
        max_attempts=1,
        all_categories_first_pass=False,
        small_datasets_first=False,
        excluded_datasets=(),
        excluded_tasks=(excluded,),
        validate_results=False,
        account="account",
        partition="compute",
        qos="qos",
        max_active_tasks=10,
        memory_mb_levels=(32000,),
        time_limit="1-00:00:00",
        module="python",
        conda_sh=tmp_path / "conda.sh",
        env_name="tsml-eval",
        numba_cache_dir=tmp_path / "numba",
        categories=(controller.Category("IntervalBased", ("CIF",)),),
    )
    tasks = tuple(
        controller._iter_tasks(
            config.categories[0],
            ("ProblemA", "ProblemB"),
            config.resamples,
            config.excluded_tasks,
        )
    )

    assert [task.dataset for task in tasks] == ["ProblemB"]
    rows = controller._classifier_email_rows(
        config, ("ProblemA", "ProblemB"), controller.SlurmSnapshot({}, 0)
    )
    assert rows == [("IntervalBased", "CIF", 0, 1, 0)]
