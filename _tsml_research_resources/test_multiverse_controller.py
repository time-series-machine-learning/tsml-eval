"""Tests for the recurring Multiverse controller."""

import json
from pathlib import Path

from _tsml_research_resources import multiverse_controller as controller
from tsml_eval.experiments import _get_classifier
from tsml_eval.utils.functions import str_in_nested_list


def _config(
    tmp_path,
    categories,
    resamples=3,
    max_active_tasks=3,
    all_categories_first_pass=False,
    small_datasets_first=False,
    excluded_datasets=(),
):
    dataset_file = tmp_path / "datasets.txt"
    dataset_file.write_text("ProblemA\n", encoding="utf-8", newline="\n")
    repo_dir = tmp_path / "repo"
    data_dir = tmp_path / "data"
    repo_dir.mkdir()
    data_dir.mkdir()
    return controller.ControllerConfig(
        username="tester",
        email="tester@example.com",
        repo_dir=repo_dir,
        data_dir=data_dir,
        dataset_file=dataset_file,
        results_root=tmp_path / "results",
        state_dir=tmp_path / "results" / ".controller",
        resamples=resamples,
        max_attempts=2,
        all_categories_first_pass=all_categories_first_pass,
        small_datasets_first=small_datasets_first,
        excluded_datasets=excluded_datasets,
        validate_results=False,
        account="account",
        partition="compute",
        qos="qos",
        max_active_tasks=max_active_tasks,
        memory_mb_levels=(16000, 32000, 64000, 128000),
        time_limit="7-00:00:00",
        module="python/module",
        conda_sh=tmp_path / "conda.sh",
        env_name="tsml-eval",
        numba_cache_dir=tmp_path / "numba",
        categories=categories,
    )


def _write_result(config, category, classifier, resample):
    task = controller.Task(category, classifier, "ProblemA", resample)
    result_file = controller._result_file(config, task)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text("result\n", encoding="utf-8", newline="\n")


def _state():
    return {
        "version": 1,
        "attempts": {},
        "memory_levels": {},
        "last_submitted_memory": {},
        "failures": {},
    }


def test_categories_are_strictly_ordered(tmp_path):
    """A later category must wait until every earlier result exists."""
    categories = (
        controller.Category("IntervalBased", ("CIF",)),
        controller.Category("FeatureBased", ("Catch22",)),
    )
    config = _config(tmp_path, categories, resamples=2)
    datasets = controller._read_datasets(config.dataset_file)

    snapshot = controller.SlurmSnapshot({}, 0)
    category, missing = controller._find_current_category(
        config, datasets, snapshot, _state()
    )
    assert category.name == "IntervalBased"
    assert len(missing) == 2

    _write_result(config, "IntervalBased", "CIF", 0)
    _write_result(config, "IntervalBased", "CIF", 1)
    category, missing = controller._find_current_category(
        config, datasets, snapshot, _state()
    )
    assert category.name == "FeatureBased"
    assert len(missing) == 2


def test_first_pass_interleaves_all_categories(tmp_path):
    """Breadth-first mode should offer work from every category in rotation."""
    categories = (
        controller.Category("IntervalBased", ("CIF",)),
        controller.Category("FeatureBased", ("Catch22",)),
    )
    config = _config(tmp_path, categories, resamples=2, all_categories_first_pass=True)
    datasets = controller._read_datasets(config.dataset_file)

    category, missing = controller._find_work_scope(
        config, datasets, controller.SlurmSnapshot({}, 0), _state()
    )
    ordered = controller._round_robin_categories(config, missing)

    assert category.name == "AllCategoriesFirstPass"
    assert [(task.category, task.resample) for task in ordered] == [
        ("IntervalBased", 0),
        ("FeatureBased", 0),
        ("IntervalBased", 1),
        ("FeatureBased", 1),
    ]


def test_excluded_datasets_are_removed(tmp_path):
    """Slow-problem exclusions should not enter progress totals or submissions."""
    categories = (controller.Category("IntervalBased", ("CIF",)),)
    config = _config(
        tmp_path, categories, excluded_datasets=("AustraliaRainfall_disc",)
    )

    included = controller._included_datasets(
        config, ("ProblemA", "AustraliaRainfall_disc")
    )

    assert included == ("ProblemA",)


def test_small_datasets_are_scheduled_first(tmp_path):
    """Downloaded dataset bytes should determine the scheduling order."""
    categories = (controller.Category("IntervalBased", ("CIF",)),)
    config = _config(tmp_path, categories, small_datasets_first=True)
    for dataset, content in (("Large", "12345"), ("Small", "1")):
        dataset_dir = config.data_dir / dataset
        dataset_dir.mkdir()
        (dataset_dir / f"{dataset}_TRAIN.ts").write_text(content, encoding="utf-8")

    included = controller._included_datasets(config, ("Large", "Unavailable", "Small"))

    assert included == ("Small", "Large", "Unavailable")


def test_clean_first_pass_ignores_logs_from_older_runs(tmp_path):
    """Archived state makes old failure logs irrelevant to the new pass."""
    categories = (controller.Category("IntervalBased", ("CIF",)),)
    config = _config(tmp_path, categories, all_categories_first_pass=True)
    task = controller.Task("IntervalBased", "CIF", "ProblemA", 0)
    output_dir = config.results_root / "IntervalBased" / "output" / "CIF" / "ProblemA"
    output_dir.mkdir(parents=True)
    (output_dir / "123-1.err").write_text(
        "slurmstepd: error: Detected 1 oom_kill event\n", encoding="utf-8"
    )
    state = _state()

    controller._refresh_failure_record(config, state, task)

    assert state["attempts"] == {}
    assert state["failures"] == {}
    assert controller._task_retryable(config, state, task)


def test_cycle_skips_active_task_and_fills_exact_capacity(tmp_path, monkeypatch):
    """An existing array element is not duplicated and only free slots are filled."""
    categories = (controller.Category("IntervalBased", ("CIF",)),)
    config = _config(tmp_path, categories, max_active_tasks=2)
    snapshot = controller.SlurmSnapshot(
        {("CIF_ProblemA", 1): "RUNNING"}, total_user_tasks=1
    )
    scripts = []

    monkeypatch.setattr(controller, "_query_slurm", lambda unused: snapshot)
    monkeypatch.setattr(
        controller, "_git_revision", lambda unused: ("test-branch", "abc123")
    )

    def submit(unused_config, script, dry_run):
        scripts.append(script)
        return "12345"

    monkeypatch.setattr(controller, "_submit_array", submit)

    status = controller.run_cycle(config, no_email=True)

    assert status == 0
    assert len(scripts) == 1
    assert "#SBATCH --array=2" in scripts[0]
    assert "#SBATCH --cpus-per-task=1" in scripts[0]
    assert 'export CUDA_VISIBLE_DEVICES=""' in scripts[0]
    saved_state = json.loads((config.state_dir / "state.json").read_text())
    assert saved_state["attempts"] == {
        "IntervalBased|CIF|ProblemA|0": 1,
        "IntervalBased|CIF|ProblemA|1": 1,
    }
    assert saved_state["last_submitted_memory"] == {
        "IntervalBased|CIF|ProblemA|1": 16000
    }


def test_dry_run_does_not_create_controller_state(tmp_path, monkeypatch):
    """A dry run should inspect and report without persistent writes."""
    categories = (controller.Category("IntervalBased", ("CIF",)),)
    config = _config(tmp_path, categories, resamples=1)
    monkeypatch.setattr(
        controller,
        "_query_slurm",
        lambda unused: controller.SlurmSnapshot({}, 0),
    )
    monkeypatch.setattr(
        controller, "_git_revision", lambda unused: ("test-branch", "abc123")
    )

    status = controller.run_cycle(config, dry_run=True)

    assert status == 0
    assert not config.state_dir.exists()
    assert not config.numba_cache_dir.exists()


def test_email_interval_persists_across_cycles(tmp_path, monkeypatch):
    """A successful email marker should throttle subsequent controller cycles."""
    state_dir = tmp_path / "state"
    monkeypatch.setattr(controller.time, "time", lambda: 1000.0)

    assert controller._email_due(state_dir, 14400)
    controller._record_email_sent(state_dir)
    assert not controller._email_due(state_dir, 14400)

    monkeypatch.setattr(controller.time, "time", lambda: 15400.0)
    assert controller._email_due(state_dir, 14400)


def test_report_starts_with_overall_completion(tmp_path):
    """The email body should lead with aggregate completed and expected tasks."""
    config = _config(tmp_path, (controller.Category("IntervalBased", ("CIF",)),))
    rows = [
        ("IntervalBased", 3, 0, 0, 0, 0, 0, 4),
        ("FeatureBased", 2, 0, 0, 0, 0, 0, 6),
    ]

    report = controller._compose_report(
        config,
        "branch",
        "commit",
        None,
        [],
        [],
        [],
        [],
        [],
        controller.SlurmSnapshot({}, 0),
        rows,
        [],
        [],
        _state(),
    )

    assert report.splitlines()[0] == "Complete: 5/10 (50.0%)"


def test_exhausted_category_advances_to_keep_allocation_used(tmp_path):
    """Permanent failures are reported but do not strand later useful work."""
    categories = (
        controller.Category("IntervalBased", ("CIF",)),
        controller.Category("FeatureBased", ("Catch22",)),
    )
    config = _config(tmp_path, categories, resamples=1)
    datasets = controller._read_datasets(config.dataset_file)
    state = _state()
    state["attempts"] = {"IntervalBased|CIF|ProblemA|0": config.max_attempts}

    category, missing = controller._find_current_category(
        config, datasets, controller.SlurmSnapshot({}, 0), state
    )

    assert category.name == "FeatureBased"
    assert missing == [controller.Task("FeatureBased", "Catch22", "ProblemA", 0)]


def test_oom_diagnosis_is_resample_specific(tmp_path):
    """Failure diagnosis should read the requested array element's logs."""
    categories = (controller.Category("IntervalBased", ("CIF",)),)
    config = _config(tmp_path, categories)
    task = controller.Task("IntervalBased", "CIF", "ProblemA", 1)
    output_dir = config.results_root / "IntervalBased" / "output" / "CIF" / "ProblemA"
    output_dir.mkdir(parents=True)
    (output_dir / "123-2.err").write_text(
        "slurmstepd: error: Detected 1 oom_kill event\n", encoding="utf-8"
    )

    assert controller._latest_failure_reason(config, task) == "OOM"


def test_oom_escalates_memory_and_timeout_is_terminal(tmp_path):
    """Only OOMs increase memory, while a Slurm timeout settles the task."""
    categories = (controller.Category("IntervalBased", ("CIF",)),)
    config = _config(tmp_path, categories)
    state = _state()
    oom_task = controller.Task("IntervalBased", "CIF", "ProblemA", 0)
    oom_dir = config.results_root / "IntervalBased" / "output" / "CIF" / "ProblemA"
    oom_dir.mkdir(parents=True)
    (oom_dir / "123-1.err").write_text(
        "slurmstepd: error: Detected 1 oom_kill event\n", encoding="utf-8"
    )
    state["attempts"][oom_task.state_key] = 1
    state["last_submitted_memory"][oom_task.state_key] = 16000

    controller._refresh_failure_record(config, state, oom_task)

    assert controller._task_memory(config, state, oom_task) == 32000
    assert controller._task_retryable(config, state, oom_task)
    assert state["failures"][oom_task.state_key]["events"][0]["reason"] == "OOM"
    assert state["failures"][oom_task.state_key]["events"][0]["next_memory_mb"] == 32000

    timeout_task = controller.Task("IntervalBased", "CIF", "ProblemA", 1)
    (oom_dir / "124-2.err").write_text(
        "slurmstepd: error: JOB 124 CANCELLED DUE TO TIME LIMIT\n",
        encoding="utf-8",
    )
    state["attempts"][timeout_task.state_key] = 1
    state["last_submitted_memory"][timeout_task.state_key] = 16000

    controller._refresh_failure_record(config, state, timeout_task)

    assert controller._task_memory(config, state, timeout_task) == 16000
    assert not controller._task_retryable(config, state, timeout_task)
    assert controller._task_terminal_reason(config, state, timeout_task) == "Time limit"

    external_task = controller.Task("IntervalBased", "CIF", "ProblemA", 2)
    external_snapshot = controller.SlurmSnapshot(
        {external_task.job_key: "RUNNING"},
        1,
        memory_mb={external_task.job_key: 64000},
    )
    controller._record_active_submission(
        config, state, external_task, external_snapshot
    )
    (oom_dir / "125-3.err").write_text(
        "slurmstepd: error: Detected 1 oom_kill event\n", encoding="utf-8"
    )

    controller._refresh_failure_record(config, state, external_task)

    assert controller._task_memory(config, state, external_task) == 128000
    assert controller._task_retryable(config, state, external_task)


def test_slurm_memory_parser():
    """Slurm's common requested-memory units should convert to MB."""
    assert controller._parse_memory_mb("16000M") == 16000
    assert controller._parse_memory_mb("64G") == 65536
    assert controller._parse_memory_mb("2Gc") == 2048
    assert controller._parse_memory_mb("N/A") is None


def test_later_category_active_memory_is_recorded(tmp_path):
    """Running shapelet jobs are captured while IntervalBased remains current."""
    categories = (
        controller.Category("IntervalBased", ("CIF",)),
        controller.Category("ShapeletBased", ("RDST",)),
    )
    config = _config(tmp_path, categories, resamples=1)
    datasets = controller._read_datasets(config.dataset_file)
    task = controller.Task("ShapeletBased", "RDST", "ProblemA", 0)
    snapshot = controller.SlurmSnapshot(
        {task.job_key: "RUNNING"},
        1,
        memory_mb={task.job_key: 64000},
    )
    state = _state()

    controller._record_all_active_submissions(config, datasets, snapshot, state)

    assert state["attempts"][task.state_key] == 1
    assert state["last_submitted_memory"][task.state_key] == 64000
    assert controller._task_memory(config, state, task) == 64000


def test_shipped_configuration_is_breadth_first_at_8gb():
    """The supplied Hali configuration should make one broad low-memory pass."""
    config_file = Path(controller.__file__).with_name("multiverse_controller.toml")
    config = controller._load_config(config_file)

    assert config.categories[0].name == "IntervalBased"
    assert config.resamples == 30
    assert config.max_active_tasks == 8000
    assert config.max_attempts == 1
    assert config.all_categories_first_pass
    assert config.small_datasets_first
    assert config.excluded_datasets == ("AustraliaRainfall_disc",)
    assert config.memory_mb_levels == (8000,)
    assert config.email == "ajb@uea.ac.uk"
    assert "DeepLearning" not in {category.name for category in config.categories}
    assert "LS" not in {
        classifier
        for category in config.categories
        for classifier in category.classifiers
    }
    assert sum(len(category.classifiers) for category in config.categories) == 32
    assert [category.name for category in config.categories[:4]] == [
        "IntervalBased",
        "DictionaryBased",
        "ShapeletBased",
        "ConvolutionBased",
    ]


def test_shipped_classifier_names_are_tsml_eval_aliases():
    """Every configured classifier must be accepted by classification experiments."""
    config_file = Path(controller.__file__).with_name("multiverse_controller.toml")
    config = controller._load_config(config_file)
    classifier_lists = (
        _get_classifier.convolution_based_classifiers,
        _get_classifier.deep_learning_classifiers,
        _get_classifier.dictionary_based_classifiers,
        _get_classifier.distance_based_classifiers,
        _get_classifier.feature_based_classifiers,
        _get_classifier.hybrid_classifiers,
        _get_classifier.interval_based_classifiers,
        _get_classifier.shapelet_based_classifiers,
    )

    for category in config.categories:
        for classifier in category.classifiers:
            assert any(
                str_in_nested_list(classifier_list, classifier.lower())
                for classifier_list in classifier_lists
            ), classifier
