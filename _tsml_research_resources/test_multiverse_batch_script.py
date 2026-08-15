"""Tests for the Slurm batch scripts the Multiverse controller generates."""

import pytest

from _tsml_research_resources import multiverse_controller as controller

_REQUIRED_DIRECTIVES = (
    "--partition=",
    "--time=",
    "--job-name=",
    "--array=",
    "--nodes=1",
    "--ntasks=1",
    "--cpus-per-task=",
    "--mem=",
    "--output=",
    "--error=",
)


def _config(tmp_path, **overrides):
    """Build a controller configuration with Hali CPU defaults."""
    settings = dict(
        username="tester",
        email="tester@example.com",
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
        validate_results=False,
        account="cmp",
        partition="compute",
        qos="uea-core-default",
        max_active_tasks=10,
        memory_mb_levels=(8000,),
        time_limit="7-00:00:00",
        module="python/anaconda",
        conda_sh=tmp_path / "conda.sh",
        env_name="tsml-eval",
        numba_cache_dir=tmp_path / "numba",
        categories=(controller.Category("DeepLearning", ("H-InceptionTime",)),),
    )
    settings.update(overrides)
    return controller.ControllerConfig(**settings)


def _header(config):
    """Return the #SBATCH header lines of a generated script."""
    task = controller.Task("DeepLearning", "H-InceptionTime", "STEW", 0)
    script = controller._batch_script(
        config, task, [1, 3], "abc123", 64000, prepare_directories=False
    )
    return script.split("\nset -eo pipefail")[0].splitlines()


def test_cpu_header_has_every_required_directive(tmp_path):
    """A CPU pass should request no GPU but still carry every other directive."""
    header = _header(_config(tmp_path))

    for directive in _REQUIRED_DIRECTIVES:
        assert any(directive in line for line in header), directive
    assert "#SBATCH --account=cmp" in header
    assert "#SBATCH --qos=uea-core-default" in header
    assert not any("--gres" in line for line in header)
    assert "#SBATCH --array=1,3" in header


def test_hali_gpu_header_uses_untyped_gres(tmp_path):
    """Hali has no GPU type, so the plain gpu:N form is generated."""
    header = _header(
        _config(tmp_path, partition="gpu", qos="gpu", cpus_per_task=2, gpus=1)
    )

    assert "#SBATCH --gres=gpu:1" in header
    assert "#SBATCH --cpus-per-task=2" in header
    assert "#SBATCH --nodes=1" in header


def test_iridisx_gpu_header_omits_account_and_types_gres(tmp_path):
    """IridisX needs a typed gres and no account or QoS directive."""
    header = _header(
        _config(
            tmp_path,
            account="",
            partition="swarm_a100",
            qos="",
            cpus_per_task=2,
            gpus=1,
            gres="gpu:a100swarm:1",
        )
    )

    assert "#SBATCH --gres=gpu:a100swarm:1" in header
    assert "#SBATCH --partition=swarm_a100" in header
    assert "#SBATCH --nodes=1" in header
    assert not any("--account" in line for line in header)
    assert not any("--qos" in line for line in header)
    for directive in _REQUIRED_DIRECTIVES:
        assert any(directive in line for line in header), directive


def test_gpu_job_verifies_the_environment_and_device(tmp_path):
    """A GPU job should fail loudly rather than silently train on the CPU."""
    config = _config(tmp_path, partition="gpu", qos="gpu", cpus_per_task=2, gpus=1)
    task = controller.Task("DeepLearning", "H-InceptionTime", "STEW", 0)
    script = controller._batch_script(
        config, task, [1], "abc123", 64000, prepare_directories=False
    )

    assert "unset CONDA_DEFAULT_ENV" in script
    assert "which is outside $CONDA_PREFIX" in script
    assert "TensorFlow cannot see the allocated GPU" in script
    assert 'export CUDA_VISIBLE_DEVICES=""' not in script


def test_cpu_job_disables_visible_devices(tmp_path):
    """A CPU job should not pick up a GPU that happens to be on the node."""
    script_lines = _header(_config(tmp_path))
    config = _config(tmp_path)
    task = controller.Task("DeepLearning", "H-InceptionTime", "STEW", 0)
    script = controller._batch_script(
        config, task, [1], "abc123", 8000, prepare_directories=False
    )

    assert 'export CUDA_VISIBLE_DEVICES=""' in script
    assert script_lines[0] == "#!/bin/bash"


def test_conda_sh_is_derived_when_not_configured(tmp_path):
    """An unset conda_sh should resolve from the loaded module at job runtime."""
    config = _config(tmp_path, conda_sh=None)
    task = controller.Task("DeepLearning", "H-InceptionTime", "STEW", 0)
    script = controller._batch_script(
        config, task, [1], "abc123", 8000, prepare_directories=False
    )

    assert 'conda_sh="$(dirname "$(dirname "$(command -v conda)")")' in script
    assert "ERROR: conda.sh not found" in script


def test_nodes_must_be_positive(tmp_path):
    """A node count below one is rejected rather than submitted."""
    with pytest.raises(ValueError, match="nodes must be at least 1"):
        controller._validate_config(_config(tmp_path, nodes=0))


def test_gres_without_gpus_is_rejected(tmp_path):
    """A gres string with no GPU request would silently request no GPU."""
    with pytest.raises(ValueError, match="gres is set but gpus is 0"):
        controller._validate_config(_config(tmp_path, gres="gpu:a100swarm:1"))


def test_ordered_pass_can_ignore_failure_logs_from_older_runs(tmp_path):
    """Clean log handling should not require breadth-first category scheduling."""
    config = _config(
        tmp_path,
        all_categories_first_pass=False,
        ignore_existing_failure_logs=True,
    )
    task = controller.Task("DeepLearning", "H-InceptionTime", "STEW", 0)
    error_file = (
        config.results_root
        / task.category
        / "output"
        / task.classifier
        / task.dataset
        / "123-1.err"
    )
    error_file.parent.mkdir(parents=True)
    error_file.write_text("CANCELLED\n", encoding="utf-8")
    state = controller._load_state(config.state_dir / "state.json")

    controller._refresh_failure_record(config, state, task)

    assert state["attempts"] == {}
    assert state["failures"] == {}
