"""Tests for Multiverse Slurm queue parsing."""

from types import SimpleNamespace
from unittest.mock import Mock

from _tsml_research_resources import multiverse_controller as controller


def test_query_uses_array_index_and_preserves_running(tmp_path, monkeypatch):
    """Slurm array indices must be parsed and RUNNING must beat a duplicate PD."""
    config = SimpleNamespace(username="ajb", partition="compute")
    output = (
        "CIF_ProblemA|4|RUNNING|32000M|compute042\n"
        "CIF_ProblemA|4|PENDING|32000M|(Priority)\n"
    )
    runner = Mock(return_value=SimpleNamespace(stdout=output))
    monkeypatch.setattr(controller.shutil, "which", lambda unused: "/usr/bin/squeue")
    monkeypatch.setattr(controller.subprocess, "run", runner)

    snapshot = controller._query_slurm(config)

    command = runner.call_args[0][0]
    assert "--format=%200j|%K|%T|%m|%R" in command
    assert snapshot.states[("CIF_ProblemA", 4)] == "RUNNING"
    assert snapshot.memory_mb[("CIF_ProblemA", 4)] == 32000
    assert snapshot.nodes[("CIF_ProblemA", 4)] == "compute042"
    assert snapshot.total_user_tasks == 2
