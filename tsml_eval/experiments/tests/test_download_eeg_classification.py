"""Tests for the Iridis EEG dataset download helper."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


def test_download_eeg_classification_writes_only_ready_datasets(monkeypatch, tmp_path):
    """Include staged and downloaded datasets, and report unavailable records."""
    data_dir = tmp_path / "data"
    ready_dir = data_dir / "Ready"
    ready_dir.mkdir(parents=True)
    (ready_dir / "Ready_TRAIN.ts").write_text("train")
    (ready_dir / "Ready_TEST.ts").write_text("test")
    dataset_list = tmp_path / "datasets.txt"

    datasets = ModuleType("aeon_neuro.datasets")
    datasets.__path__ = []
    collection = ModuleType("aeon_neuro.datasets.classification_datasets")
    collection.dataset_names = ["Ready", "Download", "Unavailable"]
    collection.dataset_map = {"Ready": 1, "Download": 2, "Unavailable": 49}

    def load_eeg_classification(name, split, extract_path):
        assert (name, split) == ("Download", "train")
        problem_dir = Path(extract_path) / name
        problem_dir.mkdir()
        (problem_dir / f"{name}_TRAIN.ts").write_text("train")
        (problem_dir / f"{name}_TEST.ts").write_text("test")

    datasets.load_eeg_classification = load_eeg_classification
    monkeypatch.setitem(sys.modules, "aeon_neuro.datasets", datasets)
    monkeypatch.setitem(
        sys.modules, "aeon_neuro.datasets.classification_datasets", collection
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_eeg_classification.py",
            "--data-dir",
            str(data_dir),
            "--dataset-list",
            str(dataset_list),
        ],
    )

    script = (
        Path(__file__).parents[3]
        / "_tsml_research_resources/soton/iridis/download_eeg_classification.py"
    )
    with pytest.raises(SystemExit, match="1"):
        runpy.run_path(str(script), run_name="__main__")

    assert dataset_list.read_text().splitlines() == ["Ready", "Download"]
