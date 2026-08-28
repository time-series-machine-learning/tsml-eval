"""Tests for the ConvTran Multiverse dataset selector."""

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).parent / "select_convtran_datasets.py"
_SPEC = importlib.util.spec_from_file_location("select_convtran_datasets", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_ts(path, dataset, cases, channels, timepoints, include_shape=True):
    header = [
        f"@problemName {dataset}",
        "@timestamps false",
        "@equalLength true",
    ]
    if include_shape:
        header.extend([f"@dimensions {channels}", f"@seriesLength {timepoints}"])
    header.extend(["@classLabel true 0 1", "@data"])
    channel = ",".join("0" for _ in range(timepoints))
    row = ":".join([channel] * channels + ["0"])
    path.write_text("\n".join(header + [row] * cases) + "\n")


def test_select_convtran_datasets_filters_and_orders(tmp_path):
    """Selection uses real shapes and orders feasible work smallest first."""
    data_dir = tmp_path / "data"
    for name, cases, channels, timepoints, include_shape in [
        ("Small", 3, 2, 5, False),
        ("Medium", 4, 3, 8, True),
        ("Long", 2, 1, 21, True),
        ("Expensive", 10, 1, 15, True),
    ]:
        dataset_dir = data_dir / name
        dataset_dir.mkdir(parents=True)
        _write_ts(
            dataset_dir / f"{name}_TRAIN.ts",
            name,
            cases,
            channels,
            timepoints,
            include_shape,
        )
        _write_ts(
            dataset_dir / f"{name}_TEST.ts",
            name,
            1,
            channels,
            timepoints,
            include_shape,
        )

    source = tmp_path / "source.txt"
    source.write_text("Medium\nMissing\nLong\nSmall\nExpensive\n")
    accepted, rejected, unavailable = _MODULE.select_datasets(
        source,
        data_dir,
        max_train_cases=5,
        max_timepoints=20,
        max_attention_work=1000,
    )

    assert [shape.name for shape in accepted] == ["Small", "Medium"]
    assert {item[0] for item in rejected} == {"Long", "Expensive"}
    assert unavailable == ["Missing"]
    assert accepted[0].channels == 2
    assert accepted[0].timepoints == 5


def test_select_convtran_datasets_uses_equal_length_variant(tmp_path):
    """Selection must inspect the same equal-length variant loaded by aeon."""
    name = "UnequalOriginal"
    dataset_dir = tmp_path / "data" / name
    dataset_dir.mkdir(parents=True)
    for split in ["TRAIN", "TEST"]:
        _write_ts(
            dataset_dir / f"{name}_{split}.ts",
            name,
            2,
            1,
            30,
            include_shape=True,
        )
        path = dataset_dir / f"{name}_{split}.ts"
        unequal_text = path.read_text().replace(
            "@equalLength true", "@equalLength false"
        )
        path.write_text(unequal_text)
        _write_ts(
            dataset_dir / f"{name}_eq_{split}.ts",
            name,
            2,
            1,
            10,
            include_shape=True,
        )

    source = tmp_path / "source.txt"
    source.write_text(f"{name}\n")
    accepted, rejected, unavailable = _MODULE.select_datasets(
        source,
        tmp_path / "data",
        max_train_cases=5,
        max_timepoints=20,
        max_attention_work=1000,
    )

    assert [shape.name for shape in accepted] == [name]
    assert accepted[0].timepoints == 10
    assert rejected == []
    assert unavailable == []
