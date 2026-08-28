"""Select equal-length Multiverse datasets for a conservative ConvTran pass."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetShape:
    """Shape information read without materialising a dataset."""

    name: str
    train_cases: int
    channels: int
    timepoints: int

    @property
    def attention_work(self) -> int:
        """Return a simple relative training-work proxy."""
        return self.train_cases * self.timepoints**2


def _parse_bool(value: str) -> bool:
    return value.casefold() == "true"


def _resolve_variant(dataset_dir: Path, dataset: str) -> str:
    """Mirror aeon's equal-length/no-missing local filename selection."""
    selected = dataset
    for suffix in ("_eq", "_nmv"):
        train_file = dataset_dir / f"{dataset}{suffix}_TRAIN.ts"
        test_file = dataset_dir / f"{dataset}{suffix}_TEST.ts"
        if train_file.is_file() and test_file.is_file():
            selected += suffix
    return selected


def read_train_shape(
    dataset_dir: Path, dataset: str, file_stem: str | None = None
) -> DatasetShape:
    """Read train shape from a ``.ts`` file without loading its values."""
    file_stem = dataset if file_stem is None else file_stem
    train_file = dataset_dir / f"{file_stem}_TRAIN.ts"
    if not train_file.is_file():
        raise FileNotFoundError(train_file)

    channels = None
    timepoints = None
    equal_length = None
    timestamps = None
    class_label = True
    train_cases = 0
    in_data = False

    with train_file.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lower = stripped.casefold()
            if not in_data:
                fields = stripped.split()
                tag = fields[0].casefold()
                if tag == "@dimensions":
                    channels = int(fields[1])
                elif tag == "@serieslength":
                    timepoints = int(fields[1])
                elif tag == "@equallength":
                    equal_length = _parse_bool(fields[1])
                elif tag == "@timestamps":
                    timestamps = _parse_bool(fields[1])
                elif tag == "@classlabel":
                    class_label = _parse_bool(fields[1])
                elif lower == "@data":
                    in_data = True
                continue

            train_cases += 1
            if channels is None or timepoints is None:
                fields = stripped.split(":")
                data_fields = fields[:-1] if class_label else fields
                if channels is None:
                    channels = len(data_fields)
                if timepoints is None and data_fields:
                    timepoints = len(data_fields[0].split(","))

    if timestamps:
        raise ValueError("timestamped data are not supported")
    if equal_length is False:
        raise ValueError("unequal-length data are not supported")
    if not in_data or train_cases == 0:
        raise ValueError("no training cases found")
    if channels is None or channels < 1:
        raise ValueError("number of channels could not be determined")
    if timepoints is None or timepoints < 1:
        raise ValueError("number of timepoints could not be determined")

    return DatasetShape(dataset, train_cases, channels, timepoints)


def select_datasets(
    source_list: Path,
    data_dir: Path,
    max_train_cases: int,
    max_timepoints: int,
    max_attention_work: int,
):
    """Return accepted, rejected, and unavailable datasets."""
    accepted = []
    rejected = []
    unavailable = []

    for raw_dataset in source_list.read_text().splitlines():
        dataset = raw_dataset.strip().removesuffix("\r")
        if not dataset or dataset.startswith("#"):
            continue

        dataset_dir = data_dir / dataset
        train_file = dataset_dir / f"{dataset}_TRAIN.ts"
        test_file = dataset_dir / f"{dataset}_TEST.ts"
        if not train_file.is_file() or not test_file.is_file():
            unavailable.append(dataset)
            continue

        try:
            file_stem = _resolve_variant(dataset_dir, dataset)
            shape = read_train_shape(dataset_dir, dataset, file_stem)
        except ValueError as error:
            rejected.append((dataset, str(error), None))
            continue

        reasons = []
        if shape.train_cases > max_train_cases:
            reasons.append(
                f"train cases {shape.train_cases} > limit {max_train_cases}"
            )
        if shape.timepoints > max_timepoints:
            reasons.append(
                f"timepoints {shape.timepoints} > limit {max_timepoints}"
            )
        if shape.attention_work > max_attention_work:
            reasons.append(
                "train_cases*timepoints^2 "
                f"{shape.attention_work} > limit {max_attention_work}"
            )
        if reasons:
            rejected.append((dataset, "; ".join(reasons), shape))
        else:
            accepted.append(shape)

    accepted.sort(key=lambda shape: (shape.attention_work, shape.name))
    return accepted, rejected, unavailable


def _write_lines(path: Path, lines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))


def _write_rejections(path: Path, rejected) -> None:
    lines = ["dataset\treason\ttrain_cases\tchannels\ttimepoints\twork\n"]
    for dataset, reason, shape in rejected:
        if shape is None:
            values = (dataset, reason, "", "", "", "")
        else:
            values = (
                dataset,
                reason,
                shape.train_cases,
                shape.channels,
                shape.timepoints,
                shape.attention_work,
            )
        lines.append("\t".join(map(str, values)) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def main(argv=None) -> int:
    """Run the ConvTran feasibility selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-list", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, required=True)
    parser.add_argument("--unavailable-output", type=Path, required=True)
    parser.add_argument("--max-train-cases", type=int, default=10_000)
    parser.add_argument("--max-timepoints", type=int, default=2000)
    parser.add_argument("--max-attention-work", type=int, default=1_000_000_000)
    args = parser.parse_args(argv)

    if (
        args.max_train_cases < 1
        or args.max_timepoints < 1
        or args.max_attention_work < 1
    ):
        parser.error("feasibility limits must be positive")
    if not args.source_list.is_file():
        parser.error(f"source list not found: {args.source_list}")
    if not args.data_dir.is_dir():
        parser.error(f"data directory not found: {args.data_dir}")

    accepted, rejected, unavailable = select_datasets(
        args.source_list,
        args.data_dir,
        args.max_train_cases,
        args.max_timepoints,
        args.max_attention_work,
    )
    _write_lines(args.output, (shape.name for shape in accepted))
    _write_rejections(args.rejected_output, rejected)
    _write_lines(args.unavailable_output, unavailable)

    print(f"Feasible datasets:    {len(accepted)} -> {args.output}")
    print(f"Rejected by limits:   {len(rejected)} -> {args.rejected_output}")
    print(f"Unavailable datasets: {len(unavailable)} -> {args.unavailable_output}")
    for shape in accepted:
        print(
            f"  {shape.name}: train={shape.train_cases}, channels={shape.channels}, "
            f"timepoints={shape.timepoints}, work={shape.attention_work}"
        )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
