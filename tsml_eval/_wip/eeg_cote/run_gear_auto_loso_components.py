"""Run the GEAR-Auto LOSO components for one held-out subject.

GEAR-Auto chooses one reduction for the whole pipeline, so the reducer is fitted
once per fold and shared by every component. Running the components as separate
jobs would refit the channel selector for each of them and charge that cost
repeatedly, which is what inflated the ``GEAR-Comp-Native-*`` runtimes.

Each component writes its own ``GEAR-Auto-Native-{component}`` results, so HC2 can
be assembled from them afterwards with ``FromFileHIVECOTE``. The shared reduction
cost is written once per fold to a JSON sidecar rather than being added to any
component, so summing component times and adding the sidecar gives the pipeline
cost without double counting.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from tsml_eval.experiments._channel_selection_hc2 import (
    _make_channel_transformer,
    _make_gear_transformer,
    _metadata_to_builtin,
    _selector_metadata,
)
from tsml_eval.experiments._get_classifier import _make_hc2_or_component
from tsml_eval.experiments.experiments import run_classification_experiment
from tsml_eval._wip.eeg_loso import load_loso_split

# TDE is excluded: it is the weakest component on this problem and removing it
# improves accuracy while removing a large part of the cost.
DEFAULT_COMPONENTS = ("Arsenal", "DrCIF", "STC")

# Result families, one per reduction. Detach-ROCKET is included because it gave the
# best accuracy and runtime compromise in the archive comparison. The existing
# partial DetachRocket-* results were produced per component, each refitting the
# selector, so this family is kept separate.
SELECTORS = {
    "gear-auto": "GEAR-Auto-Native",
    "detachrocket": "DetachRocket-Native",
}
DEFAULT_PROPORTION = 0.25


def _make_reducer(selector: str, n_channels: int, random_state: int, proportion: float):
    """Construct the shared reduction for one fold."""
    if selector == "gear-auto":
        # GEAR chooses its own channel budget, so proportion does not apply.
        return _make_gear_transformer(
            component="auto", random_state=random_state, n_jobs=1
        )
    return _make_channel_transformer(
        selector,
        n_channels=n_channels,
        proportion=proportion,
        random_state=random_state,
        n_jobs=1,
    )


def _apply_fit(reducer, X, y):
    """Fit the reduction, supporting both resampling and plain transformers."""
    if hasattr(reducer, "fit_resample"):
        return reducer.fit_resample(X, y)
    return reducer.fit_transform(X, y), y


def _result_files(
    results_path: Path, prefix: str, component: str, dataset: str, subject: int
):
    """Return the train and test result paths for one component and fold."""
    directory = results_path / f"{prefix}-{component}" / "Predictions" / dataset
    return (
        directory / f"trainResample{subject}.csv",
        directory / f"testResample{subject}.csv",
    )


def run_fold(
    *,
    data_path: Path,
    results_path: Path,
    held_subject: int,
    dataset: str,
    components: tuple[str, ...],
    selector: str = "gear-auto",
    proportion: float = DEFAULT_PROPORTION,
) -> None:
    """Fit one shared reduction and every outstanding component on it."""
    if selector not in SELECTORS:
        raise ValueError(f"selector must be one of {sorted(SELECTORS)}.")
    prefix = SELECTORS[selector]
    result_dataset = f"{dataset}LOSO"

    outstanding = [
        component
        for component in components
        if not all(
            path.is_file() and path.stat().st_size > 0
            for path in _result_files(
                results_path, prefix, component, result_dataset, held_subject
            )
        )
    ]
    if not outstanding:
        print(  # noqa: T201
            f"{prefix}/{result_dataset}/subject{held_subject}: "
            "complete results exist for every component; skipping."
        )
        return

    X_train, y_train, X_test, y_test, subjects = load_loso_split(
        data_path, dataset, held_subject
    )
    print(  # noqa: T201
        f"{prefix}/{result_dataset}/subject{held_subject}: "
        f"subjects={len(subjects)}, TRAIN {X_train.shape}, TEST {X_test.shape}, "
        f"components={outstanding}"
    )

    # One reduction for the whole pipeline, fitted once and reused below.
    reducer = _make_reducer(
        selector, X_train.shape[1], random_state=held_subject, proportion=proportion
    )
    start = time.perf_counter_ns()
    X_train_reduced, y_train_reduced = _apply_fit(reducer, X_train, y_train)
    transform_fit_millis = (time.perf_counter_ns() - start) / 1_000_000
    if len(y_train_reduced) != len(y_train):
        raise RuntimeError(f"{prefix} must retain every training label.")

    start = time.perf_counter_ns()
    X_test_reduced = reducer.transform(X_test)
    transform_predict_millis = (time.perf_counter_ns() - start) / 1_000_000

    print(  # noqa: T201
        f"  reduced TRAIN {X_train_reduced.shape} TEST {X_test_reduced.shape} "
        f"in {transform_fit_millis / 1000:.1f}s"
    )

    sidecar = results_path / f"{prefix}-Reduction" / result_dataset
    sidecar.mkdir(parents=True, exist_ok=True)
    (sidecar / f"reduction{held_subject}.json").write_text(
        json.dumps(
            {
                "held_subject": held_subject,
                "transform_fit_millis": transform_fit_millis,
                "transform_predict_millis": transform_predict_millis,
                "train_input_shape": list(X_train.shape),
                "train_output_shape": list(X_train_reduced.shape),
                "test_output_shape": list(X_test_reduced.shape),
                "components": list(components),
                "selector_name": selector,
                "selector": _metadata_to_builtin(_selector_metadata(reducer)),
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    for component in outstanding:
        classifier = _make_hc2_or_component(
            component=component.casefold(),
            random_state=held_subject,
            n_jobs=1,
            fit_contract=0,
            kwargs={},
        )
        if not classifier.get_tag("capability:train_estimate", False, False):
            raise RuntimeError(
                f"{type(classifier).__name__} does not advertise native train estimates."
            )
        print(f"  fitting {component} on the shared reduction")  # noqa: T201
        run_classification_experiment(
            X_train_reduced,
            y_train_reduced,
            X_test_reduced,
            y_test,
            classifier,
            results_path=str(results_path),
            classifier_name=f"{prefix}-{component}",
            dataset_name=result_dataset,
            resample_id=held_subject,
            build_train_file=True,
            build_test_file=True,
            benchmark_time=True,
        )


def main() -> None:
    """Parse arguments and run one LOSO fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("results_path", type=Path)
    parser.add_argument("held_subject", type=int)
    parser.add_argument("--dataset", default="OpenCloseFist")
    parser.add_argument("--components", nargs="+", default=list(DEFAULT_COMPONENTS))
    parser.add_argument("--selector", default="gear-auto", choices=sorted(SELECTORS))
    parser.add_argument("--proportion", type=float, default=DEFAULT_PROPORTION)
    args = parser.parse_args()
    run_fold(
        data_path=args.data_path,
        results_path=args.results_path,
        held_subject=args.held_subject,
        dataset=args.dataset,
        components=tuple(args.components),
        selector=args.selector,
        proportion=args.proportion,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
