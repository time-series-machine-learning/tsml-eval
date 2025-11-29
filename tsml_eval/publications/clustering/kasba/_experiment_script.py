import sys

import numpy as np

from tsml_eval.experiments import (
    run_clustering_experiment as tsml_clustering_experiment,
)
from tsml_eval.publications.clustering.kasba._model_configuration import (
    EXPERIMENT_MODELS,
)
from tsml_eval.publications.clustering.kasba._utils import (
    _parse_command_line_bool,
    check_experiment_results_exist,
    load_dataset_from_file,
)


def run_threaded_clustering_experiment(
    dataset: str,
    clusterer_name: str,
    dataset_path: str,
    results_path: str,
    combine_test_train: bool,
    resample_id: int,
):
    """Run clustering experiment.

    Parameters
    ----------
    dataset : str
        Dataset name.
    distance : str
        Distance string (assumed correct and final), e.g.:
        "msm", "dtw", "soft_msm", "soft_dtw",
        "soft_divergence_msm", "soft_divergence_dtw".
    clusterer_str : str
        Free-form label used only for naming/logging (not logic).
    dataset_path : str
        Path to the dataset.
    results_path : str
        Path to the results.
    averaging_method : str
        One of: "soft", "kasba", "petitjean_ba", "subgradient_ba".
    combine_test_train : bool, default=False
        Boolean indicating if data should be combined for test and train.
    resample_id : int, default=0
        Integer indicating the resample id.
    n_jobs : int default=-1
        Integer indicating the number of jobs to run in parallel.
    """
    if clusterer_name not in EXPERIMENT_MODELS:
        raise ValueError(f"Unknown clusterer_name '{clusterer_name}'")

    # Skip if results already exist
    if check_experiment_results_exist(
        model_name=clusterer_name,
        dataset=dataset,
        combine_test_train=combine_test_train,
        path_to_results=results_path,
        resample_id=resample_id,
    ):
        return (
            f"[SKIP] {clusterer_name} (resample {resample_id}): "
            f"results already exist."
        )

    X_train, y_train, X_test, y_test = load_dataset_from_file(
        dataset,
        dataset_path,
        normalize=True,
        combine_test_train=combine_test_train,
        resample_id=0,
    )
    n_clusters = np.unique(y_train).size

    factory = EXPERIMENT_MODELS[clusterer_name]
    clusterer = factory(
        n_clusters=n_clusters,
        random_state=resample_id,
        n_jobs=1,
    )

    tsml_clustering_experiment(
        X_train=X_train,
        y_train=y_train,
        clusterer=clusterer,
        results_path=results_path,
        X_test=X_test,
        y_test=y_test,
        n_clusters=n_clusters,
        clusterer_name=clusterer_name,
        dataset_name=dataset,
        resample_id=resample_id,
        data_transforms=None,
        build_test_file=not combine_test_train,
        build_train_file=True,
        benchmark_time=True,
    )
    print(f"[DONE] {clusterer_name} (resample {resample_id})")


# Boolean to toggle if running locally or via command line.
RUN_LOCALLY = True

if __name__ == "__main__":
    """NOTE: To run with command line arguments, set RUN_LOCALLY to False."""
    if RUN_LOCALLY:
        print("RUNNING WITH TEST CONFIG")

        dataset = "GunPoint"
        clusterer_name = "KASBA"
        combine_test_train = True

        dataset_path = (
            "/Users/chrisholder/Documents/Research/datasets/UCR/Univariate_ts"
        )
        results_path = "/Users/chrisholder/projects/kasba-experiments/full_results"
        run_threaded_clustering_experiment(
            dataset=dataset,
            clusterer_name=clusterer_name,
            dataset_path=dataset_path,
            results_path=results_path,
            combine_test_train=combine_test_train,
            resample_id=0,
        )

    else:
        if len(sys.argv) != 6:
            print(
                "Usage: python _clustering_experiment_all.py "
                "<dataset> <clusterer_name> <dataset_path> <result_path> "
                "<combine_test_train>"
            )
            sys.exit(1)

        dataset = str(sys.argv[1])
        clusterer_name = str(sys.argv[2])
        dataset_path = str(sys.argv[3])
        results_path = str(sys.argv[4])
        combine_test_train = _parse_command_line_bool(sys.argv[5])

        run_threaded_clustering_experiment(
            dataset=dataset,
            clusterer_name=clusterer_name,
            dataset_path=dataset_path,
            results_path=results_path,
            combine_test_train=combine_test_train,
            resample_id=1,
        )
