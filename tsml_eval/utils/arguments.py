"""tsml-eval command line argument parser."""

__maintainer__ = ["MatthewMiddlehurst"]

__all__ = [
    "parse_args",
]

import argparse

import tsml_eval


def parse_args(args):
    """Parse the command line arguments for tsml_eval.

    The following is the --help output for tsml_eval:

    usage: tsml_eval [-h] [--version] [-ow] [-pr] [-rs RANDOM_SEED] [-nj N_JOBS]
                     [-tr] [-te] [-fc FIT_CONTRACT] [-ch] [-rn] [-nc N_CLUSTERS]
                     [-kw KEY VALUE TYPE]
                     data_path results_path estimator_name dataset_name
                     resample_id

    positional arguments:
      data_path             the path to the directory storing dataset files.
      results_path          the path to the directory where results files are
                            written to.
      estimator_name        the name of the estimator to run. See the
                            set_{task}.py file for each task learning task for
                            available options.
      dataset_name          the name of the dataset to load.
                            {data_dir}/{dataset_name}/{dataset_name}_TRAIN.ts and
                            {data_dir}/{dataset_name}/{dataset_name}_TEST.ts will
                            be loaded.
      resample_id           the resample ID to use when randomly resampling the
                            data, as a random seed for estimators and the suffix
                            when writing results files. An ID of 0 will use the
                            default TRAIN/TEST split.

    options:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      -ow, --overwrite      overwrite existing results files. If False, existing
                            results files will be skipped (default: False).
      -pr, --predefined_resample
                            load a dataset file with a predefined resample. The
                            dataset file must follow the naming format
                            '{dataset_name}_{resample_id}.ts' (default: False).
      -rs RANDOM_SEED, --random_seed RANDOM_SEED
                            use a different random seed than the resample ID. If
                            None use the {resample_id} (default: None).
      -nj N_JOBS, --n_jobs N_JOBS
                            the number of jobs to run in parallel. Only used if
                            the experiments file and selected estimator allows
                            threading (default: 1).
      -tr, --train_fold     write a results file for the training data in the
                            classification and regression task (default: False).
      -te, --test_fold      write a results file for the test data in the
                            clustering task (default: False).
      -fc FIT_CONTRACT, --fit_contract FIT_CONTRACT
                            a time limit for estimator fit in minutes. Only used
                            if the estimator can contract fit (default: 0).
      -ch, --checkpoint     save the estimator fit to file periodically while
                            building. Only used if the estimator can checkpoint
                            (default: False).
      -dtn DATA_TRANSFORM_NAME, --data_transform_name DATA_TRANSFORM_NAME
                            str to pass to get_data_transform_by_name to apply a
                            transformation to the data prior to running the experiment.
                            By default no transform is applied.
                            Can be used multiple times (default: None).
      -tto, --transform_train_only
                            if set, transformations will be applied only to the
                            training dataset leaving the test dataset unchanged
                            (default: False).
      -rn, --row_normalise  normalise the data rows prior to fitting and
                            predicting. effectively the same as passing Normalizer to
                            --data_transform_name (default: False).
      -nc N_CLUSTERS, --n_clusters N_CLUSTERS
                            the number of clusters to find for clusterers which
                            have an {n_clusters} parameter. If {-1}, use the
                            number of classes in the dataset. The {n_clusters} parameter
                            for attributes will also be set. Please ensure that
                            the argument input itself has the {n_clusters} parameters
                            and is not a default such as None. (default: -1).
      -ctts, --combine_test_train_split
                            whether to use a train/test split or not. If True, the
                            train and test sets are combined and used the fit the
                            estimator. Only available for clustering
                            (default: False).
      -bt, --benchmark_time
                            run a benchmark function and save the time spent in the
                            results file (default: False).
      -wa, --write_attributes
                            write the estimator attributes to file when running
                            experiments. Will recursively write the attributes of
                            sub-estimators if present. (default: False).
      -ams ATT_MAX_SHAPE, --att_max_shape ATT_MAX_SHAPE
                            The max estimator collections shape allowed when
                            writing attributes, at 0 no estimators in collections
                            will be written, at 1 estimators in one-dimensional
                            lists will be written etc. (default: 0).
      -kw KEY VALUE TYPE, --kwargs KEY VALUE TYPE, --kwarg KEY VALUE TYPE
                            additional keyword arguments to pass to the estimator.
                            Should contain the parameter to set, the parameter
                            value, and the type of the value i.e. {--kwargs
                            n_estimators 200 int} to change the size of an
                            ensemble. Valid types are {int, float, bool, str}. Any
                            other type will be passed as a str. Can be used
                            multiple times (default: None).

    Parameters
    ----------
    args : list
        List of command line arguments to parse.

    Returns
    -------
    same_resample : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(prog="tsml_eval")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {tsml_eval.__version__}"
    )
    parser.add_argument(
        "data_path", help="the path to the directory storing dataset files."
    )
    parser.add_argument(
        "results_path",
        help="the path to the directory where results files are written to.",
    )
    parser.add_argument(
        "estimator_name",
        help="the name of the estimator to run. See the set_{task}.py file for each "
        "task learning task for available options.",
    )
    parser.add_argument(
        "dataset_name",
        help="the name of the dataset to load. "
        "{data_dir}/{dataset_name}/{dataset_name}_TRAIN.ts and "
        "{data_dir}/{dataset_name}/{dataset_name}_TEST.ts will be loaded.",
    )
    parser.add_argument(
        "resample_id",
        type=int,
        help="the resample ID to use when randomly resampling the data, as a random "
        "seed for estimators and the suffix when writing results files. An ID of "
        "0 will use the default TRAIN/TEST split.",
    )
    parser.add_argument(
        "-ow",
        "--overwrite",
        action="store_true",
        help="overwrite existing results files. If False, existing results files "
        "will be skipped (default: %(default)s).",
    )
    parser.add_argument(
        "-pr",
        "--predefined_resample",
        action="store_true",
        help="load a dataset file with a predefined resample. The dataset file must "
        "follow the naming format '{dataset_name}{resample_id}.ts' "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "-rs",
        "--random_seed",
        type=int,
        help="use a different random seed than the resample ID. If None use the "
        "{resample_id} (default: %(default)s).",
    )
    parser.add_argument(
        "-nj",
        "--n_jobs",
        type=int,
        default=1,
        help="the number of jobs to run in parallel. Only used if the experiments file "
        "and selected estimator allows threading (default: %(default)s).",
    )
    parser.add_argument(
        "-tr",
        "--train_fold",
        action="store_true",
        help="write a results file for the training data in the classification and "
        "regression task (default: %(default)s).",
    )
    parser.add_argument(
        "-te",
        "--test_fold",
        action="store_true",
        help="write a results file for the test data in the clustering task "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "-fc",
        "--fit_contract",
        type=int,
        default=0,
        help="a time limit for estimator fit in minutes. Only used if the estimator "
        "can contract fit (default: %(default)s).",
    )
    parser.add_argument(
        "-ch",
        "--checkpoint",
        action="store_true",
        help="save the estimator fit to file periodically while building. Only used if "
        "the estimator can checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "-dtn",
        "--data_transform_name",
        action="append",
        help="str to pass to get_data_transform_by_name to apply a transformation "
        "to the data prior to running the experiment. By default no transform "
        "is applied. Can be used multiple times (default: %(default)s).",
    )
    parser.add_argument(
        "-tto",
        "--transform_train_only",
        action="store_true",
        help="if set, transformations will be applied only to the training dataset, "
        "leaving the test dataset unchanged (default: %(default)s).",
    )
    parser.add_argument(
        "-rn",
        "--row_normalise",
        action="store_true",
        help="normalise the data rows prior to fitting and predicting. "
        "effectively the same as passing Normalizer to --data_transform_name "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "-nc",
        "--n_clusters",
        type=int,
        default=-1,
        help="the number of clusters to find for clusterers which have an {n_clusters} "
        "parameter. If {-1}, use the number of classes in the dataset. The "
        "{n_clusters} parameter for arguments will also be set. Please ensure that"
        "the argument input itself has the {n_clusters} parameters and is not a default"
        "such as None (default: %(default)s).",
    )
    parser.add_argument(
        "-ctts",
        "--combine_test_train_split",
        action="store_true",
        help="whether to use a train/test split or not. If True, the train and test "
        "sets are combined and used the fit the estimator. Only available for "
        "clustering (default: %(default)s).",
    )
    parser.add_argument(
        "-bt",
        "--benchmark_time",
        action="store_true",
        help="run a benchmark function and save the time spent in the results file "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "-wa",
        "--write_attributes",
        action="store_true",
        help="write the estimator attributes to file when running experiments. Will "
        "recursively write the attributes of sub-estimators if present. "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "-ams",
        "--att_max_shape",
        type=int,
        default=0,
        help="The max estimator collections shape allowed when writing attributes, at "
        "0 no estimators in collections will be written, at 1 estimators in "
        "one-dimensional lists will be written etc. (default: %(default)s).",
    )
    parser.add_argument(
        "-kw",
        "--kwargs",
        "--kwarg",
        action="append",
        nargs=3,
        metavar=("KEY", "VALUE", "TYPE"),
        help="additional keyword arguments to pass to the estimator. Should contain "
        "the parameter to set, the parameter value, and the type of the value i.e. "
        "{--kwargs n_estimators 200 int} to change the size of an ensemble. Valid "
        "types are {int, float, bool, str}. Any other type will be passed as a str. "
        "Can be used multiple times (default: %(default)s).",
    )
    args = parser.parse_args(args=args)

    kwargs = {}
    if args.kwargs is not None:
        for kwarg in args.kwargs:
            if kwarg[2] == "int":
                kwargs[kwarg[0]] = int(kwarg[1])
            elif kwarg[2] == "float":
                kwargs[kwarg[0]] = float(kwarg[1])
            elif kwarg[2] == "bool":
                kwargs[kwarg[0]] = kwarg[1].lower() == "true" or kwarg[1] == "1"
            else:
                kwargs[kwarg[0]] = kwarg[1]
    args.kwargs = kwargs

    return args
