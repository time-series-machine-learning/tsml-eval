import os

from tsml_eval.experiments import classification_experiments


def test_kwargs():
    """Test experiments with kwargs input."""
    dataset = "MinimalChinatown"
    classifier = "ROCKET"

    data_path = (
        "./tsml_eval/datasets/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../datasets/"
    )
    result_path = (
        "./test_output/kwargs/"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../../../test_output/kwargs/"
    )

    args = [
        data_path,
        result_path,
        classifier,
        dataset,
        "0",
        "--kwargs",
        "num_kernels",
        "50",
        "int",
        "-ow",
    ]

    classification_experiments.run_experiment(args)

    test_file = f"{result_path}{classifier}/Predictions/{dataset}/testResample0.csv"

    assert os.path.exists(test_file)
    os.remove(test_file)
