from tsml_eval.experiments.classification_experiments import run_experiment

TSML_EVAL_PATH = "/Users/chrisholder/projects/tsml-eval"
DATASET_LIST_PATH = (
    f"{TSML_EVAL_PATH}/_tsml_research_resources/dataset_lists/Univariate112Datasets.txt"
)
DATASET_PATH = "/Users/chrisholder/Documents/Research/datasets/UCR/Univariate_ts"
RESULT_PATH = (
    "/Users/chrisholder/Documents/Research/SOTON/local-run-mac-results/test-train-split"
)

if __name__ == "__main__":
    # Load the dataset list
    with open(DATASET_LIST_PATH) as f:
        dataset_list = f.read().splitlines()

    for i in range(len(dataset_list)):
        dataset_name = dataset_list[i]
        print(f"{i+1}/112: Running {dataset_name}")

        run_experiment(
            [DATASET_PATH, RESULT_PATH, "minirocket", dataset_name, "0", "-rn", "-tr"]
        )
