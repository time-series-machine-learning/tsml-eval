#!/bin/bash

set -euo pipefail

username="ajb"
local_path="/gpfs/home/${username}"
results_dir="${local_path}/Results/Multiverse/TestOnly/MultiverseCore"

classifiers=(
    "RDST"
    "Catch22"
    "Dummy"
    "1NN-DTW"
)

datasets=(
    "AsphaltObstaclesCoordinates"
    "AsphaltRegularityCoordinates"
    "CharacterTrajectories"
    "CounterMovementJump"
    "JapaneseVowels"
    "SpokenArabicDigits"
)

delete=false
if [[ "${1:-}" == "--delete" ]]; then
    delete=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--delete]"
    exit 2
fi

if [[ ! -d "${results_dir}" ]]; then
    echo "ERROR: results directory does not exist: ${results_dir}"
    exit 1
fi

found=0
deleted=0

for classifier in "${classifiers[@]}"; do
    for dataset in "${datasets[@]}"; do
        prediction_dir="${results_dir}/${classifier}/Predictions/${dataset}"
        result_file="${prediction_dir}/testResample0.csv"

        if [[ ! -f "${result_file}" ]]; then
            continue
        fi

        ((found += 1))
        if [[ "${delete}" == true ]]; then
            rm -- "${result_file}"
            ((deleted += 1))
            echo "Deleted ${result_file}"
        else
            echo "Would delete ${result_file}"
        fi
    done
done

if [[ "${delete}" == true ]]; then
    echo "Deleted ${deleted} of ${found} matching result files."
else
    echo "Found ${found} matching result files; no files were deleted."
    echo "Run $0 --delete to delete them."
fi
