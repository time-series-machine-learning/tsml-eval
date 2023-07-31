#!/bin/bash

# Script to run all example notebooks.
set -euxo pipefail

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"

excluded=(
  "examples/_wip/diagrams.ipynb"
  "examples/_wip/evaluation_metric_results.ipynb"
  "examples/_wip/evaluation_raw_results.ipynb"
  "tsml_eval/publications/y2023/distance_based_clustering/package_distance_timing.ipynb"
  "tsml_eval/publications/y2023/distance_based_clustering/distance_based_clustering.ipynb"
  "tsml_eval/publications/y2023/distance_based_clustering/alignment_and_paths_figures.ipynb"
)

runtimes=()

# Loop over all notebooks in the examples and publications directory.
find "examples/" "tsml_eval/publications/" -name "*.ipynb" -print0 |
  while IFS= read -r -d "" notebook; do
    # Skip notebooks in the excluded list.
    if printf "%s\0" "${excluded[@]}" | grep -Fxqz -- "$notebook"; then
      echo "Skipping: $notebook"
    # Run the notebook.
    else
      echo "Running: $notebook"
      $CMD "$notebook"

      start=`date +%s`
      $CMD "$notebook"
      end=`date +%s`

      runtimes+=($((end-start)))
    fi
  done

# print first 5 items in runtimes array
echo "Runtimes:"
echo "${runtimes[@]:0:5}"
