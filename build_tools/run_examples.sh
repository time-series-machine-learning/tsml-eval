#!/bin/bash

# Script to run all example notebooks.
set -euxo pipefail

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"

excluded=(
  "tsml_eval/publications/y2023/distance_based_clustering/package_distance_timing.ipynb"
)

shopt -s lastpipe
notebooks=()
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

      start=$(date +%s)
      $CMD "$notebook"
      end=$(date +%s)

      notebooks+=("$notebook")
      runtimes+=($((end-start)))
    fi
  done

# print runtimes and notebooks
echo "Runtimes:"
paste <(printf "%s\n" "${runtimes[@]}") <(printf "%s\n" "${notebooks[@]}")
