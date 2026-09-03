#!/bin/bash

set -euo pipefail

# Run aeon's MUSE classifier on resample 0 of all 66 Multiverse-core datasets
# on the Iridis6 batch partition. The shared task-farm controller checks for a
# valid testResample0.csv before submitting each experiment, retries resource
# failures at increasing memory tiers, and chains refills until the pass settles.
# Results are written to Results/Multiverse/DictionaryBased/MUSE.
#
# Usage:
#   bash run_multiverse_muse_classifier.sh
#   bash run_multiverse_muse_classifier.sh --dry-run
#
# To backfill all 30 resamples after the first pass:
#   MV_MAX_FOLDS=30 bash run_multiverse_muse_classifier.sh

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MV_CLASSIFIER="MUSE"
export MV_RESULTS_CATEGORY="DictionaryBased"
export MV_RESULTS_ROOT="${MV_MUSE_RESULTS_ROOT:-/iridisfs/home/ajb2u23/Results/Multiverse/DictionaryBased}"
export MV_WORKFLOW_KEY="muse"
export MV_SUBMISSION_LABEL="MVMUSE"
export MV_MAX_FOLDS="${MV_MAX_FOLDS:-1}"
export MV_START_FOLD="${MV_START_FOLD:-1}"
export MV_DATASET_LIST="${MV_DATASET_LIST:-/iridisfs/home/ajb2u23/Code/tsml-eval/_tsml_research_resources/dataset_lists/MultivariateClassification66-MultiverseMini.txt}"

exec bash "${script_dir}/run_multiverse_interval_classifier.sh" "$@"
