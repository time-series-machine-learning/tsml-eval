#!/bin/bash

set -euo pipefail

# Retry only the Multiverse-core resample-0 results still missing for MUSE.
#
# This workflow deliberately has a fresh state directory and source pin. The
# original MUSE chain was submitted before its registration source changed;
# jobs that waited in Slurm subsequently failed their provenance check before
# reaching an experiment. Results remain in the same MUSE directory, so the
# shared controller still skips every valid testResample0.csv already present.
#
# The first pass established that the remaining datasets create feature
# matrices far larger than their raw .ts files suggest (up to 543 GiB in the
# copied logs). Start every missing experiment at the 620 GiB tier. This yields
# one experiment at a time per node and schedules larger raw datasets first.
#
# Usage:
#   bash run_multiverse_muse_missing_classifier.sh
#   bash run_multiverse_muse_missing_classifier.sh --dry-run

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MV_CLASSIFIER="MUSE"
export MV_RESULTS_CATEGORY="DictionaryBased"
export MV_RESULTS_ROOT="${MV_MUSE_RESULTS_ROOT:-/iridisfs/home/ajb2u23/Results/Multiverse/DictionaryBased}"
export MV_WORKFLOW_KEY="muse-missing"
export MV_SUBMISSION_LABEL="MVMUSEMissing"
export MV_MAX_FOLDS="1"
export MV_START_FOLD="1"
export MV_DATASET_LIST="${MV_DATASET_LIST:-/iridisfs/home/ajb2u23/Code/tsml-eval/_tsml_research_resources/dataset_lists/MultivariateClassification66-MultiverseMini.txt}"

# Tier 8 is 620 GiB. Set all size classes because MUSE's transformed feature
# size, rather than the dataset's on-disk size, determines memory consumption.
export large_dataset_start_tier="8"
export medium_dataset_start_tier="8"
export default_dataset_start_tier="8"

# At the maximum tier there is nowhere further to escalate. Two attempts allow
# one retry for a transient/silent node failure without looping indefinitely on
# a deterministic estimator error.
export max_attempts_per_experiment="2"
export max_failed_attempts="2"

exec bash "${script_dir}/run_multiverse_interval_classifier.sh" "$@"
