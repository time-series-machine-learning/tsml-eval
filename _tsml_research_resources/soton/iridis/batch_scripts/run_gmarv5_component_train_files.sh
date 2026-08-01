#!/bin/bash

set -euo pipefail

# Generate the missing train prediction files for the four component-specific
# GMARv5 pipelines over the 25-problem EEG archive. Existing non-empty test
# files are retained: tsml-eval refits the pipeline, disables test-file output,
# and writes only trainResample0.csv. If a test file is genuinely missing, the
# same run safely produces both test and train files.
#
# Exactly two Slurm task-farm jobs are submitted:
#   fast: archive batches 1 and 2 (21 datasets x 4 components = 84 commands)
#   slow: archive batch 3       ( 4 datasets x 4 components = 16 commands)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export classifier_prefix="GMARv5"
export job_name_prefix="eeg-gmarv5-train"
export generate_train_files="true"
export components_only="true"
export batch_mode="fast_slow"

# Match the memory allocation used for the completed GMARv5 component runs.
# The fast job uses at most 20 concurrent processes (600 GiB). The slow job has
# only 16 commands and therefore requests 480 GiB.
export max_cpus_to_use=20
export memory_per_cpu_gib=30

exec bash "${script_dir}/run_gmar_archive_prefix.sh"
