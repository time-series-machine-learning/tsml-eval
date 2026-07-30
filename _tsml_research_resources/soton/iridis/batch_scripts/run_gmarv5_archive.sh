#!/bin/bash

set -euo pipefail

# Run the V3 guarded temporal design with DetachRocket channel selection.
# The shared runner submits GMARv5 with HC2 and all four HC2 components over
# the same 25 EEG archive problems and skips non-empty existing results.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export classifier_prefix="GMARv5"
export job_name_prefix="eeg-gmarv5"
export max_cpus_to_use=20
export memory_per_cpu_gib=30

exec bash "${script_dir}/run_gmar_archive_prefix.sh"
