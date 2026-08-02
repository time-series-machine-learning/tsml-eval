#!/bin/bash

set -euo pipefail

# Run both GEAR modes over the 25-problem EEG archive. GEAR-Auto applies the
# reduction chosen by the best proxy to HC2; GEAR-Comp learns a separate
# reduction for each of the four HC2 components.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export max_cpus_to_use=20
export memory_per_cpu_gib=30

classifier_prefix="GEAR-Auto" \
component_to_run="HC2" \
job_name_prefix="eeg-gear-auto" \
submission_label="GEAR-Auto" \
    bash "${script_dir}/run_gmar_archive_prefix.sh"

classifier_prefix="GEAR-Comp" \
components_only="true" \
job_name_prefix="eeg-gear-comp" \
submission_label="GEAR-Comp" \
    bash "${script_dir}/run_gmar_archive_prefix.sh"
