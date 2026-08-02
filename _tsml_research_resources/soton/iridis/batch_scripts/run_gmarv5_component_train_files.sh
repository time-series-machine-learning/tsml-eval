#!/bin/bash

set -euo pipefail

# Generate the missing train prediction files for the four component-specific
# GMARv5 pipelines over the 25-problem EEG archive. Existing non-empty test
# files are retained: tsml-eval refits the pipeline, disables test-file output,
# and writes only trainResample0.csv. If a test file is genuinely missing, the
# same run safely produces both test and train files.
#
# Eight component/runtime-specific Slurm task farms are submitted. This allows
# Slurm to pack the smaller jobs together on the two available nodes without
# forcing every process to inherit TDE's much larger memory request.

# Optional run-set filter. The default preserves the original eight-job run.
# In particular, use RUN_SET=drcif-tde-slow to rerun only the two expensive
# slow groups needed to fill their train prediction files.
run_set="${RUN_SET:-all}"
run_set="${run_set,,}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export classifier_prefix="GMARv5"
export generate_train_files="true"

# component|runtime group|maximum concurrent tasks|GiB per task
#
# Requests include headroom over the peak fit memory recorded in the completed
# GMARv5 test runs. Observed fast/slow maxima in GiB were Arsenal 2.8/11.7,
# DrCIF 6.4/21.2, STC 2.0/4.7, and TDE 20.7/117.2. There are 21 commands per
# fast component and four per slow component, so requesting more tasks would
# not increase concurrency.
all_run_specs=(
    "Arsenal|fast_only|21|6"
    "DrCIF|fast_only|21|12"
    "STC|fast_only|21|6"
    "TDE|fast_only|20|30"
    "Arsenal|slow_only|4|20"
    "DrCIF|slow_only|4|35"
    "STC|slow_only|4|10"
    "TDE|slow_only|4|150"
)

case "${run_set}" in
    all)
        run_specs=("${all_run_specs[@]}")
        ;;
    fast)
        run_specs=(
            "Arsenal|fast_only|21|6"
            "DrCIF|fast_only|21|12"
            "STC|fast_only|21|6"
            "TDE|fast_only|20|30"
        )
        ;;
    slow)
        run_specs=(
            "Arsenal|slow_only|4|20"
            "DrCIF|slow_only|4|35"
            "STC|slow_only|4|10"
            "TDE|slow_only|4|150"
        )
        ;;
    drcif-slow)
        run_specs=("DrCIF|slow_only|4|35")
        ;;
    tde-slow)
        run_specs=("TDE|slow_only|4|150")
        ;;
    drcif-tde-slow|slow-drcif-tde)
        run_specs=(
            "DrCIF|slow_only|4|35"
            "TDE|slow_only|4|150"
        )
        ;;
    *)
        echo "ERROR: unknown RUN_SET: ${run_set}" >&2
        echo "Use all, fast, slow, drcif-slow, tde-slow, or drcif-tde-slow." >&2
        exit 2
        ;;
esac

echo "GMARv5 component train-file run set: ${run_set}"

for spec in "${run_specs[@]}"; do
    IFS="|" read -r component runtime_group cpu_count memory_gib <<< "${spec}"

    component_slug="${component,,}"
    group_slug="${runtime_group%_only}"

    echo "Submitting ${component} ${group_slug}: " \
        "${cpu_count} tasks at ${memory_gib} GiB/task"

    component_to_run="${component}" \
    batch_mode="${runtime_group}" \
    max_cpus_to_use="${cpu_count}" \
    memory_per_cpu_gib="${memory_gib}" \
    job_name_prefix="eeg-gmarv5-train-${component_slug}" \
    submission_label="GMARv5-${component}-${group_slug}" \
        bash "${script_dir}/run_gmar_archive_prefix.sh"
done
