#!/bin/bash

set -euo pipefail

# Complete BPSO over the 25-problem EEG archive without rerunning the 24
# historical datasets. Run STAGE=test first. After those four jobs complete,
# run STAGE=train to generate LongIntervalTask train estimates by parallel CV.

stage="${STAGE:-test}"
stage="${stage,,}"

username="ajb2u23"
local_path="/iridisfs/home/${username}"
tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
python_path="/home/${username}/.conda/envs/tsml-eval/bin/python"

previous_root="${local_path}/Results/ChannelSelection/BPSO"
pipeline_root="${local_path}/Results/ChannelSelectionPipeline"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
archive_runner="${script_dir}/run_gmar_archive_prefix.sh"
parallel_runner="${script_dir}/run_gear_parallel_train_recovery.sh"

components=("Arsenal" "DrCIF" "STC" "TDE")

for required in \
    "${python_path}" \
    "${previous_root}" \
    "${pipeline_root}" \
    "${archive_runner}" \
    "${parallel_runner}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required path does not exist: ${required}" >&2
        exit 1
    fi
done

echo "Importing historical BPSO files without overwriting pipeline results..."
for component in "${components[@]}"; do
    source_dir="${previous_root}/${component}"
    destination_dir="${pipeline_root}/BPSO-${component}"
    if [[ ! -d "${source_dir}" ]]; then
        echo "ERROR: missing historical BPSO directory: ${source_dir}" >&2
        exit 1
    fi
    mkdir -p "${destination_dir}"
    cp -a -n "${source_dir}/." "${destination_dir}/"
done

export PYTHONNOUSERSITE=1
export PYTHONPATH="${aeon_dir}:${tsml_eval_dir}"
"${python_path}" - <<'PY'
import aeon_neuro

from tsml_eval.experiments._get_classifier import get_classifier_by_name

for component in ("Arsenal", "DrCIF", "STC", "TDE"):
    get_classifier_by_name(f"BPSO-{component}", random_state=0, n_jobs=1)
print("BPSO pipeline factory preflight succeeded:", aeon_neuro.__file__)
PY

case "${stage}" in
    test)
        # component|GiB for the single outstanding LongIntervalTask fit
        test_specs=(
            "Arsenal|20"
            "DrCIF|35"
            "STC|12"
            "TDE|60"
        )
        for spec in "${test_specs[@]}"; do
            IFS="|" read -r component memory_gib <<< "${spec}"
            component_slug="${component,,}"
            echo "Submitting BPSO-${component} LongIntervalTask test recovery."
            classifier_prefix="BPSO" \
            component_to_run="${component}" \
            batch_mode="slow_only" \
            generate_train_files="false" \
            max_cpus_to_use="1" \
            memory_per_cpu_gib="${memory_gib}" \
            job_name_prefix="eeg-bpso-${component_slug}" \
            submission_label="BPSO-${component}-recovery" \
                bash "${archive_runner}"
        done
        ;;
    train)
        echo "Submitting four parallel BPSO LongIntervalTask train estimates."
        RUN_SET=bpso bash "${parallel_runner}"
        ;;
    *)
        echo "ERROR: unknown STAGE '${stage}'. Use test or train." >&2
        exit 2
        ;;
esac
