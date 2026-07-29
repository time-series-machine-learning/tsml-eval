#!/bin/bash

set -euo pipefail

# Fit the shared TSelect and GMARv3 representations and four
# component-specific GMARv4 representations for the participant-disjoint
# MatchWords MEG case study.

# ==============================================================================
# Experiment configuration
# ==============================================================================

queue="batch"
max_num_submitted=200
max_cpus_to_use=6
memory_per_cpu_gib=32
memory_per_cpu="${memory_per_cpu_gib}G"
max_time="60:00:00"

username="ajb2u23"
mail="NONE"
mailto="${username}@soton.ac.uk"
local_path="/iridisfs/home/${username}"
job_name="meg-matchwords-transform"

problem="MatchWords"
random_state=0
overwrite="false"

variants_to_run=(
    "TSelect"
    "GMARv3"
    "GMARv4-Arsenal"
    "GMARv4-DrCIF"
    "GMARv4-STC"
    "GMARv4-TDE"
)

# ==============================================================================
# Repository, environment, data, and output locations
# ==============================================================================

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"

transform_script="${tsml_eval_dir}/tsml_eval/_wip/eeg_cote/generate_matchwords_transform.py"

env_name="tsml-eval"
python_path="/home/${username}/.conda/envs/${env_name}/bin/python"

input_root="${local_path}/Data/EEG"
output_root="${local_path}/Data/EEGTransforms"

run_root="${local_path}/Results/MatchWordsCaseStudy/transform-stage"
output_log_root="${run_root}/output"
numba_cache_dir="${local_path}/Code/.cache/${env_name}"

# ==============================================================================
# Validate configuration
# ==============================================================================

if [[ ! -x "${python_path}" ]]; then
    echo "ERROR: Python executable not found or not executable:"
    echo "  ${python_path}"
    exit 1
fi

if [[ ! -f "${transform_script}" ]]; then
    echo "ERROR: transform program not found:"
    echo "  ${transform_script}"
    exit 1
fi

for repository in "${tsml_eval_dir}" "${aeon_dir}"; do
    if [[ ! -d "${repository}/.git" ]]; then
        echo "ERROR: Git checkout not found:"
        echo "  ${repository}"
        exit 1
    fi
done

source_directory="${input_root}/${problem}"
standard_train="${source_directory}/${problem}_TRAIN.ts"
standard_test="${source_directory}/${problem}_TEST.ts"
duplicated_train="${standard_train}.ts"
duplicated_test="${standard_test}.ts"

if [[ ! -s "${standard_train}" && ! -s "${duplicated_train}" ]]; then
    echo "ERROR: MatchWords TRAIN file not found."
    echo "Expected ${standard_train} or ${duplicated_train}"
    exit 1
fi
if [[ ! -s "${standard_test}" && ! -s "${duplicated_test}" ]]; then
    echo "ERROR: MatchWords TEST file not found."
    echo "Expected ${standard_test} or ${duplicated_test}"
    exit 1
fi

if ((max_cpus_to_use < 1 || max_cpus_to_use > 192)); then
    echo "ERROR: max_cpus_to_use must be between 1 and 192."
    exit 1
fi
if ((memory_per_cpu_gib < 1)); then
    echo "ERROR: memory_per_cpu_gib must be positive."
    exit 1
fi

mkdir -p \
    "${output_root}" \
    "${run_root}" \
    "${output_log_root}" \
    "${numba_cache_dir}"

tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)
tsml_eval_branch=$(git -C "${tsml_eval_dir}" branch --show-current)
aeon_branch=$(git -C "${aeon_dir}" branch --show-current)

echo "Problem:           ${problem}"
echo "Variants:          ${#variants_to_run[@]}"
echo "Python:            ${python_path}"
echo "tsml-eval branch:  ${tsml_eval_branch}"
echo "tsml-eval commit:  ${tsml_eval_commit}"
echo "aeon branch:       ${aeon_branch}"
echo "aeon commit:       ${aeon_commit}"
echo "Input:             ${source_directory}"
echo "Transformed data:  ${output_root}"
echo "Memory per task:   ${memory_per_cpu}"
echo "Maximum tasks:     ${max_cpus_to_use}"
echo

echo "Available space for transformed datasets:"
df -h "${output_root}"
echo

# Check the exact source layout and all transform factories before submitting.
PYTHONNOUSERSITE=1 \
PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - "${variants_to_run[@]}" <<'PY'
import sys

import aeon
import tsml_eval
from aeon.transformations.collection.channel_selection import TSelect
from tsml_eval.experiments._channel_selection_hc2 import (
    _make_channel_transformer,
    _make_gmarv4_transformer,
)

print("Python:   ", sys.executable)
print("aeon:     ", aeon.__file__)
print("tsml-eval:", tsml_eval.__file__)
print("TSelect:  ", TSelect)

for variant in sys.argv[1:]:
    if variant == "TSelect":
        transformer = TSelect(random_state=0)
    elif variant == "GMARv3":
        transformer = _make_channel_transformer(
            selector="GuardedTemporalV3",
            n_channels=157,
            random_state=0,
            n_jobs=1,
            proxy_component="HC2",
        )
    else:
        transformer = _make_gmarv4_transformer(
            component=variant.removeprefix("GMARv4-"),
            random_state=0,
            n_jobs=1,
        )
    print(f"{variant}: {type(transformer).__name__}")

print("Transform factory check succeeded")
PY

# ==============================================================================
# Generate commands for incomplete representations
# ==============================================================================

run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${run_root}/batch-submissions/${run_id}"
command_file="${submission_dir}/generatedCommandList-${run_id}.txt"
submission_file="${submission_dir}/generatedSubmissionFile-${run_id}.sub"

mkdir -p "${submission_dir}"
: > "${command_file}"

command_count=0

for variant in "${variants_to_run[@]}"; do
    destination="${output_root}/${variant}/${problem}"
    output_train="${destination}/${problem}_TRAIN.ts"
    output_test="${destination}/${problem}_TEST.ts"
    output_summary="${destination}/transform_summary.json"

    if [[ "${overwrite,,}" != "true" ]] \
        && [[ -s "${output_train}" ]] \
        && [[ -s "${output_test}" ]] \
        && [[ -s "${output_summary}" ]]; then
        echo "${variant}: complete output exists; skipping."
        continue
    fi

    variant_log_directory="${output_log_root}/${variant}"
    mkdir -p "${variant_log_directory}"
    experiment_output="${variant_log_directory}/output-${run_id}.txt"

    command=(
        "${python_path}"
        -u
        "${transform_script}"
        "--variant"
        "${variant}"
        "--input-root"
        "${input_root}"
        "--output-root"
        "${output_root}"
        "--problem"
        "${problem}"
        "--random-state"
        "${random_state}"
    )
    if [[ "${overwrite,,}" == "true" ]]; then
        command+=("--overwrite")
    fi

    printf -v command_line '%q ' "${command[@]}"
    printf '%s> %q 2>&1\n' \
        "${command_line}" \
        "${experiment_output}" \
        >> "${command_file}"
    command_count=$((command_count + 1))
done

if ((command_count == 0)); then
    echo "All MatchWords transform outputs are complete; no job submitted."
    exit 0
fi

cpu_count=${command_count}
if ((cpu_count > max_cpus_to_use)); then
    cpu_count=${max_cpus_to_use}
fi
total_memory_gib=$((cpu_count * memory_per_cpu_gib))

# ==============================================================================
# Build and submit one task-farm job
# ==============================================================================

cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=${job_name}
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${run_id}.out
#SBATCH --error=${submission_dir}/%A-${run_id}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpu_count}
#SBATCH --mem-per-cpu=${memory_per_cpu}

# Source the cluster profile before enabling strict shell handling. Some Iridis
# profile fragments reference optional locale variables.
. /etc/profile
set -eo pipefail

cd "${tsml_eval_dir}" || exit 1

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${aeon_dir}:${tsml_eval_dir}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export PYTHONUNBUFFERED=1

export NUMBA_CACHE_DIR="${numba_cache_dir}"
mkdir -p "\${NUMBA_CACHE_DIR}"

current_tsml_eval_commit=\$(git -C "${tsml_eval_dir}" rev-parse HEAD)
current_aeon_commit=\$(git -C "${aeon_dir}" rev-parse HEAD)

if [[ "\${current_tsml_eval_commit}" != "${tsml_eval_commit}" ]]; then
    echo "ERROR: tsml-eval changed after submission."
    echo "Expected: ${tsml_eval_commit}"
    echo "Current:  \${current_tsml_eval_commit}"
    exit 1
fi
if [[ "\${current_aeon_commit}" != "${aeon_commit}" ]]; then
    echo "ERROR: aeon changed after submission."
    echo "Expected: ${aeon_commit}"
    echo "Current:  \${current_aeon_commit}"
    exit 1
fi

echo "Host:             \$(hostname)"
echo "Slurm job ID:     \${SLURM_JOB_ID}"
echo "Allocated tasks:  \${SLURM_NTASKS}"
echo "Command count:    ${command_count}"
echo "Memory per task:  ${memory_per_cpu}"
echo "tsml-eval commit: \${current_tsml_eval_commit}"
echo "aeon commit:      \${current_aeon_commit}"
echo "Command file:     ${command_file}"
echo

"${python_path}" - <<'PY'
import aeon
import tsml_eval
from aeon.transformations.collection.channel_selection import TSelect

print("aeon:     ", aeon.__file__)
print("tsml-eval:", tsml_eval.__file__)
print("TSelect:  ", TSelect)
print("Runtime import check succeeded")
PY

staskfarm "${command_file}"
EOF

while true; do
    num_jobs=$(
        squeue \
            --noheader \
            --user="${username}" \
            --partition="${queue}" \
            --states=RUNNING,PENDING |
            wc -l
    )
    if ((num_jobs < max_num_submitted)); then
        break
    fi
    echo "Waiting 60 seconds: ${num_jobs} jobs are running or pending."
    sleep 60
done

if ! sbatch_output=$(sbatch "${submission_file}"); then
    echo "ERROR: failed to submit ${submission_file}" >&2
    exit 1
fi

echo "${sbatch_output}"
echo "Submitted MatchWords transform stage:"
echo "  Commands:             ${command_count}"
echo "  Requested tasks:      ${cpu_count}"
echo "  Memory per task:      ${memory_per_cpu}"
echo "  Maximum node memory:  ${total_memory_gib} GiB"
echo "  Wall time:            ${max_time}"
echo "  Transformed datasets: ${output_root}"
echo "  Submission records:   ${submission_dir}"

# Slurm has copied the submission script. The command list must remain because
# staskfarm reads it when the queued job begins.
rm -f "${submission_file}"
