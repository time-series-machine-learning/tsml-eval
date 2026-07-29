#!/bin/bash

set -euo pipefail

# Submit one of two explicit ChannelSelectionPipeline recovery sets.
# This uses aeon-neuro from the tsml-eval environment, not a source checkout.

# ==============================================================================
# Recovery set: change this one hard-coded value
# ==============================================================================

# "missing_hc2":
#   The nine missing HC2 results for which all four current component test
#   results are already available.
#
# "reconstructable_components":
#   The 17 component results whose old and new predictions agreed on every
#   overlapping problem. Rerunning these gives definitive full-pipeline timing
#   instead of copying classifier-only timings from the old result files.
# missing_hc2 or reconstructable_components
recovery_set="reconstructable_components"

# ==============================================================================
# Experiment configuration
# ==============================================================================

max_folds=1
start_fold=1

max_num_submitted=200
queue="batch"
max_cpus_to_use=17

memory_per_cpu_gib=35
memory_per_cpu="${memory_per_cpu_gib}G"
max_time="60:00:00"

username="ajb2u23"
mail="NONE"
mailto="${username}@soton.ac.uk"

local_path="/iridisfs/home/${username}"
job_name_prefix="eeg-reconstruct17"

generate_train_files="false"
predefined_folds="false"
normalise_data="false"

# ==============================================================================
# The two exact recovery lists
# ==============================================================================

missing_hc2_tasks=(
    "GMARv2-HC2|MatchingPennies"
    "GMARv2-HC2|ShortIntervalTask"
    "CaseTimeReducer-HC2|ShortIntervalTask"
    "CaseTimeReducer-HC2|SitStand"
    "CLeVerCluster-HC2|LongIntervalTask"
    "CLeVerRank-HC2|SitStand"
    "DetachRocket-HC2|SitStand"
    "GMARv3-HC2|ShortIntervalTask"
    "GMARv3-HC2|SitStand"
)

reconstructable_component_tasks=(
    "CLeVerHybrid-TDE|LongIntervalTask"
    "CLeVerHybrid-TDE|ShortIntervalTask"
    "CLeVerRank-STC|LongIntervalTask"
    "CLeVerRank-TDE|LongIntervalTask"
    "CLeVerRank-TDE|ShortIntervalTask"
    "ECP-TDE|MatchingPennies"
    "ECP-TDE|ShortIntervalTask"
    "ECP-TDE|SitStand"
    "ECS-TDE|LongIntervalTask"
    "ECS-TDE|MatchingPennies"
    "Random-TDE|LongIntervalTask"
    "Random-TDE|ShortIntervalTask"
    "Random-TDE|SitStand"
    "Riemannian-STC|LongIntervalTask"
    "Riemannian-TDE|LongIntervalTask"
    "Riemannian-TDE|MatchingPennies"
    "Riemannian-TDE|ShortIntervalTask"
)

case "${recovery_set}" in
    missing_hc2)
        tasks_to_run=("${missing_hc2_tasks[@]}")
        expected_task_count=9
        ;;
    reconstructable_components)
        tasks_to_run=("${reconstructable_component_tasks[@]}")
        expected_task_count=17
        ;;
    *)
        echo "ERROR: unknown recovery_set: ${recovery_set}"
        echo "Use missing_hc2 or reconstructable_components."
        exit 1
        ;;
esac

if ((${#tasks_to_run[@]} != expected_task_count)); then
    echo "ERROR: ${recovery_set} should contain exactly ${expected_task_count} tasks."
    exit 1
fi

# ==============================================================================
# Repository, environment, data, and result locations
# ==============================================================================

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"

script_file_path="${tsml_eval_dir}/tsml_eval/experiments/classification_experiments.py"

env_name="tsml-eval"
python_path="/home/${username}/.conda/envs/${env_name}/bin/python"

data_dir="${local_path}/Data/EEG"
results_dir="${local_path}/Results/ChannelSelectionPipeline"
out_dir="${results_dir}/output"
numba_cache_dir="${local_path}/Code/.cache/${env_name}"

# ==============================================================================
# Validate configuration
# ==============================================================================

if [[ ! -x "${python_path}" ]]; then
    echo "ERROR: Python executable not found or not executable:"
    echo "  ${python_path}"
    exit 1
fi
if [[ ! -f "${script_file_path}" ]]; then
    echo "ERROR: classification experiment script not found:"
    echo "  ${script_file_path}"
    exit 1
fi
for repository in "${tsml_eval_dir}" "${aeon_dir}"; do
    if [[ ! -d "${repository}/.git" ]]; then
        echo "ERROR: Git checkout not found:"
        echo "  ${repository}"
        exit 1
    fi
done
if [[ ! -d "${data_dir}" ]]; then
    echo "ERROR: data directory not found: ${data_dir}"
    exit 1
fi
if ((start_fold < 1 || max_folds < start_fold)); then
    echo "ERROR: invalid fold range ${start_fold}..${max_folds}."
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
for required_command in sbatch squeue staskfarm; do
    if ! command -v "${required_command}" > /dev/null 2>&1; then
        echo "ERROR: required cluster command is unavailable: ${required_command}"
        exit 1
    fi
done

mkdir -p "${results_dir}" "${out_dir}" "${numba_cache_dir}"

tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)
tsml_eval_branch=$(git -C "${tsml_eval_dir}" branch --show-current)
aeon_branch=$(git -C "${aeon_dir}" branch --show-current)

# Validate every selected dataset and collect each distinct classifier.
declare -A seen_tasks
declare -A seen_classifiers
classifiers_to_check=()

for task in "${tasks_to_run[@]}"; do
    IFS="|" read -r classifier dataset <<< "${task}"
    if [[ -z "${classifier}" || -z "${dataset}" ]]; then
        echo "ERROR: malformed task: ${task}"
        exit 1
    fi
    if [[ -n "${seen_tasks[${task}]+present}" ]]; then
        echo "ERROR: duplicate task: ${task}"
        exit 1
    fi
    seen_tasks["${task}"]=1

    train_data="${data_dir}/${dataset}/${dataset}_TRAIN.ts"
    test_data="${data_dir}/${dataset}/${dataset}_TEST.ts"
    if [[ ! -s "${train_data}" || ! -s "${test_data}" ]]; then
        echo "ERROR: missing or empty raw data for ${dataset}."
        exit 1
    fi

    if [[ -z "${seen_classifiers[${classifier}]+present}" ]]; then
        seen_classifiers["${classifier}"]=1
        classifiers_to_check+=("${classifier}")
    fi
done

echo "Recovery set:      ${recovery_set}"
echo "Configured tasks:  ${#tasks_to_run[@]}"
echo "Python:            ${python_path}"
echo "Data directory:    ${data_dir}"
echo "Results:           ${results_dir}"
echo "Memory per task:   ${memory_per_cpu}"
echo "Maximum tasks:     ${max_cpus_to_use}"
echo "tsml-eval branch:  ${tsml_eval_branch}"
echo "tsml-eval commit:  ${tsml_eval_commit}"
echo "aeon branch:       ${aeon_branch}"
echo "aeon commit:       ${aeon_commit}"
echo

# aeon-neuro is deliberately imported from the environment. PYTHONPATH only
# supplies the source aeon and tsml-eval checkouts.
PYTHONNOUSERSITE=1 \
PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - "${classifiers_to_check[@]}" <<'PY'
import sys

import aeon
import aeon_neuro
import tsml_eval
from tsml_eval.experiments import get_classifier_by_name

print("Python:     ", sys.executable)
print("aeon:       ", aeon.__file__)
print("aeon-neuro: ", aeon_neuro.__file__)
print("tsml-eval:  ", tsml_eval.__file__)

for name in sys.argv[1:]:
    classifier = get_classifier_by_name(name, random_state=0, n_jobs=1)
    print(f"{name}: {type(classifier).__name__}")

print("Factory check succeeded")
PY

# ==============================================================================
# Convert Boolean options into tsml-eval arguments
# ==============================================================================

generate_train_arg=""
predefined_folds_arg=""
normalise_data_arg=""

if [[ "${generate_train_files,,}" == "true" ]]; then
    generate_train_arg="-tr"
fi
if [[ "${predefined_folds,,}" == "true" ]]; then
    predefined_folds_arg="-pr"
fi
if [[ "${normalise_data,,}" == "true" ]]; then
    normalise_data_arg="-rn"
fi

# ==============================================================================
# Generate commands for incomplete selected tasks
# ==============================================================================

run_id=$(date +%Y%m%d%H%M%S)
batch_id="${run_id}-${recovery_set}"
submission_dir="${results_dir}/batch-submissions/${run_id}"
command_file="${submission_dir}/generatedCommandList-${batch_id}.txt"
submission_file="${submission_dir}/generatedSubmissionFile-${batch_id}.sub"

mkdir -p "${submission_dir}"
: > "${command_file}"

command_count=0

for task in "${tasks_to_run[@]}"; do
    IFS="|" read -r classifier dataset <<< "${task}"
    mkdir -p "${out_dir}/${classifier}"

    for ((
        resample = start_fold - 1;
        resample < max_folds;
        resample++
    )); do
        test_file="${results_dir}/${classifier}/Predictions/${dataset}/testResample${resample}.csv"
        train_file="${results_dir}/${classifier}/Predictions/${dataset}/trainResample${resample}.csv"

        if [[ -s "${test_file}" ]]; then
            if [[ -z "${generate_train_arg}" || -s "${train_file}" ]]; then
                echo "Skipping complete: ${classifier}/${dataset}/resample${resample}"
                continue
            fi
        fi

        experiment_output="${out_dir}/${classifier}/output-${dataset}-${resample}-${batch_id}.txt"
        command=(
            "${python_path}"
            -u
            "${script_file_path}"
            "${data_dir}"
            "${results_dir}"
            "${classifier}"
            "${dataset}"
            "${resample}"
        )
        if [[ -n "${generate_train_arg}" ]]; then
            command+=("${generate_train_arg}")
        fi
        if [[ -n "${predefined_folds_arg}" ]]; then
            command+=("${predefined_folds_arg}")
        fi
        if [[ -n "${normalise_data_arg}" ]]; then
            command+=("${normalise_data_arg}")
        fi

        printf -v command_line "%q " "${command[@]}"
        printf "%s> %q 2>&1\n" \
            "${command_line}" \
            "${experiment_output}" \
            >> "${command_file}"
        command_count=$((command_count + 1))
    done
done

if ((command_count == 0)); then
    echo "All selected ${recovery_set} results already exist; no job submitted."
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
#SBATCH --job-name=${job_name_prefix}-${recovery_set}
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${batch_id}.out
#SBATCH --error=${submission_dir}/%A-${batch_id}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpu_count}
#SBATCH --mem-per-cpu=${memory_per_cpu}

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

echo "Recovery set:     ${recovery_set}"
echo "Host:             \$(hostname)"
echo "Slurm job ID:     \${SLURM_JOB_ID}"
echo "Allocated tasks:  \${SLURM_NTASKS}"
echo "Command count:    ${command_count}"
echo "Memory per task:  ${memory_per_cpu}"
echo "Command file:     ${command_file}"
echo "tsml-eval commit: \${current_tsml_eval_commit}"
echo "aeon commit:      \${current_aeon_commit}"
echo

"${python_path}" - <<'PY'
import aeon
import aeon_neuro
import tsml_eval

print("aeon:       ", aeon.__file__)
print("aeon-neuro: ", aeon_neuro.__file__)
print("tsml-eval:  ", tsml_eval.__file__)
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
echo "Submitted ${recovery_set}:"
echo "  Configured tasks:    ${#tasks_to_run[@]}"
echo "  Outstanding commands:${command_count}"
echo "  Requested tasks:     ${cpu_count}"
echo "  Memory per task:     ${memory_per_cpu}"
echo "  Maximum node memory: ${total_memory_gib} GiB"
echo "  Wall time:           ${max_time}"
echo "  Submission records:  ${submission_dir}"

# Slurm has copied the submission script. The command list must remain because
# staskfarm reads it when the queued job begins.
rm -f "${submission_file}"
