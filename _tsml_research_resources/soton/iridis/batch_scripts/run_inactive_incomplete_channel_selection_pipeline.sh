#!/bin/bash

set -euo pipefail

# Rerun every retained ChannelSelectionPipeline experiment that is neither:
#
#   1. complete (a non-empty testResample0.csv exists), nor
#   2. currently running, waiting inside a running task farm, or covered by a
#      pending Slurm job.
#
# GMARv2, GMARv4, GuardedMultiAxis, and the MatchWords case study are outside
# this recovery scope. The script rescans Slurm at submission time, prints the
# exact recovery set, and splits it into small independent task-farm jobs.

# ==============================================================================
# Experiment and Slurm configuration
# ==============================================================================

resample=0
queue="batch"
max_num_submitted=200

# Each recovery command is single-threaded. Ten commands at 40 GiB each request
# at most 400 GiB on an Iridis 6 node.
commands_per_job=10
memory_per_cpu_gib=40
memory_per_cpu="${memory_per_cpu_gib}G"
max_time="60:00:00"

username="ajb2u23"
mail="NONE"
mailto="${username}@soton.ac.uk"

local_path="/iridisfs/home/${username}"
job_name_prefix="eeg-inactive-recovery"

generate_train_files="false"
predefined_folds="false"
normalise_data="false"

# Change to "false" to inspect the selected tasks and generated command files
# without calling sbatch.
submit_jobs="true"

# ==============================================================================
# Retained paper scope
# ==============================================================================

datasets=(
    "EyesOpenShut"
    "FingerMovements"
    "HandMovementDirection"
    "ButtonPress"
    "LowCost"
    "FeedbackButton"
    "MindReading"
    "FibroLiverpool"
    "FibroUEA"
    "PhotoStimulation"
    "MotorImagery"
    "PronouncedSpeech"
    "InnerSpeech"
    "VisualSpeech"
    "Alzheimers"
    "FaceDetection"
    "OpenCloseFist"
    "ImaginedFeetHands"
    "ImaginedOpenCloseFist"
    "FeetHands"
    "SongFamiliarity"
    "SitStand"
    "ShortIntervalTask"
    "MatchingPennies"
    "LongIntervalTask"
)

transforms=(
    "CSP"
    "ECS"
    "ECP"
    "TSelect"
    "Random"
    "Riemannian"
    "DetachRocket"
    "CaseTimeReducer"
    "CLeVerRank"
    "CLeVerCluster"
    "CLeVerHybrid"
    "GMARv3"
)

classifiers=(
    "HC2"
    "Arsenal"
    "DrCIF"
    "STC"
    "TDE"
)

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
# Validation
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
    echo "ERROR: data directory not found:"
    echo "  ${data_dir}"
    exit 1
fi
if ((commands_per_job < 1 || commands_per_job > 192)); then
    echo "ERROR: commands_per_job must be between 1 and 192."
    exit 1
fi
if ((memory_per_cpu_gib < 1)); then
    echo "ERROR: memory_per_cpu_gib must be positive."
    exit 1
fi
if [[ "${submit_jobs}" != "true" && "${submit_jobs}" != "false" ]]; then
    echo "ERROR: submit_jobs must be true or false."
    exit 1
fi
for required_command in scontrol squeue staskfarm; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "ERROR: required cluster command is unavailable: ${required_command}"
        exit 1
    fi
done
if [[ "${submit_jobs}" == "true" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: required cluster command is unavailable: sbatch"
    exit 1
fi

mkdir -p "${results_dir}" "${out_dir}" "${numba_cache_dir}"

tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)
tsml_eval_branch=$(git -C "${tsml_eval_dir}" branch --show-current)
aeon_branch=$(git -C "${aeon_dir}" branch --show-current)

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
# Discover experiments covered by live Slurm jobs
# ==============================================================================

declare -A active_tasks=()
declare -A active_job_for_task=()

command_file_for_job() {
    local job_id="$1"
    local job_information=""
    local stdout_path=""
    local stdout_name=""
    local stdout_directory=""
    local suffix=""
    local candidate=""
    local master_output=""
    local command_file=""
    local line=""

    job_information=$(scontrol show job -o "${job_id}" 2>/dev/null || true)
    if [[ "${job_information}" =~ StdOut=([^[:space:]]+) ]]; then
        stdout_path="${BASH_REMATCH[1]}"
        stdout_name="${stdout_path##*/}"
        stdout_directory="${stdout_path%/*}"

        if [[ "${stdout_name}" == "${job_id}-"* ]]; then
            suffix="${stdout_name#"${job_id}"-}"
        elif [[ "${stdout_name}" == "%A-"* ]]; then
            suffix="${stdout_name#"%A-"}"
        fi
        suffix="${suffix%.out}"
        candidate="${stdout_directory}/generatedCommandList-${suffix}.txt"
        if [[ -f "${candidate}" ]]; then
            printf "%s" "${candidate}"
            return
        fi
    fi

    while IFS= read -r candidate; do
        master_output="${candidate}"
        break
    done < <(
        find "${results_dir}/batch-submissions" \
            -type f \
            -name "${job_id}-*.out" \
            -print \
            2>/dev/null
    )

    if [[ -n "${master_output}" ]]; then
        while IFS= read -r line; do
            if [[ "${line}" == "Command file:"* ]]; then
                command_file="${line#Command file:}"
                command_file="${command_file#"${command_file%%[![:space:]]*}"}"
            fi
        done < "${master_output}"
    fi

    if [[ -f "${command_file}" ]]; then
        printf "%s" "${command_file}"
    fi
}

log_has_terminal_failure() {
    local output_log="$1"

    [[ -s "${output_log}" ]] || return 1
    grep -Eiq \
        'out[ -]?of[ -]?memory|OUT_OF_MEMORY|oom[_-]kill|Killed process|MemoryError|Cannot allocate memory|std::bad_alloc|Traceback \(most recent call last\)|Segmentation fault|^ERROR:|slurmstepd: error:|Exception:' \
        "${output_log}"
}

refresh_active_tasks() {
    local job_id
    local job_name
    local job_state
    local lower_name
    local command_file
    local command_line
    local pipeline
    local dataset
    local command_resample
    local output_log
    local task_key
    local command_regex='classification_experiments\.py[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)[[:space:]]+([0-9]+)'
    local redirect_regex='>[[:space:]]+([^[:space:]]+)[[:space:]]+2>&1'

    active_tasks=()
    active_job_for_task=()

    while IFS="|" read -r job_id job_name job_state; do
        [[ -z "${job_id}" ]] && continue
        lower_name="${job_name,,}"
        if [[ "${lower_name}" != *eeg* \
            && "${lower_name}" != *channel* \
            && "${lower_name}" != *gmar* ]]; then
            continue
        fi

        command_file=$(command_file_for_job "${job_id}")
        if [[ ! -f "${command_file}" ]]; then
            echo "WARNING: could not map active job ${job_id} (${job_name}) to a command file." >&2
            continue
        fi

        while IFS= read -r command_line || [[ -n "${command_line}" ]]; do
            if [[ ! "${command_line}" =~ ${command_regex} ]]; then
                continue
            fi

            pipeline="${BASH_REMATCH[1]}"
            dataset="${BASH_REMATCH[2]}"
            command_resample="${BASH_REMATCH[3]}"
            if [[ "${command_resample}" != "${resample}" ]]; then
                continue
            fi

            output_log=""
            if [[ "${command_line}" =~ ${redirect_regex} ]]; then
                output_log="${BASH_REMATCH[1]}"
            fi

            # A command with a recognised terminal failure is no longer
            # active even if other commands in its task farm are still alive.
            if [[ "${job_state}" == "RUNNING" \
                && -n "${output_log}" \
                && -e "${output_log}" ]] \
                && log_has_terminal_failure "${output_log}"; then
                continue
            fi

            task_key="${pipeline}|${dataset}|${command_resample}"
            active_tasks["${task_key}"]=1
            active_job_for_task["${task_key}"]="${job_id}"
        done < "${command_file}"
    done < <(
        squeue \
            --noheader \
            --user="${username}" \
            --states=RUNNING,PENDING,CONFIGURING \
            --format="%i|%j|%T" \
            2>/dev/null
    )
}

# ==============================================================================
# Select incomplete, inactive experiments
# ==============================================================================

declare -a tasks_to_run=()
complete_count=0
active_count=0
expected_count=0

refresh_active_tasks

for classifier in "${classifiers[@]}"; do
    for transform in "${transforms[@]}"; do
        pipeline="${transform}-${classifier}"

        for dataset in "${datasets[@]}"; do
            expected_count=$((expected_count + 1))
            test_file="${results_dir}/${pipeline}/Predictions/${dataset}/testResample${resample}.csv"
            task_key="${pipeline}|${dataset}|${resample}"

            if [[ -s "${test_file}" ]]; then
                complete_count=$((complete_count + 1))
                continue
            fi

            if [[ -n "${active_tasks[${task_key}]+present}" ]]; then
                active_count=$((active_count + 1))
                continue
            fi

            tasks_to_run+=("${pipeline}|${dataset}")
        done
    done
done

echo "Inactive incomplete ChannelSelectionPipeline recovery"
echo "-----------------------------------------------------"
echo "Scope:             ${expected_count} experiments"
echo "Complete:          ${complete_count}"
echo "Active incomplete: ${active_count}"
echo "Selected to rerun: ${#tasks_to_run[@]}"
echo "Commands per job:  ${commands_per_job}"
echo "Memory per task:   ${memory_per_cpu}"
echo "Wall time per job: ${max_time}"
echo "Results:           ${results_dir}"
echo "tsml-eval branch:  ${tsml_eval_branch}"
echo "tsml-eval commit:  ${tsml_eval_commit}"
echo "aeon branch:       ${aeon_branch}"
echo "aeon commit:       ${aeon_commit}"
echo

if ((${#tasks_to_run[@]} == 0)); then
    echo "No inactive incomplete experiments found; nothing submitted."
    exit 0
fi

printf "%-27s %s\n" "PIPELINE" "DATASET"
for task in "${tasks_to_run[@]}"; do
    IFS="|" read -r pipeline dataset <<< "${task}"
    printf "%-27s %s\n" "${pipeline}" "${dataset}"
done
echo

# Validate only the datasets and classifier names that will actually run.
declare -A selected_datasets=()
declare -A selected_pipelines=()
pipelines_to_check=()

for task in "${tasks_to_run[@]}"; do
    IFS="|" read -r pipeline dataset <<< "${task}"
    selected_datasets["${dataset}"]=1
    if [[ -z "${selected_pipelines[${pipeline}]+present}" ]]; then
        selected_pipelines["${pipeline}"]=1
        pipelines_to_check+=("${pipeline}")
    fi
done

for dataset in "${!selected_datasets[@]}"; do
    train_data="${data_dir}/${dataset}/${dataset}_TRAIN.ts"
    test_data="${data_dir}/${dataset}/${dataset}_TEST.ts"
    if [[ ! -s "${train_data}" || ! -s "${test_data}" ]]; then
        echo "ERROR: missing or empty raw data for ${dataset}."
        exit 1
    fi
done

# aeon-neuro is imported from the environment. PYTHONPATH supplies only the
# source aeon and tsml-eval checkouts.
PYTHONNOUSERSITE=1 \
PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - "${pipelines_to_check[@]}" <<'PY'
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
    get_classifier_by_name(name, random_state=0, n_jobs=1)
    print(f"Factory OK: {name}")
PY

# ==============================================================================
# Generate and submit independent task-farm chunks
# ==============================================================================

run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${results_dir}/batch-submissions/${run_id}"
mkdir -p "${submission_dir}"

chunk_number=0
chunk_command_count=0
total_commands=0
prepared_jobs=0
submitted_jobs=0
command_file=""

wait_for_queue_slot() {
    local num_jobs

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
            return
        fi
        echo "Waiting 60 seconds: ${num_jobs} jobs are running or pending."
        sleep 60
    done
}

start_chunk() {
    local chunk_label

    chunk_number=$((chunk_number + 1))
    chunk_label=$(printf "%03d" "${chunk_number}")
    command_file="${submission_dir}/generatedCommandList-${run_id}-inactive-${chunk_label}.txt"
    : > "${command_file}"
    chunk_command_count=0
}

append_command() {
    local pipeline="$1"
    local dataset="$2"
    local experiment_output
    local command_line
    local -a command

    mkdir -p "${out_dir}/${pipeline}"
    experiment_output="${out_dir}/${pipeline}/output-${dataset}-${resample}-${run_id}-inactive-${chunk_number}.txt"

    command=(
        "${python_path}"
        -u
        "${script_file_path}"
        "${data_dir}"
        "${results_dir}"
        "${pipeline}"
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

    chunk_command_count=$((chunk_command_count + 1))
    total_commands=$((total_commands + 1))
}

submit_chunk() {
    local chunk_label
    local submission_file
    local total_memory_gib
    local sbatch_output

    if ((chunk_command_count == 0)); then
        return
    fi

    prepared_jobs=$((prepared_jobs + 1))
    chunk_label=$(printf "%03d" "${chunk_number}")
    submission_file="${submission_dir}/generatedSubmissionFile-${run_id}-inactive-${chunk_label}.sub"
    total_memory_gib=$((chunk_command_count * memory_per_cpu_gib))

    cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=${job_name_prefix}-${chunk_label}
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${run_id}-inactive-${chunk_label}.out
#SBATCH --error=${submission_dir}/%A-${run_id}-inactive-${chunk_label}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${chunk_command_count}
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

echo "Recovery chunk:   ${chunk_label}"
echo "Host:             \$(hostname)"
echo "Slurm job ID:     \${SLURM_JOB_ID}"
echo "Allocated tasks:  \${SLURM_NTASKS}"
echo "Command count:    ${chunk_command_count}"
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

    echo "Prepared chunk ${chunk_label}:"
    echo "  Commands:           ${chunk_command_count}"
    echo "  Memory per task:    ${memory_per_cpu}"
    echo "  Maximum job memory: ${total_memory_gib} GiB"
    echo "  Command file:       ${command_file}"

    if [[ "${submit_jobs}" == "true" ]]; then
        wait_for_queue_slot
        if ! sbatch_output=$(sbatch "${submission_file}"); then
            echo "ERROR: failed to submit ${submission_file}" >&2
            exit 1
        fi
        echo "  ${sbatch_output}"
        submitted_jobs=$((submitted_jobs + 1))
    else
        echo "  Dry run: not submitted"
    fi
    echo
}

start_chunk
for task in "${tasks_to_run[@]}"; do
    IFS="|" read -r pipeline dataset <<< "${task}"
    append_command "${pipeline}" "${dataset}"

    if ((chunk_command_count == commands_per_job)); then
        submit_chunk
        start_chunk
    fi
done
submit_chunk

echo "Recovery preparation complete."
echo "Selected experiments: ${#tasks_to_run[@]}"
echo "Commands generated:   ${total_commands}"
echo "Task-farm jobs:       ${prepared_jobs}"
echo "Jobs submitted:       ${submitted_jobs}"
echo "Submission records:   ${submission_dir}"
