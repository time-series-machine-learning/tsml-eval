#!/bin/bash
# Prepare clean full-archive data and run the remaining CPU resample-0 work.

set -eo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_full_resample0_cpu_completion.toml"
source_list="${script_dir}/dataset_lists/Multivariate133Classification-MultiverseClean.txt"
preparer="${script_dir}/prepare_multiverse_full_cpu_data.py"
controller="${script_dir}/multiverse_controller.py"
supervisor="${script_dir}/run_multiverse_controller.sh"
data_dir="/gpfs/home/${USER}/Data/Multiverse"
list_dir="/gpfs/home/${USER}/DataSetLists"
available_list="${list_dir}/MultiverseFullCPU.txt"
unavailable_list="${list_dir}/MultiverseFullCPUUnavailable.txt"
excluded_list="${list_dir}/MultiverseFullCPUExcluded.txt"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-full-resample0-cpu-completion"
session_name="multiverse-full-resample0-cpu-completion"
required_branch="ajb/hc2"

run_worker() {
    mkdir -p "$list_dir" "$data_dir" "$state_dir"
    rm -f -- "${state_dir}/STOP"

    source /etc/profile
    unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER PYTHONPATH
    module purge
    module load python/anaconda/2024.10/3.12.7
    source /gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh
    conda activate tsml-eval

    if [[ "$(basename "${CONDA_PREFIX:-none}")" != "tsml-eval" ]]; then
        echo "ERROR: failed to activate the tsml-eval environment." >&2
        exit 1
    fi
    cd "$repo_dir"

    echo "Checking the CPU experiment environment."
    echo "Python executable: $(command -v python)"
    python -c \
        "import aeon; print('Aeon version:  ' + str(aeon.__version__)); print('Aeon location: ' + str(aeon.__file__))"

    echo "Preparing all eligible equal-length, no-missing Multiverse data."
    python -u "$preparer" \
        --source "$source_list" \
        --data-dir "$data_dir" \
        --available "$available_list" \
        --unavailable "$unavailable_list" \
        --excluded "$excluded_list"

    echo "Checking the remaining CPU work without submitting it."
    python -u "$controller" --config "$config_file" --dry-run --no-email

    echo "Starting recurring controller cycles."
    exec env \
        MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_SUPERVISOR_LOG_DIR="$state_dir" \
        MULTIVERSE_CONTROLLER_INTERVAL_SECONDS=1800 \
        MULTIVERSE_EMAIL_INTERVAL_SECONDS=14400 \
        bash "$supervisor" "$config_file"
}

if [[ "${1:-}" == "--worker" ]]; then
    run_worker
fi

reset_state=false
if [[ "${1:-}" == "--reset-state" ]]; then
    reset_state=true
elif [[ -n "${1:-}" ]]; then
    echo "ERROR: unknown option: ${1}" >&2
    exit 1
fi

for command_name in git pkill screen; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

actual_branch=$(git -C "$repo_dir" branch --show-current)
if [[ "$actual_branch" != "$required_branch" ]]; then
    echo "ERROR: CPU jobs must run from ${required_branch}; found ${actual_branch}." >&2
    exit 1
fi

for required_file in \
    "$config_file" "$source_list" "$preparer" "$controller" "$supervisor"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

# Stop only earlier full-resample-0 CPU feeders. Already submitted Slurm jobs are
# retained and the new controller recognises them by classifier/dataset name.
pkill -TERM -f \
    '[r]un_multiverse_controller.sh.*multiverse_full_resample0_cpu_32gb.toml' \
    || true
pkill -TERM -f \
    '[m]ultiverse_controller.py.*multiverse_full_resample0_cpu_32gb.toml' \
    || true
pkill -TERM -f \
    '[r]un_multiverse_controller.sh.*multiverse_full_resample0_cpu_completion.toml' \
    || true
pkill -TERM -f \
    '[m]ultiverse_controller.py.*multiverse_full_resample0_cpu_completion.toml' \
    || true

for old_name in multiverse-full-resample0-cpu "$session_name"; do
    mapfile -t old_sessions < <(
        screen -ls | awk -v name="$old_name" '$1 ~ ("\\." name "$") {print $1}'
    )
    for session in "${old_sessions[@]}"; do
        echo "Closing old controller screen: ${session}"
        screen -S "$session" -X quit >/dev/null 2>&1 || true
    done
done

if [[ "$reset_state" == true && -f "${state_dir}/state.json" ]]; then
    archived_state="${state_dir}/state.before-environment-fix-$(date +%Y%m%d-%H%M%S).json"
    mv -- "${state_dir}/state.json" "$archived_state"
    echo "Archived controller attempt state: ${archived_state}"
fi

mkdir -p "$state_dir"
bootstrap_log="${state_dir}/bootstrap.log"
echo "Starting detached data preparation and controller: ${session_name}"
screen -L -Logfile "$bootstrap_log" -dmS "$session_name" \
    bash "$script_dir/start_multiverse_full_resample0_cpu_completion.sh" --worker

sleep 2
if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: detached session did not remain running." >&2
    echo "Inspect: ${bootstrap_log}" >&2
    exit 1
fi

echo "Started successfully. Existing results and active jobs will be skipped."
echo "Progress log: ${bootstrap_log}"
echo "Join screen:  screen -r ${session_name}"
screen -ls | grep -F "$session_name" || true
