#!/bin/bash
# Fill the remaining resample-0 gaps for the Multiverse paper classifiers on the 125
# datasets that leave out the eight huge problems. Runs three controllers:
#
#   A  multiverse_full125_resample0_gaps_cpu.toml       FreshPRINCE, RDST, MRHydra, RIST
#   B  multiverse_full125_hc2_components_train.toml     Arsenal, DrCIF-500, STC, TDE
#                                                       with train files, to rebuild HC2
#   C  multiverse_full125_1nn_dtw_threaded.toml         1NN-DTW at 32 CPUs per task
#
#   ./start_multiverse_full125_gaps.sh --dry-run   prepare data, report what each would
#                                                  submit, submit nothing
#   ./start_multiverse_full125_gaps.sh             the same, then start the supervisors
#
# Existing results and jobs already queued are skipped. Pending jobs are not cancelled.

set -eo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
source_list="${script_dir}/dataset_lists/Multivariate125Classification-MultiverseNoHuge.txt"
preparer="${script_dir}/prepare_multiverse_full_cpu_data.py"
controller="${script_dir}/multiverse_controller.py"
supervisor="${script_dir}/run_multiverse_controller.sh"
configs=(
    "${script_dir}/multiverse_full125_resample0_gaps_cpu.toml"
    "${script_dir}/multiverse_full125_hc2_components_train.toml"
    "${script_dir}/multiverse_full125_1nn_dtw_threaded.toml"
)
data_dir="/gpfs/home/${USER}/Data/Multiverse"
list_dir="/gpfs/home/${USER}/DataSetLists"
available_list="${list_dir}/MultiverseFull125CPU.txt"
unavailable_list="${list_dir}/MultiverseFull125CPUUnavailable.txt"
excluded_list="${list_dir}/MultiverseFull125CPUExcluded.txt"
log_dir="/gpfs/home/${USER}/Results/Multiverse/.full125-gaps"
session_name="multiverse-full125-gaps"
required_branch="ajb/hc2"

activate_environment() {
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
}

run_worker() {
    local dry_run=$1
    mkdir -p "$list_dir" "$data_dir" "$log_dir"
    activate_environment
    cd "$repo_dir"

    echo "Python executable: $(command -v python)"
    python -c \
        "import aeon; print('Aeon version:  ' + str(aeon.__version__)); print('Aeon location: ' + str(aeon.__file__))"

    echo "Preparing equal-length, no-missing data for the 125 datasets."
    python -u "$preparer" \
        --source "$source_list" \
        --data-dir "$data_dir" \
        --available "$available_list" \
        --unavailable "$unavailable_list" \
        --excluded "$excluded_list"
    echo "Datasets available: $(grep -c . "$available_list")"
    if [[ -s "$unavailable_list" ]]; then
        echo "Unavailable, so not run:"
        cat "$unavailable_list"
    fi

    for config in "${configs[@]}"; do
        echo
        echo "=== Dry run: $(basename "$config")"
        python -u "$controller" --config "$config" --dry-run --no-email
    done

    if [[ "$dry_run" == true ]]; then
        echo
        echo "Dry run only: nothing submitted."
        return
    fi

    for config in "${configs[@]}"; do
        name="multiverse-full125-$(basename "$config" .toml)"
        state_dir=$(python -c \
            "import tomllib,sys; c=tomllib.load(open(sys.argv[1],'rb'))['controller']; print(c['state_dir'].format(username=c['username']))" \
            "$config")
        mkdir -p "$state_dir"
        rm -f -- "${state_dir}/STOP"
        echo "Starting supervisor ${name}"
        screen -L -Logfile "${state_dir}/bootstrap.log" -dmS "$name" env \
            MULTIVERSE_CLEAR_PENDING_ON_START=false \
            MULTIVERSE_SUPERVISOR_LOG_DIR="$state_dir" \
            MULTIVERSE_CONTROLLER_INTERVAL_SECONDS=1800 \
            MULTIVERSE_EMAIL_INTERVAL_SECONDS=14400 \
            bash "$supervisor" "$config"
    done
    sleep 2
    screen -ls | grep -F "multiverse-full125-" || true
}

if [[ "${1:-}" == "--worker" ]]; then
    run_worker "${2:-false}"
    exit 0
fi

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
elif [[ -n "${1:-}" ]]; then
    echo "ERROR: unknown option: ${1}" >&2
    exit 1
fi

for command_name in git screen; do
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

for required_file in "$source_list" "$preparer" "$controller" "$supervisor" "${configs[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ "$dry_run" == true ]]; then
    # Short enough to run in the foreground: data preparation plus three dry runs.
    run_worker true
    exit 0
fi

mkdir -p "$log_dir"
if screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: ${session_name} is already running. Join it with: screen -r ${session_name}" >&2
    exit 1
fi
screen -L -Logfile "${log_dir}/bootstrap.log" -dmS "$session_name" \
    bash "$script_dir/start_multiverse_full125_gaps.sh" --worker false

sleep 2
echo "Started. Data preparation and dry runs log to ${log_dir}/bootstrap.log"
echo "The three supervisors start in their own screen sessions once that finishes:"
echo "  screen -ls | grep multiverse-full125-"
