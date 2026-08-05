#!/bin/bash
# Restart the one-shot Multiverse controller every hour.

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file=${1:-"${script_dir}/multiverse_controller.toml"}
interval_seconds=${2:-${MULTIVERSE_CONTROLLER_INTERVAL_SECONDS:-3600}}
email_interval_seconds=${3:-${MULTIVERSE_EMAIL_INTERVAL_SECONDS:-14400}}
clear_pending_on_start=${MULTIVERSE_CLEAR_PENDING_ON_START:-true}
python_executable=${PYTHON:-python}
log_dir="/gpfs/home/${USER}/Results/Multiverse/.controller"
log_file="${log_dir}/supervisor.log"

if [[ ! "${interval_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: interval must be a positive number of seconds" >&2
    exit 1
fi

if [[ ! "${email_interval_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: email interval must be a positive number of seconds" >&2
    exit 1
fi

if [[ "${clear_pending_on_start}" != true &&
    "${clear_pending_on_start}" != false ]]; then
    echo "ERROR: MULTIVERSE_CLEAR_PENDING_ON_START must be true or false" >&2
    exit 1
fi

if [[ ! -f "${config_file}" ]]; then
    echo "ERROR: controller configuration not found: ${config_file}" >&2
    exit 1
fi

mkdir -p "${log_dir}"
cd "${repo_dir}" || exit 1

echo "Multiverse controller supervisor started."
echo "Configuration: ${config_file}"
echo "Cycle interval: ${interval_seconds} seconds"
echo "Email interval: ${email_interval_seconds} seconds"
echo "Clear pending jobs on start: ${clear_pending_on_start}"
echo "Log: ${log_file}"

if [[ "${clear_pending_on_start}" == true ]]; then
    if ! command -v squeue >/dev/null 2>&1 ||
        ! command -v scancel >/dev/null 2>&1; then
        echo "ERROR: squeue and scancel are required to clear pending jobs" >&2
        exit 1
    fi

    pending_output=$(
        squeue --noheader --array --user="${USER}" --states=PENDING \
            --format='%i'
    )
    mapfile -t pending_ids < <(
        sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d' \
            <<< "${pending_output}"
    )
    if ((${#pending_ids[@]})); then
        echo "Cancelling ${#pending_ids[@]} pending Slurm tasks before first cycle." \
            | tee -a "${log_file}"
        scancel "${pending_ids[@]}"
    else
        echo "No pending Slurm tasks to cancel before first cycle." \
            | tee -a "${log_file}"
    fi
fi

while true; do
    echo "Controller cycle started: $(date --iso-8601=seconds)" | tee -a "${log_file}"
    "${python_executable}" -u "${script_dir}/multiverse_controller.py" \
        --config "${config_file}" \
        --email-interval-seconds "${email_interval_seconds}" \
        2>&1 | tee -a "${log_file}"
    controller_status=${PIPESTATUS[0]}
    echo "Controller exited ${controller_status}; restarting after ${interval_seconds}s." \
        | tee -a "${log_file}"
    sleep "${interval_seconds}"
done
