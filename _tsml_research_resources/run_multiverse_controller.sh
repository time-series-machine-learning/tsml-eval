#!/bin/bash
# Restart the one-shot Multiverse controller every 30 minutes.

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file=${1:-"${script_dir}/multiverse_controller.toml"}
interval_seconds=${2:-${MULTIVERSE_CONTROLLER_INTERVAL_SECONDS:-1800}}
email_interval_seconds=${3:-${MULTIVERSE_EMAIL_INTERVAL_SECONDS:-14400}}
clear_pending_on_start=${MULTIVERSE_CLEAR_PENDING_ON_START:-true}
clear_pending_partition=${MULTIVERSE_CLEAR_PENDING_PARTITION:-}
python_executable=${PYTHON:-python}
post_cycle_python=${MULTIVERSE_POST_CYCLE_PYTHON:-}
stop_when_complete=${MULTIVERSE_STOP_WHEN_COMPLETE:-true}
complete_exit_status=20
log_dir="${MULTIVERSE_SUPERVISOR_LOG_DIR:-/gpfs/home/${USER}/Results/Multiverse/.controller}"
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

if [[ "${stop_when_complete}" != true &&
    "${stop_when_complete}" != false ]]; then
    echo "ERROR: MULTIVERSE_STOP_WHEN_COMPLETE must be true or false" >&2
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
echo "Stop when complete: ${stop_when_complete}"
echo "Pending-job partition filter: ${clear_pending_partition:-all partitions}"
echo "Log: ${log_file}"

if [[ "${clear_pending_on_start}" == true ]]; then
    if ! command -v squeue >/dev/null 2>&1 ||
        ! command -v scancel >/dev/null 2>&1; then
        echo "ERROR: squeue and scancel are required to clear pending jobs" >&2
        exit 1
    fi

    pending_query=(
        --noheader
        --array
        --user="${USER}"
        --states=PENDING
    )
    if [[ -n "${clear_pending_partition}" ]]; then
        pending_query+=(--partition="${clear_pending_partition}")
    fi
    pending_output=$(squeue "${pending_query[@]}" --format='%i')
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

controller_completion_args=()
if [[ "${stop_when_complete}" == true ]]; then
    controller_completion_args+=(--exit-when-complete)
fi

run_post_cycle() {
    post_cycle_status=0
    if [[ -z "${post_cycle_python}" ]]; then
        return
    fi
    if [[ ! -f "${post_cycle_python}" ]]; then
        echo "ERROR: post-cycle Python script not found: ${post_cycle_python}" \
            | tee -a "${log_file}"
        post_cycle_status=1
        return
    fi
    "${python_executable}" -u "${post_cycle_python}" \
        2>&1 | tee -a "${log_file}"
    post_cycle_status=${PIPESTATUS[0]}
    echo "Post-cycle script exited ${post_cycle_status}." | tee -a "${log_file}"
}

# Send a true startup snapshot before the first queue-refill cycle. An interval
# of zero forces this report on every supervisor start; a successful send then
# records the normal four-hour email marker used by subsequent cycles.
echo "Sending initial controller state." | tee -a "${log_file}"
"${python_executable}" -u "${script_dir}/multiverse_controller.py" \
    --config "${config_file}" \
    --report-only \
    --email-interval-seconds 0 \
    "${controller_completion_args[@]}" \
    2>&1 | tee -a "${log_file}"
initial_report_status=${PIPESTATUS[0]}
if ((initial_report_status == complete_exit_status)); then
    run_post_cycle
    if ((post_cycle_status == 0)); then
        echo "All configured results are complete; supervisor stopping." \
            | tee -a "${log_file}"
        exit 0
    fi
    echo "Completion detected, but post-cycle work failed; continuing." \
        | tee -a "${log_file}"
elif ((initial_report_status != 0)); then
    echo "Initial controller report exited ${initial_report_status}; continuing." \
        | tee -a "${log_file}"
fi

while true; do
    echo "Controller cycle started: $(date --iso-8601=seconds)" | tee -a "${log_file}"
    "${python_executable}" -u "${script_dir}/multiverse_controller.py" \
        --config "${config_file}" \
        --email-interval-seconds "${email_interval_seconds}" \
        "${controller_completion_args[@]}" \
        2>&1 | tee -a "${log_file}"
    controller_status=${PIPESTATUS[0]}
    run_post_cycle
    if ((controller_status == complete_exit_status && post_cycle_status == 0)); then
        echo "All configured results are complete; supervisor stopping." \
            | tee -a "${log_file}"
        exit 0
    fi
    if ((controller_status == complete_exit_status)); then
        echo "Completion detected, but post-cycle work failed; retrying after ${interval_seconds}s." \
            | tee -a "${log_file}"
    else
        echo "Controller exited ${controller_status}; restarting after ${interval_seconds}s." \
            | tee -a "${log_file}"
    fi
    sleep "${interval_seconds}"
done
