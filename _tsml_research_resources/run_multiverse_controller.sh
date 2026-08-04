#!/bin/bash
# Restart the one-shot Multiverse controller every few hours.

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file=${1:-"${script_dir}/multiverse_controller.toml"}
interval_seconds=${2:-${MULTIVERSE_CONTROLLER_INTERVAL_SECONDS:-10800}}
python_executable=${PYTHON:-python}
log_dir="/gpfs/home/${USER}/Results/Multiverse/.controller"
log_file="${log_dir}/supervisor.log"

if [[ ! "${interval_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: interval must be a positive number of seconds" >&2
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
echo "Log: ${log_file}"

while true; do
    echo "Controller cycle started: $(date --iso-8601=seconds)" | tee -a "${log_file}"
    "${python_executable}" -u "${script_dir}/multiverse_controller.py" \
        --config "${config_file}" 2>&1 | tee -a "${log_file}"
    controller_status=${PIPESTATUS[0]}
    echo "Controller exited ${controller_status}; restarting after ${interval_seconds}s." \
        | tee -a "${log_file}"
    sleep "${interval_seconds}"
done
