#!/bin/bash
# Prepare the full local Multiverse CPU list and run Arsenal on resample 0.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_full_arsenal_resample0_train.toml"
source_list="${script_dir}/dataset_lists/Multivariate133Classification-MultiverseClean.txt"
preparer="${script_dir}/prepare_multiverse_full_cpu_data.py"
controller="${script_dir}/multiverse_controller.py"
supervisor="${script_dir}/run_multiverse_controller.sh"
data_dir="/gpfs/home/${USER}/Data/Multiverse"
list_dir="/gpfs/home/${USER}/DataSetLists"
available_list="${list_dir}/MultiverseFullCPU.txt"
unavailable_list="${list_dir}/MultiverseFullCPUUnavailable.txt"
excluded_list="${list_dir}/MultiverseFullCPUExcluded.txt"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-full-arsenal-resample0-train"
session_name="multiverse-full-arsenal-resample0-train"

case "${1:-}" in
    "") ;;
    --reset-state)
        reset_state=true
        ;;
    *)
        echo "ERROR: unknown option: ${1}" >&2
        echo "Usage: bash $(basename "$0") [--reset-state]" >&2
        exit 1
        ;;
esac
if [[ -n "${2:-}" ]]; then
    echo "ERROR: too many arguments." >&2
    exit 1
fi
reset_state=${reset_state:-false}

for command_name in flock git mktemp pkill python screen squeue; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

actual_branch=$(git -C "${repo_dir}" branch --show-current)
if [[ "${actual_branch}" != "ajb/hc2" ]]; then
    echo "ERROR: CPU jobs must run from ajb/hc2; found ${actual_branch:-DETACHED}." >&2
    exit 1
fi

for required_file in "${config_file}" "${source_list}" "${preparer}" \
    "${controller}" "${supervisor}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -d "${data_dir}" ]]; then
    echo "ERROR: Multiverse data directory not found: ${data_dir}" >&2
    exit 1
fi

if [[ "${reset_state}" == true && -d "${state_dir}" ]]; then
    archived_state="${state_dir}-previous-$(date +%Y%m%d-%H%M%S)"
    mv -- "${state_dir}" "${archived_state}"
    echo "Archived prior controller state: ${archived_state}"
fi
mkdir -p "${list_dir}" "${state_dir}"
rm -f -- "${state_dir}/STOP"

# This is intentionally limited to the full-Arsenal controller. Existing Slurm
# experiment jobs are retained and will be recognised by the new controller.
pkill -TERM -f '[r]un_multiverse_controller.sh.*multiverse_full_arsenal_resample0_train.toml' || true
pkill -TERM -f '[m]ultiverse_controller.py.*multiverse_full_arsenal_resample0_train.toml' || true
screen -S "${session_name}" -X quit >/dev/null 2>&1 || true

cd "${repo_dir}"
echo "Preparing the full locally available Multiverse CPU list."
python -u "${preparer}" \
    --source "${source_list}" \
    --data-dir "${data_dir}" \
    --available "${available_list}" \
    --unavailable "${unavailable_list}" \
    --excluded "${excluded_list}"

if [[ ! -s "${available_list}" ]]; then
    echo "ERROR: no eligible datasets were written to ${available_list}." >&2
    exit 1
fi

echo "Arsenal plan: full Multiverse, resample 0, with train files"
python -u "${controller}" --config "${config_file}" --dry-run --no-email

echo "Starting Arsenal controller session: ${session_name}"
screen -dmS "${session_name}" bash -lc '
    set -euo pipefail
    cd "'"${repo_dir}"'"
    flock -n "'"${state_dir}"'/supervisor.lock" \
        env MULTIVERSE_CLEAR_PENDING_ON_START=false \
            MULTIVERSE_CONTROLLER_INTERVAL_SECONDS=3600 \
            MULTIVERSE_EMAIL_INTERVAL_SECONDS=14400 \
            MULTIVERSE_SUPERVISOR_LOG_DIR="'"${state_dir}"'" \
        bash "'"${supervisor}"'" "'"${config_file}"'"
    echo "Full Multiverse Arsenal resample-0 run completed."
'

sleep 2
if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: the detached Arsenal controller session did not remain running." >&2
    exit 1
fi

screen -ls | grep -F "${session_name}" || true
