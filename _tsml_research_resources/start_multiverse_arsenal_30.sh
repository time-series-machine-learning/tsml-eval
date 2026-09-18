#!/bin/bash
# Run Arsenal over 30 resamples with train files.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
supervisor="${script_dir}/run_multiverse_controller.sh"
config_file="${script_dir}/multiverse_arsenal_30_train.toml"
session_name="multiverse-arsenal30"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-arsenal-30-train"

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

for command_name in flock git pkill python screen squeue; do
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

if [[ ! -f "${supervisor}" || ! -f "${config_file}" ]]; then
    echo "ERROR: supervisor or Arsenal configuration is missing." >&2
    exit 1
fi

if [[ "${reset_state}" == true && -d "${state_dir}" ]]; then
    archived_state="${state_dir}-previous-$(date +%Y%m%d-%H%M%S)"
    mv -- "${state_dir}" "${archived_state}"
    echo "Archived prior controller state: ${archived_state}"
fi

mkdir -p "${state_dir}"
rm -f -- "${state_dir}/STOP"

pkill -TERM -f '[r]un_multiverse_controller.sh.*multiverse_arsenal_30_train.toml' || true
pkill -TERM -f '[m]ultiverse_controller.py.*multiverse_arsenal_30_train.toml' || true
screen -S "${session_name}" -X quit >/dev/null 2>&1 || true

cd "${repo_dir}"
echo "Arsenal plan: 30 resamples with train files"
python -u "${script_dir}/multiverse_controller.py" \
    --config "${config_file}" --dry-run --no-email

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
    echo "Arsenal 30-resample run completed."
' 

sleep 2
if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: the detached Arsenal controller session did not remain running." >&2
    exit 1
fi

screen -ls | grep -F "${session_name}" || true
