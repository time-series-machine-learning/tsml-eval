#!/bin/bash
# Run Arsenal resample 0 with train files, then HC2 resample 0 without train files.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
supervisor="${script_dir}/run_multiverse_controller.sh"
arsenal_config="${script_dir}/multiverse_arsenal_resample0_train.toml"
hc2_config="${script_dir}/multiverse_hc2_resample0_no_train.toml"
session_name="multiverse-arsenal-hc2-resample0"
arsenal_state="/gpfs/home/${USER}/Results/Multiverse/.controller-arsenal-resample0-train"
hc2_state="/gpfs/home/${USER}/Results/Multiverse/.controller-hc2-resample0-no-train"

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

for required_file in "${supervisor}" "${arsenal_config}" "${hc2_config}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ "${reset_state}" == true ]]; then
    for state_dir in "${arsenal_state}" "${hc2_state}"; do
        if [[ -d "${state_dir}" ]]; then
            archived_state="${state_dir}-previous-$(date +%Y%m%d-%H%M%S)"
            mv -- "${state_dir}" "${archived_state}"
            echo "Archived prior controller state: ${archived_state}"
        fi
    done
fi

mkdir -p "${arsenal_state}" "${hc2_state}"
rm -f -- "${arsenal_state}/STOP" "${hc2_state}/STOP"

# Stop controllers from either the old 30-resample run or this run.
pkill -TERM -f '[r]un_multiverse_controller.sh.*multiverse_arsenal_30_train.toml' || true
pkill -TERM -f '[m]ultiverse_controller.py.*multiverse_arsenal_30_train.toml' || true
pkill -TERM -f '[r]un_multiverse_controller.sh.*multiverse_arsenal_resample0_train.toml' || true
pkill -TERM -f '[m]ultiverse_controller.py.*multiverse_arsenal_resample0_train.toml' || true
pkill -TERM -f '[r]un_multiverse_controller.sh.*multiverse_hc2_resample0_no_train.toml' || true
pkill -TERM -f '[m]ultiverse_controller.py.*multiverse_hc2_resample0_no_train.toml' || true
screen -S "${session_name}" -X quit >/dev/null 2>&1 || true

cd "${repo_dir}"
echo "Arsenal plan: resample 0 with train files"
python -u "${script_dir}/multiverse_controller.py" \
    --config "${arsenal_config}" --dry-run --no-email
echo "HC2 plan: resample 0 without train files"
python -u "${script_dir}/multiverse_controller.py" \
    --config "${hc2_config}" --dry-run --no-email

echo "Starting sequential Arsenal resample 0 -> HC2 resample 0 session: ${session_name}"
screen -dmS "${session_name}" bash -lc '
    set -euo pipefail
    cd "'"${repo_dir}"'"
    flock -n "'"${arsenal_state}"'/supervisor.lock" \
        env MULTIVERSE_CLEAR_PENDING_ON_START=false \
            MULTIVERSE_CONTROLLER_INTERVAL_SECONDS=3600 \
            MULTIVERSE_EMAIL_INTERVAL_SECONDS=14400 \
            MULTIVERSE_SUPERVISOR_LOG_DIR="'"${arsenal_state}"'" \
        bash "'"${supervisor}"'" "'"${arsenal_config}"'"
    flock -n "'"${hc2_state}"'/supervisor.lock" \
        env MULTIVERSE_CLEAR_PENDING_ON_START=false \
            MULTIVERSE_CONTROLLER_INTERVAL_SECONDS=3600 \
            MULTIVERSE_EMAIL_INTERVAL_SECONDS=14400 \
            MULTIVERSE_SUPERVISOR_LOG_DIR="'"${hc2_state}"'" \
        bash "'"${supervisor}"'" "'"${hc2_config}"'"
    echo "Arsenal and HC2 resample-0 runs completed."
'

sleep 2
if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: the detached controller session did not remain running." >&2
    exit 1
fi

screen -ls | grep -F "${session_name}" || true
