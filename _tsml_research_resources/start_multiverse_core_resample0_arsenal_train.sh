#!/bin/bash
# Start the resample-0 Multiverse-Core Arsenal train-file controller on HALI.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_core_resample0_arsenal_train.toml"
supervisor="${script_dir}/run_multiverse_controller.sh"
session_name="multiverse-core-arsenal-train"
required_branch="ajb/hc2"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-core-resample0-arsenal-train"
reset_state=false

case "${1:-}" in
    "") ;;
    --reset-state) reset_state=true ;;
    *)
        echo "ERROR: unknown option: ${1}" >&2
        echo "Usage: bash $(basename "$0") [--reset-state]" >&2
        exit 1
        ;;
esac

if [[ -n "${2:-}" ]]; then
    echo "ERROR: too many arguments." >&2
    echo "Usage: bash $(basename "$0") [--reset-state]" >&2
    exit 1
fi

for command_name in flock git pkill python screen squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

actual_branch=$(git -C "$repo_dir" branch --show-current)
if [[ "$actual_branch" != "$required_branch" ]]; then
    echo "ERROR: CPU jobs must run from ${required_branch}; found ${actual_branch:-DETACHED}." >&2
    exit 1
fi

for required_file in "$config_file" "$supervisor"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

echo "Stopping an earlier Arsenal train-file controller on this login node, if present."
pkill -TERM -f '[r]un_multiverse_controller.sh.*multiverse_core_resample0_arsenal_train.toml' || true
pkill -TERM -f '[m]ultiverse_controller.py.*multiverse_core_resample0_arsenal_train.toml' || true

mapfile -t old_sessions < <(
    screen -ls | awk -v screen_name="$session_name" \
        '$1 ~ ("\\." screen_name "$") {print $1}'
)
for old_session in "${old_sessions[@]}"; do
    echo "Closing screen session: ${old_session}"
    screen -S "$old_session" -X quit >/dev/null 2>&1 || true
done

if [[ "$reset_state" == true && -d "$state_dir" ]]; then
    archived_state="${state_dir}-previous-$(date +%Y%m%d-%H%M%S)"
    mv -- "$state_dir" "$archived_state"
    echo "Archived prior controller state: ${archived_state}"
fi

mkdir -p "$state_dir"
rm -f -- "${state_dir}/STOP"

cd "$repo_dir"

echo "Checking missing Multiverse-Core Arsenal train/test results."
python -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" \
    --dry-run \
    --no-email

echo "Starting detached controller session: ${session_name}"
screen -dmS "$session_name" \
    flock -n "${state_dir}/supervisor.lock" \
    env MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_CONTROLLER_INTERVAL_SECONDS=3600 \
        MULTIVERSE_EMAIL_INTERVAL_SECONDS=14400 \
        MULTIVERSE_SUPERVISOR_LOG_DIR="$state_dir" \
    bash "$supervisor" "$config_file"

sleep 2

if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: the detached controller session did not remain running." >&2
    exit 1
fi

echo
echo "Arsenal train-file controller started successfully."
screen -ls | grep -F "$session_name" || true
echo
echo "Current compute queue:"
squeue -u "$USER" -p compute
