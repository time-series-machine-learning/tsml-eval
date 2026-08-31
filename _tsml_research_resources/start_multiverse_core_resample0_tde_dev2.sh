#!/bin/bash
# Start the resample-0 Multiverse-Core TDE_Dev2 controller on HALI.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_core_resample0_tde_dev2.toml"
supervisor="${script_dir}/run_multiverse_controller.sh"
session_name="multiverse-core-tde-dev2"
required_branch="ajb/hc2"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-core-resample0-tde-dev2"
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

for command_name in git python screen pkill squeue; do
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

if [[ ! -f "$config_file" || ! -f "$supervisor" ]]; then
    echo "ERROR: controller files were not found beside this script." >&2
    exit 1
fi

echo "Stopping an earlier TDE_Dev2 controller on this login node, if present."
pkill -TERM -f "[r]un_multiverse_controller.sh.*multiverse_core_resample0_tde_dev2.toml" || true
pkill -TERM -f "[m]ultiverse_controller.py.*multiverse_core_resample0_tde_dev2.toml" || true

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

# Starting this controller explicitly clears only its own stop marker.
rm -f -- "${state_dir}/STOP"

cd "$repo_dir"

echo "Checking the missing Multiverse-Core TDE_Dev2 work."
python -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" \
    --dry-run \
    --no-email

echo "Starting detached controller session: ${session_name}"
screen -dmS "$session_name" \
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
echo "TDE_Dev2 controller started successfully."
screen -ls | grep -F "$session_name" || true
echo
echo "Current compute queue:"
squeue -u "$USER" -p compute
