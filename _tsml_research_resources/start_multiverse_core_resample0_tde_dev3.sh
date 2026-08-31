#!/bin/bash
# Start a resample-0 Multiverse-Core TDE Dev3 controller on HALI.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
supervisor="${script_dir}/run_multiverse_controller.sh"
required_branch="ajb/hc2"
variant="accuracy"
reset_state=false

for option in "$@"; do
    case "$option" in
        --uniform) variant="uniform" ;;
        --accuracy) variant="accuracy" ;;
        --reset-state) reset_state=true ;;
        *)
            echo "ERROR: unknown option: ${option}" >&2
            echo "Usage: bash $(basename "$0") [--accuracy|--uniform] [--reset-state]" >&2
            exit 1
            ;;
    esac
done

if [[ "$variant" == "uniform" ]]; then
    classifier="TDE_Dev3-Uniform"
    config_name="multiverse_core_resample0_tde_dev3_uniform.toml"
    session_name="multiverse-core-tde-dev3-uniform"
    state_name=".controller-core-resample0-tde-dev3-uniform"
else
    classifier="TDE_Dev3"
    config_name="multiverse_core_resample0_tde_dev3.toml"
    session_name="multiverse-core-tde-dev3"
    state_name=".controller-core-resample0-tde-dev3"
fi

config_file="${script_dir}/${config_name}"
state_dir="/gpfs/home/${USER}/Results/Multiverse/${state_name}"

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

echo "Stopping an earlier ${classifier} controller on this login node, if present."
pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true

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

rm -f -- "${state_dir}/STOP"
cd "$repo_dir"

echo "Checking the missing Multiverse-Core ${classifier} work."
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
echo "${classifier} controller started successfully."
screen -ls | grep -F "$session_name" || true
echo
echo "Current compute queue:"
squeue -u "$USER" -p compute
