#!/bin/bash
# Restart the Hali MultiverseCore H-InceptionTime GPU controller after
# previously submitted tasks were cancelled. Existing results and Slurm jobs
# are retained; only this controller's persistent bookkeeping is archived.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_file="${script_dir}/multiverse_core_resample0_hinception_gpu.toml"
supervisor="${script_dir}/run_multiverse_controller.sh"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-core-resample0-hinception-gpu"
session_name="multiverse-hinception-gpu"
config_name=$(basename "$config_file")

for command_name in python screen pgrep pkill squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

if [[ ! -f "$config_file" || ! -f "$supervisor" ]]; then
    echo "ERROR: controller files were not found beside this script." >&2
    exit 1
fi

echo "Stopping the existing H-InceptionTime GPU controller, if present."
pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
screen -S "$session_name" -X quit >/dev/null 2>&1 || true

if [[ -d "$state_dir" ]]; then
    archived_state="${state_dir}-cancelled-$(date +%Y%m%d-%H%M%S)"
    mv -- "$state_dir" "$archived_state"
    echo "Archived stale controller state: ${archived_state}"
else
    echo "No stale controller state directory was present."
fi

echo "Checking the work that will be submitted."
python -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" \
    --dry-run \
    --no-email

echo "Starting a fresh detached controller session: ${session_name}"
screen -dmS "$session_name" \
    env MULTIVERSE_CLEAR_PENDING_ON_START=false \
    bash "$supervisor" "$config_file"

sleep 2

if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: the detached controller session did not remain running." >&2
    exit 1
fi

echo
echo "Controller started successfully."
screen -ls | grep -F "$session_name" || true
echo
echo "Current GPU queue:"
squeue -u "$USER" -p gpu
