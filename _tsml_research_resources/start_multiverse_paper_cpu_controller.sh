#!/bin/bash
# Start the 30-resample Multiverse paper CPU controller in a detached screen.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_paper_30resamples_cpu.toml"
supervisor="${script_dir}/run_multiverse_controller.sh"
session_name="multiverse-paper-cpu"
required_branch="ajb/hc2"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-paper-30resamples-cpu"

for command_name in git python screen pkill squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

actual_branch=$(git -C "$repo_dir" branch --show-current)
if [[ "$actual_branch" != "$required_branch" ]]; then
    echo "ERROR: CPU Multiverse jobs must run from ${required_branch}." >&2
    echo "Current branch: ${actual_branch:-DETACHED}" >&2
    exit 1
fi

if [[ ! -f "$config_file" || ! -f "$supervisor" ]]; then
    echo "ERROR: controller files were not found beside this script." >&2
    exit 1
fi

# Starting this controller is an explicit request to clear a previous one-off stop.
rm -f -- "${state_dir}/STOP"

echo "Stopping known CPU Multiverse queue feeders on this login node."
cpu_configs=(
    multiverse_controller.toml
    multiverse_interval_32gb.toml
    multiverse_core_resample0_non_deep.toml
    multiverse_core_resample0_litetime_mv.toml
    multiverse_full_resample0_cpu_32gb.toml
    multiverse_full_resample0_cpu_completion.toml
    multiverse_resample0_extra_cpu.toml
    multiverse_paper_30resamples_cpu.toml
)
for config_name in "${cpu_configs[@]}"; do
    pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
    pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
done

screen_names=(multiverse-controller multiverse-interval-32gb "$session_name")
for name in "${screen_names[@]}"; do
    mapfile -t old_sessions < <(
        screen -ls | awk -v screen_name="$name" \
            '$1 ~ ("\\." screen_name "$") {print $1}'
    )
    for old_session in "${old_sessions[@]}"; do
        echo "Closing screen session: ${old_session}"
        screen -S "$old_session" -X quit >/dev/null 2>&1 || true
    done
done

echo "Checking the missing CPU work."
python -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" \
    --dry-run \
    --no-email

echo "Starting detached controller session: ${session_name}"
screen -dmS "$session_name" \
    env MULTIVERSE_CLEAR_PENDING_ON_START=true \
    MULTIVERSE_CLEAR_PENDING_PARTITION=compute \
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
echo "Current compute queue:"
squeue -u "$USER" -p compute
