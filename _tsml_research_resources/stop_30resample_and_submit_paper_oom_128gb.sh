#!/bin/bash
# Stop the paper 30-resample controller and submit resample-0 OOM jobs at 128 GB.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-paper-30resamples-cpu"
stop_marker="${state_dir}/STOP"
session_name="multiverse-paper-cpu"

for command_name in flock git pkill python scancel screen squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

actual_branch=$(git -C "$repo_dir" branch --show-current)
if [[ "$actual_branch" != "ajb/hc2" ]]; then
    echo "ERROR: CPU jobs must run from ajb/hc2; found ${actual_branch}." >&2
    exit 1
fi

mkdir -p "$state_dir"
touch "$stop_marker"
echo "Created shared controller stop marker: ${stop_marker}"

# Stop a supervisor on this login node. The shared marker prevents a supervisor
# on another login node from submitting more work on its next cycle.
pkill -TERM -f \
    '[r]un_multiverse_controller.sh.*multiverse_paper_30resamples_cpu.toml' \
    || true
pkill -TERM -f \
    '[m]ultiverse_controller.py.*multiverse_paper_30resamples_cpu.toml' \
    || true

mapfile -t sessions < <(
    screen -ls | awk \
        '$1 ~ /\.multiverse-paper-cpu$/ {print $1}'
)
for session in "${sessions[@]}"; do
    echo "Closing local screen session: ${session}"
    screen -S "$session" -X quit >/dev/null 2>&1 || true
done

# Wait until any controller cycle already submitting jobs has exited. Keep the
# lock through cancellation and replacement submission to prevent a refill race.
exec 9>"${state_dir}/controller.lock"
if ! flock -w 60 9; then
    echo "ERROR: an active controller cycle did not stop within 60 seconds." >&2
    exit 1
fi

cd "$repo_dir"
python -u \
    "${script_dir}/submit_paper_resample0_oom_128gb.py"

echo
echo "Current compute queue:"
squeue -u "$USER" -p compute
