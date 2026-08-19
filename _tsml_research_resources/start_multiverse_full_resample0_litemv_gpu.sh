#!/bin/bash
# Start the full-run LITETime-MV resample-0 jobs on Hali GPUs, covering the
# same dataset list as the CPU full pass (MultiverseAvailableCPU.txt). Assumes
# that list has already been built by start_multiverse_full_resample0_cpu.sh.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_full_resample0_litemv_gpu.toml"
supervisor="${script_dir}/run_multiverse_controller.sh"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-full-resample0-litemv-gpu"
session_name="multiverse-full-litemv-gpu"
python_executable="/gpfs/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
config_name=$(basename "$config_file")
dataset_list="/gpfs/home/${USER}/DataSetLists/MultiverseAvailableCPU.txt"

for command_name in flock git pkill screen squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

for required_file in "$config_file" "$supervisor" "$python_executable" "$dataset_list"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -x "$python_executable" ]]; then
    echo "ERROR: GPU-environment Python is not executable: ${python_executable}" >&2
    exit 1
fi

branch=$(git -C "$repo_dir" branch --show-current)
if [[ "$branch" != "ajb/gpu" ]]; then
    echo "ERROR: GPU jobs must run from ajb/gpu; found ${branch}." >&2
    exit 1
fi

echo "Stopping an earlier full-run LITETime-MV GPU controller, if present."
pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
mapfile -t old_sessions < <(
    screen -ls | awk -v name="$session_name" \
        '$1 ~ ("\\." name "$") {print $1}'
)
for old_session in "${old_sessions[@]}"; do
    echo "Closing screen session: ${old_session}"
    screen -S "$old_session" -X quit >/dev/null 2>&1 || true
done

# Start with fresh attempt bookkeeping while retaining every result and every
# already submitted Slurm task. The controller reconciles both before submitting.
if [[ -d "$state_dir" ]]; then
    archived_state="${state_dir}-previous-$(date +%Y%m%d-%H%M%S)"
    mv -- "$state_dir" "$archived_state"
    echo "Archived prior controller state: ${archived_state}"
fi
mkdir -p "$state_dir"

cd "$repo_dir"
echo "Checking missing LITETime-MV work without submitting it."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" --dry-run --no-email

echo "Starting detached full-run LITETime-MV GPU controller: ${session_name}"
screen -dmS "$session_name" \
    flock -n "${state_dir}/supervisor.lock" \
    env PYTHON="$python_executable" \
        MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_LOG_DIR="$state_dir" \
    bash "$supervisor" "$config_file"

sleep 2
if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: controller session did not remain running." >&2
    echo "Another supervisor may already hold ${state_dir}/supervisor.lock." >&2
    exit 1
fi

echo
echo "Full-run LITETime-MV GPU controller started."
screen -ls | grep -F "$session_name" || true
echo
echo "Current GPU queue:"
squeue -u "$USER" -p gpu
