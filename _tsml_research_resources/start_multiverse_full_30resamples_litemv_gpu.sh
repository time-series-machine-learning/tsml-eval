#!/bin/bash
# Start the 30-resample LITETime-MV pass over the 125 eligible Multiverse
# datasets on Hali GPUs. Run this from a Hali login node; no active Conda
# environment is required.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_full_30resamples_litemv_gpu.toml"
supervisor="${script_dir}/run_multiverse_controller.sh"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-full-30resamples-litemv-gpu"
session_name="multiverse-full-30resamples-litemv-gpu"
python_executable="/gpfs/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
dataset_list="/gpfs/home/${USER}/DataSetLists/MultiverseFullCPU.txt"

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

echo "Stopping earlier full-run LITETime-MV controllers, if present."
for old_config in \
    multiverse_full_resample0_litemv_gpu.toml \
    multiverse_full_30resamples_litemv_gpu.toml; do
    pkill -TERM -f "[r]un_multiverse_controller.sh.*${old_config}" || true
    pkill -TERM -f "[m]ultiverse_controller.py.*${old_config}" || true
done

for old_name in multiverse-full-litemv-gpu "$session_name"; do
    mapfile -t old_sessions < <(
        screen -ls | awk -v name="$old_name" '$1 ~ ("\\." name "$") {print $1}'
    )
    for old_session in "${old_sessions[@]}"; do
        echo "Closing screen session: ${old_session}"
        screen -S "$old_session" -X quit >/dev/null 2>&1 || true
    done
done

# Reset attempt bookkeeping without deleting results or cancelling Slurm jobs.
# Active jobs are reconciled before any new work is submitted.
if [[ -d "$state_dir" ]]; then
    archived_state="${state_dir}-previous-$(date +%Y%m%d-%H%M%S)"
    mv -- "$state_dir" "$archived_state"
    echo "Archived prior controller state: ${archived_state}"
fi
mkdir -p "$state_dir"

cd "$repo_dir"
echo "Checking the missing 30-resample work without submitting it."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" --dry-run --no-email

echo "Starting detached controller: ${session_name}"
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
echo "Thirty-resample LITETime-MV GPU controller started."
screen -ls | grep -F "$session_name" || true
echo
echo "Current GPU queue:"
squeue -u "$USER" -p gpu
