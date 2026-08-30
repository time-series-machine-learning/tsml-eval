#!/bin/bash
# Submit resample-0 FCN, MLP, and ResNet jobs on Hali GPUs.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_resample0_extra_deep_gpu.toml"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-resample0-extra-deep-gpu"
python_executable="/gpfs/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
dataset_list="/gpfs/home/${USER}/DataSetLists/MultiverseFullCPU.txt"

for command_name in git pkill scancel squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

for required_file in "$config_file" "$python_executable" "$dataset_list"; do
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
    echo "ERROR: GPU jobs must run from ajb/gpu; found ${branch:-DETACHED}." >&2
    exit 1
fi

echo "Stopping known LITETime-MV GPU queue feeders on this login node."
for old_config in \
    multiverse_full_resample0_litemv_gpu.toml \
    multiverse_full_30resamples_litemv_gpu.toml \
    multiverse_resample0_extra_deep_gpu.toml; do
    pkill -TERM -f "[r]un_multiverse_controller.sh.*${old_config}" || true
    pkill -TERM -f "[m]ultiverse_controller.py.*${old_config}" || true
done

mapfile -t pending_lite < <(
    squeue --noheader --array --user="$USER" --partition=gpu \
        --states=PENDING --format='%i %j' |
        awk '$2 ~ /^LITETime/ {print $1}'
)
if ((${#pending_lite[@]})); then
    echo "Cancelling ${#pending_lite[@]} pending LITETime GPU tasks."
    scancel "${pending_lite[@]}"
else
    echo "No pending LITETime GPU tasks to cancel."
fi

mkdir -p "$state_dir"
cd "$repo_dir"

echo "Checking missing FCN, MLP, and ResNet GPU work without submitting it."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" \
    --dry-run \
    --no-email

echo "Submitting one controller cycle for FCN, MLP, and ResNet."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" \
    --no-email

echo
echo "Current GPU queue:"
squeue -u "$USER" -p gpu
