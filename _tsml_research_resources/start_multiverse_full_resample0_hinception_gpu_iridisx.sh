#!/bin/bash
# Build the locally available IridisX dataset list for the full multiverse run, split
# it across IridisX's two GPU queues, and start one controller per queue.
#
# a100 (2 concurrent GPUs, no account/QoS) and swarm_a100 (8 concurrent GPUs via the
# ecs account) each get their own controller reading a disjoint half of the dataset
# list. squeue is queried per-partition (see _query_slurm in multiverse_controller.py),
# so a task queued on one partition is invisible to the other controller -- splitting
# the list up front is what keeps the two controllers from racing to submit the same
# task twice. Both controllers still sort their own half smallest-first independently
# (small_datasets_first re-measures on-disk size, it does not just follow file order),
# so alternating lines between the two halves keeps both queues busy with a size mix
# rather than one queue front-loaded with every large problem.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_a100="${script_dir}/multiverse_full_resample0_hinception_gpu_iridisx_a100.toml"
config_swarm="${script_dir}/multiverse_full_resample0_hinception_gpu_iridisx_swarm.toml"
source_list="${script_dir}/dataset_lists/Multivariate133Classification-MultiverseClean.txt"
supervisor="${script_dir}/run_multiverse_controller.sh"
data_dir="/home/${USER}/Data/Multiverse"
list_dir="/home/${USER}/DataSetLists"
available_list="${list_dir}/MultiverseAvailableGPU-IridisX.txt"
missing_list="${list_dir}/MultiverseUnavailableGPU-IridisX.txt"
excluded_list="${list_dir}/MultiverseExcludedGPU-IridisX.txt"
a100_list="${list_dir}/MultiverseAvailableGPU-IridisX-A100.txt"
swarm_list="${list_dir}/MultiverseAvailableGPU-IridisX-Swarm.txt"
state_dir_a100="/home/${USER}/Results/Multiverse/.controller-full-resample0-hinception-gpu-iridisx-a100"
state_dir_swarm="/home/${USER}/Results/Multiverse/.controller-full-resample0-hinception-gpu-iridisx-swarm"
session_a100="multiverse-full-hinception-iridisx-a100"
session_swarm="multiverse-full-hinception-iridisx-swarm"
python_executable="/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
required_branch="ajb/gpu"

for command_name in flock git mktemp pkill screen squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

for required_file in "$config_a100" "$config_swarm" "$source_list" "$supervisor" \
    "$python_executable"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -x "$python_executable" ]]; then
    echo "ERROR: GPU-environment Python is not executable: ${python_executable}" >&2
    exit 1
fi

actual_branch=$(git -C "$repo_dir" branch --show-current)
if [[ "$actual_branch" != "$required_branch" ]]; then
    echo "ERROR: GPU jobs must run from ${required_branch}; found ${actual_branch}." >&2
    exit 1
fi

if [[ ! -d "$data_dir" ]]; then
    echo "ERROR: Multiverse data directory not found: ${data_dir}" >&2
    exit 1
fi

mkdir -p "$list_dir" "$state_dir_a100" "$state_dir_swarm"
available_tmp=$(mktemp)
missing_tmp=$(mktemp)
excluded_tmp=$(mktemp)
cleanup() {
    rm -f -- "$available_tmp" "$missing_tmp" "$excluded_tmp"
}
trap cleanup EXIT

archive_count=0
available_count=0
excluded_count=0
while IFS= read -r clean_dataset || [[ -n "$clean_dataset" ]]; do
    clean_dataset=${clean_dataset//$'\r'/}
    if [[ -z "$clean_dataset" || "$clean_dataset" == \#* ]]; then
        continue
    fi

    # Results retain the archive's base problem name. The clean-list suffix
    # identifies the exact equal-length/no-missing variant that must be present.
    dataset=${clean_dataset%_eq_nmv}
    dataset=${dataset%_eq}
    dataset=${dataset%_nmv}

    if [[ "$dataset" == DREAM* || "$dataset" == S2Agri-* ]]; then
        printf '%s\n' "$dataset" >> "$excluded_tmp"
        ((excluded_count += 1))
        continue
    fi
    ((archive_count += 1))

    # Aeon's downloaded-dataset check requires the base pair even when a clean
    # variant is selected. Requiring both prevents load_classification from
    # downloading an absent record and guarantees the requested clean variant.
    if [[ -f "${data_dir}/${dataset}/${dataset}_TRAIN.ts" && \
          -f "${data_dir}/${dataset}/${dataset}_TEST.ts" && \
          -f "${data_dir}/${dataset}/${clean_dataset}_TRAIN.ts" && \
          -f "${data_dir}/${dataset}/${clean_dataset}_TEST.ts" ]]; then
        printf '%s\n' "$dataset" >> "$available_tmp"
        ((available_count += 1))
    else
        printf '%s\n' "$dataset" >> "$missing_tmp"
    fi
done < "$source_list"

missing_count=$((archive_count - available_count))
echo "Eligible archive problems: ${archive_count}"
echo "Explicitly excluded:       ${excluded_count}"
echo "Locally available:         ${available_count}"
echo "Unavailable (not queued):  ${missing_count}"
if ((excluded_count)); then
    echo "Excluded datasets:"
    sed 's/^/  /' "$excluded_tmp"
fi
if ((missing_count)); then
    echo "Unavailable datasets:"
    sed 's/^/  /' "$missing_tmp"
fi

if ((available_count == 0)); then
    echo "ERROR: no locally complete eligible datasets were found; refusing to submit." >&2
    exit 1
fi

mv -- "$available_tmp" "$available_list"
mv -- "$missing_tmp" "$missing_list"
mv -- "$excluded_tmp" "$excluded_list"
trap - EXIT

# Alternate lines between the two queues so both get a size mix rather than one
# queue receiving every dataset that happens to sort first.
awk 'NR % 2 == 1' "$available_list" > "$a100_list"
awk 'NR % 2 == 0' "$available_list" > "$swarm_list"
echo "a100 queue:       $(wc -l < "$a100_list") datasets -> ${a100_list}"
echo "swarm_a100 queue: $(wc -l < "$swarm_list") datasets -> ${swarm_list}"

# Replace only earlier copies of these two controllers.
for config in "$config_a100" "$config_swarm"; do
    config_name=$(basename "$config")
    pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
    pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
done
for session_name in "$session_a100" "$session_swarm"; do
    mapfile -t old_sessions < <(
        screen -ls | awk -v name="$session_name" \
            '$1 ~ ("\\." name "$") {print $1}'
    )
    for old_session in "${old_sessions[@]}"; do
        echo "Closing old screen session: ${old_session}"
        screen -S "$old_session" -X quit >/dev/null 2>&1 || true
    done
done

cd "$repo_dir"
echo "Checking a100 work without submitting it."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_a100" --dry-run --no-email
echo "Checking swarm_a100 work without submitting it."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_swarm" --dry-run --no-email

echo "Starting detached a100 controller: ${session_a100}"
screen -dmS "$session_a100" \
    flock -n "${state_dir_a100}/supervisor.lock" \
    env PYTHON="$python_executable" \
        MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_LOG_DIR="$state_dir_a100" \
    bash "$supervisor" "$config_a100"

echo "Starting detached swarm_a100 controller: ${session_swarm}"
screen -dmS "$session_swarm" \
    flock -n "${state_dir_swarm}/supervisor.lock" \
    env PYTHON="$python_executable" \
        MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_LOG_DIR="$state_dir_swarm" \
    bash "$supervisor" "$config_swarm"

sleep 2
for session_name in "$session_a100" "$session_swarm"; do
    if ! screen -ls | grep -Fq ".${session_name}"; then
        echo "ERROR: controller session did not remain running: ${session_name}" >&2
        echo "Another supervisor may already hold its lock file." >&2
        exit 1
    fi
done

echo
echo "Both controllers started."
echo "Available list:   ${available_list}"
echo "Unavailable list: ${missing_list}"
echo "Excluded list:    ${excluded_list}"
screen -ls | grep -E "${session_a100}|${session_swarm}" || true
echo
echo "Current a100 queue:"
squeue -u "$USER" -p a100
echo
echo "Current swarm_a100 queue:"
squeue -u "$USER" -p swarm_a100
