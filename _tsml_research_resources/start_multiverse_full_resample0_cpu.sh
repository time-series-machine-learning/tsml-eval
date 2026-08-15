#!/bin/bash
# Build the locally available CPU list and start its controller.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_full_resample0_cpu_32gb.toml"
source_list="${script_dir}/dataset_lists/Multivariate133Classification-MultiverseClean.txt"
supervisor="${script_dir}/run_multiverse_controller.sh"
data_dir="/gpfs/home/${USER}/Data/Multiverse"
list_dir="/gpfs/home/${USER}/DataSetLists"
available_list="${list_dir}/MultiverseAvailableCPU.txt"
missing_list="${list_dir}/MultiverseUnavailableCPU.txt"
excluded_list="${list_dir}/MultiverseExcludedCPU.txt"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-full-resample0-cpu-32gb"
session_name="multiverse-full-resample0-cpu"
required_branch="ajb/hc2"

for command_name in flock git mktemp pkill python screen squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

actual_branch=$(git -C "$repo_dir" branch --show-current)
if [[ "$actual_branch" != "$required_branch" ]]; then
    echo "ERROR: CPU jobs must run from ${required_branch}; found ${actual_branch}." >&2
    exit 1
fi

for required_file in "$config_file" "$source_list" "$supervisor"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -d "$data_dir" ]]; then
    echo "ERROR: Multiverse data directory not found: ${data_dir}" >&2
    exit 1
fi

mkdir -p "$list_dir" "$state_dir"
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

rm -f -- "${state_dir}/STOP"

# Replace only another copy of this controller on the current login node.
pkill -TERM -f \
    '[r]un_multiverse_controller.sh.*multiverse_full_resample0_cpu_32gb.toml' \
    || true
pkill -TERM -f \
    '[m]ultiverse_controller.py.*multiverse_full_resample0_cpu_32gb.toml' \
    || true

mapfile -t old_sessions < <(
    screen -ls | awk -v screen_name="$session_name" \
        '$1 ~ ("\\." screen_name "$") {print $1}'
)
for session in "${old_sessions[@]}"; do
    echo "Closing old screen session: ${session}"
    screen -S "$session" -X quit >/dev/null 2>&1 || true
done

cd "$repo_dir"
echo "Checking the locally available work without submitting it."
python -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" \
    --dry-run \
    --no-email

echo "Starting detached CPU controller: ${session_name}"
screen -dmS "$session_name" \
    flock -n "${state_dir}/supervisor.lock" \
    env MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_SUPERVISOR_LOG_DIR="$state_dir" \
    bash "$supervisor" "$config_file"

sleep 2
if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: controller session did not remain running." >&2
    echo "Another supervisor may already hold ${state_dir}/supervisor.lock." >&2
    exit 1
fi

echo
echo "Controller started successfully."
echo "Available list:   ${available_list}"
echo "Unavailable list: ${missing_list}"
echo "Excluded list:    ${excluded_list}"
screen -ls | grep -F "$session_name" || true
echo
echo "Current compute queue:"
squeue -u "$USER" -p compute
