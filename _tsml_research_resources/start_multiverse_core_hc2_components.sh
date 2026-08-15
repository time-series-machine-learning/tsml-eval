#!/bin/bash
# Run exact HC2 components for all missing Core HC2 results and build HC2.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_core_hc2_components.toml"
supervisor="${script_dir}/run_multiverse_controller.sh"
post_cycle_python="${script_dir}/submit_hc2_core_builds.py"
core_list="/gpfs/home/${USER}/DataSetLists/MultiverseCore.txt"
target_list="/gpfs/home/${USER}/DataSetLists/MultiverseCoreMissingHC2.txt"
active_list="/gpfs/home/${USER}/DataSetLists/MultiverseCoreActiveHC2.txt"
results_root="/gpfs/home/${USER}/Results/Multiverse"
state_dir="${results_root}/.controller-hc2-core-components"
session_name="multiverse-hc2-components"

for command_name in flock git mktemp pkill python screen squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

branch=$(git -C "$repo_dir" branch --show-current)
if [[ "$branch" != "ajb/hc2" ]]; then
    echo "ERROR: CPU jobs must run from ajb/hc2; found ${branch}." >&2
    exit 1
fi

for required_file in \
    "$config_file" "$supervisor" "$post_cycle_python" "$core_list"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

mkdir -p "$(dirname -- "$target_list")" "$state_dir"
target_tmp=$(mktemp)
active_tmp=$(mktemp)
cleanup() {
    rm -f -- "$target_tmp" "$active_tmp"
}
trap cleanup EXIT

declare -A active_jobs=()
while IFS= read -r job_name; do
    if [[ -n "$job_name" ]]; then
        active_jobs["$job_name"]=1
    fi
done < <(
    squeue --noheader --array --user="$USER" --partition=compute \
        --states=RUNNING,PENDING --format='%200j' |
        sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d'
)

complete_count=0
target_count=0
active_count=0
while IFS= read -r dataset || [[ -n "$dataset" ]]; do
    dataset=${dataset//$'\r'/}
    if [[ -z "$dataset" || "$dataset" == \#* ]]; then
        continue
    fi
    result_file="${results_root}/Hybrid/HC2/Predictions/${dataset}/testResample0.csv"
    if [[ -s "$result_file" ]]; then
        ((complete_count += 1))
    else
        printf '%s\n' "$dataset" >> "$target_tmp"
        ((target_count += 1))
        if [[ -n "${active_jobs[HC2_${dataset}]:-}" ]]; then
            printf '%s\n' "$dataset" >> "$active_tmp"
            ((active_count += 1))
        fi
    fi
done < "$core_list"

mv -- "$target_tmp" "$target_list"
mv -- "$active_tmp" "$active_list"
trap - EXIT

echo "Core HC2 results complete:      ${complete_count}"
echo "Missing and selected:           ${target_count}"
echo "Also running directly as HC2:  ${active_count}"
if ((active_count)); then
    echo "Component backups will also run for these active HC2 datasets:"
    sed 's/^/  /' "$active_list"
fi

if ((target_count == 0)); then
    echo "No missing Core HC2 work is currently available."
    exit 0
fi

rm -f -- "${state_dir}/STOP"
pkill -TERM -f \
    '[r]un_multiverse_controller.sh.*multiverse_core_hc2_components.toml' || true
pkill -TERM -f \
    '[m]ultiverse_controller.py.*multiverse_core_hc2_components.toml' || true

mapfile -t old_sessions < <(
    screen -ls | awk -v screen_name="$session_name" \
        '$1 ~ ("\\." screen_name "$") {print $1}'
)
for session in "${old_sessions[@]}"; do
    screen -S "$session" -X quit >/dev/null 2>&1 || true
done

cd "$repo_dir"
echo "Checking HC2 component work without submitting it."
python -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" --dry-run --no-email

echo "Starting detached HC2 component controller: ${session_name}"
screen -dmS "$session_name" \
    flock -n "${state_dir}/supervisor.lock" \
    env MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_SUPERVISOR_LOG_DIR="$state_dir" \
        MULTIVERSE_POST_CYCLE_PYTHON="$post_cycle_python" \
    bash "$supervisor" "$config_file"

sleep 2
if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: controller session did not remain running." >&2
    exit 1
fi

echo "Controller started. Components will begin at 32 GB."
echo "Ready component sets will automatically be combined into Hybrid/HC2 results."
screen -ls | grep -F "$session_name" || true
