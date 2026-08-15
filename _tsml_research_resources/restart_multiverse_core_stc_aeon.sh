#!/bin/bash
# Replace Core STC resample 0 with Aeon's STC, retaining recoverable backups.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_core_stc_aeon.toml"
supervisor="${script_dir}/run_multiverse_controller.sh"
core_list="/gpfs/home/${USER}/DataSetLists/MultiverseCore.txt"
results_root="/gpfs/home/${USER}/Results/Multiverse"
state_dir="${results_root}/.controller-core-stc-aeon"
session_name="multiverse-core-stc-aeon"
timestamp=$(date '+%Y%m%d-%H%M%S')
backup_root="${results_root}/.replaced-stc-early-abandon/${timestamp}"

for command_name in flock git mktemp pkill python scancel screen squeue; do
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

for required_file in "$config_file" "$supervisor" "$core_list"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 1
    fi
done

declare -A core_stc_jobs=()
while IFS= read -r dataset || [[ -n "$dataset" ]]; do
    dataset=${dataset//$'\r'/}
    if [[ -n "$dataset" && "$dataset" != \#* ]]; then
        core_stc_jobs["STC_${dataset}"]=1
    fi
done < "$core_list"

# Stop only an earlier copy of this replacement controller.
pkill -TERM -f \
    '[r]un_multiverse_controller.sh.*multiverse_core_stc_aeon.toml' || true
pkill -TERM -f \
    '[m]ultiverse_controller.py.*multiverse_core_stc_aeon.toml' || true
mapfile -t old_sessions < <(
    screen -ls | awk -v screen_name="$session_name" \
        '$1 ~ ("\\." screen_name "$") {print $1}'
)
for session in "${old_sessions[@]}"; do
    screen -S "$session" -X quit >/dev/null 2>&1 || true
done

# Cancel old-implementation STC work for Core only. HC2 and other classifiers
# are untouched, as are STC jobs for datasets outside the Core list.
cancel_ids=()
while IFS='|' read -r job_id job_name; do
    job_id=${job_id//[[:space:]]/}
    job_name=${job_name#"${job_name%%[![:space:]]*}"}
    job_name=${job_name%"${job_name##*[![:space:]]}"}
    if [[ -n "${core_stc_jobs[$job_name]:-}" ]]; then
        cancel_ids+=("$job_id")
    fi
done < <(
    squeue --noheader --array --user="$USER" --partition=compute \
        --states=RUNNING,PENDING --format='%i|%200j'
)
if ((${#cancel_ids[@]})); then
    echo "Cancelling ${#cancel_ids[@]} active Core STC tasks from the old alias."
    scancel "${cancel_ids[@]}"
    for ((attempt = 1; attempt <= 12; attempt++)); do
        remaining=0
        while IFS= read -r job_name; do
            job_name=${job_name#"${job_name%%[![:space:]]*}"}
            job_name=${job_name%"${job_name##*[![:space:]]}"}
            if [[ -n "${core_stc_jobs[$job_name]:-}" ]]; then
                ((remaining += 1))
            fi
        done < <(
            squeue --noheader --array --user="$USER" --partition=compute \
                --states=RUNNING,PENDING --format='%200j'
        )
        if ((remaining == 0)); then
            break
        fi
        if ((attempt == 12)); then
            echo "ERROR: ${remaining} cancelled Core STC tasks remain active." >&2
            echo "No result files have been moved; try again shortly." >&2
            exit 1
        fi
        sleep 5
    done
else
    echo "No active Core STC tasks need cancelling."
fi

# Preserve only Core resample-0 result files. Other datasets and resamples stay
# in place. The new run will write back to the standard ShapeletBased/STC path.
archived=0
while IFS= read -r dataset || [[ -n "$dataset" ]]; do
    dataset=${dataset//$'\r'/}
    if [[ -z "$dataset" || "$dataset" == \#* ]]; then
        continue
    fi
    prediction_dir="${results_root}/ShapeletBased/STC/Predictions/${dataset}"
    for split in test train; do
        source_file="${prediction_dir}/${split}Resample0.csv"
        if [[ -f "$source_file" ]]; then
            destination_dir="${backup_root}/ShapeletBased/STC/Predictions/${dataset}"
            mkdir -p "$destination_dir"
            mv -- "$source_file" "$destination_dir/"
            ((archived += 1))
        fi
    done
done < "$core_list"

if [[ -f "${state_dir}/state.json" ]]; then
    mkdir -p "${backup_root}/controller-state"
    mv -- "${state_dir}/state.json" "${backup_root}/controller-state/"
fi
mkdir -p "$state_dir"
rm -f -- "${state_dir}/STOP"

echo "Archived ${archived} old Core STC resample-0 files under:"
echo "  ${backup_root}"

cd "$repo_dir"
echo "Checking the replacement work without submitting it."
python -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" --dry-run --no-email

echo "Starting detached Aeon STC controller: ${session_name}"
screen -dmS "$session_name" \
    flock -n "${state_dir}/supervisor.lock" \
    env MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_SUPERVISOR_LOG_DIR="$state_dir" \
    bash "$supervisor" "$config_file"

sleep 2
if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: controller session did not remain running." >&2
    exit 1
fi

echo "Aeon STC Core rerun started at 32 GB with train and test files enabled."
screen -ls | grep -F "$session_name" || true
