#!/bin/bash
# Run the GPU deep learners on the Multiverse paper's 100 scored datasets, on Hali.
#
# Starts a detached supervisor that refills the GPU queue every hour, emails a
# progress report once a day, and stops, after one final email, once every result
# exists or every remaining job has used up its attempts. Results already on disk
# are skipped. Rerunning this is safe: it replaces the supervisor, keeps the queue
# and keeps the attempt counts. Pass --reset-state to start the counts afresh.
#
#   bash _tsml_research_resources/start_multiverse_paper100_deep_gpu_hali.sh
#
# Progress: screen -r multiverse-paper100-deep-gpu, or the supervisor.log in the
# state directory below. Stop it early with stop_multiverse_jobs.sh, or by creating
# a file named STOP in the state directory.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_paper100_resample0_deep_gpu_hali.toml"
dataset_list="${script_dir}/dataset_lists/MultivariateClassification100-MultiversePaperComplete.txt"
supervisor="${script_dir}/run_multiverse_controller.sh"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-paper100-resample0-deep-gpu"
session_name="multiverse-paper100-deep-gpu"
python_executable="/gpfs/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
data_dir="/gpfs/home/${USER}/Data/Multiverse"
config_name=$(basename "$config_file")

case "${1:-}" in
    "") reset_state=false ;;
    --reset-state) reset_state=true ;;
    *)
        echo "ERROR: unknown option: ${1}" >&2
        echo "Usage: bash $(basename "$0") [--reset-state]" >&2
        exit 1
        ;;
esac

for command_name in flock git pkill screen squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done
for required_file in "$config_file" "$dataset_list" "$supervisor" "$python_executable"; do
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
if [[ -n "$(git -C "$repo_dir" status --porcelain --untracked-files=normal)" ]]; then
    echo "ERROR: commit or discard repository changes before submission." >&2
    exit 1
fi

# A dataset with no data would fail every one of its thirteen jobs, and those
# failures would count as settled. Refuse to start until the data is in place.
missing_data=()
while IFS= read -r dataset || [[ -n "$dataset" ]]; do
    dataset=${dataset%$'\r'}
    [[ -z "$dataset" ]] && continue
    if [[ ! -d "${data_dir}/${dataset}" ]]; then
        missing_data+=("$dataset")
    fi
done < "$dataset_list"
if ((${#missing_data[@]})); then
    echo "ERROR: ${#missing_data[@]} of the 100 datasets are not in ${data_dir}:" >&2
    printf '  %s\n' "${missing_data[@]}" >&2
    exit 1
fi
echo "All 100 datasets are present in ${data_dir}."

# EmoPain is one of the 100, and an aeon that still refuses low-variance input fails
# all thirteen of its jobs in seconds. Tested by behaviour, as in the EmoPain
# starter, because aeon 1.3.0 has no such check and runs EmoPain perfectly well.
echo "Checking this aeon accepts low-variance input."
"$python_executable" - <<'PYTHON'
import sys

import numpy as np
import aeon
from aeon.classification import DummyClassifier

print(f"  aeon {aeon.__version__} at {aeon.__file__}")
X = np.random.random((6, 2, 20))
X[0, 1, :] = 1.0 + np.arange(20) * 1e-12
y = np.array(["a", "b"] * 3)
try:
    DummyClassifier().fit(X, y)
except ValueError as error:
    if "too little variation" not in str(error):
        raise
    print(
        "  This aeon refuses low-variance input, so every EmoPain job would fail.\n"
        "  Install aeon from main (#3598 or later) into tsml-eval-gpu first.",
        file=sys.stderr,
    )
    raise SystemExit(1) from None
print("  ok: low-variance input is accepted")
PYTHON

echo "Stopping an earlier paper-100 supervisor, if present."
pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
mapfile -t old_sessions < <(
    screen -ls | awk -v name="$session_name" '$1 ~ ("\\." name "$") {print $1}'
)
for old_session in "${old_sessions[@]}"; do
    echo "Closing screen session: ${old_session}"
    screen -S "$old_session" -X quit >/dev/null 2>&1 || true
done

if [[ "$reset_state" == true && -d "$state_dir" ]]; then
    archived_state="${state_dir}-previous-$(date +%Y%m%d-%H%M%S)"
    mv -- "$state_dir" "$archived_state"
    echo "Archived prior controller state: ${archived_state}"
fi
mkdir -p "$state_dir"
rm -f -- "${state_dir}/STOP"

cd "$repo_dir"
echo "Checking the missing deep-learning work without submitting it."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" --dry-run --no-email

# Hourly queue refills, a daily email, and a final email then exit once settled.
# Pending jobs from other runs are left alone.
echo "Starting detached supervisor: ${session_name}"
screen -dmS "$session_name" \
    flock -n "${state_dir}/supervisor.lock" \
    env PYTHON="$python_executable" \
        MULTIVERSE_CLEAR_PENDING_ON_START=false \
        MULTIVERSE_STOP_WHEN_COMPLETE=true \
        MULTIVERSE_CONTROLLER_INTERVAL_SECONDS=3600 \
        MULTIVERSE_EMAIL_INTERVAL_SECONDS=86400 \
        MULTIVERSE_LOG_DIR="$state_dir" \
    bash "$supervisor" "$config_file"

sleep 2
if ! screen -ls | grep -Fq ".${session_name}"; then
    echo "ERROR: supervisor session did not remain running." >&2
    echo "Another supervisor may already hold ${state_dir}/supervisor.lock." >&2
    exit 1
fi

echo
echo "Paper-100 deep-learning supervisor started. A first report is emailed now,"
echo "then one a day until every job has finished."
echo "log: ${state_dir}/supervisor.log"
echo
squeue -u "$USER" -p gpu
