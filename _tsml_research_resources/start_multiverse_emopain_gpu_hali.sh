#!/bin/bash
# Run EmoPain for the six deep learners that lack it, on Hali GPUs.
#
# EmoPain was never a compute problem: aeon raised check_collection_variance before
# fit for every aeon classifier. aeon relegated that to a warning in #3598, which
# landed eight days after v1.5.0 was tagged, so this needs an aeon newer than any
# release. The check below refuses to queue against an aeon that still raises,
# because six jobs would otherwise fail in seconds exactly as they did before.
#
# Six jobs, roughly twenty minutes of GPU time in total.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_emopain_resample0_deep_gpu_hali.toml"
supervisor="${script_dir}/run_multiverse_controller.sh"
state_dir="/gpfs/home/${USER}/Results/Multiverse/.controller-emopain-resample0-deep-gpu"
session_name="multiverse-emopain-gpu"
python_executable="/gpfs/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python"
data_dir="/gpfs/home/${USER}/Data/Multiverse"
config_name=$(basename "$config_file")

for command_name in flock git pkill screen squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done
for required_file in "$config_file" "$supervisor" "$python_executable"; do
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
if [[ -n "$(git -C "$repo_dir" status --porcelain --untracked-files=normal)" ]]; then
    echo "ERROR: commit or discard repository changes before submission." >&2
    exit 1
fi
if [[ ! -d "${data_dir}/EmoPain" ]]; then
    echo "ERROR: EmoPain data not found: ${data_dir}/EmoPain" >&2
    exit 1
fi

# The whole reason this run exists. An aeon that still raises turns six jobs into
# six identical failures, and the controller would then record them as terminal.
# Tested by behaviour rather than by version or by reading the source: aeon 1.3.0
# has no such check at all and is perfectly able to run EmoPain, so a source or
# version test would refuse a working environment.
echo "Checking this aeon accepts low-variance input."
"$python_executable" - <<'PYTHON'
import numpy as np
import aeon
from aeon.classification import DummyClassifier

print(f"  aeon {aeon.__version__} at {aeon.__file__}")
# one channel with std ~1e-12 and a nonzero range, which is exactly what aeon's
# check_collection_variance flags and what 1733 of EmoPain's pairs look like
X = np.random.random((6, 2, 20))
X[0, 1, :] = 1.0 + np.arange(20) * 1e-12
y = np.array(["a", "b"] * 3)
try:
    DummyClassifier().fit(X, y)
except ValueError as error:
    if "too little variation" not in str(error):
        raise
    raise SystemExit(
        "  This aeon refuses low-variance input, so all six EmoPain jobs would
"
        "  fail in seconds. It needs f8db0d3e2 (#3598, 2026-07-07), which is
"
        "  newer than v1.5.0. Install aeon from main into tsml-eval-gpu first."
    ) from None
print("  ok: low-variance input is accepted")
PYTHON

# EmoPain loads as a 3D array, so a loader problem would show here rather than in
# six separate job logs.
"$python_executable" - "$data_dir" <<'PYTHON'
import sys
from aeon.datasets import load_classification

X, y = load_classification("EmoPain", split="train", extract_path=sys.argv[1])
if not hasattr(X, "ndim"):
    raise SystemExit(f"  EmoPain loaded as {type(X).__name__}, expected a 3D array")
print(f"  ok: EmoPain train {X.shape}, {len(set(y))} classes")
PYTHON

echo "Stopping an earlier EmoPain controller, if present."
pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
mapfile -t old_sessions < <(
    screen -ls | awk -v name="$session_name" '$1 ~ ("\." name "$") {print $1}'
)
for old_session in "${old_sessions[@]}"; do
    echo "Closing screen session: ${old_session}"
    screen -S "$old_session" -X quit >/dev/null 2>&1 || true
done

# Earlier attempts recorded EmoPain as failed for these six. That bookkeeping is
# about the old aeon, not about the work, so start the attempt count fresh while
# keeping every result already on disk.
if [[ -d "$state_dir" ]]; then
    archived_state="${state_dir}-previous-$(date +%Y%m%d-%H%M%S)"
    mv -- "$state_dir" "$archived_state"
    echo "Archived prior controller state: ${archived_state}"
fi
mkdir -p "$state_dir"

cd "$repo_dir"
echo "Checking the missing EmoPain work without submitting it."
"$python_executable" -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" --dry-run --no-email

echo "Starting detached EmoPain GPU controller: ${session_name}"
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
echo "EmoPain GPU controller started: ConvTran, DisjointCNN, PatchMTSC, TimesNet,"
echo "TimesURL and TS2Vec, one job each. Expect them back within the hour."
echo "log: ${state_dir}/supervisor.log"
echo
squeue -u "$USER" -p gpu
