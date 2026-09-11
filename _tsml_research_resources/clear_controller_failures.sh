#!/bin/bash
# Clear a controller's terminal failure records so it will submit that work again.
#
# The controller refuses to retry two outcomes however many attempts remain:
#
#   Time limit                        never retried, at any memory tier
#   OOM at the highest memory tier    nowhere left to escalate
#
# That is right when the failure describes the work, and wrong when it describes
# something since fixed: a timeout caused by a bug in the estimator, or an OOM on
# a GPU that was shared with another job. In those cases the controller reports
# "ALL SETTLED WITH FAILURES" and submits nothing, and the records have to go
# before it will try again.
#
# Results are never touched. The controller skips any dataset that already has a
# testResample0.csv, so clearing a record cannot cause completed work to be redone.
# The state file is backed up first.
#
# Usage:
#   bash clear_controller_failures.sh --list
#   bash clear_controller_failures.sh <state-dir-name> [--dry-run]
#   bash clear_controller_failures.sh .controller-core-gapfill-deep-torch-i7-h200
#   bash clear_controller_failures.sh .controller-core-resample0-rankscl-i7-h200

set -euo pipefail

results_dir="/home/${USER}/Results/Multiverse"
dry_run=false
target=""

for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=true ;;
        --list)
            echo "Controller state directories under ${results_dir}:"
            for directory in "${results_dir}"/.controller-*/; do
                [[ -f "${directory}state.json" ]] || continue
                echo "  $(basename "$directory")"
            done
            exit 0
            ;;
        -*) echo "ERROR: unknown option: ${argument}" >&2; exit 1 ;;
        *) target="$argument" ;;
    esac
done

if [[ -z "$target" ]]; then
    echo "ERROR: name a controller state directory, or pass --list." >&2
    exit 1
fi

state_dir="${results_dir}/${target}"
state_file="${state_dir}/state.json"
if [[ ! -f "$state_file" ]]; then
    echo "ERROR: no state file at ${state_file}" >&2
    echo "Run with --list to see the available controllers." >&2
    exit 1
fi

config_name=$(basename "$target" | sed 's/^\.controller-//')
if pgrep -f "[m]ultiverse_controller.py.*${config_name}" >/dev/null 2>&1; then
    echo "ERROR: a controller for ${target} appears to be running." >&2
    echo "Stop it first, or it will rewrite the state underneath this." >&2
    exit 1
fi

# Any Python 3 will do; this only edits JSON. Prefer the experiment environment's
# interpreter when it is there, and accept a bare command name in PYTHON.
python_executable="${PYTHON:-/home/${USER}/.conda/envs/tsml-eval-gpu/bin/python}"
if ! command -v "$python_executable" >/dev/null 2>&1; then
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            python_executable="$candidate"
            break
        fi
    done
fi
if ! command -v "$python_executable" >/dev/null 2>&1; then
    echo "ERROR: no Python interpreter found; set PYTHON." >&2
    exit 1
fi

DRY_RUN="$dry_run" "$python_executable" - "$state_file" <<'PYTHON'
import json
import os
import shutil
import sys

path = sys.argv[1]
dry_run = os.environ.get("DRY_RUN") == "true"
state = json.load(open(path))
failures = state.get("failures", {})

terminal = {}
for key, record in failures.items():
    reason = record.get("last_reason")
    if reason == "Time limit":
        terminal[key] = "Time limit"
    elif reason == "OOM":
        # only terminal once the top tier has been tried; below that the
        # controller escalates on its own and the record is still useful
        events = record.get("events", [])
        if events and events[-1].get("next_memory_mb") is None:
            terminal[key] = f"OOM at {events[-1].get('memory_mb')} MB, top tier"

if not terminal:
    print("No terminal failure records; nothing to clear.")
    print(f"{len(failures)} failure record(s) remain, all still retryable.")
    raise SystemExit(0)

print(f"Terminal records in {os.path.basename(os.path.dirname(path))}:")
for key, why in sorted(terminal.items()):
    print(f"  {key}: {why}")

if dry_run:
    print(f"\n--dry-run: {len(terminal)} record(s) would be cleared, nothing written.")
    raise SystemExit(0)

backup = path + ".bak"
shutil.copy(path, backup)
for key in terminal:
    state["failures"].pop(key, None)
    state.get("attempts", {}).pop(key, None)
    # leave last_submitted_memory alone: it is what lets the controller resume at
    # the tier this task already needed rather than starting from the bottom
with open(path, "w") as handle:
    json.dump(state, handle, indent=2)
print(f"\nCleared {len(terminal)} record(s). Previous state saved to {backup}.")
print("Restart the controller and it will submit this work again.")
PYTHON
