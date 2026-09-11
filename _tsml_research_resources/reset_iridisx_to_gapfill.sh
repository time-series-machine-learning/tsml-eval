#!/bin/bash
# Clear the IridisX H200 queue of controllers that can no longer produce anything,
# then restart the deep-learner gap fill on the current checkout.
#
# Seven controllers were live while only one had useful work. Four could not
# produce a result at all:
#
#   timesnet      TimesNet is complete at 66 of 66, and EmoPain is excluded
#   disjointcnn   65 of 66; its only gap is EmoPain, which its config excludes
#   gapfill-keras LiteTIME only, and LiteTIME is withheld from the tables
#   xcm           65 of 66; its only gap is EmoPain, and this cluster's aeon
#                 still raises on low-variance input, so it retries and fails
#
# The fifth, the torch gap fill, is the one worth running but was started from
# the checkout before the gap fill was trimmed, so it is restarted rather than
# left. The two UEA completion controllers are left alone by default: they have
# real work, and whether it can proceed depends on the InsectWingbeat data being
# on this cluster. Pass --stop-uea to stop those as well.
#
# Usage:
#   bash reset_iridisx_to_gapfill.sh --dry-run   # show the plan, change nothing
#   bash reset_iridisx_to_gapfill.sh             # do it
#   bash reset_iridisx_to_gapfill.sh --stop-uea  # also stop the UEA completion pair

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
results_dir="/home/${USER}/Results/Multiverse"

dry_run=false
stop_uea=false
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=true ;;
        --stop-uea) stop_uea=true ;;
        *) echo "ERROR: unknown argument: ${argument}" >&2; exit 1 ;;
    esac
done

for command_name in git pgrep pkill scancel squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        echo "This script is meant to be run on an IridisX login node." >&2
        exit 1
    fi
done
if [[ ! -d "$results_dir" ]]; then
    echo "ERROR: results directory not found: ${results_dir}" >&2
    exit 1
fi

# controllers with nothing left to produce, and the reason for the record
obsolete=(
    "multiverse_core_resample0_timesnet_gpu_iridisx_i7_h200.toml:TimesNet is complete at 66 of 66"
    "multiverse_core_resample0_disjointcnn_gpu_iridisx_i7_h200.toml:only gap is EmoPain, which the config excludes"
    "multiverse_core_gapfill_deep_keras_gpu_iridisx_i7_h200.toml:LiteTIME is withheld from the tables"
    "multiverse_core_resample0_xcm_gpu_iridisx_i7_h200.toml:only gap is EmoPain, and this aeon still raises on it"
)
# the gap fill is stopped too, because it is running the pre-trim configuration
restart="multiverse_core_gapfill_deep_torch_gpu_iridisx_i7_h200.toml"
uea=(
    "uea_completion_resample0_deep_torch_gpu_iridisx_i7_h200.toml"
    "uea_completion_resample0_deep_keras_gpu_iridisx_i7_h200.toml"
)

run() {
    if [[ "$dry_run" == true ]]; then
        echo "    would run: $*"
    else
        "$@" || true
    fi
}

stop_controller() {
    local config="$1" reason="${2:-}"
    if ! pgrep -f "[m]ultiverse_controller.py.*${config}" >/dev/null \
        && ! pgrep -f "[r]un_multiverse_controller.sh.*${config}" >/dev/null; then
        echo "  already stopped: ${config}"
        return
    fi
    echo "  stopping ${config}${reason:+  (${reason})}"
    run pkill -TERM -f "[r]un_multiverse_controller.sh.*${config}"
    run pkill -TERM -f "[m]ultiverse_controller.py.*${config}"
}

echo "=== 1. controllers currently running ==="
pgrep -af "[m]ultiverse_controller.py" | sed 's/.*_tsml_research_resources\///; s/ .*//' \
    | sed 's/^/  /' || echo "  none"

echo
echo "=== 2. stopping controllers that cannot produce a result ==="
for entry in "${obsolete[@]}"; do
    stop_controller "${entry%%:*}" "${entry#*:}"
done
stop_controller "$restart" "running the pre-trim configuration; restarted below"
if [[ "$stop_uea" == true ]]; then
    for config in "${uea[@]}"; do
        stop_controller "$config" "--stop-uea given"
    done
else
    echo "  left alone: the two UEA completion controllers (pass --stop-uea to stop them)"
fi

sleep 2
# A killed supervisor orphans the sleep it waits in, and that orphan keeps the
# inherited descriptor on supervisor.lock open. flock -n would then fail and the
# next controller would exit silently while pgrep showed nothing running.
if command -v fuser >/dev/null 2>&1; then
    for lock in "${results_dir}"/.controller-*/supervisor.lock; do
        [[ -e "$lock" ]] || continue
        if fuser -s "$lock" 2>/dev/null \
            && ! pgrep -f "[m]ultiverse_controller.py" >/dev/null; then
            echo "  releasing orphaned lock: ${lock}"
            run fuser -k -TERM "$lock"
        fi
    done
fi

echo
echo "=== 3. the queue ==="
squeue -u "$USER" -p i7_h200 -o "%.12i %.30j %.2t %.11M" || true
pending=$(squeue -u "$USER" -p i7_h200 -t PD -h -o "%i" | wc -l)
running=$(squeue -u "$USER" -p i7_h200 -t R -h -o "%i" | wc -l)
echo "  ${running} running, ${pending} pending"
if [[ "$pending" -gt 0 ]]; then
    echo "  Cancelling the pending ones. They would die anyway: every batch script"
    echo "  re-checks the commit it was submitted at, and the pull below moves HEAD."
    echo "  Running jobs are left to finish."
    squeue -u "$USER" -p i7_h200 -t PD -h -o "%i" | while read -r job; do
        run scancel "$job"
    done
fi

echo
echo "=== 4. updating the checkout ==="
if [[ -n "$(git -C "$repo_dir" status --porcelain --untracked-files=normal)" ]]; then
    echo "  Repository is not clean:"
    git -C "$repo_dir" status --porcelain --untracked-files=normal | sed 's/^/    /'
    echo "  Stray *.keras files are checkpoints from killed Keras jobs and are safe"
    echo "  to delete; anything else needs your decision. Not pulling."
    if [[ -z "$(git -C "$repo_dir" status --porcelain --untracked-files=normal \
                | grep -v '\.keras$')" ]]; then
        run rm -f "${repo_dir}"/*.keras
        echo "  Removed the stray checkpoints; rerun to continue."
    fi
    [[ "$dry_run" == true ]] || exit 1
fi
run git -C "$repo_dir" pull --ff-only
echo "  HEAD is now $(git -C "$repo_dir" log --oneline -1)"

echo
echo "=== 5. restarting the gap fill ==="
echo "  Six experiments: ConvTran on Alzheimers, EigenWorms and PhotoStimulation,"
echo "  TS2Vec on Locust2022, Tiselac and USCActivity. ConvTran's three each close"
echo "  a dataset, taking the scored set from 56 to 59 of 64."
if [[ "$dry_run" == true ]]; then
    echo "    would run: bash ${script_dir}/start_multiverse_core_gapfill_gpu_iridisx.sh"
else
    bash "${script_dir}/start_multiverse_core_gapfill_gpu_iridisx.sh"
fi
