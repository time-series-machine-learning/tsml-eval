#!/bin/bash
set -uo pipefail

# Stop queue feeders before touching the queue. This deliberately stops only
# Multiverse controller processes and screens; it does not cancel unrelated jobs
# owned by the user.
controller_pattern='[_]tsml_research_resources/(start|run)_multiverse[^[:space:]]*|[_]tsml_research_resources/multiverse_controller[.]py'

echo "Stopping Multiverse controller processes."
pkill -TERM -f "${controller_pattern}" 2>/dev/null || true

mapfile -t multiverse_sessions < <(
    screen -ls 2>/dev/null | awk '$1 ~ /[.]multiverse/ {print $1}'
)
for session in "${multiverse_sessions[@]}"; do
    echo "Closing controller screen: ${session}"
    screen -S "${session}" -X quit >/dev/null 2>&1 || true
done

for _ in {1..10}; do
    if ! pgrep -f "${controller_pattern}" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if pgrep -f "${controller_pattern}" >/dev/null 2>&1; then
    echo "Controller processes did not exit after SIGTERM; sending SIGKILL."
    pkill -KILL -f "${controller_pattern}" 2>/dev/null || true
fi

if pgrep -f "${controller_pattern}" >/dev/null 2>&1; then
    echo "ERROR: a Multiverse controller is still running." >&2
    exit 1
fi

echo "All Multiverse controllers stopped. Existing Slurm experiment jobs were left running."
squeue --user="${USER}"
