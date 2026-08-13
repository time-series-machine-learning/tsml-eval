#!/bin/bash
set -u

# Stop queue feeders before cancelling jobs, otherwise the queue may refill.
pkill -TERM -f '[r]un_multiverse_controller.sh' 2>/dev/null || true
pkill -TERM -f '[_]tsml_research_resources/multiverse_controller.py' \
    2>/dev/null || true

for session in multiverse-controller multiverse-interval-32gb; do
    screen -S "${session}" -X quit 2>/dev/null || true
done

if pgrep -f \
    '[r]un_multiverse_controller.sh|[_]tsml_research_resources/multiverse_controller.py' \
    >/dev/null; then
    echo "ERROR: a Multiverse controller is still running; jobs were not cancelled."
    exit 1
fi

echo "Controller stopped. Cancelling all Slurm jobs owned by ${USER}."
scancel --user="${USER}"
squeue --user="${USER}"
