#!/bin/bash
# Stop the Multiverse controller and cancel unwanted running Slurm array tasks.

set -euo pipefail

if ! command -v squeue >/dev/null 2>&1; then
    echo "ERROR: squeue was not found. Run this script on a Slurm login node." >&2
    exit 1
fi

if ! command -v scancel >/dev/null 2>&1; then
    echo "ERROR: scancel was not found. Run this script on a Slurm login node." >&2
    exit 1
fi

# Stop queue feeders before cancelling, otherwise they may replace cancelled jobs.
screen -S multiverse-controller -X quit >/dev/null 2>&1 || true
pkill -f '[r]un_multiverse_controller.sh' >/dev/null 2>&1 || true
pkill -f '[_]tsml_research_resources/multiverse_controller.py' \
    >/dev/null 2>&1 || true

keep_ids=()
cancel_ids=()

while IFS='|' read -r raw_id raw_name; do
    # Slurm pads formatted fields with spaces.
    job_id=${raw_id//[[:space:]]/}
    job_name=${raw_name//[[:space:]]/}

    if [[ "${job_id}" == *_1 &&
        ( "${job_name}" == *AustraliaRainfall* ||
            "${job_name}" == *BIDMC32* ||
            "${job_name}" == H-Inception* ) ]]; then
        keep_ids+=("${job_id}")
    else
        cancel_ids+=("${job_id}")
    fi
done < <(
    squeue --noheader --array --user="${USER}" \
        --states=RUNNING --format='%i|%200j'
)

echo "Retaining ${#keep_ids[@]} running jobs:"
if ((${#keep_ids[@]})); then
    printf '  %s\n' "${keep_ids[@]}"
else
    echo "  none"
fi

echo "Cancelling ${#cancel_ids[@]} running jobs:"
if ((${#cancel_ids[@]})); then
    printf '  %s\n' "${cancel_ids[@]}"
    scancel "${cancel_ids[@]}"
else
    echo "  none"
fi

echo "Remaining running and pending jobs:"
squeue --array --user="${USER}" --states=RUNNING,PENDING \
    --format='%.20i %.10T %.50j'
