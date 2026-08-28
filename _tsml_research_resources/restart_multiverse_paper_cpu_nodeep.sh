#!/bin/bash
# Cancel short-running CPU jobs and restart the dataset-first paper pass.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
launcher="${script_dir}/start_multiverse_paper_cpu_controller.sh"
required_branch="ajb/hc2"
minimum_age_seconds=$((24 * 60 * 60))

for command_name in git scancel squeue; do
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

elapsed_seconds() {
    local elapsed=$1
    local days=0
    local hours minutes seconds
    if [[ "$elapsed" == *-* ]]; then
        days=${elapsed%%-*}
        elapsed=${elapsed#*-}
    fi
    IFS=: read -r hours minutes seconds <<< "$elapsed"
    if [[ -z "${seconds:-}" ]]; then
        seconds=$minutes
        minutes=$hours
        hours=0
    fi
    echo $((days * 86400 + hours * 3600 + minutes * 60 + seconds))
}

mapfile -t short_jobs < <(
    squeue --noheader --user="$USER" --partition=compute --states=RUNNING \
        --format='%i|%M' |
        while IFS='|' read -r job_id elapsed; do
            job_id=${job_id//[[:space:]]/}
            elapsed=${elapsed//[[:space:]]/}
            [[ -n "$job_id" && -n "$elapsed" ]] || continue
            if (( $(elapsed_seconds "$elapsed") < minimum_age_seconds )); then
                printf '%s|%s\n' "$job_id" "$elapsed"
            fi
        done
)

if ((${#short_jobs[@]})); then
    echo "Cancelling ${#short_jobs[@]} running compute tasks younger than 24 hours:"
    for entry in "${short_jobs[@]}"; do
        IFS='|' read -r job_id elapsed <<< "$entry"
        echo "  ${job_id} (${elapsed})"
        scancel "$job_id"
    done
else
    echo "No running compute tasks younger than 24 hours."
fi

echo "Restarting the CPU paper controller. Pending compute jobs will also be cleared."
exec bash "$launcher"
