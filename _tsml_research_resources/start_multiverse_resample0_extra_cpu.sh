#!/bin/bash
# Submit the resample-0 pass for extra non-deep multivariate aeon classifiers.

set -eo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
config_file="${script_dir}/multiverse_resample0_extra_cpu.toml"
required_branch="ajb/hc2"

activate_environment() {
    source /etc/profile
    unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER PYTHONPATH
    module purge
    module load python/anaconda/2024.10/3.12.7
    source /gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh
    conda activate tsml-eval
}

if [[ -n "${1:-}" ]]; then
    echo "ERROR: unknown option: ${1}" >&2
    exit 1
fi

for command_name in git python pkill scancel squeue; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

actual_branch=$(git -C "$repo_dir" branch --show-current)
if [[ "$actual_branch" != "$required_branch" ]]; then
    echo "ERROR: CPU jobs must run from ${required_branch}; found ${actual_branch:-DETACHED}." >&2
    exit 1
fi

echo "Stopping known CPU Multiverse queue feeders on this login node."
cpu_configs=(
    multiverse_controller.toml
    multiverse_interval_32gb.toml
    multiverse_core_resample0_non_deep.toml
    multiverse_full_resample0_cpu_32gb.toml
    multiverse_full_resample0_cpu_completion.toml
    multiverse_paper_30resamples_cpu.toml
    multiverse_resample0_extra_cpu.toml
)
for config_name in "${cpu_configs[@]}"; do
    pkill -TERM -f "[r]un_multiverse_controller.sh.*${config_name}" || true
    pkill -TERM -f "[m]ultiverse_controller.py.*${config_name}" || true
done

pending_output=$(
    squeue --noheader --array --user="$USER" --partition=compute \
        --states=PENDING --format='%i'
)
mapfile -t pending_ids < <(
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d' \
        <<< "$pending_output"
)
if ((${#pending_ids[@]})); then
    echo "Cancelling ${#pending_ids[@]} pending compute tasks before submitting the focused pass."
    scancel "${pending_ids[@]}"
else
    echo "No pending compute tasks to cancel."
fi

activate_environment
cd "$repo_dir"

echo "Checking the missing extra CPU work."
python -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" \
    --dry-run \
    --no-email

echo "Submitting one controller cycle for MUSE and REDCOMETS."
python -u "${script_dir}/multiverse_controller.py" \
    --config "$config_file" \
    --no-email

echo
echo "Current compute queue:"
squeue -u "$USER" -p compute
