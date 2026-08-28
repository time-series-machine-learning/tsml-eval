#!/bin/bash
# Run one ConvTran smoke test in an IridisX A100 allocation.

set -euo pipefail

partition=${PARTITION:-a100}
gres=${GRES:-gpu:a100:1}
account=${ACCOUNT:-}
qos=${QOS:-}

if [[ "${1:-}" != "--inside-allocation" ]]; then
    script_path=$(realpath "$0")
    dataset=${1:-AtrialFibrillation}
    resample=${2:-0}
    srun_options=(
        --partition="$partition"
        --gres="$gres"
        --cpus-per-task=2
        --mem=32G
        --time=0-02:00:00
        --nodes=1
        --job-name="ConvTran_test_${dataset}"
    )
    [[ -n "$account" ]] && srun_options+=(--account="$account")
    [[ -n "$qos" ]] && srun_options+=(--qos="$qos")
    echo "Requesting ${gres} on ${partition} for ${dataset} resample ${resample}."
    exec srun "${srun_options[@]}" \
        bash "$script_path" --inside-allocation "$dataset" "$resample"
fi

dataset=${2:-AtrialFibrillation}
resample=${3:-0}
username=${USER:?USER is not set}
repo_dir="/home/${username}/Code/tsml-eval-gpu"
data_dir="/home/${username}/Data/Multiverse"
results_dir="/home/${username}/Results/GPUTest/DeepLearning"
env_dir="/home/${username}/.conda/envs/tsml-eval-gpu"

# IridisX's ssh-x-forwarding profile hook dereferences DISPLAY under nounset.
export DISPLAY="${DISPLAY:-}"
source /etc/profile
unset CONDA_DEFAULT_ENV PYTHONPATH
module purge
module load conda/python3
conda_sh="$(dirname "$(dirname "$(command -v conda)")")/etc/profile.d/conda.sh"
if [[ ! -f "$conda_sh" ]]; then
    echo "ERROR: conda.sh not found at ${conda_sh}" >&2
    exit 1
fi
source "$conda_sh"
if (( ${CONDA_SHLVL:-0} > 0 )); then
    conda deactivate
fi
conda activate "$env_dir"
if [[ "$(command -v python)" != "${env_dir}/bin/python" ]]; then
    echo "ERROR: expected ${env_dir}/bin/python, got $(command -v python)" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$repo_dir"
if [[ "$(git branch --show-current)" != "ajb/gpu" ]]; then
    echo "ERROR: expected branch ajb/gpu, found $(git branch --show-current)." >&2
    exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    echo "ERROR: commit or discard repository changes before the smoke test." >&2
    exit 1
fi
echo "Host:       $(hostname)"
echo "Slurm job:  ${SLURM_JOB_ID:-unknown}"
echo "Python:     $(command -v python)"
echo "Repository: $repo_dir"
nvidia-smi -L

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see the allocated GPU.")
print("GPU:", torch.cuda.get_device_name(0))
PY

python -u -m tsml_eval.experiments.classification_experiments \
    "$data_dir" "$results_dir" ConvTran "$dataset" "$resample" \
    -ow -kw device cuda str -kw verbose true bool

result_file="${results_dir}/ConvTran/Predictions/${dataset}/testResample${resample}.csv"
if [[ ! -s "$result_file" ]]; then
    echo "ERROR: expected result was not created: ${result_file}" >&2
    exit 1
fi
echo "Completed successfully: ${result_file}"
