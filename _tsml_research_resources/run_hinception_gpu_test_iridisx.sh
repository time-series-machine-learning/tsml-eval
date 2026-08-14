#!/bin/bash
# Submit and run one H-InceptionTime experiment in a correctly sized IridisX GPU
# allocation. Run this script from an IridisX login node; no Conda activation is
# required beforehand.
#
# This mirrors run_hinception_gpu_test.sh, which is the same check on Hali. Keep the
# two in step. IridisX differences: no account or QoS, /home rather than /gpfs, the
# conda/python3 module, and the swarm_a100 partition whose gres type is a100swarm.

set -eo pipefail

if [[ "${1:-}" != "--inside-allocation" ]]; then
    script_path=$(realpath "$0")
    dataset=${1:-AtrialFibrillation}
    resample=${2:-0}

    exec srun \
        --partition=swarm_a100 \
        --gres=gpu:a100swarm:1 \
        --cpus-per-task=2 \
        --mem=64G \
        --time=0-04:00:00 \
        --nodes=1 \
        --job-name="HI_test_${dataset}" \
        bash "$script_path" --inside-allocation "$dataset" "$resample"
fi

dataset=${2:-AtrialFibrillation}
resample=${3:-0}
username=${USER:?USER is not set}
repo_dir="/home/${username}/Code/tsml-eval-gpu"
data_dir="/home/${username}/Data/Multiverse"
# A dedicated tree, not the paper results. If a paper result already existed the
# experiment would skip training and the check would pass without testing anything,
# and a check that did train would pollute completeness reporting
results_dir="/home/${username}/Results/GPUTest/DeepLearning"
env_name=tsml-eval-gpu

env_dir="/home/${username}/.conda/envs/${env_name}"

source /etc/profile
set -u

# Mirrors the Hali smoke test. Conda state inherited from the submitting shell is
# dropped, the environment is activated by absolute path rather than by name, and the
# interpreter is verified. A module reload can otherwise leave a stale
# CONDA_DEFAULT_ENV behind, making the activation a no-op and the job run base Python.
unset CONDA_DEFAULT_ENV PYTHONPATH
module purge
module load conda/python3

# Derived rather than hardcoded, as the IridisX conda.sh location is not pinned down.
# The resolved path is what the controller TOMLs would need for an explicit conda_sh
conda_sh="$(dirname "$(dirname "$(command -v conda)")")/etc/profile.d/conda.sh"
if [[ ! -f "$conda_sh" ]]; then
    echo "ERROR: conda.sh not found at $conda_sh" >&2
    exit 1
fi
echo "conda.sh:   $conda_sh"
source "$conda_sh"
if (( ${CONDA_SHLVL:-0} > 0 )); then
    conda deactivate
fi
conda activate "$env_dir"

if [[ "$(command -v python)" != "${env_dir}/bin/python" ]]; then
    echo "ERROR: the GPU Conda environment did not activate correctly." >&2
    echo "Expected Python: ${env_dir}/bin/python" >&2
    echo "Actual Python:   $(command -v python)" >&2
    exit 1
fi

cuda_lib_dirs=$(find "$CONDA_PREFIX/lib" -type d -path '*/site-packages/nvidia/*/lib' -print | paste -sd:)
if [[ -z "$cuda_lib_dirs" ]]; then
    echo "ERROR: pip-installed CUDA libraries were not found in $CONDA_PREFIX." >&2
    exit 1
fi
export LD_LIBRARY_PATH="${cuda_lib_dirs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

ptxas_path=$(find "$CONDA_PREFIX/lib" -type f -path '*/site-packages/nvidia/*/bin/ptxas' -print -quit)
if [[ -n "$ptxas_path" ]]; then
    export PATH="$(dirname "$ptxas_path"):$PATH"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export PYTHONUNBUFFERED=1

cd "$repo_dir"

echo "Host:       $(hostname)"
echo "Slurm job:  ${SLURM_JOB_ID:-unknown}"
echo "Repository: $repo_dir"
echo "Dataset:    $dataset"
echo "Resample:   $resample"
nvidia-smi -L

python - <<'PY'
import tensorflow as tf

devices = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("GPUs:", devices)
if not devices:
    raise SystemExit("ERROR: TensorFlow cannot see the allocated GPU.")
PY

python -u -m tsml_eval.experiments.classification_experiments \
    "$data_dir" \
    "$results_dir" \
    H-InceptionTime \
    "$dataset" \
    "$resample"

result_file="${results_dir}/H-InceptionTime/Predictions/${dataset}/testResample${resample}.csv"
if [[ ! -s "$result_file" ]]; then
    echo "ERROR: expected result was not created: $result_file" >&2
    exit 1
fi

echo "Completed successfully: $result_file"
