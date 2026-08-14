#!/bin/bash
# Submit and run one H-InceptionTime experiment in a correctly sized Hali GPU
# allocation. Run this script from a Hali login node; no Conda activation is
# required beforehand.

set -eo pipefail

if [[ "${1:-}" != "--inside-allocation" ]]; then
    script_path=$(realpath "$0")
    dataset=${1:-STEW}
    resample=${2:-0}

    exec srun \
        --account=cmp \
        --partition=gpu \
        --qos=gpu \
        --gres=gpu:1 \
        --cpus-per-task=2 \
        --mem=64G \
        --time=0-04:00:00 \
        --job-name="HI_test_${dataset}" \
        bash "$script_path" --inside-allocation "$dataset" "$resample"
fi

dataset=${2:-STEW}
resample=${3:-0}
username=${USER:?USER is not set}
repo_dir="/gpfs/home/${username}/Code/tsml-eval-gpu"
data_dir="/gpfs/home/${username}/Data/Multiverse"
results_dir="/gpfs/home/${username}/Results/Multiverse/DeepLearning"
env_name=tsml-eval-gpu
env_dir="/gpfs/home/${username}/.conda/envs/${env_name}"

source /etc/profile
set -u
module purge
module load python/anaconda/2024.10/3.12.7
source /gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh
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
echo "Python:     $(command -v python)"
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
