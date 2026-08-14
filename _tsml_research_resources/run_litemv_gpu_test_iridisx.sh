#!/bin/bash
# Run one LITETime-MV experiment in an IridisX GPU allocation, as an end to end check
# that the tsml-eval-gpu environment trains on the GPU. Run from an IridisX login
# node; no Conda activation is required beforehand.
#
#   sh run_litemv_gpu_test_iridisx.sh                 # BasicMotions, resample 0
#   sh run_litemv_gpu_test_iridisx.sh ArticularyWordRecognition 0
#
# Results go to Results/GPUTest so a throwaway check never lands in the real
# Multiverse or UCR result directories.
#
# LITETime-MV is the multivariate LITE variant, so the default problem is
# multivariate. The classifier name must be spelled LITETime-MV, which is the only
# alias _get_classifier.py accepts for it.

set -eo pipefail

partition=swarm_a100
gres=gpu:a100swarm:1

if [[ "${1:-}" != "--inside-allocation" ]]; then
    script_path=$(realpath "$0")
    dataset=${1:-BasicMotions}
    resample=${2:-0}

    echo "Requesting ${gres} on ${partition} for ${dataset} resample ${resample}."
    echo "If this queues for a long time, try partition=scavenger_4a100 instead."

    exec srun \
        --partition="${partition}" \
        --gres="${gres}" \
        --cpus-per-task=2 \
        --mem=32G \
        --time=0-01:00:00 \
        --nodes=1 \
        --job-name="LITEMV_test_${dataset}" \
        bash "$script_path" --inside-allocation "$dataset" "$resample"
fi

dataset=${2:-BasicMotions}
resample=${3:-0}
username=${USER:?USER is not set}
repo_dir="/home/${username}/Code/tsml-eval-gpu"
data_dir="/home/${username}/Data/Multiverse"
results_dir="/home/${username}/Results/GPUTest/DeepLearning"
env_name=tsml-eval-gpu

source /etc/profile
set -u
module purge
module load conda/python3

# Derive conda.sh from the loaded module rather than hardcoding it. The resolved path
# is what the controller TOMLs need for their conda_sh setting, so it is printed below
conda_root=$(dirname "$(dirname "$(command -v conda)")")
conda_sh="${conda_root}/etc/profile.d/conda.sh"
if [[ ! -f "$conda_sh" ]]; then
    echo "ERROR: conda.sh not found at ${conda_sh}" >&2
    echo "Locate it with: module load conda/python3 && echo \$CONDA_EXE" >&2
    exit 1
fi
echo "conda.sh:   $conda_sh   <- use this for conda_sh in the controller TOMLs"
source "$conda_sh"
conda activate "$env_name"

cuda_lib_dirs=$(find "$CONDA_PREFIX/lib" -type d -path '*/site-packages/nvidia/*/lib' -print | paste -sd:)
if [[ -z "$cuda_lib_dirs" ]]; then
    echo "ERROR: pip-installed CUDA libraries were not found in $CONDA_PREFIX." >&2
    echo "Install them with: pip install \"tensorflow[and-cuda]\"" >&2
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
echo "Repository: $repo_dir ($(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD))"
echo "Dataset:    $dataset"
echo "Resample:   $resample"
echo "ptxas:      ${ptxas_path:-not found}"
nvidia-smi -L

python - <<'PY'
import tensorflow as tf

devices = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("GPUs:", devices)
if not devices:
    raise SystemExit("ERROR: TensorFlow cannot see the allocated GPU.")
PY

# Confirm the classifier name resolves before spending an allocation on training
python - <<'PY'
from tsml_eval.experiments._get_classifier import get_classifier_by_name

classifier = get_classifier_by_name("LITETime-MV", random_state=0)
print("Classifier:", type(classifier).__name__, "use_litemv:", classifier.use_litemv)
PY

python -u -m tsml_eval.experiments.classification_experiments \
    "$data_dir" \
    "$results_dir" \
    LITETime-MV \
    "$dataset" \
    "$resample"

result_file="${results_dir}/LITETime-MV/Predictions/${dataset}/testResample${resample}.csv"
if [[ ! -s "$result_file" ]]; then
    echo "ERROR: expected result was not created: $result_file" >&2
    exit 1
fi

echo "Completed successfully: $result_file"
head -3 "$result_file"
