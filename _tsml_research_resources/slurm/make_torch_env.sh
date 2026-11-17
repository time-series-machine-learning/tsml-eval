#!/bin/bash
# Make the PyTorch env for aeon-neuro's deep learners (EEGNet, DeepConvNet,
# EEGConformer) on Iridis X. It is separate from tsml-eval-gpu, which is
# TensorFlow-only (make_tensorflow_only.sh): two CUDA stacks in one env break both.
#
#   bash make_torch_env.sh
#
# from this directory (_tsml_research_resources/slurm), on a login node. Creates
# ~/.conda/envs/tsml-eval-torch, or reuses it; safe to run again.
#
# torch comes from PyTorch's CUDA 12.6 wheel index, not PyPI, whose default wheels
# are built for a newer CUDA than the node driver may support. The CUDA libraries
# are pip packages inside the env; no CUDA module is needed.
set -euo pipefail
# keep pip to this env, ignoring packages installed with `pip install --user`
export PYTHONNOUSERSITE=1

env_name="${1:-tsml-eval-torch}"
cuda="${TORCH_CUDA:-cu126}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"

module load conda/python3 2>/dev/null || true
env_dir="$HOME/.conda/envs/$env_name"
if [[ ! -x "$env_dir/bin/python" ]]; then
    echo "Creating $env_dir"
    conda create -y -q -p "$env_dir" python=3.12
fi
py="$env_dir/bin/python"

"$py" -m pip install -q --upgrade pip
echo "Installing torch ($cuda)"
"$py" -m pip install -q torch --index-url "https://download.pytorch.org/whl/$cuda"
echo "Installing tsml-eval from $repo with aeon-neuro"
"$py" -m pip install -q -e "$repo[neuro]"
# Nothing is uninstalled: a package found in ~/.local is shared by every env of the
# same Python version, so removing it here would remove it from tsml-eval-gpu too.

echo
"$py" -m pip check || true
"$py" -m pip list 2>/dev/null | grep -i -E '^(torch|braindecode|aeon|nvidia)' || true
echo
echo "Test on a GPU node (one line), it should end with True:"
echo "  srun -p i7_h200 --nodes=1 --gres=gpu:h200:1 --time=00:10:00 --pty $py -c \"import torch, aeon_neuro; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\""
