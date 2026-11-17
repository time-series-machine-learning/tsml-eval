#!/bin/bash
# Make a conda env TensorFlow-only: remove PyTorch and its CUDA 13 libraries, then
# restore the CUDA 12 libraries TensorFlow asks for. Two CUDA stacks in one env stop
# TensorFlow loading its GPU libraries.
#
#   bash make_tensorflow_only.sh ~/.conda/envs/tsml-eval-gpu
#
# from this directory (_tsml_research_resources/slurm). Safe to run again.
#
# Afterwards `pip check` reports that braindecode and aeon-neuro need torch. That does
# not matter for aeon's TensorFlow deep learners; run aeon-neuro's PyTorch models from
# a separate env.
set -uo pipefail

env_dir="${1:?usage: bash make_tensorflow_only.sh <conda env directory>}"
py="$env_dir/bin/python"
if [[ ! -x "$py" ]]; then
    echo "No python at $py" >&2
    exit 1
fi

tf=$("$py" -c "import importlib.metadata as m; print(m.version('tensorflow'))" 2>/dev/null)
if [[ -z "$tf" ]]; then
    echo "TensorFlow is not installed in $env_dir" >&2
    exit 1
fi
echo "TensorFlow $tf in $env_dir"

# torch and friends, every -cu13 package, and the unsuffixed nvidia packages (CUDA
# 13's cuBLAS, runtime and so on). nvidia-ml-py is not CUDA.
remove=$(
    "$py" -m pip list --format=freeze 2>/dev/null | awk -F'==' '
        tolower($1) ~ /^torch/ { print $1; next }
        tolower($1) ~ /^nvidia-.*-cu13$/ { print $1; next }
        tolower($1) ~ /^nvidia-/ && tolower($1) !~ /-cu12$/ && $1 != "nvidia-ml-py" { print $1 }
    '
)
for package in $remove; do
    echo "Removing $package"
    "$py" -m pip uninstall -y -q "$package" || echo "  could not remove $package"
done

# Reinstall TensorFlow's CUDA 12 libraries at the versions it requires: shared files
# may have been overwritten by the CUDA 13 packages, or removed with them.
echo "Restoring TensorFlow's CUDA 12 libraries"
"$py" -m pip install -q "tensorflow[and-cuda]==$tf"
cu12=$("$py" -m pip list --format=freeze 2>/dev/null | grep -i -E '^nvidia-.*-cu12==' || true)
if [[ -n "$cu12" ]]; then
    # shellcheck disable=SC2086
    "$py" -m pip install -q --force-reinstall --no-deps $cu12
fi

echo
echo "Now in the env:"
"$py" -m pip list 2>/dev/null | grep -i -E '^(tensorflow|torch|nvidia)' || true
left=$("$py" -m pip list --format=freeze 2>/dev/null | awk -F'==' '
    tolower($1) ~ /^torch/ || tolower($1) ~ /^nvidia-.*-cu13$/ ||
    (tolower($1) ~ /^nvidia-/ && tolower($1) !~ /-cu12$/ && $1 != "nvidia-ml-py") { print $1 }')
if [[ -n "$left" ]]; then
    echo
    echo "Still installed, remove by hand: $left"
fi
echo
echo "Test on a GPU node (one line):"
echo "  srun -p i7_h200 --nodes=1 --gres=gpu:h200:1 --time=00:10:00 --pty $py -c \"import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))\""
