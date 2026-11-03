#!/bin/bash
# Make a conda env TensorFlow-only: remove PyTorch and its CUDA 13 libraries, then
# reinstall TensorFlow's CUDA 12 libraries in case the two stacks overwrote shared
# files. Two CUDA stacks in one env stop TensorFlow loading its GPU libraries.
#
#   bash make_tensorflow_only.sh ~/.conda/envs/tsml-eval-gpu
#
# from this directory (_tsml_research_resources/slurm).
#
# Afterwards `pip check` reports that braindecode and aeon-neuro need torch. That does
# not matter for aeon's TensorFlow deep learners; run aeon-neuro's PyTorch models from
# a separate env.
set -euo pipefail

env_dir="${1:?usage: bash make_tensorflow_only.sh <conda env directory>}"
py="$env_dir/bin/python"
if [[ ! -x "$py" ]]; then
    echo "No python at $py" >&2
    exit 1
fi

installed() {
    "$py" -m pip list --format=freeze 2>/dev/null | cut -d= -f1
}

# torch and friends, every -cu13 package, and the unsuffixed nvidia packages at
# version 13 (CUDA 13's cuBLAS, runtime and so on). nvidia-ml-py is not CUDA.
remove=$(
    "$py" -m pip list --format=freeze 2>/dev/null | awk -F'==' '
        tolower($1) ~ /^torch/ { print $1; next }
        tolower($1) ~ /^nvidia-.*-cu13$/ { print $1; next }
        tolower($1) ~ /^nvidia-/ && tolower($1) !~ /-cu12$/ && $1 != "nvidia-ml-py" { print $1 }
    '
)
if [[ -n "$remove" ]]; then
    echo "Removing:"; echo "$remove" | sed 's/^/  /'
    # shellcheck disable=SC2086
    "$py" -m pip uninstall -y $remove
else
    echo "Nothing to remove."
fi

cu12=$(installed | grep -i -E '^nvidia-.*-cu12$' || true)
if [[ -n "$cu12" ]]; then
    echo "Reinstalling TensorFlow's CUDA 12 libraries:"; echo "$cu12" | sed 's/^/  /'
    # shellcheck disable=SC2086
    "$py" -m pip install --force-reinstall --no-deps $cu12
else
    echo "No CUDA 12 libraries found: installing tensorflow[and-cuda]"
    tf=$("$py" -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || true)
    "$py" -m pip install "tensorflow[and-cuda]${tf:+==$tf}"
fi

echo
echo "Now in the env:"
"$py" -m pip list 2>/dev/null | grep -i -E '^(tensorflow|torch|nvidia)' || true
echo
echo "Test on a GPU node (one line):"
echo "  srun -p i7_h200 --nodes=1 --gres=gpu:h200:1 --time=00:10:00 --pty $py -c \"import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))\""
