# IridisX Python

Setting up the `tsml-eval-gpu` conda environment on IridisX.

This follows `../iridis/iridis_python.md`, which covers Iridis 6. Read that first for
the general installation and Slurm guidance. This file only covers what is different
for a GPU environment on IridisX.

**Settings that have not been confirmed on IridisX are marked TODO-IRIDISX. Run
`iridisx_probe.sh` on the login node first.**

Server address: `loginX003.iridis.soton.ac.uk`. `loginX001` also has GPUs (4x L4
24GB), `loginX002` is CPU-only. `/home`, `/scratch` and the module shares are global,
so it does not matter which you use for setup.

You need to be on a Soton network machine or the VPN to connect.

**Do not run experiments on the login nodes**, it can crash them. The GPUs on
`loginX001`/`loginX003` are for short checks like the TensorFlow device test below,
not for training.

## Why a separate environment

The CPU environment is called `tsml-eval` and the GPU environment `tsml-eval-gpu`.
Keeping them separate avoids a GPU-enabled TensorFlow being pulled into CPU jobs,
where it wastes memory and start-up time, and lets the GPU environment pin whatever
TensorFlow/CUDA combination IridisX actually supports without disturbing CPU runs.

The submission scripts in `gpu_scripts/` activate `tsml-eval-gpu`.

## 1. Conda storage on scratch

Conda environments hold a large number of small files and will hit the home-drive
inode limit. Symlink `.conda` to scratch before creating anything:

>mkdir -p ~/scratch/.conda

>ln -s ~/scratch/.conda ~/.conda

## 2. Create the environment

>module load conda/python3   # TODO-IRIDISX: confirm the module name

>conda init bash

>conda create -n tsml-eval-gpu python=3.13

>conda activate tsml-eval-gpu

Do not do this in an interactive session, the installation steps need internet access.

## 3. Install a GPU TensorFlow

TODO-IRIDISX: confirm from the probe output which CUDA/cuDNN modules exist, and
whether IridisX expects you to load them or to let pip install the bundled CUDA
libraries.

The self-contained pip route avoids depending on the module stack:

>pip install "tensorflow[and-cuda]"

If IridisX provides its own CUDA modules and prefers you use them, load them in the
job script instead by adding a `module load` line next to the conda module.

Verify the environment sees a GPU. This one check is short enough to run on
`loginX001` or `loginX003`, which have L4 GPUs:

>python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

This must print a non-empty list. If it prints `[]`, the environment is running on
CPU and there is no point submitting jobs.

Note that the login node L4s are a different architecture to the A100s the jobs will
actually run on, so also confirm on a real allocation before an archive run, either
with an interactive job or by reading the `nvidia-smi` output in the first job's
`.out` file.

## 4. Install tsml-eval

>cd tsml-eval

>pip install --editable .

`aeon` deep learners come from the `aeon` dependency. To use a development branch:

>pip uninstall aeon

>pip install git+https://github.com/aeon-toolkit/aeon.git@main

## 5. Before running scripts

**Scripts will not run properly whilst the conda environment is active.**

>conda deactivate

Then submit as normal:

>sh gpu_classification_experiments_ucr.sh
