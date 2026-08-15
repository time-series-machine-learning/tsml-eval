# IridisX Python

Setting up the `tsml-eval-gpu` conda environment on IridisX.

This follows `../iridis/iridis_python.md`, which covers Iridis 6. Read that first for
the general installation and Slurm guidance. This file only covers what is different
for a GPU environment on IridisX.

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

>mkdir -p /scratch/$USER/.conda

>ln -s /scratch/$USER/.conda ~/.conda

## 2. Create the environment

Either build a fresh environment:

>module load conda/python3

>conda init bash

>conda create -n tsml-eval-gpu python=3.13

>conda activate tsml-eval-gpu

or clone the existing CPU environment, which keeps the numpy/numba/aeon versions the
CPU results were produced with:

>conda create --name tsml-eval-gpu --clone tsml-eval

**A clone inherits the CPU environment's editable install**, which points at the CPU
checkout. If the GPU work runs off a different branch, see section 4.

Do not do this in an interactive session, the installation steps need internet access.

## 3. Install a GPU TensorFlow

IridisX provides CUDA modules from `cuda/11.8.0` up to `cuda/13.3.1`, but the
self-contained pip route is simpler and avoids matching TensorFlow to a module
version:

>pip uninstall -y tensorflow tensorflow-cpu

>pip install "tensorflow[and-cuda]"

There is no separate cuDNN module, which is a further reason to prefer the pip route.
If you do want the module stack instead, add a `module load cuda/12.8.0` (or whichever
version matches your TensorFlow build) next to the conda module in the job script.

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

The GPU jobs run off their own checkout so the branch can differ from the CPU runs:

>git clone https://github.com/time-series-machine-learning/tsml-eval.git ~/Code/tsml-eval-gpu

>cd ~/Code/tsml-eval-gpu

>git switch ajb/gpu

>pip uninstall -y tsml-eval

>pip install --editable .

The `pip uninstall` matters when the environment was cloned from the CPU one, as it
drops the inherited pointer to the CPU checkout. Confirm which code the environment
resolves to:

>python -c "import tsml_eval; print(tsml_eval.__file__)"

This must print a path under `~/Code/tsml-eval-gpu`. The `script_file_path` variable
in the submission scripts must point at the same checkout, otherwise the script that
runs and the package it imports come from different branches.

## 5. Install a pinned aeon

**The GPU environment installs `aeon` from PyPI at a pinned version, unlike the CPU
environment which uses an editable checkout at `~/Code/aeon`.**

Install the same version Hali's GPU environment uses, so results from the two
clusters are comparable. Check it first:

>ssh hali "conda activate tsml-eval-gpu && python -c 'import aeon; print(aeon.__version__)'"

Then, substituting that version:

>pip uninstall -y aeon

>pip install "aeon==1.5.0"

>python -c "import aeon; print(aeon.__file__, aeon.__version__)"

The path must be inside the environment, not under `~/Code`.

`tsml-eval` requires `aeon>=1.0.0,<1.6.0`. 1.5.0, released 29/06/2026, is the newest
release inside that bound and is the right default for a fresh setup. Deep learner
internals change between minor versions, so if Hali already holds results on an
earlier version, match that instead of taking the newest.

The editable checkout exists so aeon branches can be switched for CPU work. The GPU
passes do not track a branch, so they gain nothing from it and inherit two problems:

- A cloned environment points at the shared checkout, so whatever branch the CPU work
  left it on is what GPU jobs import. This produced:

      File ".../Code/tsml-eval-gpu/tsml_eval/experiments/__init__.py", line 22
        from tsml_eval.experiments._get_clusterer import get_clusterer_by_name
      File ".../Code/aeon/aeon/base/_base.py", line 46
        from aeon.utils.validation._dependencies import _check_estimator_deps
      ModuleNotFoundError: No module named 'aeon.utils.validation'

  Any `tsml_eval.experiments` import reaches aeon through `_get_clusterer`, so a
  half-switched aeon tree breaks every experiment, not only the clustering ones.
- The controller pins the `tsml-eval` commit and refuses to run if the repository
  moves under a submitted job, but there is no equivalent guard for aeon. A shared
  checkout means switching an aeon branch silently changes the algorithm inside jobs
  that are already queued.

Pin an exact version rather than a range, so a reinstall months later cannot silently
move the algorithm code underneath a part-finished set of results.

## 6. Before running scripts

**Scripts will not run properly whilst the conda environment is active.**

>conda deactivate

Then submit as normal:

>sh gpu_classification_experiments_ucr.sh
