# Iridis 5 Python
##### Last updated: 07/09/2025

Installation guide for Python packages on Iridis 5 and useful slurm commands.

The [Iridis wiki](https://sotonac.sharepoint.com/teams/HPCCommunityWiki) provides a lot of useful information and getting started guides for using Iridis.

Server address: iridis5.soton.ac.uk

Alternatively, you can connect to one of the specific login nodes:
- iridis5_a.soton.ac.uk (usually busy)
- iridis5_b.soton.ac.uk
- iridis5_c.soton.ac.uk
- iridis5_d.soton.ac.uk (AMD CPU architecture)

This guide only covers Iridis 5 currently, but should be applicable to Iridis 6 or X.

There is a Southampton Microsoft Teams group called "HPC Community" where you can ask questions if needed.

## Windows interaction with Iridis

You need to be on a Soton network machine or have the VPN running to connect to Iridis. Connect to one of the addresses listed above.

The recommended way of connecting to Iridis in our group is using Putty for a SSH command-line interface and WinSCP for FTP file management.

Copies of data files used in experiments must be stored on the cluster, the best place to put these files is on your user area scratch storage. It is a good idea to create shortcuts to and from your scratch drive. Alternatively, you can read from someone else's directory (i.e. `/mainfs/scratch/mbm1g23/`).

## Installing on the cluster

Complete these steps sequentially for a fresh installation.

By default, commands will be run on the login node. Beyond simple commands or scripts, an interactive session should be started. For most usage (including setting up and running submission scripts as guided here) you will not require one.

>sinteractive

__DO NOT__ enter interactive mode for commands that require an internet connection (i.e. steps 1-4), as it will not work.

### 1. Clone the code from GitHub

The default location for files should be your user area. Either copy over the code files you want to run manually or clone them from a GitHub page.

>git clone GITHUB_LINK

e.g. https://github.com/time-series-machine-learning/tsml-eval

### 2. Activate an Iridis Python installation

Python is activated by default, but it is good practice to manually select the version used. The Iridis module should be loaded before creating and editing an environment.

>module load anaconda/py3.10

This will be different on each cluster. Iridis 6 is `conda/python3` for example.

You may also need to run the following to use some conda commands:

>conda init bash

You can check the current version using:

>conda --version

>python --version

### 3. Create a conda environment

#### 3.1. Set up scratch symbolic link for conda

Installing complex conda packages on your main home drive will quickly see you hitting the limit on the number of files you can store. To avoid this, it is recommended you create a symbolic link to your scratch storage (this assumes you are in your home directory and there is a symlink to your scratch drive).

>mkdir /scratch/<username>/.conda

>ln -s /scratch/<username>/.conda ~/.conda

Hitting this limit is very annoying, as it will prevent you from creating new conda environments, installing new packages or saving results file (or doing anything really).

For conda related storage guidance, see the [related HPC webpage](https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Conda.aspx#conda-and-inodes).

#### 3.2. Create environment

Create a new environment with a name of your choice. Replace PYTHON_VERSION with 3.12 by default.

>conda create -n ENV_NAME python=PYTHON_VERSION

Activate the new environment.

>conda activate ENV_NAME

Your environment should be listed now when you use the following command:

>conda info --envs

#### 3.3. Removing an environment (not required for first setup)

At some point you may want to remove an environment, either because it is no longer
needed or you want to start fresh. You can do this with the following command:

>conda remove -n ENV_NAME --all

### 4. Install package and dependencies

Install the package and required dependencies. The following are examples for a few packages and scenarios.

After installation, the installed packages can be viewed with:

>pip list

>conda list

Note that you can install a specific GitHub branch for packages such as `aeon` like so. It is important to uninstall any existing version first.

>pip uninstall aeon

>pip install git+https://github.com/aeon-toolkit/aeon.git@main

or

> pip install git+https://github.com/aeon-toolkit/aeon.git@ajb/stc

#### 4.1. tsml-eval CPU

Move to the package directory i.e.

>cd tsml-eval

This will have a `pyproject.toml` file. Run:

>pip install --editable .

For release specific dependency versions you can also run (replace `requirements.txt` with the relevant file):

>pip install -r requirements.txt

Extras may be required, install as needed i.e.:

>pip install esig tsfresh

For some extras you may need a gcc installation i.e.:

>module add gcc/11.1.0

Most extra dependencies can be installed with the `all_extras` dependency set:

>pip install -e .[all_extras]

Some dependencies are unstable, so the following may fail to install.

>pip install -e .[all_extras,unstable_extras]

If any a dependency install is "Killed", it is likely the session has run out of memory. Either give it more memory, or use a non-cached package i.e.

>pip install PACKAGE_NAME --no-cache-dir

#### 5.1. tsml-eval GPU

Currently the recommended way to run GPU jobs on Iridis is using an apptainer container built from an NVIDIA tensorflow docker image. Pulling the docker image will likely require an [NVIDIA NGC account](https://catalog.ngc.nvidia.com/) and API key.

>module load apptainer/1.3.3

>export APPTAINER_DOCKER_USERNAME='$oauthtoken'

>export APPTAINER_DOCKER_PASSWORD=PUT_YOUR_API_KEY_HERE

Pull the image you want, this can be image which has the necessary dependencies but was last tested with:

>apptainer pull docker://nvcr.io/nvidia/tensorflow:25.02-tf2-py3

Create a writable sandbox from the image. This is probably large with a lot of files so will be best on scratch:

>apptainer build --sandbox scratch/tensorflow_sandbox/ tensorflow_25.02-tf2-py3.sif

Open a shell in the container:

>apptainer shell --writable scratch/tensorflow_sandbox

Install `tsml-eval` like the above instructions, this does not have to be in the sandbox:

>cd tsml-eval

>pip install --editable .

## Running experiments

For running jobs on Iridis, we recommend using *copies* of the submission scripts provided in this folder.

**NOTE: Scripts will not run properly if done whilst the conda environment is active.**

Disable the conda environment before running scripts if you have installed packages:

>conda deactivate

### Running `tsml-eval` CPU experiments

For CPU experiments start with one of the following scripts:

>classification_experiments.sh

>regression_experiments.sh

>clustering_experiments.sh

These scripts can be run from the command line with the following command:

>sh classification_experiments.sh

You may need to use `dos2unix` to convert the line endings to unix format.

The default queue for CPU jobs is _batch_. Be sure to swap the _queue_alias_ to _serial_ in the script if you want to use this, as the number of jobs submitted won't be tracked properly otherwise.

Do not run threaded code on the cluster without requesting the correct amount of CPUs or reserving a whole node, as there is nothing to stop the job from using the CPU resources allocated to others. The default python file in the scripts attempts to avoid threading as much as possible. You should ensure processes are not intentionally using multiple threads if you change it.

Requesting memory for a job will allocate it all on the jobs assigned node. New jobs will not be submitted to a node if the total allocated memory exceeds the amount available for the node. As such, requesting too much memory can block new jobs from using the node. This is ok if the memory is actually being used, but large amounts of memory should not be requested unless you know it will be required for the jobs you are submitting. Iridis is a shared resource, and instantly requesting hundreds of GB will hurt the overall efficiency of the cluster.

### Running `tsml-eval` CPU experiments on the Iridis 5 batch queue

If you submit less than 20 tasks when requesting the _batch_ queue, your job will be redirected to the _serial_ queue. This has a much smaller job limit which you will reach quickly when submitting a lot of jobs. If you submit a single task in each submission, you will only be running ~32 jobs at once.

To get around this, you can use the batch submission scripts provided in the `batch_scripts` folder. These scripts submit multiple tasks in a single job, allowing you to run many more experiments at once.

>taskfarm_classification_experiments.sh

>taskfarm_regression_experiments.sh

>taskfarm_clustering_experiments.sh

They are named this as they use the `staskfarm` utility to run different processes over multiple threads. Read through the configuration as it is slightly different to the serial scripts. You can split task groupings by dataset by loading from a directory of submission scripts and keep classifiers separate with a variable.

### Running `tsml-eval` GPU experiments

For GPU experiments use one of the following scripts:

>gpu_classification_experiments.sh

>gpu_regression_experiments.sh

>gpu_clustering_experiments.sh

It is recommended you use different environments for CPU and GPU jobs. Using an apptainer container this will be standard, make sure to set the path to your sandbox in the script.

The default queue for GPU jobs is _gpu_.

## Monitoring jobs on Iridis

list processes for user (mind that the quotes may not be the correct ones)

>squeue -u USERNAME --format="%12i %15P %20j %10u %10t %10M %10D %20R" -r

__Tip__: to simplify and just use 'queue' in the terminal to run the above command, add this to the .bashrc file located in your home:

>alias queue='squeue -u USERNAME --format="%12i %15P %20j %10u %10t %10M %10D %20R" -r'

To kill all user jobs:

>scancel -u USERNAME

To delete all jobs on a queue it’s:

>scancel -p QUEUE

To delete one job it’s:

>scancel 11133013_1

To delete jobs in a specific job ID range use the `range_scancel` script:

>sh range_scancel.sh

## Helpful links

conda cheat sheet:
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

Queue names:
https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Iridis%205%20Job-submission-and-Limits-Quotas.aspx
