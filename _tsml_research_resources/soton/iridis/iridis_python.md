# Iridis Python

Installation guide for Python packages on Iridis and useful slurm commands.

The [Iridis wiki](https://sotonac.sharepoint.com/teams/HPCCommunityWiki) provides a lot of useful information and getting started guides for using Iridis.

Server address: iridis5.soton.ac.uk

Alternatively, you can connect to one of the specific login nodes:
- iridis5_a.soton.ac.uk (usually busy)
- iridis5_b.soton.ac.uk
- iridis5_c.soton.ac.uk
- iridis5_d.soton.ac.uk (AMD CPU architecture)

This guide only covers Iridis 5 cureently, not Iridis 6 or X.

There is a Southampton Microsoft Teams group called "HPC Community" where you can ask questions if needed.

## Windows interaction with Iridis

You need to be on a Soton network machine or have the VPN running to connect to Iridis. Connect to one of the addresses listed above.

The recommended way of connecting to Iridis is using Putty as a command-line interface and WinSCP for file management.

Copies of data files used in experiments must be stored on the cluster, the best place to put these files is on your user area scratch storage. It is a good idea to create shortcuts to and from your scratch drive. Alternatively, you can read from someone else's directory (i.e. `/mainfs/scratch/mbm1g23/`).

## Installing on the cluster

Complete these steps sequentially for a fresh installation.

By default, commands will be run on the login node. Beyond simple commands or scripts, an interactive session should be started. For most usage here you will not require one.

>sinteractive

__DO NOT__ enter interactive mode for commands that require an internet connection (i.e. steps 1-4), as only the login nodes have one.

### 1. Clone the code from GitHub

The default location for files should be your user area. Either copy over the code files you want to run manually or clone them from a GitHub page.

>git clone GITHUB_LINK

e.g. https://github.com/time-series-machine-learning/tsml-eval

### 2. Activate an Iridis Python installation

Python is activated by default, but it is good practice to manually select the version used. The Iridis module should be loaded before creating and editing an environment.

>module load anaconda/py3.10

You may also need to run the following to use some conda commands:

>conda init bash

You can check the current version using:

>conda --version

>python --version

### 3. Create a conda environment

#### 3.1. Set up scratch symbolic link for conda

Installing complex conda packages on your main home drive will quickly see you hitting the limit on the number of files you can store. To avoid this, it is recommended you create a symbolic link to your scratch storage.

>mkdir /scratch/<username>/.conda

>ln -s /scratch/<username>/.conda ~/.conda

Hitting this limit is very annoying, as it will prevent you from creating new conda environments or installing new packages (or doing anything really).

For conda related storage guidance, see the [related HPC webpage](https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Conda.aspx#conda-and-inodes)

### 3.2. Create environment

Create a new environment with a name of your choice. Replace PYTHON_VERSION with 3.11.

>conda create -n ENV_NAME python=PYTHON_VERSION

Activate the new environment.

>conda activate ENV_NAME

Your environment should be listed now when you use the following command:

>conda info --envs

### 4. Install package and dependencies

Install the package and required dependencies. The following are examples for a few packages and scenarios.

After installation, the installed packages can be viewed with:

>pip list

>conda list

Note that you can install a specific GitHub branch for packages such as `aeon` like so

>pip uninstall aeon

>pip install git+https://github.com/aeon-toolkit/aeon.git@main

or

> pip install git+https://github.com/aeon-toolkit/aeon.git@ajb/stc

#### 4.1. tsml-eval CPU

Move to the package directory and run:

>pip install --editable .

For release specific dependency versions you can also run:

>pip install -r requirements.txt

Extras may be required, install as needed i.e.:

>pip install esig tsfresh

For some extras you may need a gcc installation i.e.:

>module add gcc/11.1.0

Most extra dependencies can be installed with the all_extras dependency set:

>pip install -e .[all_extras]

Some dependencies are unstable, so the following may fail to install.

>pip install -e .[all_extras,unstable_extras]

If any a dependency install is "Killed", it is likely the interactive session has run out of memory. Either give it more memory, or use a non-cached package i.e.

>pip install PACKAGE_NAME --no-cache-dir

#### 5.1. tsml-eval GPU

It is recommended to use a different environment for GPU jobs. Move to the package directory and install the required packages for GPU jobs:

>pip install --editable . tensorflow[and-cuda] tensorrt

# Running experiments

For running jobs on Iridis, we recommend using copies of the submission scripts provided in this folder.

**NOTE: Scripts will not run properly if done whilst the conda environment is active.**

## Running `tsml-eval` CPU experiments

For CPU experiments start with one of the following scripts:

>classification_experiments.sh

>regression_experiments.sh

>clustering_experiments.sh

You may need to use `dos2unix` to convert the line endings to unix format.

The default queue for CPU jobs is _batch_. Be sure to swap the _queue_alias_ to _serial_ in the script if you want to use this, as the number of jobs submitted won't be tracked properly otherwise.

Do not run threaded code on the cluster without reserving whole nodes, as there is nothing to stop the job from using
the CPU resources allocated to others. The default python file in the scripts attempts to avoid threading as much as possible. You should ensure processes are not intentionally using multiple threads if you change it.

Requesting memory for a job will allocate it all on the jobs assigned node. New jobs will not be submitted to a node if the total allocated memory exceeds the amount available for the node. As such, requesting too much memory can block new jobs from using the node. This is ok if the memory is actually being used, but large amounts of memory should not be requested unless you know it will be required for the jobs you are submitting. Iridis is a shared resource, and instantly requesting hundreds of GB will hurt the overall efficiency of the cluster.

## Running `tsml-eval` GPU experiments

For GPU experiments use one of the following scripts:

>gpu_classification_experiments.sh

>gpu_regression_experiments.sh

>gpu_clustering_experiments.sh

It is recommended you use different environments for CPU and GPU jobs.

The default queue for GPU jobs is _gpu_.

## Monitoring jobs on Iridis

list processes for user (mind that the quotes may not be the correct ones)

>squeue -u USERNAME --format="%12i %15P %20j %10u %10t %10M %10D %20R" -r

__Tip__: to simplify and just use 'queue' in the terminal to run the above command, add this to the .bashrc file located in your home:

>alias queue='squeue -u USERNAME --format="%12i %15P %20j %10u %10t %10M %10D %20R" -r'

To kill all user jobs

>scancel -u USERNAME

To delete all jobs on a queue it’s:

>scancel -p QUEUE

To delete one job it’s:

>scancel 11133013_1

## Helpful links

conda cheat sheet:
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

Queue names:
https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Iridis%205%20Job-submission-and-Limits-Quotas.aspx
