# CMP CPU Server

_aka_ the Kraken

Unlike ADA, the CMP CPU server is not a cluster, but a single server with many cores and a lot of memory. There are much fewer restrictions on what can be run regarding runtime and memory. The server is still a shared resource and should be treated as such, however.

Server address: cmpcpusvr.uea.ac.uk

## Windows interaction with the Kraken

You need to be on a UEA network machine or have the VPN running to connect to the Kraken. Connect to cmpcpusvr.uea.ac.uk.

The recommended way of connecting to the Kraken is using Putty as a command-line interface and WinSCP for file management.

The tsml research group has a shared storage space on the Kraken under /storage/tsml/ where you can store and read datasets from.

## Installing on the Kraken

Complete these steps sequentially for a fresh install.

### 1. Clone the code from GitHub

The default location for files should be your user area. Either copy over the code files you want to run manually or clone them from a GitHub page.

>git clone GITHUB_LINK

e.g. https://github.com/time-series-machine-learning/tsml-eval

### 2. Create a conda environment

Create a new environment with a name of your choice. Replace PYTHON_VERSION with 3.10 for CPU jobs and 3.8 for GPU jobs.

>conda create -n ENV_NAME python=PYTHON_VERSION

Activate the new environment.

>conda activate ENV_NAME

Your environment should be listed now when you use the following command:

>conda info --envs

### 3. Install package and dependencies

Move to the package directory and run:

>pip install --editable .

For release specific dependency versions you can also run:

>pip install -r requirements.txt

Extras may be required, install as needed i.e.:

>pip install esig tsfresh

After installation, the installed packages can be viewed with:

>pip list

>conda list

## Running tsml-eval on the Kraken

Make sure the correct conda environment is activated prior to running any code.

> conda activate ENV_NAME

Python code can be run directly from the command line:

> python PYTHON_FILE

i.e. to run DrCIF on ItalyPowerDemand resample 0 with 6 threads, something similar to the following would be run:

> python Code/tsml_eval/experiments/threaded_classification_experiments.py tsml/TSCProblems2018TS results/ DrCIF ItalyPowerDemand 0 6

To run many single threaded experiments in parallel you can use the GNU parallel tool.

organisation:
code in Code directory
have a file for each input parameter
Code/ClassificationInputFiles
Code/ClusteringInputFiles

> parallel --delimiter "\n" --verbose --jobs 90% --memfree 30G --arg-file SubmissionFiles/data_dir.txt --arg-file SubmissionFiles/results_dir.txt --arg-file SubmissionFiles/classifiers.txt --arg-file SubmissionFiles/datasets.txt --arg-file SubmissionFiles/resamples.txt --arg-file SubmissionFiles/generate_train_files.txt --arg-file SubmissionFiles/predefined_folds.txt python Code/tsml_eval/experiments/classification_experiments.py > output.txt

More information on the parallel command can be found at https://www.gnu.org/software/parallel/.

## Monitoring jobs on the Kraken

To list current resource usage and processes:

> htop

To view the processes for a single user:

> htop -u USERNAME

To kill all processes for a single user:

> pkill -u USERNAME
