The basic scenario is that I have a set of results file (not collated) on the cluster
and I want to run another experiment to compare results. This file describes my
specific workflow and is mainly for my reference so I dont forget, but it can be
adapted to other needs. It is not optimal! It also reproduces some material in
ada_python

Some directories for reference in the workflow:

Results directory: <root>ResultsWorkingArea/STC_COMPARE
Estimators to compare against: REF, MAIN
New results: STC


From scratch:
logon to ada.uea.ac.uk


1. Run experiments with tsml-eval using a specific branch of aeon
2. Collate results on the cluster

Whilst on UEA network/VPN
Files required
DATA:

connect to ada.uea.ac.uk:
set up code and enviroment
> interactive
> git clone https://github.com/time-series-machine-learning/tsml-eval
> source /gpfs/software/ada/python/anaconda/2019.10/3.7/etc/profile.d/conda.sh
> conda create -n tsml-evaL python=3.10
> conda activate tsml-eval
> cd tsml-eval
> pip install -e .[all_extras]
> conda deactivate tsml-eval

Set up script (best to copy it to avoid conflicts)
> cd _tsm_group_experiments/uea/ada
> cp classification_experiments.sh ../../../../

# Set results path
results_dir=$local_path"ResultsWorkingArea/STC_COMPARE/"
out_dir=$local_path"ResultsWorkingArea/STC_COMPARE/output/"

Edit file to change
 # No train files
generate_train_files=""
 # Choose classifiers
for classifier in STC

> sh classification_experiments.sh

To change the branch for aeon
> interactive
> source /gpfs/software/ada/python/anaconda/2019.10/3.7/etc/profile.d/conda.sh
> conda activate tsml-eval
> pip uninstall aeon
> pip install git+https://github.com/aeon-toolkit/aeon.git@ajb/stc
> conda deactivate
> sh classification_experiments.sh

To collate results: edit collate results to match those you want then

> interactive
> source /gpfs/software/ada/python/anaconda/2019.10/3.7/etc/profile.d/conda.sh
> conda activate tsml-eval
> python  python tsml-eval/tsml_eval/_wip/experiments/collate_results.py
